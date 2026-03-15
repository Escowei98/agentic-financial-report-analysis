"""
Hybrid Retriever for System 1 (Monolith RAG).

Combines BM25 (sparse) and ChromaDB (dense) retrieval via weighted
Reciprocal Rank Fusion (RRF), followed by FlashRank reranking for final scoring.

Non-agentic — all components are stateless scoring/ranking models.
"""

import logging
from collections import defaultdict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Vector Store
# ---------------------------------------------------------------------------

def build_vectorstore(
    documents: list[Document],
    embeddings: Embeddings,
    persist_directory: str,
    collection_name: str = "sec_10k_filings",
) -> Chroma:
    """
    Create or load a ChromaDB vector store from documents.

    Args:
        documents: Chunked LangChain Documents.
        embeddings: Embedding model instance.
        persist_directory: Path for ChromaDB persistence.
        collection_name: ChromaDB collection name.

    Returns:
        ChromaDB vector store instance.
    """
    logger.info(
        "Building vectorstore: %d docs → %s/%s",
        len(documents), persist_directory, collection_name,
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    logger.info("Vectorstore built: %d documents indexed", len(documents))
    return vectorstore


# ---------------------------------------------------------------------------
#  BM25 Retriever (with score tracking)
# ---------------------------------------------------------------------------

class BM25RetrieverWithScores(BaseRetriever):
    """
    BM25 retriever that tracks relevance scores in document metadata.

    Wraps rank_bm25.BM25Okapi and exposes scores via
    doc.metadata["bm25_score"] for debugging and analysis.
    """

    documents: list[Document]
    """The full list of documents to search."""

    top_k: int = 20
    """Number of top documents to return."""

    # Private attributes (not part of pydantic schema)
    _bm25: BM25Okapi | None = None
    _tokenized_corpus: list[list[str]] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context) -> None:
        """Initialize BM25 index after pydantic model creation."""
        self._tokenized_corpus = [
            doc.page_content.lower().split() for doc in self.documents
        ]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info("BM25 index built: %d documents", len(self.documents))

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        """Retrieve top-k documents by BM25 score."""
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score (descending)
        top_indices = scores.argsort()[-self.top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return documents with non-zero score
                doc = self.documents[idx].model_copy(deep=True)
                doc.metadata["bm25_score"] = float(scores[idx])
                results.append(doc)

        logger.debug(
            "BM25 retrieved %d docs (top score: %.3f)",
            len(results), scores[top_indices[0]] if len(top_indices) > 0 else 0,
        )
        return results


# ---------------------------------------------------------------------------
#  Hybrid Retriever (BM25 + Dense) via Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

class HybridRetriever(BaseRetriever):
    """
    Weighted hybrid retriever combining BM25 and dense retrieval.

    Uses Reciprocal Rank Fusion (RRF) with configurable weights
    to merge results from both retrievers.
    """

    bm25_retriever: BM25RetrieverWithScores
    """BM25 sparse retriever."""

    dense_retriever: BaseRetriever
    """Dense (embedding-based) retriever."""

    bm25_weight: float = 0.3
    """Weight for BM25 results in fusion."""

    dense_weight: float = 0.7
    """Weight for dense results in fusion."""

    rrf_k: int = 60
    """RRF constant (standard value = 60)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        """
        Retrieve documents using weighted Reciprocal Rank Fusion.

        RRF score for doc d from retriever r:
            score_r(d) = weight_r / (rrf_k + rank_r(d))

        Final score: sum of weighted RRF scores across retrievers.
        """
        # Get results from both retrievers
        bm25_docs = self.bm25_retriever.invoke(query)
        dense_docs = self.dense_retriever.invoke(query)

        # Build fused scores using Reciprocal Rank Fusion
        doc_scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, Document] = {}

        # Score BM25 results
        for rank, doc in enumerate(bm25_docs):
            doc_key = doc.page_content[:200]  # Use content prefix as key
            doc_scores[doc_key] += self.bm25_weight / (self.rrf_k + rank + 1)
            doc_map[doc_key] = doc

        # Score dense results
        for rank, doc in enumerate(dense_docs):
            doc_key = doc.page_content[:200]
            doc_scores[doc_key] += self.dense_weight / (self.rrf_k + rank + 1)
            if doc_key not in doc_map:
                doc_map[doc_key] = doc

        # Sort by fused score (descending)
        sorted_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)

        results = []
        for key in sorted_keys:
            doc = doc_map[key]
            doc.metadata["rrf_score"] = doc_scores[key]
            results.append(doc)

        logger.debug(
            "Hybrid RRF: %d BM25 + %d dense → %d fused docs",
            len(bm25_docs), len(dense_docs), len(results),
        )
        return results


def build_hybrid_retriever(
    vectorstore: Chroma,
    documents: list[Document],
    bm25_weight: float = 0.3,
    pre_rerank_top_k: int = 20,
) -> HybridRetriever:
    """
    Build a HybridRetriever combining BM25 and dense retrieval.

    Args:
        vectorstore: ChromaDB vector store.
        documents: Full document list (for BM25 index).
        bm25_weight: Weight for BM25 results (dense_weight = 1 - bm25_weight).
        pre_rerank_top_k: Number of results each retriever returns before reranking.

    Returns:
        HybridRetriever combining both retrieval strategies via RRF.
    """
    dense_weight = round(1.0 - bm25_weight, 2)

    # BM25 retriever with score tracking
    bm25_retriever = BM25RetrieverWithScores(
        documents=documents,
        top_k=pre_rerank_top_k,
    )

    # Dense retriever from ChromaDB
    dense_retriever = vectorstore.as_retriever(
        search_kwargs={"k": pre_rerank_top_k},
    )

    # Combine with weighted RRF
    hybrid = HybridRetriever(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
    )

    logger.info(
        "Hybrid retriever built: bm25_weight=%.2f, dense_weight=%.2f, pre_rerank_k=%d",
        bm25_weight, dense_weight, pre_rerank_top_k,
    )
    return hybrid


# ---------------------------------------------------------------------------
#  FlashRank Reranker (local cross-encoder, no API key needed)
# ---------------------------------------------------------------------------

# Module-level singleton to avoid reloading the model on every call (~1-2s).
_flashrank_ranker = None


def _get_flashrank_ranker(model_name: str = "ms-marco-MiniLM-L-12-v2"):
    """Return a cached FlashRank Ranker instance (singleton)."""
    global _flashrank_ranker
    if _flashrank_ranker is None:
        from flashrank import Ranker
        _flashrank_ranker = Ranker(model_name=model_name)
        logger.info("FlashRank ranker loaded: %s", model_name)
    return _flashrank_ranker


def rerank_documents(
    query: str,
    documents: list[Document],
    top_n: int = 5,
    model: str = "ms-marco-MiniLM-L-12-v2",
) -> list[Document]:
    """
    Rerank documents using FlashRank (local cross-encoder).

    Non-agentic — stateless scoring model that sorts by relevance.
    Runs entirely locally, no API key required.

    Args:
        query: Search query.
        documents: Pre-retrieved documents to rerank.
        top_n: Number of top documents to return after reranking.
        model: FlashRank model name.

    Returns:
        Top-n documents sorted by rerank relevance score.
    """
    from flashrank import RerankRequest

    ranker = _get_flashrank_ranker(model)

    # Convert LangChain Documents → FlashRank passage dicts
    passages = [
        {"id": i, "text": doc.page_content}
        for i, doc in enumerate(documents)
    ]

    rerank_request = RerankRequest(query=query, passages=passages)
    ranked_results = ranker.rerank(rerank_request)

    # Map back to LangChain Documents, keeping top_n
    reranked_docs = []
    for result in ranked_results[:top_n]:
        original_idx = result["id"]
        doc = documents[original_idx].model_copy(deep=True)
        doc.metadata["rerank_score"] = result["score"]
        reranked_docs.append(doc)

    logger.debug(
        "Reranked %d → %d docs (model=%s)",
        len(documents), len(reranked_docs), model,
    )
    return reranked_docs


# ---------------------------------------------------------------------------
#  Full Retrieval Pipeline
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    retriever: HybridRetriever,
    post_rerank_top_k: int = 5,
    reranker_model: str = "ms-marco-MiniLM-L-12-v2",
    reranker_enabled: bool = True,
) -> list[Document]:
    """
    Full retrieval pipeline: Hybrid Retrieve → Rerank.

    Args:
        query: Search query.
        retriever: HybridRetriever instance.
        post_rerank_top_k: Final number of documents after reranking.
        reranker_model: FlashRank model for reranking.
        reranker_enabled: Whether to apply reranking.

    Returns:
        Final list of relevant documents.
    """
    # Step 1: Hybrid retrieval (BM25 + Dense via RRF)
    pre_rerank_docs = retriever.invoke(query)
    logger.info("Hybrid retrieval: %d documents", len(pre_rerank_docs))

    if not pre_rerank_docs:
        logger.warning("No documents retrieved for query: %s", query[:100])
        return []

    # Step 2: Rerank (optional)
    if reranker_enabled and len(pre_rerank_docs) > post_rerank_top_k:
        final_docs = rerank_documents(
            query=query,
            documents=pre_rerank_docs,
            top_n=post_rerank_top_k,
            model=reranker_model,
        )
    else:
        final_docs = pre_rerank_docs[:post_rerank_top_k]

    logger.info("Final retrieval: %d documents", len(final_docs))
    return final_docs
