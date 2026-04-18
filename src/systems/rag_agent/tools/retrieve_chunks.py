"""
Retrieve Chunks Tool for System 2 (Agent RAG).

Wraps the shared hybrid retrieval pipeline (BM25 + Dense + FlashRank rerank)
as a LangChain tool. Uses the SAME retrieval stack as System 1 (Monolith).
"""

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _format_chunks(docs: list[Document]) -> str:
    """Format retrieved documents into a readable string for the LLM."""
    if not docs:
        return "No relevant chunks found for this query."

    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        source = (
            f"{meta.get('ticker', '?')} FY{meta.get('fiscal_year', '?')} "
            f"— {meta.get('section_name', 'Unknown Section')}"
        )
        parts.append(
            f"[Chunk {i}] Source: {source}\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(parts)


def create_retrieve_chunks_tool(
    retriever: Any,
    retrieval_config: dict,
    reranker_config: dict,
) -> Any:
    """
    Factory that creates a retrieve_chunks tool bound to a specific retriever.

    Args:
        retriever: Pre-built HybridRetriever instance.
        retrieval_config: Retrieval settings (post_rerank_top_k, etc.).
        reranker_config: Reranker settings (model, enabled).

    Returns:
        A LangChain @tool function.
    """
    from src.common.retrieval import retrieve as retrieval_fn

    @tool
    def retrieve_chunks(query: str) -> str:
        """Search across all SEC 10-K filings using hybrid retrieval (BM25 + dense + reranking).

        Use this for general questions when you don't know the exact section or company.
        Returns the top-k most relevant text chunks with source metadata.

        Args:
            query: Natural language search query about SEC filings.
        """
        docs = retrieval_fn(
            query=query,
            retriever=retriever,
            post_rerank_top_k=retrieval_config.get("post_rerank_top_k", 5),
            reranker_model=reranker_config.get("model", "ms-marco-MiniLM-L-12-v2"),
            reranker_enabled=reranker_config.get("enabled", True),
        )

        logger.info(
            "retrieve_chunks: query=%r → %d docs", query[:80], len(docs),
        )
        return _format_chunks(docs)

    return retrieve_chunks
