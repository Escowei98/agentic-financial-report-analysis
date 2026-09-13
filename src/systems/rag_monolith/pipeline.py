"""
End-to-End RAG Pipeline for System 1 (Monolith RAG).

Orchestrates: Chunking → Vectorstore → Hybrid Retrieval → Reranking → Generation.
All components are non-agentic (no tool-use, no autonomous decisions).
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI

from src.common.answer_format_convention import ANSWER_FORMAT_CONVENTION
from src.common.few_shot_examples import FewShotScenario, render_few_shot_examples
from src.common.reasoning_chain_convention import REASONING_CHAIN_CONVENTION
from src.common.config import load_config
from src.common.ingestion import ProcessedFiling
from src.common.llm_client import get_embeddings, get_llm
from src.common.non_answerability_convention import NON_ANSWERABILITY_CONVENTION
from src.common.retrieval import (
    HybridRetriever,
    build_hybrid_retriever,
    build_vectorstore,
    load_or_build_documents,
    retrieve,
)
from src.common.utils import RunMetrics, TokenUsage, compute_filings_hash, extract_text

logger = logging.getLogger(__name__)

def _s1_steps(scenario: FewShotScenario) -> list[str]:
    """The monolith's idiom: no tools, read the prefixed context chunks."""
    if scenario.key == "single_fact":
        return ["Read the chunk(s) prefixed [AAPL, FY2024, Risk Factors] in the Context."]
    if scenario.key == "comparison":
        return [
            "Read the chunk prefixed [AAPL, FY2024, Financial Statements] -> total net sales = $391,035M.",
            "Read the chunk prefixed [MSFT, FY2024, Financial Statements] -> total revenue = $245,122M.",
            "Subtract the two figures: 391,035 - 245,122 = 145,913.",
        ]
    return [
        "Read the chunks prefixed [MSFT, FY2024, Financial Statements]; the income "
        "statement lists FY2024, FY2023 and FY2022 side by side (chunks from the "
        "FY2022 filing cover any year it does not show).",
        "Extract net income for each year and compute the cumulative change from "
        "the FY2022 and FY2024 figures.",
    ]


# Rendered once at import. Brace-free by construction (see few_shot_examples.py),
# which matters here and nowhere else: this string goes through
# ChatPromptTemplate, where a stray brace is a template variable.
FEW_SHOT_EXAMPLES = render_few_shot_examples(
    intro=(
        "These examples show the expected pattern for three common query "
        "types. Read the relevant chunks in the Context and answer from them."
    ),
    steps_for=_s1_steps,
)

# Standard RAG prompt — deterministic, no agentic behavior
RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a financial analyst assistant. Answer the user's question "
        "based ONLY on the provided context from SEC 10-K filings. "
        "If the context does not contain the answer, say so explicitly. "
        "Be precise with numbers — include exact figures from the filings. "
        "Each context chunk below is prefixed with its source in "
        "[TICKER, FYYEAR, SECTION] form. FYYEAR is the fiscal year of the "
        "filing the chunk comes from; a filing's statements also carry the two "
        "prior fiscal years in comparative columns, so a chunk tagged FY2024 "
        "may contain FY2023 and FY2022 figures. Always cite the source using "
        "\"({{TICKER}}, FY{{YEAR}}, {{SECTION_NAME}})\" in your answer. "
        "IMPORTANT: ALWAYS reply in English. Use standard English number formatting (e.g. 1,000.50). "
        "\n\n## Examples\n\n"
        + FEW_SHOT_EXAMPLES
        + "\n\n## Output Rules\n\n"
        + ANSWER_FORMAT_CONVENTION
        + NON_ANSWERABILITY_CONVENTION
        + REASONING_CHAIN_CONVENTION
    ),
    (
        "human",
        "Context:\n{context}\n\n---\n\nQuestion: {question}"
    ),
])


@dataclass
class RAGResult:
    """Result from a single RAG pipeline query."""

    answer: str
    contexts: list[str] = field(default_factory=list)
    context_documents: list[Document] = field(default_factory=list)
    retrieval_scores: dict = field(default_factory=dict)
    metrics: RunMetrics = field(default_factory=RunMetrics)


class MonolithRAGPipeline:
    """
    System 1: Simple RAG Pipeline (non-agentic).

    Chunking → ChromaDB + BM25 → EnsembleRetriever → FlashRank Rerank → Gemini Generation.

    Usage:
        pipeline = MonolithRAGPipeline(config_override={"chunk_size": 1500})
        pipeline.build(filings)
        result = pipeline.query("What was AAPL's revenue in FY2024?")
    """

    def __init__(self, config_override: dict | None = None):
        """
        Initialize pipeline with config from rag_monolith.yaml + optional overrides.

        Args:
            config_override: Parameter overrides (e.g., from Optuna trial).
                Keys: chunk_size, chunk_overlap_pct, bm25_weight,
                      pre_rerank_top_k, post_rerank_top_k
        """
        self.config = load_config("rag_monolith")
        self._apply_overrides(config_override or {})

        self._retriever: HybridRetriever | None = None
        self._documents: list[Document] | None = None
        self._vectorstore: Chroma | None = None
        self._llm: ChatVertexAI | None = None

    def _apply_overrides(self, overrides: dict) -> None:
        """Apply parameter overrides from Optuna or manual config."""
        chunking = self.config.get("chunking", {})
        retrieval = self.config.get("retrieval", {})
        reranker = self.config.get("reranker", {})

        # Map flat override keys to nested config
        if "chunk_size" in overrides:
            chunking["chunk_size"] = overrides["chunk_size"]
        if "chunk_overlap_pct" in overrides:
            chunking["chunk_overlap_pct"] = overrides["chunk_overlap_pct"]
        if "bm25_weight" in overrides:
            retrieval["bm25_weight"] = overrides["bm25_weight"]
            retrieval["dense_weight"] = round(1.0 - overrides["bm25_weight"], 2)
        if "pre_rerank_top_k" in overrides:
            retrieval["pre_rerank_top_k"] = overrides["pre_rerank_top_k"]
        if "post_rerank_top_k" in overrides:
            retrieval["post_rerank_top_k"] = overrides["post_rerank_top_k"]

        self.config["chunking"] = chunking
        self.config["retrieval"] = retrieval
        self.config["reranker"] = reranker

    @property
    def params(self) -> dict:
        """Return current parameter configuration as flat dict."""
        chunking = self.config.get("chunking", {})
        retrieval = self.config.get("retrieval", {})
        return {
            "chunk_size": chunking.get("chunk_size", 1000),
            "chunk_overlap_pct": chunking.get("chunk_overlap_pct", 0.20),
            "bm25_weight": retrieval.get("bm25_weight", 0.3),
            "pre_rerank_top_k": retrieval.get("pre_rerank_top_k", 20),
            "post_rerank_top_k": retrieval.get("post_rerank_top_k", 5),
        }

    def build(self, filings: Sequence[ProcessedFiling]) -> None:
        """
        Build the RAG pipeline from processed filings.

        Steps:
          1. Chunk filings into Documents (or load from cache)
          2. Build ChromaDB vectorstore (or load from cache)
          3. Build hybrid retriever with BM25 (or load from cache)
          4. Initialize LLM

        All intermediate artefacts are persisted under a content-addressed
        directory so that repeated runs with the same filings and config
        skip the expensive embedding / indexing work entirely.

        Args:
            filings: Processed SEC 10-K filings.
        """
        chunking = self.config.get("chunking", {})
        retrieval = self.config.get("retrieval", {})
        vs_config = self.config.get("vectorstore", {})

        chunk_size = chunking.get("chunk_size", 1000)
        overlap_pct = chunking.get("chunk_overlap_pct", 0.20)
        bm25_weight = retrieval.get("bm25_weight", 0.3)
        pre_rerank_k = retrieval.get("pre_rerank_top_k", 20)

        logger.info(
            "Building pipeline: chunk_size=%d, overlap=%.0f%%, bm25_weight=%.2f, pre_k=%d",
            chunk_size, overlap_pct * 100, bm25_weight, pre_rerank_k,
        )

        # Compute content-addressed cache path
        filings_hash = compute_filings_hash(filings)
        base_dir = vs_config.get("persist_directory", "data/vectorstores/rag_monolith")
        config_hash = f"cs{chunk_size}_ov{int(overlap_pct*100)}"
        persist_dir = str(Path(base_dir) / config_hash / filings_hash)

        # Step 1: Chunk (with disk cache)
        self._documents = load_or_build_documents(
            filings,
            persist_directory=persist_dir,
            chunk_size=chunk_size,
            overlap_pct=overlap_pct,
        )

        # Step 2: Vectorstore (auto-caches via ChromaDB persistence)
        embeddings = get_embeddings("rag_monolith")

        self._vectorstore = build_vectorstore(
            documents=self._documents,
            embeddings=embeddings,
            persist_directory=persist_dir,
            collection_name=vs_config.get("collection_name", "sec_10k_filings"),
        )

        # Step 3: Hybrid retriever (BM25 index cached via persist_dir)
        self._retriever = build_hybrid_retriever(
            vectorstore=self._vectorstore,
            documents=self._documents,
            bm25_weight=bm25_weight,
            pre_rerank_top_k=pre_rerank_k,
            persist_directory=persist_dir,
        )

        # Step 4: LLM
        self._llm = get_llm("rag_monolith")

        logger.info("Pipeline built successfully: %d total chunks", len(self._documents))

    def query(self, question: str) -> RAGResult:
        """
        Run a single query through the RAG pipeline.

        Steps: Retrieve → Rerank → Generate answer.

        Args:
            question: User question about SEC filings.

        Returns:
            RAGResult with answer, contexts, scores, and metrics.
        """
        if self._retriever is None or self._llm is None:
            raise RuntimeError("Pipeline not built. Call .build(filings) first.")

        retrieval_config = self.config.get("retrieval", {})
        reranker_config = self.config.get("reranker", {})

        start_time = time.perf_counter()

        # Step 1: Retrieve + Rerank
        docs = retrieve(
            query=question,
            retriever=self._retriever,
            post_rerank_top_k=retrieval_config.get("post_rerank_top_k", 5),
            reranker_model=reranker_config.get("model", "rerank-v3.5"),
            reranker_enabled=reranker_config.get("enabled", True),
        )

        # Step 2: Build context from retrieved documents.
        # `contexts` (returned to the caller for RAGAS) stays raw page_content;
        # the prompt gets a metadata-prefixed version so the LLM has grounded
        # ticker/year/section info to cite from. This is a deterministic
        # formatting step on already-attached chunk metadata (see
        # src/common/chunker.py) — it does not add tool-use or autonomous
        # decisions, so S1 remains a fixed one-shot retrieve-then-generate
        # pipeline.
        contexts = [doc.page_content for doc in docs]
        labeled_contexts = [
            f"[{doc.metadata.get('ticker', '?')}, FY{doc.metadata.get('fiscal_year', '?')}, "
            f"{doc.metadata.get('section_name', '?')}]\n{doc.page_content}"
            for doc in docs
        ]
        context_str = "\n\n---\n\n".join(labeled_contexts)

        # Step 3: Generate answer
        prompt = RAG_PROMPT.format_messages(
            context=context_str,
            question=question,
        )
        response = self._llm.invoke(prompt)
        answer = extract_text(response.content)

        # Step 4: Collect metrics
        elapsed = time.perf_counter() - start_time

        # Extract token usage from response metadata
        usage_meta = getattr(response, "usage_metadata", None)
        token_usage = TokenUsage()
        if usage_meta:
            token_usage = TokenUsage(
                prompt_tokens=usage_meta.get("input_tokens", 0),
                completion_tokens=usage_meta.get("output_tokens", 0),
                total_tokens=usage_meta.get("total_tokens", 0),
            )

        # Collect BM25 scores for debugging
        retrieval_scores = {}
        for i, doc in enumerate(docs):
            if "bm25_score" in doc.metadata:
                retrieval_scores[f"doc_{i}_bm25"] = doc.metadata["bm25_score"]

        metrics = RunMetrics(
            query=question,
            system_name="rag_monolith",
            token_usage=token_usage,
            latency_seconds=elapsed,
            num_steps=1,  # Non-agentic: always 1 step
        )

        return RAGResult(
            answer=answer,
            contexts=contexts,
            context_documents=docs,
            retrieval_scores=retrieval_scores,
            metrics=metrics,
        )
