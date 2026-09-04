"""
End-to-End RAG Pipeline for System 2 (Agent RAG).

Orchestrates: Build (Vectorstore + Tools + Agent) → Query (Agent Loop
→ Reflection → Optional Correction Loop → Metrics).

Uses LangGraph ReAct agent with tool-use for autonomous retrieval decisions.
Same retrieval stack as System 1 (Monolith) — only the control flow differs.
Optionally augmented with a Reflexion-style external verifier (Shinn et al.
2023) that can trigger one correction round per query.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langchain_google_vertexai import ChatVertexAI

from src.common.agent import build_agent
from src.common.config import load_config
from src.common.ingestion import ProcessedFiling
from src.common.llm_client import get_embeddings, get_llm
from src.common.message_parsing import parse_agent_messages
from src.common.reflection import (
    ReflectionVerdict,
    build_reflection_chain,
    run_reflection_pass,
)
from src.common.retrieval import (
    HybridRetriever,
    build_hybrid_retriever,
    build_vectorstore,
    load_or_build_documents,
)
from src.common.tools.calculate import calculate
from src.common.tools.list_filings import create_list_filings_tool
from src.common.utils import RunMetrics, TokenUsage, compute_filings_hash
from src.systems.rag_agent.agent import SYSTEM_PROMPT
from src.systems.rag_agent.tools.retrieve_chunks import create_retrieve_chunks_tool
from src.systems.rag_agent.tools.search_section import create_search_section_tool

logger = logging.getLogger(__name__)


@dataclass
class AgentRAGResult:
    """Result from a single Agent RAG query."""

    answer: str
    contexts: list[str] = field(default_factory=list)
    context_documents: list[Document] = field(default_factory=list)
    tool_calls_log: list[dict] = field(default_factory=list)
    metrics: RunMetrics = field(default_factory=RunMetrics)
    reflection_verdict: ReflectionVerdict | None = None
    was_revised: bool = False


class AgentRAGPipeline:
    """
    System 2: Single-Agent RAG Pipeline (agentic).

    Chunking → Shared ChromaDB + BM25 → LangGraph Agent → Tool-Use Loop
    → Optional Reflection-Verifier → Final Answer.

    The agent autonomously decides when/how to use retrieval tools,
    unlike System 1 which always follows a fixed pipeline. An optional
    Reflexion-style external verifier can trigger exactly one correction
    round per query (single-pass policy). Reflection is enabled by default
    but can be disabled via `reflection.enabled=false` in the config for
    ablation experiments.

    Usage:
        pipeline = AgentRAGPipeline()
        pipeline.build(filings)
        result = pipeline.query("What was AAPL's revenue in FY2024?")
    """

    def __init__(self, config_override: dict | None = None):
        """
        Initialize pipeline with config from rag_agent.yaml + optional overrides.

        Args:
            config_override: Parameter overrides for retrieval and agent
                settings. Also supports 'reflection_enabled' to toggle
                the Reflexion verifier at runtime (e.g. for ablation).
        """
        self.config = load_config("rag_agent")
        self._apply_overrides(config_override or {})

        self._agent = None
        self._reflection_chain: Runnable | None = None
        self._retriever: HybridRetriever | None = None
        self._vectorstore: Chroma | None = None
        self._documents: list[Document] | None = None
        self._filings: list[ProcessedFiling] | None = None
        self._llm: ChatVertexAI | None = None

    def _apply_overrides(self, overrides: dict) -> None:
        """Apply parameter overrides (same structure as monolith for consistency)."""
        chunking = self.config.get("chunking", {})
        retrieval = self.config.get("retrieval", {})
        agent_cfg = self.config.get("agent", {})
        reflection_cfg = self.config.get("reflection", {})

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
        if "max_iterations" in overrides:
            agent_cfg["max_iterations"] = overrides["max_iterations"]
        if "reflection_enabled" in overrides:
            reflection_cfg["enabled"] = overrides["reflection_enabled"]

        self.config["chunking"] = chunking
        self.config["retrieval"] = retrieval
        self.config["agent"] = agent_cfg
        self.config["reflection"] = reflection_cfg

    @property
    def params(self) -> dict:
        """Return current parameter configuration as flat dict."""
        chunking = self.config.get("chunking", {})
        retrieval = self.config.get("retrieval", {})
        agent_cfg = self.config.get("agent", {})
        reflection_cfg = self.config.get("reflection", {})
        return {
            "chunk_size": chunking.get("chunk_size", 2000),
            "chunk_overlap_pct": chunking.get("chunk_overlap_pct", 0.10),
            "bm25_weight": retrieval.get("bm25_weight", 0.3),
            "pre_rerank_top_k": retrieval.get("pre_rerank_top_k", 15),
            "post_rerank_top_k": retrieval.get("post_rerank_top_k", 5),
            "max_iterations": agent_cfg.get("max_iterations", 5),
            "reflection_enabled": reflection_cfg.get("enabled", True),
        }

    def build(self, filings: Sequence[ProcessedFiling]) -> None:
        """
        Build the Agent RAG pipeline from processed filings.

        Steps:
          1. Chunk filings into Documents (or load from cache)
          2. Build ChromaDB vectorstore (or load from cache)
          3. Build hybrid retriever with BM25 (or load from cache)
          4. Create tools with injected dependencies
          5. Build LangGraph agent
          6. (optional) Build reflection chain

        All intermediate artefacts are persisted under a content-addressed
        directory so that repeated runs with the same filings and config
        skip the expensive embedding / indexing work entirely.

        Args:
            filings: Processed SEC 10-K filings.
        """
        self._filings = list(filings)
        chunking = self.config.get("chunking", {})
        retrieval = self.config.get("retrieval", {})
        reranker = self.config.get("reranker", {})
        vs_config = self.config.get("vectorstore", {})
        agent_cfg = self.config.get("agent", {})
        reflection_cfg = self.config.get("reflection", {})

        chunk_size = chunking.get("chunk_size", 2000)
        overlap_pct = chunking.get("chunk_overlap_pct", 0.10)
        bm25_weight = retrieval.get("bm25_weight", 0.3)
        pre_rerank_k = retrieval.get("pre_rerank_top_k", 15)

        logger.info(
            "Building agent pipeline: chunk_size=%d, overlap=%.0f%%, "
            "bm25_weight=%.2f, pre_k=%d",
            chunk_size, overlap_pct * 100, bm25_weight, pre_rerank_k,
        )

        # Compute content-addressed cache path
        filings_hash = compute_filings_hash(filings)
        base_dir = vs_config.get("persist_directory", "data/vectorstores/rag_monolith")
        config_hash = f"cs{chunk_size}_ov{int(overlap_pct * 100)}"
        persist_dir = str(Path(base_dir) / config_hash / filings_hash)

        # Step 1: Chunk (with disk cache)
        self._documents = load_or_build_documents(
            filings,
            persist_directory=persist_dir,
            chunk_size=chunk_size,
            overlap_pct=overlap_pct,
        )

        # Step 2: Vectorstore (auto-caches via ChromaDB persistence)
        embeddings = get_embeddings("rag_agent")

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

        # Step 4: Create tools
        tools = [
            create_retrieve_chunks_tool(
                retriever=self._retriever,
                retrieval_config=retrieval,
                reranker_config=reranker,
            ),
            create_search_section_tool(
                vectorstore=self._vectorstore,
                retrieval_config=retrieval,
                reranker_config=reranker,
            ),
            calculate,  # Stateless tool, no factory needed
            create_list_filings_tool(filings=self._filings),
        ]

        # Step 5: Build agent
        self._llm = get_llm("rag_agent")
        recursion_limit = agent_cfg.get("recursion_limit", 12)

        self._agent = build_agent(
            llm=self._llm,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            recursion_limit=recursion_limit,
        )

        # Step 6: Build reflection chain (optional, gated by config)
        if reflection_cfg.get("enabled", True):
            self._reflection_chain = build_reflection_chain(
                llm=self._llm, include_raw=True,
            )
            logger.info("Reflection chain enabled (single-pass policy)")
        else:
            self._reflection_chain = None
            logger.info("Reflection chain disabled (ablation mode)")

        logger.info(
            "Agent pipeline built: %d chunks, %d tools, recursion_limit=%d",
            len(self._documents), len(tools), recursion_limit,
        )

    def query(self, question: str) -> AgentRAGResult:
        """
        Run a single query through the Agent RAG pipeline.

        Flow:
            1. agent.invoke() — ReAct loop produces a draft answer.
            2. If reflection is enabled: reflection_chain evaluates the
               draft. On 'accept' the draft becomes final. On 'revise' the
               agent runs once more with feedback injected as a HumanMessage
               (single-pass policy; no further reflection).
            3. Final answer and aggregated metrics are returned.

        Args:
            question: User question about SEC filings.

        Returns:
            AgentRAGResult with answer, tool call log, metrics, and
            reflection metadata.
        """
        if self._agent is None or self._llm is None:
            raise RuntimeError("Pipeline not built. Call .build(filings) first.")

        agent_cfg = self.config.get("agent", {})
        recursion_limit = agent_cfg.get("recursion_limit", 12)

        start_time = time.perf_counter()

        # --- First pass: standard ReAct loop -------------------------------
        first_result = self._agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": recursion_limit},
        )
        messages = first_result.get("messages", [])

        # --- Optional reflection + single correction pass ------------------
        verdict: ReflectionVerdict | None = None
        was_revised = False
        reflection_prompt_tokens = 0
        reflection_completion_tokens = 0

        if self._reflection_chain is not None:
            (
                messages,
                verdict,
                was_revised,
                reflection_prompt_tokens,
                reflection_completion_tokens,
            ) = run_reflection_pass(
                agent=self._agent,
                messages=messages,
                question=question,
                reflection_chain=self._reflection_chain,
                recursion_limit=recursion_limit,
                empty_contexts_placeholder="(no retrieval contexts captured)",
            )

        elapsed = time.perf_counter() - start_time

        # --- Aggregate final outputs from the (possibly revised) history ---
        answer, tool_calls_log, contexts, token_usage = parse_agent_messages(
            messages
        )

        # Fold reflection-LLM tokens into the total usage for accurate cost tracking.
        total_prompt = token_usage.prompt_tokens + reflection_prompt_tokens
        total_completion = (
            token_usage.completion_tokens + reflection_completion_tokens
        )
        aggregated_tokens = TokenUsage(
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_prompt + total_completion,
        )

        metrics = RunMetrics(
            query=question,
            system_name="rag_agent",
            token_usage=aggregated_tokens,
            latency_seconds=elapsed,
            num_steps=len(tool_calls_log),
            tool_calls=[tc["tool"] for tc in tool_calls_log],
            corrections=1 if was_revised else 0,
        )

        logger.info(
            "Agent query completed: %d tool calls, %.2fs, %d tokens "
            "(reflection=%s, revised=%s)",
            len(tool_calls_log), elapsed, aggregated_tokens.total_tokens,
            "on" if self._reflection_chain else "off",
            was_revised,
        )

        return AgentRAGResult(
            answer=answer,
            contexts=contexts,
            tool_calls_log=tool_calls_log,
            metrics=metrics,
            reflection_verdict=verdict,
            was_revised=was_revised,
        )
