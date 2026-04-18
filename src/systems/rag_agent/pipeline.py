"""
End-to-End RAG Pipeline for System 2 (Agent RAG).

Orchestrates: Build (Vectorstore + Tools + Agent) → Query (Agent Loop → Metrics).

Uses LangGraph ReAct agent with tool-use for autonomous retrieval decisions.
Same retrieval stack as System 1 (Monolith) — only the control flow differs.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.common.config import load_config
from src.common.ingestion import ProcessedFiling
from src.common.llm_client import get_embeddings, get_llm
from src.common.retrieval import build_hybrid_retriever, build_vectorstore
from src.common.utils import RunMetrics, TokenUsage
from src.systems.rag_agent.agent import build_agent
from src.systems.rag_agent.tools.calculate import calculate
from src.systems.rag_agent.tools.list_filings import create_list_filings_tool
from src.systems.rag_agent.tools.retrieve_chunks import create_retrieve_chunks_tool
from src.systems.rag_agent.tools.search_section import create_search_section_tool
from src.systems.rag_monolith.chunker import chunk_filings

logger = logging.getLogger(__name__)


@dataclass
class AgentRAGResult:
    """Result from a single Agent RAG query."""

    answer: str
    contexts: list[str] = field(default_factory=list)
    context_documents: list[Document] = field(default_factory=list)
    tool_calls_log: list[dict] = field(default_factory=list)
    metrics: RunMetrics = field(default_factory=RunMetrics)


class AgentRAGPipeline:
    """
    System 2: Single-Agent RAG Pipeline (agentic).

    Chunking → Shared ChromaDB + BM25 → LangGraph Agent → Tool-Use Loop → Answer.

    The agent autonomously decides when/how to use retrieval tools,
    unlike System 1 which always follows a fixed pipeline.

    Usage:
        pipeline = AgentRAGPipeline()
        pipeline.build(filings)
        result = pipeline.query("What was AAPL's revenue in FY2024?")
    """

    def __init__(self, config_override: dict | None = None):
        """
        Initialize pipeline with config from rag_agent.yaml + optional overrides.

        Args:
            config_override: Parameter overrides for retrieval settings.
        """
        self.config = load_config("rag_agent")
        self._apply_overrides(config_override or {})

        self._agent = None
        self._retriever = None
        self._vectorstore = None
        self._documents = None
        self._filings = None
        self._llm = None

    def _apply_overrides(self, overrides: dict) -> None:
        """Apply parameter overrides (same structure as monolith for consistency)."""
        chunking = self.config.get("chunking", {})
        retrieval = self.config.get("retrieval", {})
        agent_cfg = self.config.get("agent", {})

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

        self.config["chunking"] = chunking
        self.config["retrieval"] = retrieval
        self.config["agent"] = agent_cfg

    @property
    def params(self) -> dict:
        """Return current parameter configuration as flat dict."""
        chunking = self.config.get("chunking", {})
        retrieval = self.config.get("retrieval", {})
        agent_cfg = self.config.get("agent", {})
        return {
            "chunk_size": chunking.get("chunk_size", 2000),
            "chunk_overlap_pct": chunking.get("chunk_overlap_pct", 0.10),
            "bm25_weight": retrieval.get("bm25_weight", 0.3),
            "pre_rerank_top_k": retrieval.get("pre_rerank_top_k", 15),
            "post_rerank_top_k": retrieval.get("post_rerank_top_k", 5),
            "max_iterations": agent_cfg.get("max_iterations", 5),
        }

    def build(self, filings: Sequence[ProcessedFiling]) -> None:
        """
        Build the Agent RAG pipeline from processed filings.

        Steps:
          1. Chunk filings into Documents (same as monolith)
          2. Build ChromaDB vectorstore (shared with monolith)
          3. Build hybrid retriever (BM25 + Dense)
          4. Create tools with injected dependencies
          5. Build LangGraph agent

        Args:
            filings: Processed SEC 10-K filings.
        """
        self._filings = list(filings)
        chunking = self.config.get("chunking", {})
        retrieval = self.config.get("retrieval", {})
        reranker = self.config.get("reranker", {})
        vs_config = self.config.get("vectorstore", {})
        agent_cfg = self.config.get("agent", {})

        chunk_size = chunking.get("chunk_size", 2000)
        overlap_pct = chunking.get("chunk_overlap_pct", 0.10)
        bm25_weight = retrieval.get("bm25_weight", 0.3)
        pre_rerank_k = retrieval.get("pre_rerank_top_k", 15)

        logger.info(
            "Building agent pipeline: chunk_size=%d, overlap=%.0f%%, "
            "bm25_weight=%.2f, pre_k=%d",
            chunk_size, overlap_pct * 100, bm25_weight, pre_rerank_k,
        )

        # Step 1: Chunk (identical to monolith)
        self._documents = chunk_filings(
            filings,
            chunk_size=chunk_size,
            overlap_pct=overlap_pct,
        )

        # Step 2: Vectorstore (SHARED directory with monolith)
        embeddings = get_embeddings("rag_agent")
        base_dir = vs_config.get("persist_directory", "data/vectorstores/rag_monolith")
        config_hash = f"cs{chunk_size}_ov{int(overlap_pct * 100)}"
        persist_dir = str(Path(base_dir) / config_hash)

        self._vectorstore = build_vectorstore(
            documents=self._documents,
            embeddings=embeddings,
            persist_directory=persist_dir,
            collection_name=vs_config.get("collection_name", "sec_10k_filings"),
        )

        # Step 3: Hybrid retriever
        self._retriever = build_hybrid_retriever(
            vectorstore=self._vectorstore,
            documents=self._documents,
            bm25_weight=bm25_weight,
            pre_rerank_top_k=pre_rerank_k,
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
                top_k=retrieval.get("post_rerank_top_k", 5),
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
            recursion_limit=recursion_limit,
        )

        logger.info(
            "Agent pipeline built: %d chunks, %d tools, recursion_limit=%d",
            len(self._documents), len(tools), recursion_limit,
        )

    def query(self, question: str) -> AgentRAGResult:
        """
        Run a single query through the Agent RAG pipeline.

        The agent decides autonomously which tools to call and in what order.

        Args:
            question: User question about SEC filings.

        Returns:
            AgentRAGResult with answer, tool call log, and metrics.
        """
        if self._agent is None or self._llm is None:
            raise RuntimeError("Pipeline not built. Call .build(filings) first.")

        agent_cfg = self.config.get("agent", {})
        recursion_limit = agent_cfg.get("recursion_limit", 12)

        start_time = time.perf_counter()

        # Invoke agent
        result = self._agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config={"recursion_limit": recursion_limit},
        )

        elapsed = time.perf_counter() - start_time

        # Extract results from message history
        messages = result.get("messages", [])
        answer, tool_calls_log, contexts, token_usage = self._parse_messages(messages)

        metrics = RunMetrics(
            query=question,
            system_name="rag_agent",
            token_usage=token_usage,
            latency_seconds=elapsed,
            num_steps=len(tool_calls_log),
            tool_calls=[tc["tool"] for tc in tool_calls_log],
        )

        logger.info(
            "Agent query completed: %d tool calls, %.2fs, %d tokens",
            len(tool_calls_log), elapsed, token_usage.total_tokens,
        )

        return AgentRAGResult(
            answer=answer,
            contexts=contexts,
            tool_calls_log=tool_calls_log,
            metrics=metrics,
        )

    def _parse_messages(
        self, messages: list,
    ) -> tuple[str, list[dict], list[str], TokenUsage]:
        """
        Parse LangGraph agent message history to extract results and metrics.

        Returns:
            Tuple of (answer, tool_calls_log, contexts, token_usage)
        """
        answer = ""
        tool_calls_log = []
        contexts = []
        total_prompt = 0
        total_completion = 0

        for msg in messages:
            if isinstance(msg, AIMessage):
                # Accumulate token usage from all AI messages
                usage = getattr(msg, "usage_metadata", None)
                if usage:
                    total_prompt += usage.get("input_tokens", 0)
                    total_completion += usage.get("output_tokens", 0)

                # Track tool calls made by the agent
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls_log.append({
                            "tool": tc["name"],
                            "args": tc["args"],
                        })

                # Last AI message with content is the final answer
                if msg.content and not msg.tool_calls:
                    answer = msg.content

            elif isinstance(msg, ToolMessage):
                # Tool outputs serve as context
                if msg.content:
                    contexts.append(msg.content)

        token_usage = TokenUsage(
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_prompt + total_completion,
        )

        return answer, tool_calls_log, contexts, token_usage
