"""
End-to-End Pipeline for System 3 (Single-Agent Long Context).

Replaces the retrieval stack of S2 with a static prompt-resident
knowledge base: every processed filing is inlined into the agent's
system prompt. The agent then operates over the full corpus directly,
optionally invoking `calculate` for arithmetic and `list_filings` for
metadata inspection. An optional Reflexion-style verifier identical to
S2's can trigger one correction round per query.

Architecture (flow):
    build()  -> compose system_prompt(filings) + agent + (reflection)
    query()  -> agent.invoke()
                -> [optional] reflection_chain.invoke()
                    -> if 'revise': inject feedback, agent.invoke() again
                -> aggregate metrics, return result

The class signature mirrors `AgentRAGPipeline` so that the evaluation
notebooks can switch backends with minimal code changes.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Sequence

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.common.config import load_config
from src.common.ingestion import ProcessedFiling
from src.common.llm_client import get_llm
from src.common.reflection import (
    ReflectionVerdict,
    build_reflection_chain,
    generate_feedback_message,
    unpack_reflection_result,
)
from src.common.utils import RunMetrics, TokenUsage, extract_text
from src.systems.long_context.agent import build_agent
from src.systems.long_context.prompt import build_system_prompt
from src.systems.rag_agent.tools.calculate import calculate
from src.systems.rag_agent.tools.list_filings import create_list_filings_tool

logger = logging.getLogger(__name__)


@dataclass
class LongContextResult:
    """Result from a single Long-Context Agent query."""

    answer: str
    contexts: list[str] = field(default_factory=list)
    tool_calls_log: list[dict] = field(default_factory=list)
    metrics: RunMetrics = field(default_factory=RunMetrics)
    reflection_verdict: ReflectionVerdict | None = None
    was_revised: bool = False


class LongContextPipeline:
    """
    System 3: Single-Agent Long Context Pipeline.

    Inline-Filings System Prompt -> LangGraph Agent -> Tool-Use Loop
    -> Optional Reflection Verifier -> Final Answer.

    Architecture vs. S2:
      - SAME: LangGraph ReAct agent, single-pass reflection, message
              parsing, RunMetrics shape.
      - DIFFERENT: no chunker, no embeddings, no vectorstore, no
                   retriever, no retrieval tools. The full corpus lives
                   in the system prompt instead.

    Usage:
        pipeline = LongContextPipeline()
        pipeline.build(filings)
        result = pipeline.query("What was AAPL's revenue in FY2024?")
    """

    def __init__(self, config_override: dict | None = None):
        """
        Initialize pipeline with config from long_context.yaml + overrides.

        Args:
            config_override: Optional override dict. Supported keys:
                'max_iterations', 'recursion_limit', 'reflection_enabled'.
        """
        self.config = load_config("long_context")
        self._apply_overrides(config_override or {})

        self._agent = None
        self._reflection_chain = None
        self._llm = None
        self._filings: list[ProcessedFiling] | None = None
        self._system_prompt: str | None = None

    def _apply_overrides(self, overrides: dict) -> None:
        """Apply runtime overrides (parity with S2's override surface)."""
        agent_cfg = self.config.get("agent", {})
        reflection_cfg = self.config.get("reflection", {})

        if "max_iterations" in overrides:
            agent_cfg["max_iterations"] = overrides["max_iterations"]
        if "recursion_limit" in overrides:
            agent_cfg["recursion_limit"] = overrides["recursion_limit"]
        if "reflection_enabled" in overrides:
            reflection_cfg["enabled"] = overrides["reflection_enabled"]

        self.config["agent"] = agent_cfg
        self.config["reflection"] = reflection_cfg

    @property
    def params(self) -> dict:
        """Return the active runtime parameters as a flat dict."""
        agent_cfg = self.config.get("agent", {})
        reflection_cfg = self.config.get("reflection", {})
        return {
            "max_iterations": agent_cfg.get("max_iterations", 5),
            "recursion_limit": agent_cfg.get("recursion_limit", 12),
            "reflection_enabled": reflection_cfg.get("enabled", True),
        }

    def build(self, filings: Sequence[ProcessedFiling]) -> None:
        """
        Build the Long-Context pipeline from processed filings.

        Steps:
            1. Compose the static system prompt (filings inlined).
            2. Initialize the LLM (Gemini 2.5 Flash via Vertex AI).
            3. Create the (non-retrieval) tool set.
            4. Build the LangGraph ReAct agent.
            5. (optional) Build the reflection chain.

        Args:
            filings: Processed SEC 10-K filings.
        """
        self._filings = list(filings)

        agent_cfg = self.config.get("agent", {})
        reflection_cfg = self.config.get("reflection", {})

        recursion_limit = agent_cfg.get("recursion_limit", 12)

        # Step 1: Compose the static system prompt with all filings inlined.
        self._system_prompt = build_system_prompt(self._filings)

        # Step 2: LLM
        self._llm = get_llm("long_context")

        # Step 3: Non-retrieval tool set
        tools = [
            calculate,
            create_list_filings_tool(filings=self._filings),
        ]

        # Step 4: Agent
        self._agent = build_agent(
            llm=self._llm,
            tools=tools,
            system_prompt=self._system_prompt,
            recursion_limit=recursion_limit,
        )

        # Step 5: Reflection chain (optional, gated by config)
        if reflection_cfg.get("enabled", True):
            self._reflection_chain = build_reflection_chain(
                llm=self._llm, include_raw=True,
            )
            logger.info("S3 reflection chain enabled (single-pass policy)")
        else:
            self._reflection_chain = None
            logger.info("S3 reflection chain disabled (ablation mode)")

        logger.info(
            "S3 pipeline built: %d filings inlined, %d tools, "
            "recursion_limit=%d",
            len(self._filings), len(tools), recursion_limit,
        )

    def query(self, question: str) -> LongContextResult:
        """
        Run a single query through the Long-Context pipeline.

        Flow:
            1. agent.invoke() with the question; ReAct loop produces a
               draft answer (the filings are already in the system prompt).
            2. If reflection is enabled: reflection_chain evaluates the
               draft. On 'accept' the draft becomes final. On 'revise'
               the agent runs once more with feedback injected as a
               HumanMessage (single-pass policy).
            3. Aggregate token usage (agent + reflection) and return.

        Args:
            question: User question about the filings.

        Returns:
            LongContextResult with answer, tool-call log, metrics, and
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
            draft_answer, draft_tool_calls, draft_contexts, _ = (
                self._parse_messages(messages)
            )

            chain_result = self._reflection_chain.invoke({
                "question": question,
                "answer": draft_answer,
                "contexts": "\n\n---\n\n".join(draft_contexts)
                if draft_contexts
                else "(no tool outputs; agent answered directly from "
                "the inlined filings)",
                "tool_calls": _format_tool_calls_for_prompt(draft_tool_calls),
            })
            verdict, reflection_prompt_tokens, reflection_completion_tokens = (
                unpack_reflection_result(chain_result)
            )
            logger.info(
                "S3 reflection verdict: status=%s, issues=%s",
                verdict.status, verdict.issues,
            )

            feedback_msg = generate_feedback_message(verdict)
            if feedback_msg is not None:
                revised_messages = messages + [feedback_msg]
                second_result = self._agent.invoke(
                    {"messages": revised_messages},
                    config={"recursion_limit": recursion_limit},
                )
                messages = second_result.get("messages", revised_messages)
                was_revised = True

        elapsed = time.perf_counter() - start_time

        # --- Aggregate final outputs ---------------------------------------
        answer, tool_calls_log, contexts, token_usage = self._parse_messages(
            messages
        )

        # Fold reflection LLM tokens into total usage for accurate cost.
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
            system_name="long_context",
            token_usage=aggregated_tokens,
            latency_seconds=elapsed,
            num_steps=len(tool_calls_log),
            tool_calls=[tc["tool"] for tc in tool_calls_log],
            corrections=1 if was_revised else 0,
        )

        logger.info(
            "S3 query done: %d tool calls, %.2fs, %d tokens "
            "(reflection=%s, revised=%s)",
            len(tool_calls_log), elapsed, aggregated_tokens.total_tokens,
            "on" if self._reflection_chain else "off",
            was_revised,
        )

        return LongContextResult(
            answer=answer,
            contexts=contexts,
            tool_calls_log=tool_calls_log,
            metrics=metrics,
            reflection_verdict=verdict,
            was_revised=was_revised,
        )

    def _parse_messages(
        self, messages: list,
    ) -> tuple[str, list[dict], list[str], TokenUsage]:
        """
        Parse LangGraph agent message history into result components.

        Returns:
            Tuple of (answer, tool_calls_log, contexts, token_usage).

        Note on `contexts`:
            For S3 the answer is grounded in the inlined system prompt,
            not in tool outputs. The 'contexts' list therefore typically
            stays empty (only `list_filings`/`calculate` tool outputs
            land here, and only when the agent invokes them). This is
            architecturally correct: feeding the entire ~600k-token
            filings block into the RAGAS contexts would be misleading.
            For RAGAS scoring of S3 the evaluation notebook should
            build the contexts field from ground-truth source sections
            instead — see the upcoming S3 baseline notebook for the
            chosen convention.
        """
        answer = ""
        tool_calls_log: list[dict] = []
        contexts: list[str] = []
        total_prompt = 0
        total_completion = 0

        tool_call_index: dict[str, int] = {}  # tool_call_id -> index into tool_calls_log

        for msg in messages:
            if isinstance(msg, AIMessage):
                usage = getattr(msg, "usage_metadata", None)
                if usage:
                    total_prompt += usage.get("input_tokens", 0)
                    total_completion += usage.get("output_tokens", 0)

                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls_log.append({
                            "tool": tc["name"],
                            "args": tc["args"],
                            "result": "",
                        })
                        tool_call_index[tc["id"]] = len(tool_calls_log) - 1

                if msg.content and not msg.tool_calls:
                    answer = extract_text(msg.content)

            elif isinstance(msg, ToolMessage):
                # Linked back to tool_calls_log via tool_call_id so the
                # trajectory formatter can show calculate/list_filings
                # outputs, not just which tool was called.
                if msg.content:
                    result_text = extract_text(msg.content)
                    contexts.append(result_text)
                    idx = tool_call_index.get(msg.tool_call_id)
                    if idx is not None:
                        tool_calls_log[idx]["result"] = result_text

        token_usage = TokenUsage(
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_prompt + total_completion,
        )

        return answer, tool_calls_log, contexts, token_usage


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _format_tool_calls_for_prompt(tool_calls: list[dict]) -> str:
    """Render tool_calls_log as a readable string for the reflection prompt."""
    if not tool_calls:
        return "(no tool calls)"
    return "\n".join(
        f"- {tc['tool']}({tc['args']})" for tc in tool_calls
    )
