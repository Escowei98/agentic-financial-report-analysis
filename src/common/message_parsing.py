"""
Shared parsing of LangGraph agent message histories (S2, S3, S4).

Every agentic system needs the same four things out of a finished agent run:
the final answer, a tool-call log, the tool outputs, and the aggregated token
usage. That loop previously existed five times — as a private method on the
S2 and S3 pipelines and inline three times inside the S4 node functions — with
identical semantics but drifting comments and annotations.

The token figures feed NF-5 (efficiency) and the tool-call log feeds both the
process metrics and the judge-facing trajectory (thesis 3.6.2), so a silent
divergence between these copies would have shown up as an architectural
difference in the results. One implementation removes that failure mode.
"""

import logging

from langchain_core.messages import AIMessage, ToolMessage

from src.common.utils import TokenUsage, extract_text

logger = logging.getLogger(__name__)


def parse_agent_messages(
    messages: list,
) -> tuple[str, list[dict], list[str], TokenUsage]:
    """
    Parse a LangGraph agent message history into its result components.

    Args:
        messages: Message history returned by a compiled agent invocation.

    Returns:
        Tuple of (answer, tool_calls_log, contexts, token_usage).

        - answer: content of the last AI message that carries content and
          makes no tool call.
        - tool_calls_log: one entry per tool call with name, arguments and —
          linked back via ``tool_call_id`` — the tool's own output, so the
          trajectory formatter can show what a tool actually returned rather
          than only which tool was invoked.
        - contexts: the raw tool outputs. Meaningful as grounding evidence
          only for the retrieval-based systems; for S3/S4 the answer is
          grounded in the inlined filings, which never pass through here
          (see `src/evaluation/eval_runner.py`).
        - token_usage: summed over every AI message in the history.
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
                    if tc["id"] is not None:
                        tool_call_index[tc["id"]] = len(tool_calls_log) - 1

            if msg.content and not msg.tool_calls:
                answer = extract_text(msg.content)

        elif isinstance(msg, ToolMessage):
            if msg.content:
                result_text = extract_text(msg.content)
                contexts.append(result_text)
                idx = (
                    tool_call_index.get(msg.tool_call_id)
                    if msg.tool_call_id is not None
                    else None
                )
                if idx is not None:
                    tool_calls_log[idx]["result"] = result_text

    token_usage = TokenUsage(
        prompt_tokens=total_prompt,
        completion_tokens=total_completion,
        total_tokens=total_prompt + total_completion,
    )

    return answer, tool_calls_log, contexts, token_usage


def format_tool_calls_for_prompt(tool_calls: list[dict]) -> str:
    """Render tool_calls_log as a readable string for the reflection prompt."""
    if not tool_calls:
        return "(no tool calls)"
    return "\n".join(
        f"- {tc['tool']}({tc['args']})" for tc in tool_calls
    )
