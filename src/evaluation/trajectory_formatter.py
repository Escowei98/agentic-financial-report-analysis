"""
Trajectory rendering for the four systems.

Turns a pipeline result object into a readable account of what the system did.
This is documentation of a run, not an input to any metric.

WHY IT SURVIVED THE 2026-09-08 REASONING REWRITE
------------------------------------------------
These formatters used to feed the Core/Agentic judge, which is gone: judging
each architecture off its own trace format is what produced the differential
bias that retired it. Reasoning is now scored off the uniform chain the
systems emit (src/common/reasoning_chain_convention.py), not off this.

The output is still needed, though, and by two consumers that are not metrics:
`eval_runner` stores it per query in `detailed_results`, from which the
human-rating page is built, and diagnostic write-ups such as
data/results/judge_validation/s3_long_context_review.md read it directly. A
rater or a reader still needs to see what the system actually did.

`_WITHHELD_NOTE` and `_PRIMARY_SOURCE_TOOLS` therefore stay in force. No judge
reads this any more, but a human rater does, and the asymmetry they guard
against -- only S1 and S2 can show retrieved filing text at all -- applies to
a human reading the page exactly as it did to the judge.
"""

import logging

logger = logging.getLogger(__name__)


# Truncation limit for evidence text shown in trajectories (chunk_size in
# rag_monolith.yaml/rag_agent.yaml is 1000 chars; this leaves headroom so a
# full chunk is very rarely cut). Kept as a safety cap, not a normal-case
# limit — see EVAL_DECISION_LOG.md [2026-08-01] "Trajectory truncation was
# hiding retrieved evidence from the judge".
_EVIDENCE_CHAR_LIMIT = 1500


# Tools whose output IS primary filing text. Their results are withheld from
# the trajectory so that no architecture shows the judge more source text than
# another can (see _WITHHELD_NOTE). Tools that return a computed value or a
# catalogue (calculate, list_filings) are not evidence and stay visible.
_PRIMARY_SOURCE_TOOLS = {"retrieve_chunks", "search_section"}

_WITHHELD_NOTE = (
    "Evidence text is deliberately not shown. Only S1 and S2 can surface "
    "retrieved filing passages at all: S3 holds the whole corpus in its system "
    "prompt and S4 sees it through its specialists, so neither has an "
    "observable retrieval step to report. Showing the passages for the two "
    "architectures that have them would have made Evidence Faithfulness a "
    "measure of the instrumentation rather than of the system — S3 scored 1.89 "
    "against 3.07-3.36 for the others on exactly that artefact. Judge grounding "
    "from the answer's own citations, which every system produces in the same "
    "(TICKER, FYYEAR, SECTION) form. See EVAL_DECISION_LOG.md [2026-09-06]."
)


def format_trajectory(result, system_name: str) -> str:
    """
    Extract a human-readable trajectory from a pipeline result object.

    Converts the internal tool_calls_log, specialist_outputs, reflection
    verdicts etc. into a structured text representation that the LLM judge
    can evaluate.

    Args:
        result: One of RAGResult, AgentRAGResult, LongContextResult,
                or MultiAgentResult.
        system_name: One of 'rag_monolith', 'rag_agent', 'long_context',
                     'multi_agent'.

    Returns:
        A formatted string describing the reasoning trajectory.
    """
    if system_name == "rag_monolith":
        return _format_s1_trajectory(result)
    elif system_name == "rag_agent":
        return _format_s2_trajectory(result)
    elif system_name == "long_context":
        return _format_s3_trajectory(result)
    elif system_name == "multi_agent":
        return _format_s4_trajectory(result)
    else:
        return f"[Unknown system: {system_name}]"


# Truncation limit for evidence text shown in trajectories (chunk_size in
# rag_monolith.yaml/rag_agent.yaml is 1000 chars; this leaves headroom so a
# full chunk is very rarely cut). Kept as a safety cap, not a normal-case
# limit — see EVAL_DECISION_LOG.md [2026-08-01] "Trajectory truncation was
# hiding retrieved evidence from the judge".
_EVIDENCE_CHAR_LIMIT = 1500


# Tools whose output IS primary filing text. Their results are withheld from
# the trajectory so that no architecture shows the judge more source text than
# another can (see _WITHHELD_NOTE). Tools that return a computed value or a
# catalogue (calculate, list_filings) are not evidence and stay visible.
_PRIMARY_SOURCE_TOOLS = {"retrieve_chunks", "search_section"}

_WITHHELD_NOTE = (
    "Evidence text is deliberately not shown. Only S1 and S2 can surface "
    "retrieved filing passages at all: S3 holds the whole corpus in its system "
    "prompt and S4 sees it through its specialists, so neither has an "
    "observable retrieval step to report. Showing the passages for the two "
    "architectures that have them would have made Evidence Faithfulness a "
    "measure of the instrumentation rather than of the system — S3 scored 1.89 "
    "against 3.07-3.36 for the others on exactly that artefact. Judge grounding "
    "from the answer's own citations, which every system produces in the same "
    "(TICKER, FYYEAR, SECTION) form. See EVAL_DECISION_LOG.md [2026-09-06]."
)


def _context_sources(result) -> list[str]:
    """Ticker / fiscal year / section for each retrieved chunk, no text.

    Reads the metadata the chunker already attaches (see
    src/common/chunker.py); `contexts` itself carries only raw page content.
    """
    docs = getattr(result, "context_documents", None) or []
    sources = []
    for doc in docs:
        meta = getattr(doc, "metadata", None) or {}
        ticker = meta.get("ticker", "?")
        year = meta.get("fiscal_year", "?")
        section = meta.get("section_name", "?")
        sources.append(f"{ticker} FY{year}, {section}")
    return sources


def _format_tool_result(tool_name: str, tool_result) -> str:
    """Render one tool result, withholding primary filing text."""
    name = str(tool_name or "").strip().lower()
    if name in _PRIMARY_SOURCE_TOOLS:
        if isinstance(tool_result, str) and tool_result:
            return f"Result: {len(tool_result)} chars of filing text (withheld)"
        return "Result: none"
    if isinstance(tool_result, str) and tool_result:
        return f"Result: {_truncate(tool_result)}"
    return "Result: none"


def _truncate(text: str, limit: int = _EVIDENCE_CHAR_LIMIT) -> str:
    """Truncate text to `limit` chars, appending '...' only if actually cut."""
    text = text.replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _format_s1_trajectory(result) -> str:
    """S1 (RAG Monolith): Minimal trajectory — retrieval + generation."""
    lines = ["## Trajectory"]
    lines.append("Steps: 1 (single retrieval + generation pass)")

    # Show retrieved contexts summary
    num_contexts = len(result.contexts) if hasattr(result, "contexts") else 0
    lines.append(f"Retrieved contexts: {num_contexts} chunks")

    # Provenance without the passage text. S1 has exactly one step, and what
    # it retrieved is the only observable artefact of that step — withholding
    # it along with the text would have left S1 as the only system whose
    # process cannot be seen at all, while S4 keeps its full plan/delegation
    # trace and S2 exposes ticker/year/section through its tool arguments.
    # This line restores that parity and exposes no filing text.
    # See EVAL_DECISION_LOG.md [2026-09-06].
    for i, source in enumerate(_context_sources(result), 1):
        lines.append(f"  Context {i}: {source}")

    lines.append(f"Total tokens: {result.metrics.token_usage.total_tokens}")
    return "\n".join(lines)


def _format_s2_trajectory(result) -> str:
    """S2 (Agent RAG): ReAct loop with retrieval tools."""
    lines = ["## Trajectory"]

    # Tool calls
    if hasattr(result, "tool_calls_log") and result.tool_calls_log:
        lines.append(f"Total tool calls: {len(result.tool_calls_log)}")
        for i, tc in enumerate(result.tool_calls_log, 1):
            tool_name = tc.get("tool", tc.get("name", "unknown"))
            tool_args = tc.get("args", tc.get("arguments", {}))
            tool_result = tc.get("result", tc.get("output", ""))
            lines.append(f"Step {i}: {tool_name}({_format_args(tool_args)})")
            lines.append(f"  → {_format_tool_result(tool_name, tool_result)}")
    else:
        lines.append("Tool calls: None (direct answer without tool use)")

    # Reflection
    _append_reflection(lines, result)

    lines.append(f"Total tokens: {result.metrics.token_usage.total_tokens}")
    return "\n".join(lines)


def _format_s3_trajectory(result) -> str:
    """S3 (Long Context): ReAct loop with full document context, no retrieval."""
    lines = ["## Trajectory"]
    lines.append("Context: All SEC 10-K filings inlined in system prompt (no retrieval)")

    # Tool calls
    if hasattr(result, "tool_calls_log") and result.tool_calls_log:
        lines.append(f"Total tool calls: {len(result.tool_calls_log)}")
        for i, tc in enumerate(result.tool_calls_log, 1):
            tool_name = tc.get("tool", tc.get("name", "unknown"))
            tool_args = tc.get("args", tc.get("arguments", {}))
            tool_result = tc.get("result", tc.get("output", ""))
            lines.append(f"Step {i}: {tool_name}({_format_args(tool_args)})")
            lines.append(f"  → {_format_tool_result(tool_name, tool_result)}")
    else:
        lines.append("Tool calls: None (direct answer from full context)")

    # Reflection
    _append_reflection(lines, result)

    lines.append(f"Total tokens: {result.metrics.token_usage.total_tokens}")
    return "\n".join(lines)


def _format_s4_trajectory(result) -> str:
    """S4 (Multi-Agent): Supervisor → Specialists → Synthesizer → Reflection."""
    lines = ["## Trajectory"]

    # Supervisor plan
    if hasattr(result, "supervisor_plan") and result.supervisor_plan:
        lines.append("### Supervisor Plan")
        lines.append(_truncate(result.supervisor_plan))

    # Delegation requests
    if hasattr(result, "delegation_requests") and result.delegation_requests:
        lines.append(f"### Delegations ({len(result.delegation_requests)} specialists invoked)")
        for i, dr in enumerate(result.delegation_requests, 1):
            ticker = dr.get("ticker", dr.get("tickers", "?"))
            sections = dr.get("sections", dr.get("section", "?"))
            sub_q = dr.get("sub_question", dr.get("question", "?"))
            lines.append(f"  Delegation {i}: ticker={ticker}, sections={sections}")
            lines.append(f"    Sub-question: {sub_q}")

    # Specialist outputs
    if hasattr(result, "specialist_outputs") and result.specialist_outputs:
        lines.append("### Specialist Outputs")
        for spec_id, output in result.specialist_outputs.items():
            lines.append(f"  [{spec_id}]: {_truncate(output)}")

    # Tool calls (from specialists/synthesizer)
    if hasattr(result, "tool_calls_log") and result.tool_calls_log:
        lines.append(f"### Tool Calls (across all agents): {len(result.tool_calls_log)}")
        for i, tc in enumerate(result.tool_calls_log, 1):
            tool_name = tc.get("tool", tc.get("name", "unknown"))
            tool_args = tc.get("args", tc.get("arguments", {}))
            tool_result = tc.get("result", tc.get("output", ""))
            lines.append(f"  {i}. {tool_name}({_format_args(tool_args)})")
            lines.append(f"     → {_format_tool_result(tool_name, tool_result)}")

    # Reflection
    _append_reflection(lines, result)

    # Token breakdown
    if hasattr(result, "token_breakdown") and result.token_breakdown:
        lines.append("### Token Breakdown")
        for role, usage in result.token_breakdown.items():
            if hasattr(usage, "total_tokens"):
                lines.append(f"  {role}: {usage.total_tokens} tokens")
            else:
                lines.append(f"  {role}: {usage} tokens")

    lines.append(f"Total tokens: {result.metrics.token_usage.total_tokens}")
    return "\n".join(lines)


def _format_args(args) -> str:
    """Format tool arguments into a compact string."""
    if isinstance(args, dict):
        parts = [f"{k}={v!r}" for k, v in args.items()]
        return ", ".join(parts)
    return str(args)


def _append_reflection(lines: list[str], result) -> None:
    """Append reflection verdict info if available."""
    if hasattr(result, "reflection_verdict") and result.reflection_verdict is not None:
        verdict = result.reflection_verdict
        verdict_str = verdict.value if hasattr(verdict, "value") else str(verdict)
        lines.append(f"Reflection verdict: {verdict_str}")
        if hasattr(result, "was_revised"):
            lines.append(f"Was revised after reflection: {result.was_revised}")
    else:
        lines.append("Reflection: Not performed or not available")
