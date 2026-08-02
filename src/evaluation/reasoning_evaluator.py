"""
Reasoning Quality Evaluator (Two-Tier LLM-as-a-Judge).

Evaluates the *process quality* of system responses using two tiers:

  Tier 1 — Core Score (S1–S4):
    - Logical Soundness: Are reasoning steps coherent and well-sequenced?
    - Synthesis Quality: Are multiple sources correctly integrated?
    - Evidence Faithfulness: Are claims grounded in cited document evidence?

  Tier 2 — Agentic Score (S2–S4 only):
    - Tool Selection: Are the right tools used at the right time?
    - Error Recovery: Does the system detect and correct mistakes?

This complements the outcome-based RAGAS evaluation (ragas_evaluator.py)
with a process-based assessment. The Core Score is the primary comparison
metric across all systems; the Agentic Score serves as supplementary
diagnostics explaining *why* agentic systems score differently.

Scientific basis:
  - Zheng et al. (2023), "Judging LLM-as-a-Judge with MT-Bench", NeurIPS
  - Liu et al. (2024), "AgentBench", ICLR
  - Zhuge et al. (2024), "Agent-as-a-Judge", ICML 2025
  - Jiao et al. (2024), "AgentPRM: Process-Based Reward Models"
"""

import json
import logging
import re
from dataclasses import dataclass, field

from src.common.llm_client import get_judge_llm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CoreReasoningScores:
    """Tier 1 — Core Score: applicable to ALL systems (S1–S4)."""

    logical_soundness: float = 0.0       # 1-5
    synthesis_quality: float = 0.0       # 1-5
    evidence_faithfulness: float = 0.0   # 1-5
    rationales: dict[str, str] = field(default_factory=dict)

    @property
    def composite(self) -> float:
        """Equal-weighted mean of the 3 core dimensions."""
        return (
            self.logical_soundness
            + self.synthesis_quality
            + self.evidence_faithfulness
        ) / 3.0

    def to_dict(self) -> dict:
        return {
            "logical_soundness": round(self.logical_soundness, 2),
            "synthesis_quality": round(self.synthesis_quality, 2),
            "evidence_faithfulness": round(self.evidence_faithfulness, 2),
            "core_composite": round(self.composite, 2),
            "rationales": self.rationales,
        }


@dataclass
class AgenticReasoningScores:
    """Tier 2 — Agentic Score: only for S2–S4 (systems with tools & reflection)."""

    tool_selection: float = 0.0     # 1-5
    error_recovery: float = 0.0    # 1-5
    rationales: dict[str, str] = field(default_factory=dict)

    @property
    def composite(self) -> float:
        """Equal-weighted mean of the 2 agentic dimensions."""
        return (self.tool_selection + self.error_recovery) / 2.0

    def to_dict(self) -> dict:
        return {
            "tool_selection": round(self.tool_selection, 2),
            "error_recovery": round(self.error_recovery, 2),
            "agentic_composite": round(self.composite, 2),
            "rationales": self.rationales,
        }


@dataclass
class ReasoningEvalResult:
    """Combined evaluation result for one query × one system."""

    query_id: int = 0
    system_name: str = ""
    core: CoreReasoningScores = field(default_factory=CoreReasoningScores)
    agentic: AgenticReasoningScores | None = None  # None for S1

    def to_dict(self) -> dict:
        result = {
            "query_id": self.query_id,
            "system_name": self.system_name,
            "core": self.core.to_dict(),
        }
        if self.agentic is not None:
            result["agentic"] = self.agentic.to_dict()
        return result


# ---------------------------------------------------------------------------
#  System name constants
# ---------------------------------------------------------------------------

# Systems that have agentic capabilities (tools, reflection)
AGENTIC_SYSTEMS = {"rag_agent", "long_context", "multi_agent"}

# Systems without agentic capabilities
NON_AGENTIC_SYSTEMS = {"rag_monolith"}

# Tool inventories per system (used in Agentic Judge prompt)
SYSTEM_TOOLS = {
    "rag_agent": [
        "search_section(ticker, section, query) — Hybrid BM25+dense retrieval within a specific filing section",
        "retrieve_chunks(query, top_k) — General vector similarity search across all chunks",
        "calculate(expression) — Safe arithmetic evaluator for financial calculations",
        "list_filings() — Lists all available SEC 10-K filings with metadata",
    ],
    "long_context": [
        "calculate(expression) — Safe arithmetic evaluator for financial calculations",
        "list_filings() — Lists all available SEC 10-K filings with metadata",
    ],
    "multi_agent": [
        "delegate_to_specialist(ticker, sections, sub_question) — Delegates a sub-question to a specialist agent with scoped filing access",
        "calculate(expression) — Safe arithmetic evaluator (available to Specialists and Synthesizer)",
        "list_filings() — Lists all available SEC 10-K filings with metadata",
    ],
}


# ---------------------------------------------------------------------------
#  Trajectory Formatting
# ---------------------------------------------------------------------------

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


def _truncate(text: str, limit: int = _EVIDENCE_CHAR_LIMIT) -> str:
    """Truncate text to `limit` chars, appending '...' only if actually cut."""
    text = text.replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _format_s1_trajectory(result) -> str:
    """S1 (RAG Monolith): Minimal trajectory — retrieval + generation."""
    lines = ["## System 1 Trajectory (RAG Monolith — Non-Agentic)"]
    lines.append("Steps: 1 (single retrieval + generation pass)")

    # Show retrieved contexts summary
    num_contexts = len(result.contexts) if hasattr(result, "contexts") else 0
    lines.append(f"Retrieved contexts: {num_contexts} chunks")

    if hasattr(result, "contexts") and result.contexts:
        for i, ctx in enumerate(result.contexts, 1):
            lines.append(f"  Context {i}: {_truncate(ctx)}")

    lines.append(f"Total tokens: {result.metrics.token_usage.total_tokens}")
    return "\n".join(lines)


def _format_s2_trajectory(result) -> str:
    """S2 (Agent RAG): ReAct loop with retrieval tools."""
    lines = ["## System 2 Trajectory (Agent RAG — Single-Agent, RAG-based)"]

    # Tool calls
    if hasattr(result, "tool_calls_log") and result.tool_calls_log:
        lines.append(f"Total tool calls: {len(result.tool_calls_log)}")
        for i, tc in enumerate(result.tool_calls_log, 1):
            tool_name = tc.get("tool", tc.get("name", "unknown"))
            tool_args = tc.get("args", tc.get("arguments", {}))
            tool_result = tc.get("result", tc.get("output", ""))
            lines.append(f"Step {i}: {tool_name}({_format_args(tool_args)})")
            if isinstance(tool_result, str) and tool_result:
                lines.append(f"  → Result: {_truncate(tool_result)}")
    else:
        lines.append("Tool calls: None (direct answer without tool use)")

    # Reflection
    _append_reflection(lines, result)

    lines.append(f"Total tokens: {result.metrics.token_usage.total_tokens}")
    return "\n".join(lines)


def _format_s3_trajectory(result) -> str:
    """S3 (Long Context): ReAct loop with full document context, no retrieval."""
    lines = ["## System 3 Trajectory (Long Context — Single-Agent, Full-Context)"]
    lines.append("Context: All SEC 10-K filings inlined in system prompt (no retrieval)")

    # Tool calls
    if hasattr(result, "tool_calls_log") and result.tool_calls_log:
        lines.append(f"Total tool calls: {len(result.tool_calls_log)}")
        for i, tc in enumerate(result.tool_calls_log, 1):
            tool_name = tc.get("tool", tc.get("name", "unknown"))
            tool_args = tc.get("args", tc.get("arguments", {}))
            tool_result = tc.get("result", tc.get("output", ""))
            lines.append(f"Step {i}: {tool_name}({_format_args(tool_args)})")
            if isinstance(tool_result, str) and tool_result:
                lines.append(f"  → Result: {_truncate(tool_result)}")
    else:
        lines.append("Tool calls: None (direct answer from full context)")

    # Reflection
    _append_reflection(lines, result)

    lines.append(f"Total tokens: {result.metrics.token_usage.total_tokens}")
    return "\n".join(lines)


def _format_s4_trajectory(result) -> str:
    """S4 (Multi-Agent): Supervisor → Specialists → Synthesizer → Reflection."""
    lines = ["## System 4 Trajectory (Multi-Agent — Supervisor + Specialists + Synthesizer)"]

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
            if isinstance(tool_result, str) and tool_result:
                lines.append(f"     → Result: {_truncate(tool_result)}")

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


# ---------------------------------------------------------------------------
#  Judge Prompts
# ---------------------------------------------------------------------------

CORE_JUDGE_PROMPT = """\
You are an expert evaluator assessing the reasoning quality of a financial \
analysis system. You will evaluate the system's reasoning PROCESS, not just \
its final answer.

## Task
Evaluate the following system trajectory on exactly 3 dimensions. \
For each dimension, you MUST first write a detailed rationale (2-4 sentences) \
explaining your assessment, and THEN assign a score from 1 to 5.

## Inputs
**User Question:** {question}
**System Answer:** {answer}
**Reference Answer (Ground Truth):** {ground_truth}
**System Trajectory:**
{trajectory}

## Rubric

Note on thoroughness: a step that provides supporting evidence a careful \
analyst would reasonably include (e.g., showing year-by-year figures before \
stating an overall multi-year trend) is NOT "unnecessary" or "redundant" \
under Logical Soundness, and extra correct detail is NOT a synthesis defect \
under Synthesis Quality. Only penalize steps that are irrelevant, \
contradictory, or that do not support the final answer.

### 1. Logical Soundness (logische_stringenz)
Do the reasoning steps build logically on each other? Is the argumentation \
chain coherent and well-sequenced?
- 5: All steps are logically necessary and correctly sequenced; no unnecessary steps
- 4: Logic is correct but contains 1 unnecessary or redundant step
- 3: Core logic is correct but sequencing is suboptimal or an intermediate step is missing
- 2: Logical gap: a key step is missing or contradicts a previous one
- 1: No recognizable logical structure; steps are arbitrary or contradictory

### 2. Synthesis Quality (synthese_qualitaet)
Are information from different sources correctly integrated into a coherent answer?
- 5: All relevant sources identified and correctly synthesized into a coherent answer
- 4: Synthesis is correct but one marginal additional source was not included
- 3: Core facts from sources present, but the integration is superficial or incomplete
- 2: A main source is missing from the synthesis, distorting the answer
- 1: No synthesis recognizable; answer is based on only one source despite multi-source question

### 3. Evidence Faithfulness (evidenztreue)
Are reasoning steps and intermediate results supported by concrete evidence \
from the financial filings (e.g., citations like "(AAPL, FY2024, Income Statement)")?
- 5: Every factual claim is backed by a concrete source citation; no unsupported assertions
- 4: All key facts are cited; 1 minor fact without explicit source reference
- 3: Core numbers are cited, but derivation or interpretation contains unsupported conclusions
- 2: Multiple facts asserted without source reference; origin of numbers unclear
- 1: No source references; answer contains numbers or claims without recognizable evidence basis

## Output Format
Respond with ONLY a valid JSON object (no markdown, no extra text):
{{
  "logical_soundness": {{
    "rationale": "<your reasoning>",
    "score": <1-5>
  }},
  "synthesis_quality": {{
    "rationale": "<your reasoning>",
    "score": <1-5>
  }},
  "evidence_faithfulness": {{
    "rationale": "<your reasoning>",
    "score": <1-5>
  }}
}}
"""


AGENTIC_JUDGE_PROMPT = """\
You are an expert evaluator assessing the agentic capabilities of a financial \
analysis system. You will evaluate how effectively the system uses its tools \
and handles errors during reasoning.

## Task
Evaluate the following system trajectory on exactly 2 dimensions. \
For each dimension, you MUST first write a detailed rationale (2-4 sentences) \
explaining your assessment, and THEN assign a score from 1 to 5.

## Inputs
**User Question:** {question}
**System Answer:** {answer}
**Reference Answer (Ground Truth):** {ground_truth}
**System Trajectory:**
{trajectory}

**Available Tools for this System:**
{available_tools}

## Rubric

### 1. Tool Selection Accuracy (werkzeugauswahl)
Is the right tool used at the right time with correct parameters? \
Are all necessary tools called and no unnecessary ones?
- 5: Every tool call is purposeful, correctly parameterized, and no needed tool was omitted
- 4: All needed tools were used, but 1 call has suboptimal parameters
- 3: One needed tool was not used OR one unnecessary tool was called
- 2: Multiple incorrect or missing tool calls
- 1: No tools used despite need, OR exclusively wrong tools used

### 2. Error Recovery / Self-Reflection (fehlerkorrektur)
How does the system react to errors, empty results, or ambiguous \
intermediate findings?
- 5: Error detected and correctly fixed, OR no errors occurred and reflection correctly confirms
- 4: Error detected, correction is partially successful
- 3: No error occurred, but no reflection was performed despite being available
- 2: Error occurred but was not detected or not corrected
- 1: Error detected, but the "correction" worsens the result

## Output Format
Respond with ONLY a valid JSON object (no markdown, no extra text):
{{
  "tool_selection": {{
    "rationale": "<your reasoning>",
    "score": <1-5>
  }},
  "error_recovery": {{
    "rationale": "<your reasoning>",
    "score": <1-5>
  }}
}}
"""


# ---------------------------------------------------------------------------
#  JSON Parsing
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """
    Robustly extract a JSON object from LLM output.

    Handles common issues: markdown code fences, trailing text, etc.
    """
    # Strip markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Remove opening fence (with optional language tag)
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        # Remove closing fence
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("Failed to parse JSON from judge output: %s", text[:200])
    return {}


def _safe_score(data: dict, key: str) -> tuple[float, str]:
    """Extract a score and rationale from parsed judge JSON, with fallbacks."""
    entry = data.get(key, {})
    if isinstance(entry, dict):
        score = entry.get("score", 0.0)
        rationale = entry.get("rationale", "")
    elif isinstance(entry, (int, float)):
        score = float(entry)
        rationale = ""
    else:
        score = 0.0
        rationale = ""

    # Clamp to valid range
    score = max(1.0, min(5.0, float(score))) if score > 0 else 0.0
    return score, rationale


# ---------------------------------------------------------------------------
#  Evaluation Functions
# ---------------------------------------------------------------------------

def evaluate_reasoning(
    question: str,
    answer: str,
    trajectory: str,
    ground_truth: str,
    system_name: str,
    query_id: int = 0,
) -> ReasoningEvalResult:
    """
    Evaluate the reasoning quality of a single system response.

    Runs the Core Judge on all systems. Additionally runs the Agentic Judge
    on S2–S4 (systems with tool-use and reflection capabilities).

    Args:
        question: The user's query.
        answer: The system's generated answer.
        trajectory: Formatted trajectory string (from format_trajectory).
        ground_truth: The gold-standard reference answer.
        system_name: One of 'rag_monolith', 'rag_agent', 'long_context',
                     'multi_agent'.
        query_id: Gold standard item ID for tracking.

    Returns:
        ReasoningEvalResult with Core scores (always) and Agentic scores
        (for S2–S4 only).
    """
    # Coerce answer to plain string (handles Gemini Content parts)
    answer = _to_str(answer)

    judge_llm = get_judge_llm(max_tokens=2048)

    # --- Tier 1: Core Score (all systems) ---
    core_prompt = CORE_JUDGE_PROMPT.format(
        question=question,
        answer=answer,
        ground_truth=ground_truth,
        trajectory=trajectory,
    )

    logger.debug("Running Core Judge for %s (query_id=%d)", system_name, query_id)
    core_response = judge_llm.invoke(core_prompt)
    core_text = _to_str(core_response.content)
    core_data = _extract_json(core_text)

    ls_score, ls_rationale = _safe_score(core_data, "logical_soundness")
    sq_score, sq_rationale = _safe_score(core_data, "synthesis_quality")
    ef_score, ef_rationale = _safe_score(core_data, "evidence_faithfulness")

    core_scores = CoreReasoningScores(
        logical_soundness=ls_score,
        synthesis_quality=sq_score,
        evidence_faithfulness=ef_score,
        rationales={
            "logical_soundness": ls_rationale,
            "synthesis_quality": sq_rationale,
            "evidence_faithfulness": ef_rationale,
        },
    )

    # --- Tier 2: Agentic Score (S2–S4 only) ---
    agentic_scores = None
    if system_name in AGENTIC_SYSTEMS:
        tools_list = SYSTEM_TOOLS.get(system_name, [])
        tools_str = "\n".join(f"- {t}" for t in tools_list)

        agentic_prompt = AGENTIC_JUDGE_PROMPT.format(
            question=question,
            answer=answer,
            ground_truth=ground_truth,
            trajectory=trajectory,
            available_tools=tools_str,
        )

        logger.debug("Running Agentic Judge for %s (query_id=%d)", system_name, query_id)
        agentic_response = judge_llm.invoke(agentic_prompt)
        agentic_text = _to_str(agentic_response.content)
        agentic_data = _extract_json(agentic_text)

        ts_score, ts_rationale = _safe_score(agentic_data, "tool_selection")
        er_score, er_rationale = _safe_score(agentic_data, "error_recovery")

        agentic_scores = AgenticReasoningScores(
            tool_selection=ts_score,
            error_recovery=er_score,
            rationales={
                "tool_selection": ts_rationale,
                "error_recovery": er_rationale,
            },
        )

    return ReasoningEvalResult(
        query_id=query_id,
        system_name=system_name,
        core=core_scores,
        agentic=agentic_scores,
    )


def evaluate_reasoning_batch(
    gold_items: list,
    results: list,
    system_name: str,
) -> list[ReasoningEvalResult]:
    """
    Evaluate reasoning quality for a batch of query results.

    Args:
        gold_items: List of GoldStandardItem objects.
        results: List of pipeline result objects (matching gold_items 1:1).
        system_name: System identifier.

    Returns:
        List of ReasoningEvalResult objects (one per query).
    """
    if len(gold_items) != len(results):
        raise ValueError(
            f"Length mismatch: {len(gold_items)} gold items, "
            f"{len(results)} results"
        )

    eval_results = []
    total = len(gold_items)

    for i, (item, result) in enumerate(zip(gold_items, results)):
        logger.info(
            "[%s] Evaluating reasoning %d/%d (id=%d, type=%s)",
            system_name, i + 1, total, item.id, item.fa_type,
        )

        try:
            trajectory = format_trajectory(result, system_name)
            answer = result.answer if hasattr(result, "answer") else str(result)

            eval_result = evaluate_reasoning(
                question=item.question,
                answer=answer,
                trajectory=trajectory,
                ground_truth=item.ground_truth,
                system_name=system_name,
                query_id=item.id,
            )
            eval_results.append(eval_result)

            logger.info(
                "[%s] id=%d → Core: LS=%.1f SQ=%.1f EF=%.1f (composite=%.2f)%s",
                system_name,
                item.id,
                eval_result.core.logical_soundness,
                eval_result.core.synthesis_quality,
                eval_result.core.evidence_faithfulness,
                eval_result.core.composite,
                (
                    " | Agentic: TS=%.1f ER=%.1f (composite=%.2f)"
                    % (
                        eval_result.agentic.tool_selection,
                        eval_result.agentic.error_recovery,
                        eval_result.agentic.composite,
                    )
                    if eval_result.agentic
                    else ""
                ),
            )

        except Exception as e:
            logger.error(
                "[%s] Error evaluating reasoning for id=%d: %s",
                system_name, item.id, e,
            )
            # Append a zero-score result to preserve alignment
            eval_results.append(ReasoningEvalResult(
                query_id=item.id,
                system_name=system_name,
            ))

    # Summary
    if eval_results:
        valid = [r for r in eval_results if r.core.logical_soundness > 0]
        if valid:
            avg_core = sum(r.core.composite for r in valid) / len(valid)
            logger.info(
                "[%s] Reasoning evaluation complete: %d/%d successful, "
                "avg Core composite=%.2f",
                system_name, len(valid), total, avg_core,
            )
            agentic_valid = [r for r in valid if r.agentic is not None]
            if agentic_valid:
                avg_agentic = sum(
                    r.agentic.composite for r in agentic_valid
                ) / len(agentic_valid)
                logger.info(
                    "[%s] Avg Agentic composite=%.2f (%d items)",
                    system_name, avg_agentic, len(agentic_valid),
                )

    return eval_results


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _to_str(value) -> str:
    """Coerce a value to a plain string (handles Gemini Content parts)."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [
            p.get("text", str(p)) if isinstance(p, dict) else str(p)
            for p in value
        ]
        return "\n".join(parts)
    return str(value)
