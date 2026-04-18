"""
Reflexion-Style Reflection Verifier for System 2 (Agent RAG).

Implements an external answer-verification chain following the Reflexion
pattern (Shinn et al. 2023). After the ReAct agent produces a draft answer,
a separate LLM call evaluates the answer against four dimensions and
returns a structured verdict. If the verdict is "revise", the agent
receives natural-language feedback and runs one more iteration.

This module is intentionally decoupled from the core ReAct loop:
    agent.invoke()  -> Draft Answer
                    -> Reflection Chain (this module)
                    -> If revise: inject HumanMessage, agent.invoke() again
                    -> Final Answer

The verifier is disabled by default in tests (`reflection.enabled=False`
in the pipeline config override) and can be ablated via config to compare
the raw ReAct baseline against the reflection-augmented variant.
"""

import logging
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Structured Verdict Schema
# ---------------------------------------------------------------------------

class ReflectionVerdict(BaseModel):
    """Structured output from the reflection verifier."""

    status: Literal["accept", "revise"] = Field(
        description=(
            "Final verdict on the candidate answer. 'accept' if the answer "
            "is grounded, numerically accurate, complete, and properly "
            "cited. 'revise' otherwise."
        ),
    )
    feedback: str = Field(
        default="",
        description=(
            "Actionable, concrete feedback for the agent. Empty string when "
            "status='accept'. When status='revise', must describe exactly "
            "what is wrong and what the agent should do next."
        ),
    )
    issues: list[str] = Field(
        default_factory=list,
        description=(
            "Optional list of issue tags for analysis, e.g. "
            "'missing_citation', 'numerical_mismatch', 'incomplete', "
            "'unsupported_claim'. Empty when status='accept'."
        ),
    )


# ---------------------------------------------------------------------------
#  Reflection Prompt
# ---------------------------------------------------------------------------

REFLECTION_PROMPT = """\
You are a meticulous financial-analysis reviewer. Your job is to verify
a draft answer produced by a tool-using agent against the retrieved
source contexts from SEC 10-K filings. You do NOT rewrite the answer
yourself — you only decide whether the answer can be accepted as-is
or must be revised by the agent.

## What you receive

- The original user question.
- The candidate answer the agent produced.
- The full list of retrieved source contexts the agent saw (tool outputs).
- The tool calls the agent made (for reference).

## Evaluation criteria

Check the candidate answer against ALL four dimensions. Any single
failure means the answer must be revised.

1. **Groundedness**: Every factual claim in the answer must be traceable
   to the provided contexts. If the answer contains a claim that is not
   supported by any context, flag as 'unsupported_claim'.

2. **Numerical accuracy**: Every number (revenue, percentages, ratios)
   must match exactly what appears in the source contexts. Off-by-one
   typos, unit confusions ($M vs $B), or fabricated numbers are critical
   failures. Flag as 'numerical_mismatch'.

3. **Completeness**: If the question has multiple parts (e.g. comparison
   of two companies, yes/no plus reasoning), all parts must be addressed.
   Flag as 'incomplete' if any sub-question is skipped.

4. **Citations**: Each claim should reference its source (company ticker,
   fiscal year, section) so the user can verify. Flag as
   'missing_citation' if claims lack source attribution.

## Output format

Return a structured verdict:
- status='accept' ONLY if all four dimensions are satisfied.
- status='revise' + concrete, actionable feedback otherwise. The feedback
  must tell the agent WHAT to fix and HOW (e.g. "Re-retrieve MSFT's
  FY2024 Financial Statements to verify the revenue figure; the answer
  states $245B but the context shows $245.1B").

Be strict but fair: do not reject answers for stylistic preferences. Only
reject when one of the four dimensions is genuinely violated.
"""


_USER_TEMPLATE = """\
# Original question
{question}

# Candidate answer
{answer}

# Retrieved source contexts
{contexts}

# Tool calls made by the agent
{tool_calls}

Now evaluate the candidate answer and return your structured verdict.
"""


# ---------------------------------------------------------------------------
#  Chain Factory
# ---------------------------------------------------------------------------

def build_reflection_chain(
    llm: BaseChatModel,
    include_raw: bool = True,
) -> Runnable:
    """
    Build a reflection chain that takes (question, answer, contexts, tool_calls)
    and returns a ReflectionVerdict.

    Uses `llm.with_structured_output(ReflectionVerdict)` for reliable
    Pydantic parsing. The same LLM instance can be reused from the main
    agent — no separate client is required.

    Args:
        llm: Chat model instance. Should be the same model family as the
             agent to keep cost tracking consistent.
        include_raw: When True (default), the chain returns a dict
            `{"raw": AIMessage, "parsed": ReflectionVerdict,
              "parsing_error": Exception | None}`. This exposes the raw
            `usage_metadata` on the AIMessage for accurate token counting.
            When False, the chain returns the ReflectionVerdict directly
            (legacy, no token metadata).

    Returns:
        A LangChain Runnable.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", REFLECTION_PROMPT),
        ("human", _USER_TEMPLATE),
    ])
    structured_llm = llm.with_structured_output(
        ReflectionVerdict,
        include_raw=include_raw,
    )
    chain = prompt | structured_llm
    logger.info(
        "Reflection chain built (with_structured_output=ReflectionVerdict, "
        "include_raw=%s)", include_raw,
    )
    return chain


# ---------------------------------------------------------------------------
#  Feedback Message Generation
# ---------------------------------------------------------------------------

_FEEDBACK_TEMPLATE = (
    "Reviewer feedback on your previous answer:\n\n"
    "{feedback}\n\n"
    "Issues flagged: {issues}\n\n"
    "Please revise your answer addressing these issues. Use additional "
    "tool calls if you need new evidence — do not restate the same "
    "answer. When you are done, produce the revised final answer."
)


def generate_feedback_message(verdict: ReflectionVerdict) -> HumanMessage | None:
    """
    Convert a reflection verdict into a HumanMessage for the agent.

    Only 'revise' verdicts produce a message. 'accept' verdicts return
    None — the caller uses this as a signal to skip the second
    agent.invoke() and finalize the original answer.

    Args:
        verdict: Reflection verdict from the reflection chain.

    Returns:
        A HumanMessage with structured feedback, or None if the verdict
        was 'accept'.
    """
    if verdict.status == "accept":
        return None

    issues_str = ", ".join(verdict.issues) if verdict.issues else "none listed"
    content = _FEEDBACK_TEMPLATE.format(
        feedback=verdict.feedback.strip() or "(no specific feedback provided)",
        issues=issues_str,
    )
    return HumanMessage(content=content)


# ---------------------------------------------------------------------------
#  Reflection Token Extraction
# ---------------------------------------------------------------------------

def unpack_reflection_result(
    chain_result: Any,
) -> tuple[ReflectionVerdict, int, int]:
    """
    Unpack a reflection chain result into (verdict, prompt_tokens, completion_tokens).

    Handles both output formats:
    - `include_raw=True`: dict `{"raw": AIMessage, "parsed": ReflectionVerdict, ...}`
    - `include_raw=False`: bare ReflectionVerdict (no token metadata)

    Parse-error resilience:
        If `parsed` is None (structured-output parse error), a conservative
        accept-fallback verdict is returned (`status="accept"`,
        `issues=["parse_error"]`). This prevents a single malformed LLM
        response from crashing a long benchmark run. The original exception
        is logged at warning level and token metadata is still extracted
        from `raw` when available, so cost tracking stays accurate.

    Args:
        chain_result: Return value of `build_reflection_chain(...).invoke(...)`.

    Returns:
        Tuple of (verdict, prompt_tokens, completion_tokens). Token counts
        are 0 when metadata is unavailable.
    """
    if isinstance(chain_result, ReflectionVerdict):
        return chain_result, 0, 0

    if isinstance(chain_result, dict):
        parsed = chain_result.get("parsed")
        raw = chain_result.get("raw")
        usage = getattr(raw, "usage_metadata", None) if raw else None
        prompt_tokens = usage.get("input_tokens", 0) if usage else 0
        completion_tokens = usage.get("output_tokens", 0) if usage else 0

        if parsed is None:
            err = chain_result.get("parsing_error")
            logger.warning(
                "Reflection chain failed to parse verdict (%s). Falling back "
                "to conservative accept-verdict so the query does not crash.",
                err,
            )
            fallback = ReflectionVerdict(
                status="accept",
                feedback="",
                issues=["parse_error"],
            )
            return fallback, prompt_tokens, completion_tokens

        return parsed, prompt_tokens, completion_tokens

    raise TypeError(
        f"Unexpected reflection chain result type: {type(chain_result)}"
    )
