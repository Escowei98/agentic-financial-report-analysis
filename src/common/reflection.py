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

from src.common.message_parsing import format_tool_calls_for_prompt, parse_agent_messages

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
   A justified refusal is COMPLETE, not incomplete. When the question cannot
   be answered from the contexts -- the data lies outside them, the question
   assumes something they contradict, they do not report the item, or the
   question names no single company/fiscal year -- an answer that says so and
   names which of those applies has fully addressed the question. Do not send
   it back for a figure the sources cannot supply. Do, however, flag as
   'unsupported_claim' a refusal that goes further than the sources allow,
   e.g. one asserting that something did not happen when the contexts are
   merely silent about it.

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


# ---------------------------------------------------------------------------
#  Shared single-pass reflection + correction round (S2, S3)
# ---------------------------------------------------------------------------

def run_reflection_pass(
    *,
    agent: Any,
    messages: list,
    question: str,
    reflection_chain: Runnable,
    recursion_limit: int,
    empty_contexts_placeholder: str,
    contexts_preamble: str = "",
    log_prefix: str = "",
) -> tuple[list, ReflectionVerdict, bool, int, int]:
    """
    Run the verifier on a draft answer and, if it asks for a revision,
    re-invoke the agent exactly once with the feedback appended.

    Shared verbatim by System 2 and System 3. Both re-invoke their ReAct
    agent with the *full* prior message history plus the feedback message,
    so both are exposed to the same failure mode and both need the same
    guard (see below). Keeping this in one place is what makes the symmetry
    of the self-correction capability between S2 and S3 structural — thesis
    3.5.1 requires the agentic systems not to differ in self-correction, and
    two hand-maintained copies had already drifted apart on exactly that
    point.

    System 4 does not use this helper: its correction round is a conditional
    edge back to the synthesizer node, which builds a fresh agent and passes
    a single message rather than replaying a tool-call history. It shares the
    chain, the verdict schema, the feedback generation and the single-pass
    policy, but not the re-invocation mechanics, because it has none.

    Args:
        agent: Compiled LangGraph agent to re-invoke on a `revise` verdict.
        messages: Message history of the first (draft) pass.
        question: Original user question.
        reflection_chain: Chain from `build_reflection_chain`.
        recursion_limit: Recursion limit for the re-invocation.
        empty_contexts_placeholder: System-specific wording used when the
            draft produced no tool outputs. Kept as a parameter rather than
            unified, because the two systems' placeholders differ in the
            prompt text that actually reaches the verifier and changing
            either one would alter measured behaviour.
        contexts_preamble: Text placed above the contexts block describing
            what this architecture's tool outputs actually are. Empty for a
            retrieval system, whose tool outputs ARE the source evidence the
            verifier's groundedness criterion assumes. Non-empty for a
            long-context system, whose evidence sits inlined in the system
            prompt and never passes through here: without the preamble the
            verifier reads a bare `calculate` result under the heading
            "Retrieved source contexts" and necessarily concludes that
            nothing is grounded. See EVAL_DECISION_LOG.md [2026-09-08].
        log_prefix: Optional prefix for log lines (e.g. "S3 ").

    Returns:
        Tuple of (messages, verdict, was_revised, prompt_tokens,
        completion_tokens). `messages` is the possibly revised history.
    """
    draft_answer, draft_tool_calls, draft_contexts, _ = parse_agent_messages(messages)

    contexts_block = (
        "\n\n---\n\n".join(draft_contexts)
        if draft_contexts
        else empty_contexts_placeholder
    )
    if contexts_preamble:
        contexts_block = f"{contexts_preamble}\n\n{contexts_block}"

    chain_result = reflection_chain.invoke({
        "question": question,
        "answer": draft_answer,
        "contexts": contexts_block,
        "tool_calls": format_tool_calls_for_prompt(draft_tool_calls),
    })
    verdict, prompt_tokens, completion_tokens = unpack_reflection_result(chain_result)
    logger.info(
        "%sReflection verdict: status=%s, issues=%s",
        log_prefix, verdict.status, verdict.issues,
    )

    was_revised = False
    feedback_msg = generate_feedback_message(verdict)
    if feedback_msg is not None:
        revised_messages = messages + [feedback_msg]
        try:
            second_result = agent.invoke(
                {"messages": revised_messages},
                config={"recursion_limit": recursion_limit},
            )
            messages = second_result.get("messages", revised_messages)
            was_revised = True
        except Exception as e:
            # Known upstream flakiness: Gemini 2.5's per-tool-call "thought
            # signature" occasionally fails to round-trip through
            # langchain-google-vertexai when the full message history (incl.
            # prior tool calls) is resent, surfacing as InvalidArgument
            # "must include at least one parts field". Not reproducible
            # deterministically per-question, so it can't be fixed at the
            # message-construction level. Fall back to the pre-revision
            # draft rather than losing the query outright.
            logger.warning(
                "%sRevision re-invoke failed (%s: %s); keeping pre-revision "
                "draft answer.",
                log_prefix, type(e).__name__, e,
            )

    return messages, verdict, was_revised, prompt_tokens, completion_tokens
