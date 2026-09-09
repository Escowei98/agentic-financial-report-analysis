"""
Shared reasoning-chain instruction, injected symmetrically into all four
systems' prompts (S1-S4) so the same artefact is measured everywhere.

Third constant in the same family as `answer_format_convention.py` and
`non_answerability_convention.py`, under the same discipline: defined once,
imported everywhere, never copy-pasted, so the four systems receive
byte-identical wording.

WHY THIS EXISTS
---------------
Reasoning quality used to be judged off `format_trajectory()`, i.e. off
whatever each architecture happened to expose -- a single retrieval pass for
S1, a ReAct tool log for S2, a supervisor plan plus specialist outputs for S4.
Those are not the same artefact, and the human validation of 2026-09-08
measured what that costs: the judge scored `long_context` 2.58-2.75 points
below the human raters on a 1-5 scale while staying within 0.42 for
`rag_monolith` (see data/results/judge_validation/validation_report_reserve.md).
A differential bias of that size makes a cross-architecture comparison
impossible, whatever the pooled agreement says.

The fix is to stop reading the trace and start reading a chain the systems
emit in one format. What the chain must contain is therefore identical for all
four; where in the architecture it originates is not constrained, because that
is precisely the variable under study.

AN ALTERNATIVE THAT WAS REJECTED
--------------------------------
Segmenting free-text answers into steps with a second LLM. That normaliser's
segmentation decisions would land directly in the denominator of
`groundedness_run` and `validity_run` -- an uncontrolled model in the middle
of the measurement. Asking the systems for a segmented chain moves that
decision into the object under study, where it is observable and symmetric.

WHAT THIS COSTS, AND WHY IT IS ACCEPTED
---------------------------------------
This is a chain-of-thought intervention. It acts on all four systems
identically, but it does change their behaviour and may change their answer
accuracy; numbers from runs before it are not comparable to numbers after.
And it elicits a *verbalised* derivation, which need not be the process that
actually produced the answer (Turpin et al. 2023). Both belong in the
limitations of the thesis, not in the metric.

NO CURLY BRACES IN THIS STRING
-----------------------------
S1 builds its prompt with `ChatPromptTemplate.from_messages`, which reads
`{...}` as a template variable -- that is why the citation form is written
`{{TICKER}}` there and not `{TICKER}`. A convention carrying single braces
would raise at prompt-render time in S1 and nowhere else, i.e. it would break
exactly one of the four systems. The citation form is therefore conveyed by
the worked example instead of by a brace placeholder. Guarded by
tests/test_systems/test_component_symmetry.py.

See docs/decisions/REASONING_QUALITY_SPEC.md sections 4 and 6.
"""

REASONING_CHAIN_CONVENTION = (
    "\nAfter your answer, and separated from it, append a section headed "
    "exactly \"## Reasoning\". List there, as a numbered list, the steps that "
    "carry your answer -- one claim or one inference per step, in the order "
    "they support the conclusion.\n"
    "Tag each step with its kind:\n"
    "[E] a step that states something the filings say. Every [E] step must "
    "carry the source of that statement, in the same ticker / fiscal year / "
    "section form the answer uses -- see the example below.\n"
    "[I] a step that draws a conclusion from earlier steps and adds no new "
    "statement about the filings. An [I] step carries no source.\n"
    "Write the steps you actually relied on, including the ones you would "
    "normally leave unsaid because they seem obvious. Do not add steps you "
    "did not use, and do not merge two claims into one step.\n"
    "Example:\n"
    "## Reasoning\n"
    "1. [E] Total net sales for FY2020 were $274,515M. "
    "(AAPL, FY2020, item_8_income_stmt)\n"
    "2. [E] Total net sales for FY2024 were $391,035M. "
    "(AAPL, FY2024, item_8_income_stmt)\n"
    "3. [I] Sales therefore rose over the period.\n"
    "The answer above this section must stand on its own; a reader who stops "
    "before \"## Reasoning\" must still have the complete answer."
)
