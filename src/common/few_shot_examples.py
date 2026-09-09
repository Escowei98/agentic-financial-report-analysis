"""
Shared few-shot scenarios, rendered into every system's prompt in that
system's own idiom.

Fourth constant in the family of `answer_format_convention.py`,
`non_answerability_convention.py` and `reasoning_chain_convention.py`, and
under the same discipline: the scenario texts are defined once, imported
everywhere, never copy-pasted.

WHY THIS EXISTS
---------------
Until 2026-09-09, S2 and S3 each carried three worked examples in their
system prompt -- question, tool pattern, model answer with citation -- while
S4 carried none in any of its three nodes and S1 carried none at all. Worked
examples steer exactly the things the metrics read: citation form
(`citation_accuracy`), whether arithmetic goes through `calculate`
(`exact_match` on FA-3/FA-4), and the shape of the answer. The three shared
conventions were symmetric by construction; the examples were not, because
they had never been treated as a shared component. For the S3-vs-S4
comparison (FF3) that is a confounder pointing in S3's favour, on the very
question the multi-agent design is meant to answer.

WHAT IS SHARED AND WHAT IS NOT
------------------------------
Shared: the three scenarios -- the user question and the model answer,
byte-identical in every prompt. They mirror the answerable strata the gold
standard actually contains (single fact, cross-company comparison with
arithmetic, multi-year trend). S2's former third example (an exploratory
`retrieve_chunks` question) is dropped: the gold standard holds no
exploratory item, and the scenario cannot be expressed for a system without
a search tool.

Per system: the step lines between question and answer. A retrieval agent
calls `search_section`, a long-context agent reads its inlined filings, a
supervisor delegates, a synthesizer combines specialist findings, and the
non-agentic monolith reads its context block. That difference IS the
variable under study and is deliberately not normalised away.

Every LLM call sees exactly three examples. S4 renders them in each of its
three nodes, but a node is one call and sees only its own three -- the same
count as the single-prompt systems.

NO CURLY BRACES IN THE SHARED TEXT
----------------------------------
S1 renders through `ChatPromptTemplate`, S4's supervisor and specialist
through `str.format`. A brace in a scenario would raise at render time in
some systems and not others. Guarded by
tests/test_systems/test_component_symmetry.py.

See EVAL_DECISION_LOG.md [2026-09-09] "Few-Shot-Beispiele symmetrisiert".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class FewShotScenario:
    key: str
    title: str
    question: str
    answer: str


SCENARIO_SINGLE_FACT = FewShotScenario(
    key="single_fact",
    title="Single-company targeted question",
    question="What are the key cybersecurity risks Apple identifies in its FY2024 10-K?",
    answer=(
        "Apple's FY2024 10-K identifies the following cybersecurity risks "
        "(AAPL, FY2024, Risk Factors): (1) ... (2) ... (3) ..."
    ),
)

SCENARIO_COMPARISON = FewShotScenario(
    key="comparison",
    title="Cross-company comparison with math",
    question="What is the difference in total revenue between AAPL and MSFT for FY2024?",
    answer=(
        "AAPL FY2024 total net sales were $391,035M (AAPL, FY2024, Financial "
        "Statements); MSFT FY2024 total revenue was $245,122M (MSFT, FY2024, "
        "Financial Statements). Difference: $145,913M (~$145.9B)."
    ),
)

SCENARIO_TREND = FewShotScenario(
    key="trend",
    title="Multi-year trend",
    question="How did Microsoft's net income evolve from FY2022 to FY2024?",
    answer=(
        "Microsoft's net income grew from $X in FY2022 (MSFT, FY2022, Financial "
        "Statements) to $Y in FY2023 (MSFT, FY2023, Financial Statements) to $Z "
        "in FY2024 (MSFT, FY2024, Financial Statements) -- an overall rising "
        "trend, a cumulative change of W % over the period."
    ),
)

FEW_SHOT_SCENARIOS: tuple[FewShotScenario, ...] = (
    SCENARIO_SINGLE_FACT,
    SCENARIO_COMPARISON,
    SCENARIO_TREND,
)

# The arithmetic of the comparison scenario, so every system that has a
# `calculate` tool shows the identical call and the identical return.
COMPARISON_EXPRESSION = "391035 - 245122"
COMPARISON_RESULT = "145913"

# Distinctive phrase used by the symmetry tests to make sure the scenario
# text lives here and nowhere else under src/.
PRIVATE_COPY_MARKER = "identifies the following cybersecurity risks"


def render_few_shot_examples(
    intro: str,
    steps_for: Callable[[FewShotScenario], Sequence[str]],
) -> str:
    """Render the three scenarios with system-specific step lines.

    Args:
        intro: One or two sentences framing the examples for this system
            (the analogue of the framing sentence each convention gets).
        steps_for: Returns the step lines for a scenario, in this system's
            idiom, without the trailing answer line -- that line is added
            here so the answer text is guaranteed byte-identical.
    """
    blocks = [intro.strip()]
    for number, scenario in enumerate(FEW_SHOT_SCENARIOS, start=1):
        lines = [f"### Example {number} — {scenario.title}", f'User: "{scenario.question}"']
        lines.extend(f"-> {step}" for step in steps_for(scenario))
        lines.append(f'-> Answer: "{scenario.answer}"')
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
