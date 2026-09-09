"""
System 4: Multi-Agent Long Context Prompts.

Contains the prompts for the Supervisor, Specialist, and Synthesizer agents.
Includes dynamic formatting logic for the parametrized specialists.

Convention placement, and one documented asymmetry
--------------------------------------------------
ANSWER_FORMAT_CONVENTION sits in the two nodes that emit user-facing text
(Specialist, Synthesizer). NON_ANSWERABILITY_CONVENTION sits in those two AND
in the Supervisor -- a deliberate FP-5 exception.

The Supervisor is the only node holding the corpus metadata: the specialists
see filing content but not the corpus boundary, and the synthesizer sees only
specialist answers. Withholding the convention from it would leave S4 as the
one architecture with no node able to recognise an out-of-corpus question,
which would make the FA-Refusal comparison measure a prompt-placement
artefact instead of the multi-agent topology H3 is about.

REASONING_CHAIN_CONVENTION sits in the Synthesizer ALONE. It is the node that
emits the answer under evaluation, and the spec measures the derivation that
carries the final answer, not the orchestration around it
(REASONING_QUALITY_SPEC.md section 4.4). Giving it to the specialists as well
would produce one chain per specialist plus the synthesizer's -- S4 would be
the only architecture emitting several competing chains per question, and
whichever one the parser picked would be an instrumentation choice rather than
a property of the system.

The convention text itself is byte-identical everywhere; only the framing
sentence above it differs per node, exactly as it already does for the answer
format convention.

Note a consequence that is architectural rather than fixable here: when the
Supervisor declines without delegating, the Synthesizer receives no specialist
output and therefore cannot relay the Supervisor's specific diagnosis -- it
falls back on the convention with only the question in hand. That is the
context isolation H3 examines, and it is reported, not engineered around. The
same path can leave S4 without a reasoning chain on such an item; that is
harmless for the metric, which is computed on the answerable stratum only
(REASONING_QUALITY_SPEC.md section 8.4), but it is counted in
`chain_emission_rate` either way.
"""

from __future__ import annotations

import logging
from typing import Sequence

from src.common.answer_format_convention import ANSWER_FORMAT_CONVENTION
from src.common.corpus_coverage import describe_coverage
from src.common.few_shot_examples import (
    COMPARISON_EXPRESSION,
    COMPARISON_RESULT,
    FewShotScenario,
    render_few_shot_examples,
)
from src.common.ingestion import ProcessedFiling, fiscal_year_from_metadata
from src.common.non_answerability_convention import NON_ANSWERABILITY_CONVENTION
from src.common.reasoning_chain_convention import REASONING_CHAIN_CONVENTION

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Few-shot examples, once per node in that node's idiom
#
#  The scenarios (question + answer) are the shared constant; only the step
#  lines differ. Each node is one LLM call and sees exactly three examples,
#  the same count S2 and S3 see in their single prompt. The supervisor's
#  version ends where its turn ends -- the answer line is labelled as the
#  synthesizer's output so the supervisor is not tempted to write it.
# ---------------------------------------------------------------------------

def _supervisor_steps(scenario: FewShotScenario) -> list[str]:
    if scenario.key == "single_fact":
        return [
            'delegate_to_specialist(tickers=["AAPL"], sections=["Risk Factors"], '
            'sub_question="Which cybersecurity risks does the FY2024 10-K identify?")',
            "End your turn. The Synthesizer then produces the final answer:",
        ]
    if scenario.key == "comparison":
        return [
            'delegate_to_specialist(tickers=["AAPL"], sections=["Financial Statements"], '
            'sub_question="Total net sales for FY2024")',
            'delegate_to_specialist(tickers=["MSFT"], sections=["Financial Statements"], '
            'sub_question="Total revenue for FY2024")',
            "End your turn. The Synthesizer computes the difference and produces "
            "the final answer:",
        ]
    return [
        'delegate_to_specialist(tickers=["MSFT"], sections=["Financial Statements"], '
        'sub_question="Net income for FY2022, FY2023 and FY2024")',
        "End your turn. The Synthesizer computes the cumulative change and "
        "produces the final answer:",
    ]


def _specialist_steps(scenario: FewShotScenario) -> list[str]:
    if scenario.key == "single_fact":
        return ["Read the AAPL FY2024 'Risk Factors' section from your Available Data block."]
    if scenario.key == "comparison":
        return [
            "Read AAPL FY2024 'Financial Statements' -> total net sales = $391,035M.",
            "Read MSFT FY2024 'Financial Statements' -> total revenue = $245,122M.",
            f'calculate(expression="{COMPARISON_EXPRESSION}") -> "{COMPARISON_RESULT}"',
        ]
    return [
        "Read MSFT FY2024 'Financial Statements'; the income statement lists "
        "FY2024, FY2023 and FY2022 side by side (fall back to the FY2022 filing "
        "for any year it does not show).",
        "Extract net income for each year.",
        'calculate(expression="(<FY2024 value> - <FY2022 value>) / <FY2022 value> * 100")',
    ]


def _synthesizer_steps(scenario: FewShotScenario) -> list[str]:
    if scenario.key == "single_fact":
        return [
            "Specialist (AAPL, Risk Factors) reported the cybersecurity risks with "
            "citation (AAPL, FY2024, Risk Factors).",
        ]
    if scenario.key == "comparison":
        return [
            "Specialist (AAPL, Financial Statements) reported total net sales = "
            "$391,035M (AAPL, FY2024, Financial Statements).",
            "Specialist (MSFT, Financial Statements) reported total revenue = "
            "$245,122M (MSFT, FY2024, Financial Statements).",
            f'calculate(expression="{COMPARISON_EXPRESSION}") -> "{COMPARISON_RESULT}"',
        ]
    return [
        "Specialist (MSFT, Financial Statements) reported net income for FY2022, "
        "FY2023 and FY2024, each with its citation.",
        'calculate(expression="(<FY2024 value> - <FY2022 value>) / <FY2022 value> * 100")',
    ]


SUPERVISOR_FEW_SHOT_EXAMPLES = render_few_shot_examples(
    intro=(
        "These examples show the expected delegation pattern for three common "
        "query types. Your turn ends once the specialists have reported; the "
        "final answer shown is what the Synthesizer produces from their findings."
    ),
    steps_for=_supervisor_steps,
)

SPECIALIST_FEW_SHOT_EXAMPLES = render_few_shot_examples(
    intro=(
        "These examples show the expected pattern for three common sub-question "
        "types. Read the filing sections in your Available Data block directly "
        "and invoke a tool only when arithmetic is required."
    ),
    steps_for=_specialist_steps,
)

SYNTHESIZER_FEW_SHOT_EXAMPLES = render_few_shot_examples(
    intro=(
        "These examples show the expected pattern for three common query types. "
        "Combine the specialists' findings, keep their citations, and invoke a "
        "tool only when cross-specialist arithmetic is required."
    ),
    steps_for=_synthesizer_steps,
)


# ---------------------------------------------------------------------------
#  Supervisor Prompt
# ---------------------------------------------------------------------------

SUPERVISOR_PROMPT = """\
Role: You are an orchestrator for financial analysis. You do NOT analyze
data yourself — you decompose questions and delegate to specialists.

You have access to metadata about available SEC 10-K filings. You do NOT
have direct access to the filing contents. To answer the user's question,
you must spawn specialized analysts using the `delegate_to_specialist` tool.

Available Filings Metadata:
{list_filings_output}

Tools:
1. list_filings(): Refresh or view the metadata overview of all filings.
2. delegate_to_specialist(tickers, sections, sub_question): 
   Spawn a specialist who sees ONLY the specified tickers × sections.

Strategy:
- Identify which tickers and sections are needed to answer the question.
- Decompose complex questions into focused sub-questions.
- For cross-company questions (e.g. comparing AAPL and MSFT): separate 
  specialists per ticker (same section) or one specialist for both tickers
  depending on the complexity of the comparison.
- For cross-section questions: combine sections for a single ticker if 
  they are closely related, or separate if independent.
- Do NOT try to answer the question yourself or guess the data. You must
  use `delegate_to_specialist` to get the actual data from the filings.
- After receiving the specialist outputs, end your turn so the Synthesizer
  can aggregate the results. You do not need to synthesize the final answer.

Output:
Whenever you delegate, provide a brief reasoning for your decomposition strategy.

## Examples

""" + SUPERVISOR_FEW_SHOT_EXAMPLES + """

Non-Answerability:
You hold the corpus metadata and the specialists do not, so you are the only
node that can see whether a question falls outside the corpus altogether. When
it does, do not delegate: state the finding and end your turn.
""" + NON_ANSWERABILITY_CONVENTION + "\n"


# ---------------------------------------------------------------------------
#  Specialist Prompt
# ---------------------------------------------------------------------------

SPECIALIST_PROMPT_TEMPLATE = """\
Role: You are a specialized financial analyst. You have direct access to
a focused subset of SEC 10-K filings shown below. Answer the sub-question
assigned to you by the supervisor using ONLY this data.

Your Scope: Tickers: {tickers} | Sections: {sections}

## Output Rules
- **Language & Numbers:** ALWAYS reply in English. Use standard English number formatting (e.g. 1,000.50). NEVER use German number formatting.
- Be precise with numbers — quote exact figures from the filings.
- Always cite the source using "({{TICKER}}, FY{{YEAR}}, {{SECTION_NAME}})".
- If the data is not in your inlined filings, say so explicitly. Do NOT
  invent or extrapolate.
- For any calculation, use the calculate tool. Never compute mentally.
- Keep your answer focused on the sub-question. The supervisor relies on
  your exact extraction.
- """ + ANSWER_FORMAT_CONVENTION + NON_ANSWERABILITY_CONVENTION + """

## Tools
1. **calculate(expression)** — Safe arithmetic evaluator. ALWAYS use it
   for any non-trivial calculation. Example: calculate('(391035 - 245122) / 245122')

## Examples

""" + SPECIALIST_FEW_SHOT_EXAMPLES + """

## Available Data (Filtered to your scope)
{inlined_filing_sections}
"""


# ---------------------------------------------------------------------------
#  Synthesizer Prompt
# ---------------------------------------------------------------------------

SYNTHESIZER_PROMPT = """\
Role: You are a synthesizer for financial analysis. You receive partial
answers from specialized financial analysts and must aggregate them into
a single, coherent final answer to the original user question.

Original Question: {user_query}

## Specialist Outputs
The supervisor delegated the task to the following specialists. Here are
their findings:

{formatted_specialist_outputs}

## Tools
1. **calculate(expression)** — Safe arithmetic evaluator. ALWAYS use it
   for any cross-specialist arithmetic (e.g., comparing numbers from
   different specialists). Never compute mentally.

## Examples

""" + SYNTHESIZER_FEW_SHOT_EXAMPLES + """

## Output Rules
- **Language & Numbers:** ALWAYS reply in English. Use standard English number formatting (e.g. 1,000.50). NEVER use German number formatting.
- Synthesize the findings into a clear, direct answer to the original question.
- Preserve the citations provided by the specialists: "({{TICKER}}, FY{{YEAR}}, {{SECTION_NAME}})".
- Do NOT invent data beyond what the specialists provided.
- If the specialists could not find the answer, state that the information
  is not available in the given filings.
- """ + ANSWER_FORMAT_CONVENTION + NON_ANSWERABILITY_CONVENTION + REASONING_CHAIN_CONVENTION + "\n"


# ---------------------------------------------------------------------------
#  Dynamic Formatting Logic
# ---------------------------------------------------------------------------

def _format_single_filing_filtered(
    filing: ProcessedFiling, sections: list[str]
) -> str:
    """Render one filing as a Markdown block, filtering for specific sections."""
    ticker = filing.metadata.ticker
    year = fiscal_year_from_metadata(filing)
    company = filing.metadata.company_name

    # Filter sections. If "Full Text" is requested or no specific sections exist,
    # we might need to fallback. Assuming filing.sections is populated.
    content_blocks = []
    included_sections = []

    if filing.sections:
        for sec in sections:
            if sec in filing.sections:
                content_blocks.append(f"## {sec}\n{filing.sections[sec]}")
                included_sections.append(sec)
            elif sec == "Full Text":
                # Special case if they want the whole thing
                content_blocks.append(filing.to_markdown(include_identifiers=False))
                included_sections.append("Full Text")
    else:
        # If the filing wasn't chunked by section, just include the full text
        content_blocks.append(filing.to_markdown(include_identifiers=False))
        included_sections.append("Full Text (Unsectioned)")

    if not content_blocks:
        return f"### {ticker} FY{year} - {company}\n\n(Requested sections not found in this filing.)"

    # Same coverage line list_filings prints, so the inlined filing says
    # which prior years it reports (see src/common/corpus_coverage.py).
    header = (
        f"### {ticker} FY{year} - {company}\n\n"
        f"Coverage: {describe_coverage(year)}\n"
    )
    sections_line = f"Sections Included: {', '.join(included_sections)}\n\n"
    body = "\n\n".join(content_blocks)

    return header + sections_line + body


def build_specialist_prompt(
    tickers: list[str],
    sections: list[str],
    filings: Sequence[ProcessedFiling]
) -> str:
    """
    Compose the system prompt for a parametrised specialist.
    
    Filters the filings to only include the requested tickers and sections.
    """
    # 1. Filter filings by ticker
    # If tickers is empty, maybe include all? Let's assume it should match.
    target_tickers = {t.upper() for t in tickers}
    filtered_filings = [
        f for f in filings
        if not target_tickers or f.metadata.ticker.upper() in target_tickers
    ]

    # 2. Sort for determinism
    sorted_filings = sorted(
        filtered_filings,
        key=lambda f: (f.metadata.ticker, fiscal_year_from_metadata(f)),
    )

    # 3. Format with section filtering
    blocks = [_format_single_filing_filtered(f, sections) for f in sorted_filings]
    if not blocks:
        filings_block = "(No filings matched the requested scope.)"
    else:
        filings_block = "\n\n---\n\n".join(blocks)

    prompt = SPECIALIST_PROMPT_TEMPLATE.format(
        tickers=", ".join(tickers) if tickers else "ALL",
        sections=", ".join(sections) if sections else "ALL",
        inlined_filing_sections=filings_block
    )

    logger.debug(
        "Built specialist prompt for %s x %s: %d chars",
        tickers, sections, len(prompt)
    )
    return prompt
