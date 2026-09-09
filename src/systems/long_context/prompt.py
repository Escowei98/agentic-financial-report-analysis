"""
System 3: Long-Context System Prompt Construction.

Builds the agent's system prompt by inlining ALL processed filings into
a single Markdown block under '## Available Filings'. Replaces the
retrieval stack of S1/S2 with a static prompt-resident knowledge base.

The resulting prompt is computed once at pipeline build time and reused
for every query (the `Filings` block is identical across queries; only
the user question varies). With Vertex AI implicit prefix caching, the
shared prefix is automatically reused server-side across consecutive
calls — this module does not implement explicit caching.
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
from src.common.reasoning_chain_convention import REASONING_CHAIN_CONVENTION
from src.common.ingestion import ProcessedFiling, fiscal_year_from_metadata
from src.common.non_answerability_convention import NON_ANSWERABILITY_CONVENTION

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Static prompt fragments
# ---------------------------------------------------------------------------

ROLE_AND_INSTRUCTIONS = """\
You are a financial analyst assistant. You have direct access to the
full text of every available SEC 10-K filing in the section "Available
Filings" below. ALL relevant data is already in your context — do NOT
ask for retrieval, do NOT invent data, do NOT speculate beyond what is
written in the filings.

Answer the user's question by reading and citing the relevant filing(s)
in your context. Quote precise numbers exactly as they appear; never
estimate or round mentally. Always cite the source using the format
"({TICKER}, FY{YEAR}, {SECTION_NAME})".
IMPORTANT: ALWAYS reply in English. Use standard English number formatting (e.g. 1,000.50). NEVER use German number formatting.
"""


TOOL_USAGE_GUIDANCE = """\
You have two tools available:

1. **list_filings()** — Returns a structured overview of every filing in
   the knowledge base (ticker, fiscal year, sections present). Use it
   only if you are unsure which filings exist or which sections one
   contains. Most questions can be answered without this tool because
   the filings are already inlined below.

2. **calculate(expression)** — Safe arithmetic evaluator. ALWAYS use it
   for any non-trivial calculation: differences, percentages, ratios,
   YoY growth, averages. NEVER compute numbers in your head.
   Example: calculate('(391000 - 365000) / 365000 * 100')
"""


def _s3_steps(scenario: FewShotScenario) -> list[str]:
    if scenario.key == "single_fact":
        return ["Read the AAPL FY2024 'Risk Factors' section from the Available Filings block."]
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


FEW_SHOT_EXAMPLES = render_few_shot_examples(
    intro=(
        "These examples show the expected pattern for three common query "
        "types. Notice that the agent reads the filings directly from the "
        "prompt and only invokes a tool when arithmetic is required."
    ),
    steps_for=_s3_steps,
)


OUTPUT_RULES = """\
## Output Rules

- Be precise with numbers — quote exact figures from the filings.
- Always cite the source using "({TICKER}, FY{YEAR}, {SECTION_NAME})".
- If the data is not in the inlined filings, say so explicitly. Do NOT
  invent or extrapolate.
- For any calculation, use the calculate tool. Never compute mentally.
- """ + ANSWER_FORMAT_CONVENTION + NON_ANSWERABILITY_CONVENTION + REASONING_CHAIN_CONVENTION + "\n"


# ---------------------------------------------------------------------------
#  Filing block formatting
# ---------------------------------------------------------------------------

def _format_single_filing(filing: ProcessedFiling) -> str:
    """
    Render one filing as a Markdown block.

    Format:
        ### {TICKER} FY{YEAR} - {COMPANY_NAME}

        Sections: {section1}, {section2}, ...

        {full markdown content}
    """
    ticker = filing.metadata.ticker
    year = fiscal_year_from_metadata(filing)
    company = filing.metadata.company_name
    section_names = list(filing.sections.keys()) if filing.sections else ["Full Text"]

    header = f"### {ticker} FY{year} - {company}\n\n"
    # Same coverage line list_filings prints, so the inlined filing says
    # which prior years it reports (see src/common/corpus_coverage.py).
    coverage_line = f"Coverage: {describe_coverage(year)}\n"
    sections_line = f"Sections: {', '.join(section_names)}\n\n"
    body = filing.to_markdown(include_identifiers=False)

    return header + coverage_line + sections_line + body


def _format_filings_block(filings: Sequence[ProcessedFiling]) -> str:
    """
    Render the full 'Available Filings' block.

    Filings are sorted deterministically by (ticker, fiscal_year) so the
    prompt is stable across builds — this is required for Vertex AI
    implicit prefix caching to actually hit.
    """
    if not filings:
        return "(No filings loaded.)"

    sorted_filings = sorted(
        filings,
        key=lambda f: (f.metadata.ticker, fiscal_year_from_metadata(f)),
    )
    blocks = [_format_single_filing(f) for f in sorted_filings]
    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def build_system_prompt(filings: Sequence[ProcessedFiling]) -> str:
    """
    Compose the full S3 system prompt with all filings inlined.

    The prompt structure is:
        1. Role & instructions
        2. Tool usage guidance
        3. Available Filings (sorted, full Markdown)
        4. Few-shot examples
        5. Output rules

    The filings block dominates the prompt size (~600k tokens for the
    current 12-filing knowledge base). This function is intended to be
    called once at pipeline build time; the result is then reused as a
    static system prompt for every query.

    Args:
        filings: Processed SEC 10-K filings.

    Returns:
        Composed system prompt string.
    """
    filings_block = _format_filings_block(filings)
    prompt = (
        ROLE_AND_INSTRUCTIONS
        + "\n\n## Tools\n\n"
        + TOOL_USAGE_GUIDANCE
        + "\n\n## Available Filings\n\n"
        + filings_block
        + "\n\n## Examples\n\n"
        + FEW_SHOT_EXAMPLES
        + "\n\n"
        + OUTPUT_RULES
    )

    logger.info(
        "Built S3 system prompt: %d filings, %d total prompt characters",
        len(filings), len(prompt),
    )
    return prompt
