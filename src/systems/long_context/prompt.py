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

from src.common.ingestion import ProcessedFiling

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


FEW_SHOT_EXAMPLES = """\
These examples show the expected pattern for three common query types.
Notice that the agent reads the filings directly from the prompt and
only invokes a tool when arithmetic is required.

### Example 1 — Single-company targeted question
User: "What are the key cybersecurity risks Apple identifies in its FY2024 10-K?"
-> Read the AAPL FY2024 'Risk Factors' section from the Available Filings
   block.
-> Answer: "Apple's FY2024 10-K identifies the following cybersecurity
   risks (AAPL, FY2024, Risk Factors): (1) ... (2) ... (3) ..."

### Example 2 — Cross-company comparison with math
User: "What is the difference in total revenue between AAPL and MSFT for FY2024?"
-> Read AAPL FY2024 'Financial Statements' -> total net sales = $391,035M.
-> Read MSFT FY2024 'Financial Statements' -> total revenue = $245,122M.
-> calculate(expression="391035 - 245122") -> "145913"
-> Answer: "AAPL FY2024 total net sales were $391,035M (AAPL, FY2024,
   Financial Statements); MSFT FY2024 total revenue was $245,122M
   (MSFT, FY2024, Financial Statements). Difference: $145,913M
   (~$145.9B)."

### Example 3 — Multi-year trend
User: "How did Microsoft's net income evolve from FY2022 to FY2024?"
-> Read MSFT FY2022, FY2023, FY2024 'Financial Statements'.
-> Extract net income for each year.
-> Answer: "Microsoft's net income grew from $X in FY2022 (MSFT, FY2022,
   Financial Statements) to $Y in FY2023 (MSFT, FY2023, Financial
   Statements) to $Z in FY2024 (MSFT, FY2024, Financial Statements),
   a cumulative change of W % (use calculate for the percentage)."
"""


OUTPUT_RULES = """\
## Output Rules

- Be precise with numbers — quote exact figures from the filings.
- Always cite the source using "({TICKER}, FY{YEAR}, {SECTION_NAME})".
- If the data is not in the inlined filings, say so explicitly. Do NOT
  invent or extrapolate.
- For any calculation, use the calculate tool. Never compute mentally.
"""


# ---------------------------------------------------------------------------
#  Filing block formatting
# ---------------------------------------------------------------------------

def _fiscal_year_from_metadata(filing: ProcessedFiling) -> str:
    """
    Derive a 4-digit fiscal-year string from filing metadata.

    Falls back to the filing-date year if `fiscal_year_end` is missing.
    """
    fye = filing.metadata.fiscal_year_end or ""
    if len(fye) >= 4 and fye[:4].isdigit():
        return fye[:4]
    fd = filing.metadata.filing_date or ""
    if len(fd) >= 4 and fd[:4].isdigit():
        return fd[:4]
    return "unknown"


def _format_single_filing(filing: ProcessedFiling) -> str:
    """
    Render one filing as a Markdown block.

    Format:
        ### {TICKER} FY{YEAR} - {COMPANY_NAME}

        Sections: {section1}, {section2}, ...

        {full markdown content}
    """
    ticker = filing.metadata.ticker
    year = _fiscal_year_from_metadata(filing)
    company = filing.metadata.company_name
    section_names = list(filing.sections.keys()) if filing.sections else ["Full Text"]

    header = f"### {ticker} FY{year} - {company}\n\n"
    sections_line = f"Sections: {', '.join(section_names)}\n\n"
    body = filing.to_markdown()

    return header + sections_line + body


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
        key=lambda f: (f.metadata.ticker, _fiscal_year_from_metadata(f)),
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
