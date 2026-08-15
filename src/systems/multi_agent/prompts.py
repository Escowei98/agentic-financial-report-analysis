"""
System 4: Multi-Agent Long Context Prompts.

Contains the prompts for the Supervisor, Specialist, and Synthesizer agents.
Includes dynamic formatting logic for the parametrized specialists.
"""

from __future__ import annotations

import logging
from typing import Sequence

from src.common.answer_format_convention import ANSWER_FORMAT_CONVENTION
from src.common.ingestion import ProcessedFiling
from src.systems.long_context.prompt import _fiscal_year_from_metadata

logger = logging.getLogger(__name__)


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
"""


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
- """ + ANSWER_FORMAT_CONVENTION + """

## Tools
1. **calculate(expression)** — Safe arithmetic evaluator. ALWAYS use it
   for any non-trivial calculation. Example: calculate('(391035 - 245122) / 245122')

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

## Output Rules
- **Language & Numbers:** ALWAYS reply in English. Use standard English number formatting (e.g. 1,000.50). NEVER use German number formatting.
- Synthesize the findings into a clear, direct answer to the original question.
- Preserve the citations provided by the specialists: "({{TICKER}}, FY{{YEAR}}, {{SECTION_NAME}})".
- Do NOT invent data beyond what the specialists provided.
- If the specialists could not find the answer, state that the information
  is not available in the given filings.
- """ + ANSWER_FORMAT_CONVENTION + "\n"


# ---------------------------------------------------------------------------
#  Dynamic Formatting Logic
# ---------------------------------------------------------------------------

def _format_single_filing_filtered(
    filing: ProcessedFiling, sections: list[str]
) -> str:
    """Render one filing as a Markdown block, filtering for specific sections."""
    ticker = filing.metadata.ticker
    year = _fiscal_year_from_metadata(filing)
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

    header = f"### {ticker} FY{year} - {company}\n\n"
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
        key=lambda f: (f.metadata.ticker, _fiscal_year_from_metadata(f)),
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
