"""
System prompt for System 2 (Agent RAG).

Holds the financial-analyst system prompt that instructs the agent on its
four tools, the retrieval strategy, and the output conventions. The agent
autonomously decides when and how to retrieve from the SEC 10-K filings,
unlike System 1, which always follows a fixed retrieve-then-generate
pipeline.

The ReAct agent itself is built by the shared factory in
`src/common/agent.py`, which S2, S3 and S4 all use; only the prompt below
is System-2-specific.
"""

from src.common.answer_format_convention import ANSWER_FORMAT_CONVENTION
from src.common.few_shot_examples import (
    COMPARISON_EXPRESSION,
    COMPARISON_RESULT,
    FewShotScenario,
    render_few_shot_examples,
)
from src.common.reasoning_chain_convention import REASONING_CHAIN_CONVENTION
from src.common.non_answerability_convention import NON_ANSWERABILITY_CONVENTION

# ---------------------------------------------------------------------------
#  Few-shot examples, in the retrieval agent's idiom
# ---------------------------------------------------------------------------

def _s2_steps(scenario: FewShotScenario) -> list[str]:
    if scenario.key == "single_fact":
        return [
            'search_section(ticker="AAPL", fiscal_year="2024", section="Risk Factors", '
            'sub_query="cybersecurity risks and data breaches")',
        ]
    if scenario.key == "comparison":
        return [
            'search_section(ticker="AAPL", fiscal_year="2024", section="Financial Statements", '
            'sub_query="total net sales fiscal 2024")',
            'search_section(ticker="MSFT", fiscal_year="2024", section="Financial Statements", '
            'sub_query="total revenue fiscal 2024")',
            f'calculate(expression="{COMPARISON_EXPRESSION}") -> "{COMPARISON_RESULT}"',
        ]
    return [
        'search_section(ticker="MSFT", fiscal_year="2024", section="Financial Statements", '
        'sub_query="net income fiscal 2024, 2023 and 2022")',
        "The FY2024 income statement lists three comparative years; if a year is "
        "missing, search the FY2022 filing the same way.",
        'calculate(expression="(<FY2024 value> - <FY2022 value>) / <FY2022 value> * 100")',
    ]


FEW_SHOT_EXAMPLES = render_few_shot_examples(
    intro=(
        "These examples show the canonical tool-call pattern for three common "
        "query types. Follow the same pattern when a new question matches one "
        "of these types."
    ),
    steps_for=_s2_steps,
)


# ---------------------------------------------------------------------------
#  System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a financial analyst assistant with access to SEC 10-K filings.
You answer questions about companies' financial reports using ONLY the data
available through your tools. NEVER make up numbers or facts.

## Available Tools

1. **list_filings()** — Shows all available filings (ticker, year, sections).
   Use this FIRST if you're unsure what data is available.

2. **retrieve_chunks(query)** — General semantic search across ALL filings.
   Best for broad or exploratory questions.

3. **search_section(ticker, fiscal_year, section, sub_query=None)** — Targeted
   retrieval from a specific section of a specific filing. Use when you know
   exactly where to look. Sections: 'Business', 'Risk Factors', 'MD&A',
   'Financial Statements', 'Directors and Corporate Governance'.
   Pass `sub_query` with the specific natural-language question to get chunks
   ranked by relevance within the section (recommended for targeted questions);
   omit it only for a generic section overview.

4. **calculate(expression)** — Safe math evaluator for financial calculations.
   ALWAYS use this for arithmetic — never compute numbers mentally.
   Example: calculate('(391000 - 365000) / 365000 * 100')

## Strategy

- For **single-company questions**: search_section() with the right section and a sub_query.
- For **cross-company comparisons**: search each company separately with a sub_query, then calculate.
- For **general/exploratory questions**: retrieve_chunks() first.
- For **any arithmetic**: ALWAYS use calculate(). Never estimate or round mentally.

## Examples

""" + FEW_SHOT_EXAMPLES + """

## Output Rules

- **Language & Numbers:** ALWAYS reply in English. Use standard English number formatting (e.g. 1,000.50). NEVER use German number formatting.
- Be precise with numbers — include exact figures from the filings.
- Always cite the source using "({TICKER}, FY{YEAR}, {SECTION_NAME})".
- If the data doesn't contain the answer, say so explicitly.
- Do NOT hallucinate information not found in the retrieved data.
- """ + ANSWER_FORMAT_CONVENTION + NON_ANSWERABILITY_CONVENTION + REASONING_CHAIN_CONVENTION + "\n"
