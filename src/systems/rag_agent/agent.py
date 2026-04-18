"""
LangGraph ReAct Agent for System 2 (Agent RAG).

Defines the agent graph using LangGraph's create_react_agent
with a financial-analyst system prompt and 4 tools.

The agent autonomously decides when/how to retrieve information
from SEC 10-K filings, unlike System 1 which always follows a
fixed retrieve-then-generate pipeline.
"""

import logging
from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)

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

3. **search_section(ticker, fiscal_year, section)** — Targeted retrieval from
   a specific section of a specific filing. Use when you know exactly where
   to look. Sections: 'Business', 'Risk Factors', 'MD&A',
   'Financial Statements', 'Directors and Corporate Governance'.

4. **calculate(expression)** — Safe math evaluator for financial calculations.
   ALWAYS use this for arithmetic — never compute numbers mentally.
   Example: calculate('(391000 - 365000) / 365000 * 100')

## Strategy

- For **single-company questions**: search_section() with the right section.
- For **cross-company comparisons**: search each company separately, then calculate.
- For **general/exploratory questions**: retrieve_chunks() first.
- For **any arithmetic**: ALWAYS use calculate(). Never estimate or round mentally.

## Output Rules

- Be precise with numbers — include exact figures from the filings.
- Always cite the source (company, fiscal year, section) in your answer.
- If the data doesn't contain the answer, say so explicitly.
- Do NOT hallucinate information not found in the retrieved data.
"""


# ---------------------------------------------------------------------------
#  Agent Builder
# ---------------------------------------------------------------------------

def build_agent(
    llm: BaseChatModel,
    tools: Sequence[Any],
    recursion_limit: int = 12,
) -> Any:
    """
    Build a LangGraph ReAct agent with financial analysis tools.

    Args:
        llm: Chat model instance (Gemini 2.0 Flash via Vertex AI).
        tools: List of LangChain tool functions.
        recursion_limit: Max graph recursion depth (prevents infinite loops).
            Should be > 2 * max_iterations to allow for tool calls + responses.

    Returns:
        Compiled LangGraph agent (CompiledStateGraph).
    """
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
    )

    logger.info(
        "Agent built: %d tools, recursion_limit=%d",
        len(tools), recursion_limit,
    )
    return agent
