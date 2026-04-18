"""
List Filings Tool for System 2 (Agent RAG).

Allows the agent to inspect what SEC 10-K filings are available
in the knowledge base before deciding how to search.

This enables informed planning: the agent can first check which
companies, years, and sections are available, then target specific
sections with search_section().
"""

import logging
from typing import Any

from langchain_core.tools import tool

from src.common.ingestion import ProcessedFiling

logger = logging.getLogger(__name__)


def _format_filings_overview(filings: list[ProcessedFiling]) -> str:
    """Format filing metadata into a readable overview."""
    if not filings:
        return "No filings loaded in the knowledge base."

    lines = ["Available SEC 10-K Filings:", ""]

    for filing in filings:
        m = filing.metadata
        fy = m.fiscal_year_end[:4] if m.fiscal_year_end else "unknown"
        sections = list(filing.sections.keys()) if filing.sections else ["Full Text"]

        lines.append(f"• {m.ticker} (FY{fy})")
        lines.append(f"  Company: {m.company_name}")
        lines.append(f"  Filed: {m.filing_date}")
        lines.append(f"  Sections: {', '.join(sections)}")
        lines.append("")

    return "\n".join(lines)


def create_list_filings_tool(
    filings: list[ProcessedFiling],
) -> Any:
    """
    Factory that creates a list_filings tool bound to loaded filings.

    Args:
        filings: List of processed SEC filings loaded at build time.

    Returns:
        A LangChain @tool function.
    """
    # Pre-compute the overview string (static data)
    overview = _format_filings_overview(filings)

    @tool
    def list_filings() -> str:
        """List all available SEC 10-K filings in the knowledge base.

        Returns ticker, fiscal year, filing date, and available sections
        for each filing. Use this FIRST to understand what data is available
        before making targeted searches with search_section().
        """
        logger.info("list_filings: returning overview of %d filings", len(filings))
        return overview

    return list_filings
