"""
Search Section Tool for System 2 (Agent RAG).

Allows the agent to retrieve chunks from a specific section of a specific
SEC 10-K filing using ChromaDB metadata filtering.

This is a key architectural advantage over System 1: the agent can
make targeted retrievals instead of blindly searching all chunks.
"""

import logging
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Valid section names (must match chunker.py metadata)
VALID_SECTIONS = [
    "Business",
    "Risk Factors",
    "MD&A",
    "Financial Statements",
    "Directors and Corporate Governance",
    "Full Text",
]


def _format_section_results(docs: list[Document], ticker: str, section: str) -> str:
    """Format section search results for LLM consumption."""
    if not docs:
        return (
            f"No chunks found for {ticker} in section '{section}'. "
            f"Try using retrieve_chunks() for a broader search, "
            f"or check available filings with list_filings()."
        )

    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        parts.append(
            f"[{ticker} — {section} — Chunk {i}]\n{doc.page_content}"
        )

    header = f"Found {len(docs)} chunks from {ticker} — {section}:"
    return header + "\n\n---\n\n".join([""] + parts)


def create_search_section_tool(
    vectorstore: Chroma,
    top_k: int = 10,
) -> Any:
    """
    Factory that creates a search_section tool bound to a specific vectorstore.

    Args:
        vectorstore: Pre-built ChromaDB vectorstore with section metadata.
        top_k: Number of chunks to return per section search.

    Returns:
        A LangChain @tool function.
    """

    @tool
    def search_section(ticker: str, fiscal_year: str, section: str) -> str:
        """Retrieve chunks from a specific section of a specific SEC 10-K filing.

        Use this when you know which company, year, and section contains the answer.
        More precise than retrieve_chunks() — directly targets a known section.

        Args:
            ticker: Company ticker symbol (e.g., 'AAPL', 'MSFT', 'AMZN', 'GOOGL').
            fiscal_year: Fiscal year as string (e.g., '2024', '2023').
            section: SEC 10-K section name. Must be one of:
                     'Business', 'Risk Factors', 'MD&A', 'Financial Statements',
                     'Directors and Corporate Governance'
        """
        # Normalize inputs
        ticker_upper = ticker.upper().strip()
        fy = fiscal_year.strip()

        # Validate section
        if section not in VALID_SECTIONS:
            return (
                f"Invalid section '{section}'. "
                f"Valid sections: {', '.join(VALID_SECTIONS)}"
            )

        # Build ChromaDB metadata filter
        where_filter = {
            "$and": [
                {"ticker": {"$eq": ticker_upper}},
                {"fiscal_year": {"$eq": fy}},
                {"section_name": {"$eq": section}},
            ]
        }

        try:
            # Use similarity_search with metadata filter
            # Empty query string → returns all matching chunks
            docs = vectorstore.similarity_search(
                query=f"{ticker_upper} {section}",
                k=top_k,
                filter=where_filter,
            )
        except Exception as e:
            logger.error(
                "search_section failed: ticker=%s, fy=%s, section=%s: %s",
                ticker_upper, fy, section, e,
            )
            return f"Error searching section: {e}"

        logger.info(
            "search_section: %s FY%s %s → %d chunks",
            ticker_upper, fy, section, len(docs),
        )
        return _format_section_results(docs, ticker_upper, section)

    return search_section
