"""
Unit tests for the list_filings tool.
"""

from src.common.ingestion import FilingMetadata, ProcessedFiling
from src.common.tools.list_filings import create_list_filings_tool


def _make_filing(ticker: str, fy: str, sections: list[str]) -> ProcessedFiling:
    """Create a minimal ProcessedFiling for testing."""
    return ProcessedFiling(
        metadata=FilingMetadata(
            ticker=ticker,
            company_name=f"{ticker} Inc.",
            cik="0001234567",
            filing_date="2025-02-15",
            accession_number="0001234567-25-000001",
            fiscal_year_end=f"{fy}-12-31",
        ),
        sections={s: f"Content of {s}" for s in sections},
        full_text="Full text content",
    )


class TestListFilings:
    """Tests for list_filings tool."""

    def test_lists_single_filing(self):
        filings = [_make_filing("AAPL", "2024", ["Business", "MD&A"])]
        tool = create_list_filings_tool(filings)
        result = tool.invoke({})

        assert "AAPL" in result
        assert "FY2024" in result
        assert "Business" in result
        assert "MD&A" in result

    def test_lists_multiple_filings(self):
        filings = [
            _make_filing("AAPL", "2024", ["Business"]),
            _make_filing("MSFT", "2024", ["Risk Factors"]),
        ]
        tool = create_list_filings_tool(filings)
        result = tool.invoke({})

        assert "AAPL" in result
        assert "MSFT" in result

    def test_empty_filings(self):
        tool = create_list_filings_tool([])
        result = tool.invoke({})

        assert "No filings" in result

    def test_shows_company_name(self):
        filings = [_make_filing("GOOGL", "2023", ["Business"])]
        tool = create_list_filings_tool(filings)
        result = tool.invoke({})

        assert "GOOGL Inc." in result

    def test_shows_filing_date(self):
        filings = [_make_filing("AAPL", "2024", ["Business"])]
        tool = create_list_filings_tool(filings)
        result = tool.invoke({})

        assert "2025-02-15" in result
