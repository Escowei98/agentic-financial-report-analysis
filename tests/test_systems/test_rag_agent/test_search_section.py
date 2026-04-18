"""
Unit tests for the search_section tool.
"""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.systems.rag_agent.tools.search_section import (
    VALID_SECTIONS,
    create_search_section_tool,
)


def _make_mock_vectorstore(docs: list[Document] | None = None):
    """Create a mock ChromaDB vectorstore."""
    mock = MagicMock()
    mock.similarity_search.return_value = docs or []
    return mock


class TestSearchSection:
    """Tests for section-filtered retrieval tool."""

    def test_valid_section_returns_results(self):
        docs = [
            Document(
                page_content="Revenue was $391B in FY2024",
                metadata={"ticker": "AAPL", "fiscal_year": "2024", "section_name": "MD&A"},
            ),
        ]
        vs = _make_mock_vectorstore(docs)
        tool = create_search_section_tool(vs, top_k=5)

        result = tool.invoke({
            "ticker": "AAPL",
            "fiscal_year": "2024",
            "section": "MD&A",
        })

        assert "Revenue was $391B" in result
        assert "AAPL" in result

    def test_invalid_section_returns_error(self):
        vs = _make_mock_vectorstore()
        tool = create_search_section_tool(vs, top_k=5)

        result = tool.invoke({
            "ticker": "AAPL",
            "fiscal_year": "2024",
            "section": "Nonexistent Section",
        })

        assert "Invalid section" in result
        assert "Valid sections" in result

    def test_no_results_suggests_alternatives(self):
        vs = _make_mock_vectorstore([])
        tool = create_search_section_tool(vs, top_k=5)

        result = tool.invoke({
            "ticker": "AAPL",
            "fiscal_year": "2024",
            "section": "Business",
        })

        assert "No chunks found" in result
        assert "retrieve_chunks" in result

    def test_ticker_normalized_to_uppercase(self):
        vs = _make_mock_vectorstore([])
        tool = create_search_section_tool(vs, top_k=5)

        tool.invoke({
            "ticker": "aapl",
            "fiscal_year": "2024",
            "section": "Business",
        })

        # Verify the filter used uppercase
        call_args = vs.similarity_search.call_args
        where_filter = call_args.kwargs.get("filter", call_args[1].get("filter", {}))
        conditions = where_filter.get("$and", [])
        ticker_filter = next(c for c in conditions if "ticker" in c)
        assert ticker_filter["ticker"]["$eq"] == "AAPL"

    def test_uses_correct_metadata_filter(self):
        vs = _make_mock_vectorstore([])
        tool = create_search_section_tool(vs, top_k=5)

        tool.invoke({
            "ticker": "MSFT",
            "fiscal_year": "2023",
            "section": "Risk Factors",
        })

        # Verify ChromaDB was called with correct filter
        vs.similarity_search.assert_called_once()
        call_args = vs.similarity_search.call_args
        where_filter = call_args.kwargs.get("filter", call_args[1].get("filter", {}))
        conditions = where_filter.get("$and", [])

        assert {"ticker": {"$eq": "MSFT"}} in conditions
        assert {"fiscal_year": {"$eq": "2023"}} in conditions
        assert {"section_name": {"$eq": "Risk Factors"}} in conditions

    def test_all_valid_sections_accepted(self):
        """All sections from VALID_SECTIONS should be accepted."""
        vs = _make_mock_vectorstore([])
        tool = create_search_section_tool(vs, top_k=5)

        for section in VALID_SECTIONS:
            result = tool.invoke({
                "ticker": "AAPL",
                "fiscal_year": "2024",
                "section": section,
            })
            assert "Invalid section" not in result
