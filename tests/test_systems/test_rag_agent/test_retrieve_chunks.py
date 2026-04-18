"""
Unit tests for the retrieve_chunks tool.
"""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.systems.rag_agent.tools.retrieve_chunks import (
    _format_chunks,
    create_retrieve_chunks_tool,
)


def _make_mock_docs() -> list[Document]:
    """Create sample documents for testing."""
    return [
        Document(
            page_content="Apple reported total revenue of $391 billion in FY2024",
            metadata={
                "ticker": "AAPL",
                "fiscal_year": "2024",
                "section_name": "Financial Statements",
            },
        ),
        Document(
            page_content="iPhone sales showed year-over-year growth",
            metadata={
                "ticker": "AAPL",
                "fiscal_year": "2024",
                "section_name": "MD&A",
            },
        ),
    ]


class TestRetrieveChunks:
    """Tests for retrieve_chunks tool."""

    def test_tool_calls_retrieve_and_formats(self):
        """Tool should call the shared retrieve function and format results."""
        mock_docs = _make_mock_docs()

        with patch("src.common.retrieval.retrieve", return_value=mock_docs) as mock_ret:
            tool = create_retrieve_chunks_tool(
                retriever=MagicMock(),
                retrieval_config={"post_rerank_top_k": 5},
                reranker_config={"model": "ms-marco-MiniLM-L-12-v2", "enabled": True},
            )
            result = tool.invoke({"query": "Apple revenue FY2024"})

            # Verify retrieve was called
            mock_ret.assert_called_once()
            # Verify output is formatted
            assert "AAPL" in result
            assert "$391 billion" in result

    def test_format_chunks_with_results(self):
        """Test the formatting function directly."""
        docs = _make_mock_docs()
        result = _format_chunks(docs)

        assert "Chunk 1" in result
        assert "Chunk 2" in result
        assert "AAPL" in result
        assert "FY2024" in result
        assert "Financial Statements" in result
        assert "MD&A" in result
        assert "$391 billion" in result

    def test_format_chunks_empty(self):
        """Empty docs should return a helpful message."""
        result = _format_chunks([])
        assert "No relevant chunks" in result

    def test_format_chunks_missing_metadata(self):
        """Should handle docs with missing metadata gracefully."""
        docs = [Document(page_content="Some content", metadata={})]
        result = _format_chunks(docs)

        assert "Some content" in result
        assert "?" in result  # Missing metadata shows ?
