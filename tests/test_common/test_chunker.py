"""
Unit tests for MonolithRAG chunker module.
"""

from langchain_core.documents import Document

from src.common.chunker import chunk_filings
from src.common.ingestion import FilingMetadata, ProcessedFiling


def _make_filing(
    ticker: str = "TEST",
    sections: dict[str, str] | None = None,
    full_text: str = "",
) -> ProcessedFiling:
    """Helper to create a test ProcessedFiling."""
    metadata = FilingMetadata(
        ticker=ticker,
        company_name=f"{ticker} Inc.",
        cik="0001234567",
        filing_date="2024-10-31",
        accession_number="0001234567-24-000001",
        fiscal_year_end="2024-09-30",
    )
    return ProcessedFiling(
        metadata=metadata,
        sections=sections or {},
        full_text=full_text,
    )


class TestChunkFilings:
    """Tests for chunk_filings()."""

    def test_basic_chunking(self):
        """Chunks a simple filing and checks output type."""
        filing = _make_filing(
            sections={"Business": "A" * 3000, "Risk Factors": "B" * 2000}
        )
        docs = chunk_filings([filing], chunk_size=1000, overlap_pct=0.20)

        assert len(docs) > 0
        assert all(isinstance(d, Document) for d in docs)

    def test_metadata_attached(self):
        """Each chunk should have correct metadata fields."""
        filing = _make_filing(
            ticker="AAPL",
            sections={"Business": "Apple designs products. " * 200},
        )
        docs = chunk_filings([filing], chunk_size=500, overlap_pct=0.10)

        assert len(docs) > 0
        for doc in docs:
            assert doc.metadata["ticker"] == "AAPL"
            assert doc.metadata["section_name"] == "Business"
            assert doc.metadata["fiscal_year"] == "2024"
            assert "chunk_index" in doc.metadata

    def test_chunk_index_sequential(self):
        """Chunk indices should be sequential within a section."""
        filing = _make_filing(
            sections={"MD&A": "Revenue grew significantly. " * 500},
        )
        docs = chunk_filings([filing], chunk_size=500, overlap_pct=0.10)

        indices = [d.metadata["chunk_index"] for d in docs]
        assert indices == list(range(len(docs)))

    def test_overlap_calculation(self):
        """Overlap should be chunk_size * overlap_pct."""
        filing = _make_filing(
            sections={"Business": "X" * 5000},
        )

        # With 20% overlap of 1000 = 200 char overlap
        docs_20 = chunk_filings([filing], chunk_size=1000, overlap_pct=0.20)
        # With 10% overlap of 1000 = 100 char overlap → more chunks
        docs_10 = chunk_filings([filing], chunk_size=1000, overlap_pct=0.10)

        # Less overlap → more chunks (since less reuse)
        assert len(docs_10) >= len(docs_20)

    def test_different_chunk_sizes(self):
        """Larger chunks → fewer documents."""
        filing = _make_filing(
            sections={"Business": "Z" * 5000},
        )

        docs_small = chunk_filings([filing], chunk_size=500, overlap_pct=0.10)
        docs_large = chunk_filings([filing], chunk_size=2000, overlap_pct=0.10)

        assert len(docs_small) > len(docs_large)

    def test_full_text_fallback(self):
        """Filing without sections should chunk full_text instead."""
        filing = _make_filing(
            sections={},
            full_text="Full filing text. " * 300,
        )
        docs = chunk_filings([filing], chunk_size=500, overlap_pct=0.10)

        assert len(docs) > 0
        assert all(d.metadata["section_name"] == "Full Text" for d in docs)

    def test_multiple_filings(self):
        """Chunks from multiple filings should all be present."""
        filing_a = _make_filing(
            ticker="AAPL",
            sections={"Business": "Apple content. " * 200},
        )
        filing_b = _make_filing(
            ticker="MSFT",
            sections={"Business": "Microsoft content. " * 200},
        )

        docs = chunk_filings([filing_a, filing_b], chunk_size=500, overlap_pct=0.10)
        tickers = {d.metadata["ticker"] for d in docs}
        assert tickers == {"AAPL", "MSFT"}

    def test_empty_filings_list(self):
        """Empty input should return empty output."""
        docs = chunk_filings([], chunk_size=1000, overlap_pct=0.10)
        assert docs == []
