"""
Unit tests for the hybrid retriever module.
"""

import pytest
from langchain_core.documents import Document

from src.common.retrieval import (
    BM25RetrieverWithScores,
    HybridRetriever,
    build_hybrid_retriever,
    retrieve,
)
class TestBM25RetrieverWithScores:
    """Tests for BM25RetrieverWithScores."""

    @pytest.fixture
    def sample_documents(self) -> list[Document]:
        """Create a small set of test documents."""
        return [
            Document(page_content="Apple reported total revenue of $391 billion in FY2024", metadata={"ticker": "AAPL"}),
            Document(page_content="Microsoft cloud services grew significantly in fiscal year 2024", metadata={"ticker": "MSFT"}),
            Document(page_content="Amazon's AWS segment reported strong growth in cloud computing", metadata={"ticker": "AMZN"}),
            Document(page_content="Apple's iPhone sales declined slightly in the fourth quarter", metadata={"ticker": "AAPL"}),
            Document(page_content="Google's advertising revenue reached new highs in 2024", metadata={"ticker": "GOOGL"}),
        ]

    def test_retrieves_relevant_docs(self, sample_documents):
        """Should return documents relevant to the query."""
        retriever = BM25RetrieverWithScores(documents=sample_documents, top_k=3)
        results = retriever.invoke("Apple revenue FY2024")

        assert len(results) > 0
        # Apple-related docs should rank higher
        assert any("Apple" in doc.page_content for doc in results)

    def test_bm25_scores_tracked(self, sample_documents):
        """BM25 scores should be stored in metadata."""
        retriever = BM25RetrieverWithScores(documents=sample_documents, top_k=3)
        results = retriever.invoke("cloud computing growth")

        for doc in results:
            assert "bm25_score" in doc.metadata
            assert doc.metadata["bm25_score"] > 0

    def test_scores_descending(self, sample_documents):
        """Results should be sorted by BM25 score descending."""
        retriever = BM25RetrieverWithScores(documents=sample_documents, top_k=5)
        results = retriever.invoke("revenue growth")

        scores = [doc.metadata["bm25_score"] for doc in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_results(self, sample_documents):
        """Should return at most top_k results."""
        retriever = BM25RetrieverWithScores(documents=sample_documents, top_k=2)
        results = retriever.invoke("Apple")

        assert len(results) <= 2

    def test_no_results_for_unrelated_query(self, sample_documents):
        """Query with no matching terms should return empty or low-score results."""
        retriever = BM25RetrieverWithScores(documents=sample_documents, top_k=3)
        results = retriever.invoke("xyznonexistentterm")

        # BM25 should return 0-score docs which are filtered out
        assert len(results) == 0

    def test_original_metadata_preserved(self, sample_documents):
        """Original document metadata should be preserved alongside bm25_score."""
        retriever = BM25RetrieverWithScores(documents=sample_documents, top_k=3)
        results = retriever.invoke("Apple revenue")

        for doc in results:
            assert "ticker" in doc.metadata  # Original metadata preserved
            assert "bm25_score" in doc.metadata  # Score added
