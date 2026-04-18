"""
Unit tests for the search_section tool.
"""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.systems.rag_agent.tools.search_section import (
    VALID_SECTIONS,
    create_search_section_tool,
)

# Reranker is disabled in most tests to keep assertions focused on the
# pre-rerank similarity_search path. A dedicated test exercises the
# reranker-enabled branch.
_RETRIEVAL_CFG = {"pre_rerank_top_k": 15, "post_rerank_top_k": 5}
_RERANKER_CFG = {"enabled": False, "model": "ms-marco-MiniLM-L-12-v2"}


def _make_mock_vectorstore(docs: list[Document] | None = None):
    """Create a mock ChromaDB vectorstore."""
    mock = MagicMock()
    mock.similarity_search.return_value = docs or []
    return mock


def _make_tool(
    vs,
    retrieval_cfg: dict | None = None,
    reranker_cfg: dict | None = None,
):
    return create_search_section_tool(
        vectorstore=vs,
        retrieval_config=retrieval_cfg or _RETRIEVAL_CFG,
        reranker_config=reranker_cfg or _RERANKER_CFG,
    )


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
        tool = _make_tool(vs)

        result = tool.invoke({
            "ticker": "AAPL",
            "fiscal_year": "2024",
            "section": "MD&A",
        })

        assert "Revenue was $391B" in result
        assert "AAPL" in result

    def test_invalid_section_returns_error(self):
        vs = _make_mock_vectorstore()
        tool = _make_tool(vs)

        result = tool.invoke({
            "ticker": "AAPL",
            "fiscal_year": "2024",
            "section": "Nonexistent Section",
        })

        assert "Invalid section" in result
        assert "Valid sections" in result

    def test_no_results_suggests_alternatives(self):
        vs = _make_mock_vectorstore([])
        tool = _make_tool(vs)

        result = tool.invoke({
            "ticker": "AAPL",
            "fiscal_year": "2024",
            "section": "Business",
        })

        assert "No chunks found" in result
        assert "retrieve_chunks" in result

    def test_ticker_normalized_to_uppercase(self):
        vs = _make_mock_vectorstore([])
        tool = _make_tool(vs)

        tool.invoke({
            "ticker": "aapl",
            "fiscal_year": "2024",
            "section": "Business",
        })

        call_args = vs.similarity_search.call_args
        where_filter = call_args.kwargs.get("filter", call_args[1].get("filter", {}))
        conditions = where_filter.get("$and", [])
        ticker_filter = next(c for c in conditions if "ticker" in c)
        assert ticker_filter["ticker"]["$eq"] == "AAPL"

    def test_uses_correct_metadata_filter(self):
        vs = _make_mock_vectorstore([])
        tool = _make_tool(vs)

        tool.invoke({
            "ticker": "MSFT",
            "fiscal_year": "2023",
            "section": "Risk Factors",
        })

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
        tool = _make_tool(vs)

        for section in VALID_SECTIONS:
            result = tool.invoke({
                "ticker": "AAPL",
                "fiscal_year": "2024",
                "section": section,
            })
            assert "Invalid section" not in result

    def test_pre_rerank_top_k_is_used_for_candidate_pool(self):
        """similarity_search should be called with pre_rerank_top_k, not post_."""
        vs = _make_mock_vectorstore([])
        tool = _make_tool(
            vs,
            retrieval_cfg={"pre_rerank_top_k": 12, "post_rerank_top_k": 3},
        )

        tool.invoke({
            "ticker": "AAPL",
            "fiscal_year": "2024",
            "section": "Business",
        })

        call_args = vs.similarity_search.call_args
        assert call_args.kwargs.get("k") == 12


class TestSubQueryParameter:
    """Tests for the optional sub_query parameter (ranking query selection)."""

    def test_sub_query_used_when_provided(self):
        vs = _make_mock_vectorstore([])
        tool = _make_tool(vs)

        tool.invoke({
            "ticker": "AAPL",
            "fiscal_year": "2024",
            "section": "Risk Factors",
            "sub_query": "What are Apple's antitrust risks?",
        })

        call_args = vs.similarity_search.call_args
        assert call_args.kwargs.get("query") == "What are Apple's antitrust risks?"

    def test_fallback_query_when_sub_query_is_none(self):
        vs = _make_mock_vectorstore([])
        tool = _make_tool(vs)

        tool.invoke({
            "ticker": "AAPL",
            "fiscal_year": "2024",
            "section": "Risk Factors",
        })

        call_args = vs.similarity_search.call_args
        assert call_args.kwargs.get("query") == "AAPL Risk Factors"

    def test_fallback_query_when_sub_query_is_empty_string(self):
        vs = _make_mock_vectorstore([])
        tool = _make_tool(vs)

        tool.invoke({
            "ticker": "AAPL",
            "fiscal_year": "2024",
            "section": "Risk Factors",
            "sub_query": "   ",
        })

        call_args = vs.similarity_search.call_args
        assert call_args.kwargs.get("query") == "AAPL Risk Factors"


class TestRerankerIntegration:
    """Tests for the optional FlashRank reranking branch."""

    def test_reranker_called_when_enabled_and_pool_large_enough(self):
        # Need len(candidates) > post_rerank_top_k for reranker to trigger
        candidates = [
            Document(page_content=f"chunk {i}", metadata={"ticker": "AAPL"})
            for i in range(10)
        ]
        vs = _make_mock_vectorstore(candidates)

        reranked = candidates[:3]
        with patch(
            "src.common.retrieval.rerank_documents",
            return_value=reranked,
        ) as mock_rerank:
            tool = _make_tool(
                vs,
                retrieval_cfg={"pre_rerank_top_k": 15, "post_rerank_top_k": 3},
                reranker_cfg={
                    "enabled": True,
                    "model": "ms-marco-MiniLM-L-12-v2",
                },
            )

            tool.invoke({
                "ticker": "AAPL",
                "fiscal_year": "2024",
                "section": "Risk Factors",
                "sub_query": "cybersecurity risks",
            })

            mock_rerank.assert_called_once()
            kwargs = mock_rerank.call_args.kwargs
            assert kwargs["query"] == "cybersecurity risks"
            assert kwargs["top_n"] == 3
            assert kwargs["model"] == "ms-marco-MiniLM-L-12-v2"
            assert kwargs["documents"] == candidates

    def test_reranker_skipped_when_pool_too_small(self):
        """If candidate pool <= post_rerank_top_k, reranker must be skipped."""
        # 3 candidates, post_rerank_top_k = 5 -> reranker has nothing to prune
        candidates = [
            Document(page_content=f"chunk {i}", metadata={"ticker": "AAPL"})
            for i in range(3)
        ]
        vs = _make_mock_vectorstore(candidates)

        with patch(
            "src.common.retrieval.rerank_documents",
        ) as mock_rerank:
            tool = _make_tool(
                vs,
                retrieval_cfg={"pre_rerank_top_k": 15, "post_rerank_top_k": 5},
                reranker_cfg={
                    "enabled": True,
                    "model": "ms-marco-MiniLM-L-12-v2",
                },
            )

            tool.invoke({
                "ticker": "AAPL",
                "fiscal_year": "2024",
                "section": "Business",
            })

            mock_rerank.assert_not_called()

    def test_reranker_skipped_when_disabled(self):
        candidates = [
            Document(page_content=f"chunk {i}", metadata={"ticker": "AAPL"})
            for i in range(10)
        ]
        vs = _make_mock_vectorstore(candidates)

        with patch(
            "src.common.retrieval.rerank_documents",
        ) as mock_rerank:
            tool = _make_tool(
                vs,
                retrieval_cfg={"pre_rerank_top_k": 15, "post_rerank_top_k": 3},
                reranker_cfg={
                    "enabled": False,
                    "model": "ms-marco-MiniLM-L-12-v2",
                },
            )

            tool.invoke({
                "ticker": "AAPL",
                "fiscal_year": "2024",
                "section": "Risk Factors",
            })

            mock_rerank.assert_not_called()
