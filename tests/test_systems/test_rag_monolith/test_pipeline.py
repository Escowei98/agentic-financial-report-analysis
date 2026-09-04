"""
Unit tests for the MonolithRAGPipeline (System 1).
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.common.ingestion import FilingMetadata, ProcessedFiling
from src.systems.rag_monolith.pipeline import MonolithRAGPipeline, RAGResult


def _make_mock_filings(num: int = 1) -> list[ProcessedFiling]:
    return [
        ProcessedFiling(
            metadata=FilingMetadata(
                ticker=f"TICK{i}",
                company_name=f"Company {i}",
                cik=str(1000 + i),
                filing_date="2024-01-01",
                accession_number=f"001-{i}",
                fiscal_year_end="2023-12-31",
            ),
            sections={"MD&A": f"MD&A {i}"},
            full_text=f"Total text {i}",
        )
        for i in range(num)
    ]


def _make_mock_docs() -> list[Document]:
    return [
        Document(
            page_content="Revenue was 391000",
            metadata={"ticker": "AAPL", "fiscal_year": "2024", "section_name": "MD&A", "bm25_score": 5.2},
        )
    ]


class TestMonolithRAGPipeline:
    """Tests for the non-agentic S1 pipeline."""

    def test_pipeline_build_and_query(self):
        pipeline = MonolithRAGPipeline(
            config_override={"chunk_size": 100, "chunk_overlap_pct": 0.0}
        )
        filings = _make_mock_filings()

        mock_response = MagicMock()
        mock_response.content = "Apple's revenue was 391,000."
        mock_response.usage_metadata = {
            "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
        }
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch("src.systems.rag_monolith.pipeline.build_vectorstore"), \
             patch("src.systems.rag_monolith.pipeline.build_hybrid_retriever"), \
             patch("src.systems.rag_monolith.pipeline.get_llm", return_value=mock_llm), \
             patch("src.systems.rag_monolith.pipeline.get_embeddings"), \
             patch("src.systems.rag_monolith.pipeline.retrieve", return_value=_make_mock_docs()):

            pipeline.build(filings)

            assert pipeline._retriever is not None
            assert pipeline._llm is not None

            result: RAGResult = pipeline.query("What is AAPL revenue?")

            assert result.answer == "Apple's revenue was 391,000."
            assert result.contexts == ["Revenue was 391000"]
            assert len(result.context_documents) == 1

            assert result.metrics.num_steps == 1
            assert result.metrics.system_name == "rag_monolith"
            assert result.metrics.token_usage.prompt_tokens == 100
            assert result.metrics.token_usage.completion_tokens == 20

            assert result.retrieval_scores == {"doc_0_bm25": 5.2}

    def test_query_before_build_raises_error(self):
        pipeline = MonolithRAGPipeline()
        with pytest.raises(RuntimeError, match="not built"):
            pipeline.query("test")

    def test_prompt_includes_labeled_context_and_question(self):
        """The generation prompt must carry ticker/year/section-labeled context."""
        pipeline = MonolithRAGPipeline(config_override={"chunk_size": 100, "chunk_overlap_pct": 0.0})

        mock_response = MagicMock()
        mock_response.content = "answer"
        mock_response.usage_metadata = None
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch("src.systems.rag_monolith.pipeline.build_vectorstore"), \
             patch("src.systems.rag_monolith.pipeline.build_hybrid_retriever"), \
             patch("src.systems.rag_monolith.pipeline.get_llm", return_value=mock_llm), \
             patch("src.systems.rag_monolith.pipeline.get_embeddings"), \
             patch("src.systems.rag_monolith.pipeline.retrieve", return_value=_make_mock_docs()):

            pipeline.build(_make_mock_filings())
            pipeline.query("What is AAPL revenue?")

            prompt_messages = mock_llm.invoke.call_args.args[0]
            human_content = prompt_messages[-1].content
            assert "[AAPL, FY2024, MD&A]" in human_content
            assert "What is AAPL revenue?" in human_content

    def test_no_usage_metadata_defaults_to_zero_tokens(self):
        pipeline = MonolithRAGPipeline(config_override={"chunk_size": 100, "chunk_overlap_pct": 0.0})

        mock_response = MagicMock()
        mock_response.content = "answer"
        mock_response.usage_metadata = None
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch("src.systems.rag_monolith.pipeline.build_vectorstore"), \
             patch("src.systems.rag_monolith.pipeline.build_hybrid_retriever"), \
             patch("src.systems.rag_monolith.pipeline.get_llm", return_value=mock_llm), \
             patch("src.systems.rag_monolith.pipeline.get_embeddings"), \
             patch("src.systems.rag_monolith.pipeline.retrieve", return_value=_make_mock_docs()):

            pipeline.build(_make_mock_filings())
            result = pipeline.query("test")

            assert result.metrics.token_usage.total_tokens == 0


class TestMonolithRAGPipelineParams:
    """Tests for config override plumbing (Optuna-driven ablations)."""

    def test_bm25_weight_override_recomputes_dense_weight(self):
        pipeline = MonolithRAGPipeline(config_override={"bm25_weight": 0.7})
        assert pipeline.params["bm25_weight"] == 0.7
        assert pipeline.config["retrieval"]["dense_weight"] == 0.3

    def test_defaults_used_when_no_override(self):
        pipeline = MonolithRAGPipeline()
        params = pipeline.params
        assert params["chunk_size"] == 1000
        assert params["bm25_weight"] == 0.5
