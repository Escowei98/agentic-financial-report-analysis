"""
Unit tests for the AgentRAGPipeline.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from src.systems.rag_agent.pipeline import AgentRAGPipeline, AgentRAGResult
from src.common.ingestion import FilingMetadata, ProcessedFiling

def _make_mock_filings(num: int = 2) -> list[ProcessedFiling]:
    return [
        ProcessedFiling(
            metadata=FilingMetadata(
                ticker=f"TICK{i}",
                company_name=f"Company {i}",
                cik=str(1000 + i),
                filing_date="2024-01-01",
                accession_number=f"001-{i}",
                fiscal_year_end="2023-12-31"
            ),
            sections={"MD&A": f"MD&A {i}", "Risk Factors": f"Risk {i}"},
            full_text=f"Total text {i}",
        )
        for i in range(num)
    ]


class TestAgentRAGPipeline:
    """Tests for the Agent RAG pipeline."""

    @pytest.fixture
    def mock_agent(self):
        """Mock the compiled LangGraph agent."""
        mock = MagicMock(spec=CompiledStateGraph)
        # Mock a simple tool-use loop response
        mock.invoke.return_value = {
            "messages": [
                HumanMessage(content="What is AAPL revenue?"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "search_section", "args": {"ticker": "AAPL", "fiscal_year": "2024", "section": "MD&A"}, "id": "1"}],
                    usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
                ),
                ToolMessage(
                    content="Found 1 chunks from AAPL — MD&A:\n[AAPL — MD&A — Chunk 1]\nRevenue was 391000",
                    tool_call_id="1",
                    name="search_section"
                ),
                AIMessage(
                    content="Apple's revenue was 391,000.",
                    usage_metadata={"input_tokens": 150, "output_tokens": 20, "total_tokens": 170},
                )
            ]
        }
        return mock

    def test_pipeline_build_and_query(self, mock_agent):
        """Test the full pipeline flow with a mocked agent."""
        pipeline = AgentRAGPipeline(
            config_override={"chunk_size": 100, "chunk_overlap_pct": 0.0}
        )

        filings = _make_mock_filings(num=1)

        # Mock the build dependencies
        with patch("src.systems.rag_agent.pipeline.build_vectorstore"), \
             patch("src.systems.rag_agent.pipeline.build_hybrid_retriever"), \
             patch("src.systems.rag_agent.pipeline.get_llm"), \
             patch("src.systems.rag_agent.pipeline.get_embeddings"), \
             patch("src.systems.rag_agent.pipeline.build_agent", return_value=mock_agent):
            
            pipeline.build(filings)
            
            # Agent should be set
            assert pipeline._agent is not None
            assert pipeline._llm is not None

            # Test query parsing
            result: AgentRAGResult = pipeline.query("What is AAPL revenue?")

            # Verify Answer
            assert result.answer == "Apple's revenue was 391,000."
            
            # Verify Tool Calls extracted
            assert len(result.tool_calls_log) == 1
            assert result.tool_calls_log[0]["tool"] == "search_section"
            assert result.tool_calls_log[0]["args"]["ticker"] == "AAPL"

            # Verify Context extracted
            assert len(result.contexts) == 1
            assert "Revenue was 391000" in result.contexts[0]

            # Verify Metrics
            metrics = result.metrics
            assert metrics.num_steps == 1
            assert metrics.tool_calls == ["search_section"]
            
            # Tokens should be cumulative: (100+150) prompt, (10+20) completion
            assert metrics.token_usage.prompt_tokens == 250
            assert metrics.token_usage.completion_tokens == 30
            assert metrics.token_usage.total_tokens == 280

    def test_query_before_build_raises_error(self):
        pipeline = AgentRAGPipeline()
        with pytest.raises(RuntimeError, match="not built"):
            pipeline.query("test")
