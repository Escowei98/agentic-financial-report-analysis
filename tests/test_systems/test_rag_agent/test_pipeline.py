"""
Unit tests for the AgentRAGPipeline.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from src.common.ingestion import FilingMetadata, ProcessedFiling
from src.common.reflection import ReflectionVerdict
from src.systems.rag_agent.pipeline import AgentRAGPipeline, AgentRAGResult


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


def _default_agent_messages() -> dict:
    """Canonical single-pass agent response for mocks."""
    return {
        "messages": [
            HumanMessage(content="What is AAPL revenue?"),
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "search_section",
                    "args": {"ticker": "AAPL", "fiscal_year": "2024", "section": "MD&A"},
                    "id": "1",
                }],
                usage_metadata={
                    "input_tokens": 100, "output_tokens": 10, "total_tokens": 110,
                },
            ),
            ToolMessage(
                content=(
                    "Found 1 chunks from AAPL — MD&A:\n"
                    "[AAPL — MD&A — Chunk 1]\nRevenue was 391000"
                ),
                tool_call_id="1",
                name="search_section",
            ),
            AIMessage(
                content="Apple's revenue was 391,000.",
                usage_metadata={
                    "input_tokens": 150, "output_tokens": 20, "total_tokens": 170,
                },
            ),
        ]
    }


class TestAgentRAGPipeline:
    """Tests for the Agent RAG pipeline."""

    @pytest.fixture
    def mock_agent(self):
        """Mock the compiled LangGraph agent (single-pass response)."""
        mock = MagicMock(spec=CompiledStateGraph)
        mock.invoke.return_value = _default_agent_messages()
        return mock

    def test_pipeline_build_and_query(self, mock_agent):
        """Test the full pipeline flow with a mocked agent and reflection disabled."""
        pipeline = AgentRAGPipeline(
            config_override={
                "chunk_size": 100,
                "chunk_overlap_pct": 0.0,
                "reflection_enabled": False,
            }
        )

        filings = _make_mock_filings(num=1)

        with patch("src.systems.rag_agent.pipeline.build_vectorstore"), \
             patch("src.systems.rag_agent.pipeline.build_hybrid_retriever"), \
             patch("src.systems.rag_agent.pipeline.get_llm"), \
             patch("src.systems.rag_agent.pipeline.get_embeddings"), \
             patch("src.systems.rag_agent.pipeline.build_agent", return_value=mock_agent):

            pipeline.build(filings)

            assert pipeline._agent is not None
            assert pipeline._llm is not None
            assert pipeline._reflection_chain is None

            result: AgentRAGResult = pipeline.query("What is AAPL revenue?")

            assert result.answer == "Apple's revenue was 391,000."

            assert len(result.tool_calls_log) == 1
            assert result.tool_calls_log[0]["tool"] == "search_section"
            assert result.tool_calls_log[0]["args"]["ticker"] == "AAPL"

            assert len(result.contexts) == 1
            assert "Revenue was 391000" in result.contexts[0]

            metrics = result.metrics
            assert metrics.num_steps == 1
            assert metrics.tool_calls == ["search_section"]

            # Tokens cumulative: (100+150) prompt, (10+20) completion
            assert metrics.token_usage.prompt_tokens == 250
            assert metrics.token_usage.completion_tokens == 30
            assert metrics.token_usage.total_tokens == 280

            # No reflection -> no verdict, no correction
            assert result.reflection_verdict is None
            assert result.was_revised is False
            assert metrics.corrections == 0

    def test_query_before_build_raises_error(self):
        pipeline = AgentRAGPipeline()
        with pytest.raises(RuntimeError, match="not built"):
            pipeline.query("test")


class TestAgentRAGPipelineReflection:
    """Tests for the Reflexion-style verifier integration."""

    @pytest.fixture
    def built_pipeline(self, request):
        """Factory fixture: build a pipeline with mocked heavy dependencies.

        The fixture returns a (pipeline, mock_agent) tuple. Reflection is
        enabled by default; tests override via config_override.
        """
        def _build(enable_reflection: bool = True):
            config_override = {
                "chunk_size": 100,
                "chunk_overlap_pct": 0.0,
                "reflection_enabled": enable_reflection,
            }
            pipeline = AgentRAGPipeline(config_override=config_override)

            mock_agent = MagicMock(spec=CompiledStateGraph)
            mock_agent.invoke.return_value = _default_agent_messages()

            with patch("src.systems.rag_agent.pipeline.build_vectorstore"), \
                 patch("src.systems.rag_agent.pipeline.build_hybrid_retriever"), \
                 patch("src.systems.rag_agent.pipeline.get_llm"), \
                 patch("src.systems.rag_agent.pipeline.get_embeddings"), \
                 patch("src.systems.rag_agent.pipeline.build_agent",
                       return_value=mock_agent), \
                 patch("src.systems.rag_agent.pipeline.build_reflection_chain") as mock_build_rc:
                mock_build_rc.return_value = MagicMock(name="reflection_chain")
                pipeline.build(_make_mock_filings(num=1))

            return pipeline, mock_agent

        return _build

    def test_pipeline_reflection_disabled_skips_verifier(self, built_pipeline):
        pipeline, mock_agent = built_pipeline(enable_reflection=False)

        assert pipeline._reflection_chain is None

        result = pipeline.query("What is AAPL revenue?")

        # Only one agent invocation and no reflection metadata
        assert mock_agent.invoke.call_count == 1
        assert result.was_revised is False
        assert result.reflection_verdict is None
        assert result.metrics.corrections == 0

    def test_pipeline_query_with_reflection_accept(self, built_pipeline):
        pipeline, mock_agent = built_pipeline(enable_reflection=True)

        # Reflection chain returns an 'accept' verdict (include_raw=True shape).
        accept_verdict = ReflectionVerdict(status="accept", feedback="", issues=[])
        raw_ai = AIMessage(
            content="accepted",
            usage_metadata={"input_tokens": 40, "output_tokens": 5, "total_tokens": 45},
        )
        pipeline._reflection_chain = MagicMock()
        pipeline._reflection_chain.invoke.return_value = {
            "raw": raw_ai,
            "parsed": accept_verdict,
            "parsing_error": None,
        }

        result = pipeline.query("What is AAPL revenue?")

        # Exactly one agent.invoke because reflection accepted
        assert mock_agent.invoke.call_count == 1
        assert result.was_revised is False
        assert result.metrics.corrections == 0
        assert result.reflection_verdict is not None
        assert result.reflection_verdict.status == "accept"

        # Reflection tokens must be added to the aggregated usage
        # Agent tokens: 250 prompt + 30 completion. Reflection: +40 prompt, +5 completion.
        assert result.metrics.token_usage.prompt_tokens == 290
        assert result.metrics.token_usage.completion_tokens == 35
        assert result.metrics.token_usage.total_tokens == 325

    def test_pipeline_query_with_reflection_revise(self, built_pipeline):
        pipeline, mock_agent = built_pipeline(enable_reflection=True)

        # Reflection returns 'revise' -> a second agent pass must happen.
        revise_verdict = ReflectionVerdict(
            status="revise",
            feedback="The revenue number is missing units.",
            issues=["missing_citation"],
        )
        raw_ai = AIMessage(
            content="revise",
            usage_metadata={"input_tokens": 60, "output_tokens": 15, "total_tokens": 75},
        )
        pipeline._reflection_chain = MagicMock()
        pipeline._reflection_chain.invoke.return_value = {
            "raw": raw_ai,
            "parsed": revise_verdict,
            "parsing_error": None,
        }

        # Second agent.invoke returns a revised answer (new final AIMessage at end).
        revised_messages = _default_agent_messages()["messages"] + [
            HumanMessage(content="Reviewer feedback..."),
            AIMessage(
                content="Apple's revenue was $391,000 million (AAPL, 2024, MD&A).",
                usage_metadata={
                    "input_tokens": 200, "output_tokens": 25, "total_tokens": 225,
                },
            ),
        ]
        mock_agent.invoke.side_effect = [
            _default_agent_messages(),
            {"messages": revised_messages},
        ]

        result = pipeline.query("What is AAPL revenue?")

        # Two agent invocations must have happened
        assert mock_agent.invoke.call_count == 2
        assert result.was_revised is True
        assert result.metrics.corrections == 1
        assert result.reflection_verdict is not None
        assert result.reflection_verdict.status == "revise"
        assert result.answer.startswith("Apple's revenue was $391,000 million")

        # Final message history contains all messages, so tokens aggregate:
        # First pass AI messages: 100+10, 150+20
        # Revised final AI message: 200+25
        # Reflection: 60+15
        # Total prompt = 100+150+200+60 = 510
        # Total completion = 10+20+25+15 = 70
        assert result.metrics.token_usage.prompt_tokens == 510
        assert result.metrics.token_usage.completion_tokens == 70
