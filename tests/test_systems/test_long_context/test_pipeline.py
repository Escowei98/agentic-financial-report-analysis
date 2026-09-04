"""
Unit tests for the LongContextPipeline (System 3).
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from src.common.ingestion import FilingMetadata, ProcessedFiling
from src.common.reflection import ReflectionVerdict
from src.systems.long_context.pipeline import LongContextPipeline, LongContextResult


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


def _default_agent_messages() -> dict:
    """Canonical single-pass agent response for mocks (no tool use)."""
    return {
        "messages": [
            HumanMessage(content="What is AAPL revenue?"),
            AIMessage(
                content="Apple's revenue was 391,000.",
                usage_metadata={
                    "input_tokens": 500_000, "output_tokens": 20, "total_tokens": 500_020,
                },
            ),
        ]
    }


def _agent_messages_with_tool_call() -> dict:
    return {
        "messages": [
            HumanMessage(content="What is AAPL revenue?"),
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "calculate",
                    "args": {"expression": "391000 * 1.1"},
                    "id": "1",
                }],
                usage_metadata={
                    "input_tokens": 500_000, "output_tokens": 10, "total_tokens": 500_010,
                },
            ),
            ToolMessage(content="430100", tool_call_id="1", name="calculate"),
            AIMessage(
                content="Apple's projected revenue is 430,100.",
                usage_metadata={
                    "input_tokens": 500_100, "output_tokens": 20, "total_tokens": 500_120,
                },
            ),
        ]
    }


class TestLongContextPipeline:
    """Tests for the S3 pipeline: inlined filings, no retrieval stack."""

    @pytest.fixture
    def mock_agent(self):
        mock = MagicMock(spec=CompiledStateGraph)
        mock.invoke.return_value = _default_agent_messages()
        return mock

    def test_pipeline_build_and_query(self, mock_agent):
        pipeline = LongContextPipeline(config_override={"reflection_enabled": False})
        filings = _make_mock_filings()

        with patch("src.systems.long_context.pipeline.build_system_prompt", return_value="SYSTEM PROMPT") as mock_build_prompt, \
             patch("src.systems.long_context.pipeline.get_llm"), \
             patch("src.systems.long_context.pipeline.build_agent", return_value=mock_agent) as mock_build_agent:

            pipeline.build(filings)

            mock_build_prompt.assert_called_once_with(filings)
            assert pipeline._agent is not None
            assert pipeline._reflection_chain is None

            # No retrieval tools: only calculate + list_filings.
            _, kwargs = mock_build_agent.call_args
            tool_names = [getattr(t, "name", t) for t in kwargs["tools"]]
            assert "calculate" in tool_names
            assert not any("retrieve" in str(n) or "search_section" in str(n) for n in tool_names)

            result: LongContextResult = pipeline.query("What is AAPL revenue?")

            assert result.answer == "Apple's revenue was 391,000."
            assert result.metrics.system_name == "long_context"
            assert result.metrics.num_steps == 0
            assert result.reflection_verdict is None
            assert result.was_revised is False

    def test_pipeline_tracks_tool_calls(self, mock_agent):
        mock_agent.invoke.return_value = _agent_messages_with_tool_call()
        pipeline = LongContextPipeline(config_override={"reflection_enabled": False})

        with patch("src.systems.long_context.pipeline.build_system_prompt", return_value="SYSTEM PROMPT"), \
             patch("src.systems.long_context.pipeline.get_llm"), \
             patch("src.systems.long_context.pipeline.build_agent", return_value=mock_agent):

            pipeline.build(_make_mock_filings())
            result = pipeline.query("What is AAPL projected revenue?")

            assert result.metrics.num_steps == 1
            assert result.metrics.tool_calls == ["calculate"]
            assert result.contexts == ["430100"]
            assert result.metrics.token_usage.prompt_tokens == 1_000_100

    def test_query_before_build_raises_error(self):
        pipeline = LongContextPipeline()
        with pytest.raises(RuntimeError, match="not built"):
            pipeline.query("test")


class TestLongContextPipelineReflection:
    """Tests for the Reflexion-style verifier integration (mirrors S2)."""

    @pytest.fixture
    def built_pipeline(self):
        def _build(enable_reflection: bool = True):
            pipeline = LongContextPipeline(config_override={"reflection_enabled": enable_reflection})

            mock_agent = MagicMock(spec=CompiledStateGraph)
            mock_agent.invoke.return_value = _default_agent_messages()

            with patch("src.systems.long_context.pipeline.build_system_prompt", return_value="SYSTEM PROMPT"), \
                 patch("src.systems.long_context.pipeline.get_llm"), \
                 patch("src.systems.long_context.pipeline.build_agent", return_value=mock_agent), \
                 patch("src.systems.long_context.pipeline.build_reflection_chain") as mock_build_rc:
                mock_build_rc.return_value = MagicMock(name="reflection_chain")
                pipeline.build(_make_mock_filings())

            return pipeline, mock_agent

        return _build

    def test_reflection_disabled_skips_verifier(self, built_pipeline):
        pipeline, mock_agent = built_pipeline(enable_reflection=False)

        assert pipeline._reflection_chain is None
        result = pipeline.query("What is AAPL revenue?")

        assert mock_agent.invoke.call_count == 1
        assert result.was_revised is False
        assert result.metrics.corrections == 0

    def test_reflection_accept_keeps_draft(self, built_pipeline):
        pipeline, mock_agent = built_pipeline(enable_reflection=True)

        accept_verdict = ReflectionVerdict(status="accept", feedback="", issues=[])
        raw_ai = AIMessage(
            content="accepted",
            usage_metadata={"input_tokens": 40, "output_tokens": 5, "total_tokens": 45},
        )
        pipeline._reflection_chain = MagicMock()
        pipeline._reflection_chain.invoke.return_value = {
            "raw": raw_ai, "parsed": accept_verdict, "parsing_error": None,
        }

        result = pipeline.query("What is AAPL revenue?")

        assert mock_agent.invoke.call_count == 1
        assert result.was_revised is False
        assert result.reflection_verdict.status == "accept"
        assert result.metrics.token_usage.prompt_tokens == 500_040

    def test_reflection_revise_triggers_second_pass(self, built_pipeline):
        pipeline, mock_agent = built_pipeline(enable_reflection=True)

        revise_verdict = ReflectionVerdict(
            status="revise", feedback="Missing units.", issues=["missing_citation"],
        )
        raw_ai = AIMessage(
            content="revise",
            usage_metadata={"input_tokens": 60, "output_tokens": 15, "total_tokens": 75},
        )
        pipeline._reflection_chain = MagicMock()
        pipeline._reflection_chain.invoke.return_value = {
            "raw": raw_ai, "parsed": revise_verdict, "parsing_error": None,
        }

        revised_messages = _default_agent_messages()["messages"] + [
            HumanMessage(content="Reviewer feedback..."),
            AIMessage(
                content="Apple's revenue was $391,000 million (AAPL, 2024, MD&A).",
                usage_metadata={"input_tokens": 200, "output_tokens": 25, "total_tokens": 225},
            ),
        ]
        mock_agent.invoke.side_effect = [
            _default_agent_messages(),
            {"messages": revised_messages},
        ]

        result = pipeline.query("What is AAPL revenue?")

        assert mock_agent.invoke.call_count == 2
        assert result.was_revised is True
        assert result.metrics.corrections == 1
        assert result.answer.startswith("Apple's revenue was $391,000 million")


class TestLongContextPipelineParams:
    """Tests for config override plumbing."""

    def test_max_iterations_override(self):
        pipeline = LongContextPipeline(config_override={"max_iterations": 3})
        assert pipeline.params["max_iterations"] == 3

    def test_defaults_used_when_no_override(self):
        pipeline = LongContextPipeline()
        params = pipeline.params
        assert params["recursion_limit"] == 12
        assert params["reflection_enabled"] is True
