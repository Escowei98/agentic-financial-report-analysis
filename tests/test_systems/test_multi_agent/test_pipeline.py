"""
Unit tests for the MultiAgentPipeline (System 4).

Unlike S1-S3, System 4's control flow is a compiled LangGraph StateGraph
with several nodes (supervisor -> specialist(s) -> synthesizer ->
[reflection]) rather than a single ReAct loop, so `build_agent` is mocked
per-role (dispatched on which system prompt it was called with) instead of
once. This also exercises the `Annotated[list[dict], add]` /
`Annotated[..., add_messages]` reducers on `MultiAgentState` — the pipeline
code itself carries inline comments doubting whether tool_calls_log
aggregates correctly across nodes, so pinning the actual aggregated output
here is the point of these tests, not just incidental coverage.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.common.ingestion import FilingMetadata, ProcessedFiling
from src.systems.multi_agent.pipeline import MultiAgentPipeline, MultiAgentResult


def _make_mock_filings() -> list[ProcessedFiling]:
    return [
        ProcessedFiling(
            metadata=FilingMetadata(
                ticker="AAPL",
                company_name="Apple Inc.",
                cik="1000",
                filing_date="2024-01-01",
                accession_number="001-1",
                fiscal_year_end="2023-12-31",
            ),
            sections={"MD&A": "Apple MD&A text"},
            full_text="Apple full text",
        )
    ]


def _make_fake_build_agent(specialist_answer: str, synthesizer_answers: list[str]):
    """
    Returns a `build_agent` stand-in dispatching on the system_prompt's
    role marker, plus the list `synthesizer_calls` recording each call's
    input messages (so tests can assert what the reflection loop actually
    fed back into the synthesizer).
    """
    synthesizer_calls: list[list] = []
    state = {"synth_call_count": 0}

    def fake_build_agent(llm, tools, system_prompt, recursion_limit=12):
        if system_prompt.startswith("Role: You are an orchestrator"):
            def supervisor_invoke(input_state, config=None):
                question = input_state["messages"][0].content
                delegate_tool = next(t for t in tools if t.name == "delegate_to_specialist")
                tool_result = delegate_tool.invoke({
                    "tickers": ["AAPL"],
                    "sections": ["MD&A"],
                    "sub_question": question,
                })
                return {"messages": [
                    HumanMessage(content=question),
                    AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "delegate_to_specialist",
                            "args": {"tickers": ["AAPL"], "sections": ["MD&A"], "sub_question": question},
                            "id": "d1",
                        }],
                        usage_metadata={"input_tokens": 200, "output_tokens": 30, "total_tokens": 230},
                    ),
                    ToolMessage(content=tool_result, tool_call_id="d1", name="delegate_to_specialist"),
                    AIMessage(
                        content="Delegated to an AAPL MD&A specialist.",
                        usage_metadata={"input_tokens": 250, "output_tokens": 20, "total_tokens": 270},
                    ),
                ]}
            mock = MagicMock()
            mock.invoke.side_effect = supervisor_invoke
            return mock

        if system_prompt.startswith("Role: You are a specialized financial analyst"):
            mock = MagicMock()
            mock.invoke.return_value = {"messages": [
                HumanMessage(content="sub-question"),
                AIMessage(
                    content=specialist_answer,
                    usage_metadata={"input_tokens": 300, "output_tokens": 50, "total_tokens": 350},
                ),
            ]}
            return mock

        if system_prompt.startswith("Role: You are a synthesizer"):
            def synthesizer_invoke(input_state, config=None):
                synthesizer_calls.append(input_state["messages"])
                idx = state["synth_call_count"]
                state["synth_call_count"] += 1
                answer = synthesizer_answers[min(idx, len(synthesizer_answers) - 1)]
                return {"messages": [
                    AIMessage(
                        content=answer,
                        usage_metadata={"input_tokens": 150, "output_tokens": 40, "total_tokens": 190},
                    ),
                ]}
            mock = MagicMock()
            mock.invoke.side_effect = synthesizer_invoke
            return mock

        raise AssertionError(f"Unexpected system prompt passed to build_agent: {system_prompt[:60]!r}")

    return fake_build_agent, synthesizer_calls


class TestMultiAgentPipelineNoReflection:
    """Supervisor -> Specialist -> Synthesizer, reflection disabled."""

    def test_pipeline_build_and_query_aggregates_across_nodes(self):
        pipeline = MultiAgentPipeline(config_override={"reflection_enabled": False})
        fake_build_agent, _ = _make_fake_build_agent(
            specialist_answer="Apple's FY2024 MD&A revenue was $391,035 million (AAPL, FY2024, MD&A).",
            synthesizer_answers=["Apple's revenue was approximately $391,035 million (AAPL, FY2024, MD&A)."],
        )

        with patch("src.systems.multi_agent.pipeline.get_llm"), \
             patch("src.systems.multi_agent.pipeline.build_agent", side_effect=fake_build_agent):
            pipeline.build(_make_mock_filings())
            assert pipeline._reflection_chain is None

            result: MultiAgentResult = pipeline.query("What was AAPL's FY2024 MD&A revenue?")

        assert result.answer == "Apple's revenue was approximately $391,035 million (AAPL, FY2024, MD&A)."
        assert result.num_specialists_invoked == 1
        assert len(result.delegation_requests) == 1

        # tool_calls_log must contain the supervisor's delegate_to_specialist
        # call, aggregated across the supervisor_node -> synthesizer_node
        # `add`-reducer boundary rather than being overwritten.
        assert result.metrics.tool_calls == ["delegate_to_specialist"]
        assert result.metrics.num_steps == 1

        # Token breakdown must carry one entry per role and sum correctly:
        # specialist(300,50) + supervisor(450,50) + synthesizer(150,40)
        assert result.metrics.token_usage.prompt_tokens == 900
        assert result.metrics.token_usage.completion_tokens == 140
        assert result.metrics.token_usage.total_tokens == 1040
        assert set(result.token_breakdown.keys()) >= {"supervisor", "synthesizer"}

        assert result.was_revised is False
        assert result.reflection_verdict is None
        assert result.metrics.corrections == 0

    def test_query_before_build_raises_error(self):
        pipeline = MultiAgentPipeline()
        with pytest.raises(RuntimeError, match="not built"):
            pipeline.query("test")


class TestMultiAgentPipelineReflection:
    """Reflection-verifier integration for the supervisor/synthesizer graph.

    `build_agent` is invoked lazily inside the graph's node functions at
    *query* time (not just once at build time, unlike S1-S3), so the mock
    patches must stay active across both `.build()` and `.query()` — patching
    only around `.build()` would silently fall back to the real
    `create_react_agent` once the context manager exits.
    """

    def _run(
        self,
        reflection_return_value: dict,
        max_iterations_override: int | None = None,
        synthesizer_answers: list[str] | None = None,
    ):
        pipeline = MultiAgentPipeline(config_override={"reflection_enabled": True})
        fake_build_agent, synthesizer_calls = _make_fake_build_agent(
            specialist_answer="Apple's FY2024 MD&A revenue was $391,035 million (AAPL, FY2024, MD&A).",
            synthesizer_answers=synthesizer_answers or [
                "Apple's revenue was about $391B (AAPL, FY2024, MD&A).",
                "Apple's revenue was $391,035 million, i.e. ~$391B (AAPL, FY2024, MD&A).",
            ],
        )

        with patch("src.systems.multi_agent.pipeline.get_llm"), \
             patch("src.systems.multi_agent.pipeline.build_agent", side_effect=fake_build_agent), \
             patch("src.systems.multi_agent.pipeline.build_reflection_chain") as mock_build_rc:
            mock_build_rc.return_value = MagicMock(name="reflection_chain")
            pipeline.build(_make_mock_filings())

            if max_iterations_override is not None:
                pipeline.config["reflection"]["max_iterations"] = max_iterations_override
            pipeline._reflection_chain.invoke.return_value = reflection_return_value

            result = pipeline.query("What was AAPL's FY2024 MD&A revenue?")

        return result, synthesizer_calls

    def test_accept_verdict_finishes_after_one_synthesis_pass(self):
        from src.common.reflection import ReflectionVerdict
        result, synthesizer_calls = self._run({
            "raw": AIMessage(content="ok", usage_metadata={"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}),
            "parsed": ReflectionVerdict(status="accept", feedback="", issues=[]),
            "parsing_error": None,
        })

        assert len(synthesizer_calls) == 1
        assert result.was_revised is False
        assert result.reflection_verdict.status == "accept"
        assert result.answer == "Apple's revenue was about $391B (AAPL, FY2024, MD&A)."

    def test_revise_verdict_with_default_config_triggers_one_correction_round(self):
        """
        `reflection.max_iterations` defaults to 1 (configs/multi_agent.yaml
        has no explicit value). `reflection_router` compares against
        `reflection_iterations` *after* `_reflection_node_func` has already
        incremented it, so the router must use `iterations <= max_iterations`
        (not `<`) for a single 'revise' verdict to trigger exactly one
        re-synthesis pass — matching S2/S3's single-pass policy via
        `run_reflection_pass`. Regression test for that off-by-one.
        """
        from src.common.reflection import ReflectionVerdict
        result, synthesizer_calls = self._run({
            "raw": AIMessage(content="revise", usage_metadata={"input_tokens": 60, "output_tokens": 15, "total_tokens": 75}),
            "parsed": ReflectionVerdict(status="revise", feedback="Cite the exact figure.", issues=["missing_citation"]),
            "parsing_error": None,
        })

        assert len(synthesizer_calls) == 2  # initial pass + one correction round
        # Second synthesizer call must have received the reviewer feedback
        # as its input message (last message in the fed-back state).
        second_call_messages = synthesizer_calls[1]
        assert "Cite the exact figure" in second_call_messages[-1].content

        assert result.reflection_verdict.status == "revise"
        assert result.was_revised is True
        assert result.metrics.corrections == 1
        assert result.answer == "Apple's revenue was $391,035 million, i.e. ~$391B (AAPL, FY2024, MD&A)."

        # Synthesizer token usage must accumulate across both passes via
        # _merge_token_usage rather than being overwritten by the second call.
        assert result.token_breakdown["synthesizer"].prompt_tokens == 300  # 150 + 150
        assert result.token_breakdown["synthesizer"].completion_tokens == 80  # 40 + 40

    def test_revise_verdict_stops_after_max_iterations_correction_rounds(self):
        """A verdict that keeps saying 'revise' must not loop forever — it
        stops once `max_iterations` correction rounds have run."""
        from src.common.reflection import ReflectionVerdict
        result, synthesizer_calls = self._run(
            {
                "raw": AIMessage(content="revise", usage_metadata={"input_tokens": 60, "output_tokens": 15, "total_tokens": 75}),
                "parsed": ReflectionVerdict(status="revise", feedback="Cite the exact figure.", issues=["missing_citation"]),
                "parsing_error": None,
            },
            max_iterations_override=2,
            synthesizer_answers=[
                "Apple's revenue was about $391B (AAPL, FY2024, MD&A).",
                "Apple's revenue was $391,035 million (AAPL, FY2024, MD&A).",
                "Apple's revenue was exactly $391,035 million (AAPL, FY2024, MD&A).",
            ],
        )

        assert len(synthesizer_calls) == 3  # initial pass + 2 correction rounds, then capped
        assert result.was_revised is True
        assert result.answer == "Apple's revenue was exactly $391,035 million (AAPL, FY2024, MD&A)."

        # Synthesizer token usage accumulates across all three passes.
        assert result.token_breakdown["synthesizer"].prompt_tokens == 450  # 150 * 3
        assert result.token_breakdown["synthesizer"].completion_tokens == 120  # 40 * 3
