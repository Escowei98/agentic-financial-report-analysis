"""
Unit tests for the two-tier LLM-as-a-judge reasoning evaluator
(src/evaluation/reasoning_evaluator.py).

Covers the pure formatting/parsing helpers directly (no LLM calls), plus
`evaluate_reasoning`/`evaluate_reasoning_batch` with a mocked judge LLM.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.gold_standard_loader import GoldStandardItem
from src.evaluation.reasoning_evaluator import (
    AgenticReasoningScores,
    CoreReasoningScores,
    ReasoningEvalResult,
    _extract_json,
    _format_args,
    _safe_score,
    _to_str,
    _truncate,
    evaluate_reasoning,
    evaluate_reasoning_batch,
    format_trajectory,
)


def _make_result(**kwargs) -> SimpleNamespace:
    """Minimal stand-in for a pipeline result (RAGResult/AgentRAGResult/...)."""
    defaults = dict(metrics=SimpleNamespace(token_usage=SimpleNamespace(total_tokens=100)))
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestCoreAndAgenticScores:
    """Tests for the dataclass composite/to_dict logic."""

    def test_core_composite_is_equal_weighted_mean(self):
        scores = CoreReasoningScores(logical_soundness=5, synthesis_quality=3, evidence_faithfulness=4)
        assert scores.composite == pytest.approx(4.0)

    def test_core_to_dict_rounds_and_includes_rationales(self):
        scores = CoreReasoningScores(
            logical_soundness=5, synthesis_quality=3.333, evidence_faithfulness=4,
            rationales={"logical_soundness": "clear"},
        )
        d = scores.to_dict()
        assert d["synthesis_quality"] == 3.33
        assert d["core_composite"] == pytest.approx(4.11, abs=0.01)
        assert d["rationales"] == {"logical_soundness": "clear"}

    def test_agentic_composite_is_equal_weighted_mean(self):
        scores = AgenticReasoningScores(tool_selection=4, error_recovery=2)
        assert scores.composite == 3.0

    def test_reasoning_eval_result_omits_agentic_when_none(self):
        result = ReasoningEvalResult(query_id=1, system_name="rag_monolith", core=CoreReasoningScores())
        d = result.to_dict()
        assert "agentic" not in d

    def test_reasoning_eval_result_includes_agentic_when_present(self):
        result = ReasoningEvalResult(
            query_id=1, system_name="rag_agent",
            core=CoreReasoningScores(), agentic=AgenticReasoningScores(),
        )
        assert "agentic" in result.to_dict()


class TestToStr:
    def test_string_passthrough(self):
        assert _to_str("hello") == "hello"

    def test_list_of_text_dicts_joined_with_newline(self):
        assert _to_str([{"text": "a"}, {"text": "b"}]) == "a\nb"

    def test_list_with_non_dict_items(self):
        assert _to_str(["a", {"text": "b"}]) == "a\nb"

    def test_other_type_falls_back_to_str(self):
        assert _to_str(42) == "42"


class TestTruncate:
    def test_short_text_untouched(self):
        assert _truncate("short text", limit=100) == "short text"

    def test_text_at_exact_limit_untouched(self):
        text = "x" * 50
        assert _truncate(text, limit=50) == text

    def test_text_over_limit_is_cut_with_ellipsis(self):
        text = "x" * 60
        result = _truncate(text, limit=50)
        assert result == "x" * 50 + "..."

    def test_newlines_replaced_with_spaces(self):
        assert _truncate("line1\nline2", limit=100) == "line1 line2"


class TestFormatArgs:
    def test_dict_args_formatted_as_key_value_pairs(self):
        assert _format_args({"ticker": "AAPL", "year": 2024}) == "ticker='AAPL', year=2024"

    def test_non_dict_args_stringified(self):
        assert _format_args("raw-arg") == "raw-arg"


class TestFormatTrajectoryDispatch:
    def test_dispatches_to_s1_formatter(self):
        result = _make_result(contexts=[])
        text = format_trajectory(result, "rag_monolith")
        assert "System 1 Trajectory" in text

    def test_dispatches_to_s2_formatter(self):
        result = _make_result(tool_calls_log=[])
        text = format_trajectory(result, "rag_agent")
        assert "System 2 Trajectory" in text

    def test_dispatches_to_s3_formatter(self):
        result = _make_result(tool_calls_log=[])
        text = format_trajectory(result, "long_context")
        assert "System 3 Trajectory" in text

    def test_dispatches_to_s4_formatter(self):
        result = _make_result(tool_calls_log=[])
        text = format_trajectory(result, "multi_agent")
        assert "System 4 Trajectory" in text

    def test_unknown_system_returns_placeholder(self):
        text = format_trajectory(_make_result(), "some_future_system")
        assert "Unknown system" in text


class TestFormatS1Trajectory:
    def test_shows_context_count_and_content(self):
        result = _make_result(contexts=["Revenue was 391000"])
        text = format_trajectory(result, "rag_monolith")
        assert "Retrieved contexts: 1 chunks" in text
        assert "Revenue was 391000" in text

    def test_no_contexts_shows_zero(self):
        result = _make_result(contexts=[])
        text = format_trajectory(result, "rag_monolith")
        assert "Retrieved contexts: 0 chunks" in text


class TestFormatS2S3Trajectory:
    def test_no_tool_calls_shows_direct_answer_note(self):
        result = _make_result(tool_calls_log=[])
        text = format_trajectory(result, "rag_agent")
        assert "Tool calls: None" in text

    def test_tool_calls_are_rendered_with_results(self):
        result = _make_result(tool_calls_log=[
            {"tool": "search_section", "args": {"ticker": "AAPL"}, "result": "Revenue was 391000"},
        ])
        text = format_trajectory(result, "rag_agent")
        assert "search_section(ticker='AAPL')" in text
        assert "Revenue was 391000" in text

    def test_reflection_verdict_appended(self):
        result = _make_result(tool_calls_log=[], reflection_verdict="accept", was_revised=False)
        text = format_trajectory(result, "long_context")
        assert "Reflection verdict: accept" in text
        assert "Was revised after reflection: False" in text

    def test_missing_reflection_attrs_show_not_performed(self):
        result = _make_result(tool_calls_log=[])
        text = format_trajectory(result, "long_context")
        assert "Reflection: Not performed or not available" in text


class TestFormatS4Trajectory:
    def test_full_trajectory_includes_all_sections(self):
        result = _make_result(
            supervisor_plan="Delegate to AAPL specialist.",
            delegation_requests=[{"tickers": ["AAPL"], "sections": ["MD&A"], "sub_question": "Revenue?"}],
            specialist_outputs={"AAPL_MD&A_0": "Revenue was $391B"},
            tool_calls_log=[{"tool": "calculate", "args": {"expression": "1+1"}, "result": "2"}],
            reflection_verdict="revise",
            was_revised=True,
            token_breakdown={"supervisor": SimpleNamespace(total_tokens=50), "synthesizer": 30},
        )
        text = format_trajectory(result, "multi_agent")

        assert "Supervisor Plan" in text and "Delegate to AAPL specialist." in text
        assert "Delegations (1 specialists invoked)" in text
        assert "AAPL_MD&A_0" in text and "Revenue was $391B" in text
        assert "calculate(expression='1+1')" in text
        assert "Reflection verdict: revise" in text
        assert "supervisor: 50 tokens" in text
        assert "synthesizer: 30 tokens" in text


class TestExtractJson:
    def test_plain_json(self):
        assert _extract_json('{"a": 1}') == {"a": 1}

    def test_json_wrapped_in_markdown_fence_with_language_tag(self):
        text = '```json\n{"a": 1}\n```'
        assert _extract_json(text) == {"a": 1}

    def test_json_wrapped_in_bare_fence(self):
        text = '```\n{"a": 1}\n```'
        assert _extract_json(text) == {"a": 1}

    def test_json_with_surrounding_text_extracted_via_regex(self):
        text = 'Here is the result:\n{"a": 1}\nHope that helps!'
        assert _extract_json(text) == {"a": 1}

    def test_unparsable_text_returns_empty_dict(self):
        assert _extract_json("not json at all") == {}


class TestSafeScore:
    def test_valid_entry_with_rationale(self):
        data = {"logical_soundness": {"score": 4, "rationale": "solid"}}
        score, rationale = _safe_score(data, "logical_soundness")
        assert score == 4.0
        assert rationale == "solid"

    def test_bare_numeric_entry(self):
        data = {"logical_soundness": 3}
        score, _ = _safe_score(data, "logical_soundness")
        assert score == 3.0

    def test_missing_key_defaults_to_zero(self):
        score, rationale = _safe_score({}, "logical_soundness")
        assert score == 0.0
        assert rationale == ""

    def test_score_above_range_is_clamped_to_five(self):
        score, _ = _safe_score({"x": {"score": 10}}, "x")
        assert score == 5.0

    def test_positive_score_below_one_is_clamped_to_one(self):
        score, _ = _safe_score({"x": {"score": 0.5}}, "x")
        assert score == 1.0

    def test_zero_score_is_not_clamped_up(self):
        """A 0 score (e.g. judge omitted the field) stays 0, distinguishing
        'no score given' from a valid low score of 1."""
        score, _ = _safe_score({"x": {"score": 0}}, "x")
        assert score == 0.0

    def test_negative_score_treated_as_zero(self):
        score, _ = _safe_score({"x": {"score": -3}}, "x")
        assert score == 0.0


class TestEvaluateReasoning:
    def _mock_judge_response(self, payload: dict):
        response = MagicMock()
        response.content = json.dumps(payload)
        return response

    def test_core_only_for_non_agentic_system(self):
        core_payload = {
            "logical_soundness": {"score": 5, "rationale": "r1"},
            "synthesis_quality": {"score": 4, "rationale": "r2"},
            "evidence_faithfulness": {"score": 5, "rationale": "r3"},
        }
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_judge_response(core_payload)

        with patch("src.evaluation.reasoning_evaluator.get_judge_llm", return_value=mock_llm):
            result = evaluate_reasoning(
                question="q", answer="a", trajectory="t", ground_truth="gt",
                system_name="rag_monolith", query_id=1,
            )

        assert mock_llm.invoke.call_count == 1  # only the Core Judge
        assert result.agentic is None
        assert result.core.logical_soundness == 5.0

    def test_agentic_judge_runs_for_agentic_systems(self):
        core_payload = {
            "logical_soundness": {"score": 4, "rationale": "r"},
            "synthesis_quality": {"score": 4, "rationale": "r"},
            "evidence_faithfulness": {"score": 4, "rationale": "r"},
        }
        agentic_payload = {
            "tool_selection": {"score": 5, "rationale": "r"},
            "error_recovery": {"score": 3, "rationale": "r"},
        }
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            self._mock_judge_response(core_payload),
            self._mock_judge_response(agentic_payload),
        ]

        with patch("src.evaluation.reasoning_evaluator.get_judge_llm", return_value=mock_llm):
            result = evaluate_reasoning(
                question="q", answer="a", trajectory="t", ground_truth="gt",
                system_name="rag_agent", query_id=2,
            )

        assert mock_llm.invoke.call_count == 2
        assert result.agentic is not None
        assert result.agentic.tool_selection == 5.0
        assert result.agentic.error_recovery == 3.0

    def test_answer_content_list_is_coerced_to_string(self):
        """Gemini-style content parts must not break the prompt formatting."""
        core_payload = {
            "logical_soundness": {"score": 3, "rationale": "r"},
            "synthesis_quality": {"score": 3, "rationale": "r"},
            "evidence_faithfulness": {"score": 3, "rationale": "r"},
        }
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = self._mock_judge_response(core_payload)

        with patch("src.evaluation.reasoning_evaluator.get_judge_llm", return_value=mock_llm):
            evaluate_reasoning(
                question="q", answer=[{"text": "Apple's revenue was $391B"}],
                trajectory="t", ground_truth="gt", system_name="rag_monolith",
            )

        prompt_arg = mock_llm.invoke.call_args.args[0]
        assert "Apple's revenue was $391B" in prompt_arg


class TestEvaluateReasoningBatch:
    def _make_gold_item(self, item_id: int, fa_type: str = "FA-1") -> GoldStandardItem:
        return GoldStandardItem(id=item_id, question="q", ground_truth="gt", doc_refs="AAPL", fa_type=fa_type)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            evaluate_reasoning_batch([self._make_gold_item(1)], [], "rag_monolith")

    def test_batch_preserves_alignment_on_per_item_failure(self):
        """A failure evaluating one item must not shift or drop other results."""
        items = [self._make_gold_item(1), self._make_gold_item(2)]
        results = [_make_result(contexts=[]), _make_result(contexts=[])]

        call_count = {"n": 0}

        def fake_evaluate_reasoning(**kwargs):
            call_count["n"] += 1
            if kwargs["query_id"] == 1:
                raise RuntimeError("judge LLM timed out")
            return ReasoningEvalResult(
                query_id=kwargs["query_id"], system_name="rag_monolith",
                core=CoreReasoningScores(logical_soundness=4, synthesis_quality=4, evidence_faithfulness=4),
            )

        with patch("src.evaluation.reasoning_evaluator.evaluate_reasoning", side_effect=fake_evaluate_reasoning):
            eval_results = evaluate_reasoning_batch(items, results, "rag_monolith")

        assert len(eval_results) == 2
        assert eval_results[0].query_id == 1
        assert eval_results[0].core.logical_soundness == 0.0  # zero-score fallback
        assert eval_results[1].query_id == 2
        assert eval_results[1].core.logical_soundness == 4.0
