"""
Unit tests for the RAGAS evaluation wrapper (src/evaluation/ragas_evaluator.py).

`ragas.evaluate` itself is mocked out (it makes real judge-LLM calls
internally); these tests pin evaluate_run's own logic: length validation,
which metric set is requested for include_context_metrics True/False, and
EvalScores' None-handling for systems without an observable retrieval step.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.gold_standard_loader import GoldStandardItem
from src.evaluation.ragas_evaluator import EvalScores, _to_str, evaluate_run


def _make_gold_item(item_id: int = 1) -> GoldStandardItem:
    return GoldStandardItem(id=item_id, question="What was AAPL revenue?", ground_truth="391B", doc_refs="AAPL")


class TestToStr:
    def test_string_passthrough(self):
        assert _to_str("hello") == "hello"

    def test_gemini_style_content_parts_joined(self):
        assert _to_str([{"text": "hello"}, {"text": "world"}]) == "hello\nworld"

    def test_other_type_falls_back_to_str(self):
        assert _to_str(3.14) == "3.14"


class TestEvalScores:
    def test_composite_score_averages_all_five_when_all_present(self):
        scores = EvalScores(
            answer_relevancy=0.8, answer_correctness=0.6,
            context_precision=1.0, context_recall=1.0, faithfulness=1.0,
        )
        assert scores.composite_score == pytest.approx((0.8 + 0.6 + 1.0 + 1.0 + 1.0) / 5)

    def test_composite_score_averages_only_answer_metrics_when_context_metrics_none(self):
        scores = EvalScores(answer_relevancy=0.8, answer_correctness=0.6)
        assert scores.composite_score == pytest.approx((0.8 + 0.6) / 2)

    def test_composite_score_is_zero_when_nothing_computed(self):
        """Defensive default; evaluate_run always sets at least the two answer metrics."""
        scores = EvalScores(answer_relevancy=None, answer_correctness=None)  # type: ignore[arg-type]
        assert scores.composite_score == 0.0

    def test_to_dict_serializes_uncomputed_context_metrics_as_null(self):
        scores = EvalScores(answer_relevancy=0.8, answer_correctness=0.6)
        d = scores.to_dict()
        assert d["context_precision"] is None
        assert d["context_recall"] is None
        assert d["faithfulness"] is None
        assert d["answer_relevancy"] == 0.8

    def test_to_dict_rounds_computed_values(self):
        scores = EvalScores(answer_relevancy=0.123456, answer_correctness=0.6, context_precision=0.999999)
        d = scores.to_dict()
        assert d["answer_relevancy"] == 0.1235
        assert d["context_precision"] == 1.0


class TestEvaluateRun:
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            evaluate_run([_make_gold_item()], answers=[], contexts=[])

    def _mock_ragas_result(self, values: dict) -> MagicMock:
        result = MagicMock()
        result.__getitem__.side_effect = lambda key: values[key]
        return result

    def test_requests_all_five_metrics_when_context_metrics_included(self):
        items = [_make_gold_item()]
        ragas_result = self._mock_ragas_result({
            "answer_relevancy": [0.8], "answer_correctness": [0.6],
            "context_precision": [0.9], "context_recall": [0.7], "faithfulness": [1.0],
        })

        with patch("src.evaluation.ragas_evaluator.get_judge_llm"), \
             patch("src.evaluation.ragas_evaluator.get_embeddings"), \
             patch("src.evaluation.ragas_evaluator.evaluate", return_value=ragas_result) as mock_evaluate:
            scores = evaluate_run(items, answers=["391B"], contexts=[["ctx"]], include_context_metrics=True)

            _, kwargs = mock_evaluate.call_args
            from src.evaluation.ragas_evaluator import ABLATION_METRICS
            assert kwargs["metrics"] == ABLATION_METRICS

        assert scores.context_precision == 0.9
        assert scores.faithfulness == 1.0

    def test_requests_only_answer_metrics_when_context_metrics_excluded(self):
        items = [_make_gold_item()]
        ragas_result = self._mock_ragas_result({
            "answer_relevancy": [0.8], "answer_correctness": [0.6],
        })

        with patch("src.evaluation.ragas_evaluator.get_judge_llm"), \
             patch("src.evaluation.ragas_evaluator.get_embeddings"), \
             patch("src.evaluation.ragas_evaluator.evaluate", return_value=ragas_result) as mock_evaluate:
            scores = evaluate_run(items, answers=["391B"], contexts=[[]], include_context_metrics=False)

            _, kwargs = mock_evaluate.call_args
            from src.evaluation.ragas_evaluator import ANSWER_METRICS
            assert kwargs["metrics"] == ANSWER_METRICS

        assert scores.context_precision is None
        assert scores.context_recall is None
        assert scores.faithfulness is None
        assert scores.answer_relevancy == 0.8

    def test_nan_and_none_scores_are_excluded_from_mean(self):
        items = [_make_gold_item(1), _make_gold_item(2)]
        nan = float("nan")
        ragas_result = self._mock_ragas_result({
            "answer_relevancy": [0.8, nan], "answer_correctness": [0.6, None],
        })

        with patch("src.evaluation.ragas_evaluator.get_judge_llm"), \
             patch("src.evaluation.ragas_evaluator.get_embeddings"), \
             patch("src.evaluation.ragas_evaluator.evaluate", return_value=ragas_result):
            scores = evaluate_run(items, answers=["a", "b"], contexts=[[], []], include_context_metrics=False)

        assert scores.answer_relevancy == 0.8
        assert scores.answer_correctness == 0.6
