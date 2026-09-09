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


def _make_gold_item(item_id: int = 1, expected_answerable: bool = True) -> GoldStandardItem:
    return GoldStandardItem(
        id=item_id,
        question="What was AAPL revenue?",
        ground_truth="391B" if expected_answerable else "",
        doc_refs="AAPL",
        expected_answerable=expected_answerable,
    )


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
            assert [m.name for m in kwargs["metrics"]] == [
                "context_precision", "context_recall", "faithfulness",
                "answer_relevancy", "answer_correctness",
            ]

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
            assert [m.name for m in kwargs["metrics"]] == ["answer_relevancy", "answer_correctness"]

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

    def test_per_sample_scores_are_returned_for_every_requested_metric(self):
        items = [_make_gold_item(1), _make_gold_item(2)]
        ragas_result = self._mock_ragas_result({
            "answer_relevancy": [0.8, 0.4], "answer_correctness": [0.6, 0.2],
            "context_precision": [0.9, 0.5], "context_recall": [0.7, 0.3],
            "faithfulness": [1.0, 0.1],
        })

        with patch("src.evaluation.ragas_evaluator.get_judge_llm"), \
             patch("src.evaluation.ragas_evaluator.get_embeddings"), \
             patch("src.evaluation.ragas_evaluator.evaluate", return_value=ragas_result):
            scores = evaluate_run(items, answers=["a", "b"], contexts=[["c"], ["c"]], include_context_metrics=True)

        assert set(scores.per_sample) == {
            "answer_relevancy", "answer_correctness",
            "context_precision", "context_recall", "faithfulness",
        }
        assert scores.per_sample["answer_correctness"] == [0.6, 0.2]
        assert scores.per_sample["faithfulness"] == [1.0, 0.1]

    def test_per_sample_holds_only_answer_metrics_when_context_metrics_excluded(self):
        items = [_make_gold_item(1)]
        ragas_result = self._mock_ragas_result({
            "answer_relevancy": [0.8], "answer_correctness": [0.6],
        })

        with patch("src.evaluation.ragas_evaluator.get_judge_llm"), \
             patch("src.evaluation.ragas_evaluator.get_embeddings"), \
             patch("src.evaluation.ragas_evaluator.evaluate", return_value=ragas_result):
            scores = evaluate_run(items, answers=["a"], contexts=[[]], include_context_metrics=False)

        assert set(scores.per_sample) == {"answer_relevancy", "answer_correctness"}

    def test_per_sample_normalises_nan_to_none(self):
        items = [_make_gold_item(1), _make_gold_item(2)]
        ragas_result = self._mock_ragas_result({
            "answer_relevancy": [0.8, float("nan")], "answer_correctness": [0.6, None],
        })

        with patch("src.evaluation.ragas_evaluator.get_judge_llm"), \
             patch("src.evaluation.ragas_evaluator.get_embeddings"), \
             patch("src.evaluation.ragas_evaluator.evaluate", return_value=ragas_result):
            scores = evaluate_run(items, answers=["a", "b"], contexts=[[], []], include_context_metrics=False)

        assert scores.per_sample["answer_relevancy"] == [0.8, None]
        assert scores.per_sample["answer_correctness"] == [0.6, None]

    def test_answer_metrics_average_over_answerable_items_only(self):
        """A refusal item scores 0 as "noncommittal" — it must not drag the mean down."""
        items = [_make_gold_item(1), _make_gold_item(2, expected_answerable=False)]
        ragas_result = self._mock_ragas_result({
            "answer_relevancy": [0.8, 0.0], "answer_correctness": [0.6, 0.0],
        })

        with patch("src.evaluation.ragas_evaluator.get_judge_llm"), \
             patch("src.evaluation.ragas_evaluator.get_embeddings"), \
             patch("src.evaluation.ragas_evaluator.evaluate", return_value=ragas_result):
            scores = evaluate_run(items, answers=["a", "b"], contexts=[[], []], include_context_metrics=False)

        assert scores.answer_relevancy == pytest.approx(0.8)
        assert scores.answer_correctness == pytest.approx(0.6)
        assert scores.answer_metrics_n == 1
        # The refusal item's own scores are still recorded.
        assert scores.per_sample["answer_relevancy"] == [0.8, 0.0]

    def test_context_metrics_still_average_over_all_items(self):
        """Narrowing them too would change what S1/S2's numbers mean vs. S3/S4's."""
        items = [_make_gold_item(1), _make_gold_item(2, expected_answerable=False)]
        ragas_result = self._mock_ragas_result({
            "answer_relevancy": [0.8, 0.0], "answer_correctness": [0.6, 0.0],
            "context_precision": [1.0, 0.0], "context_recall": [1.0, 0.0],
            "faithfulness": [1.0, 0.0],
        })

        with patch("src.evaluation.ragas_evaluator.get_judge_llm"), \
             patch("src.evaluation.ragas_evaluator.get_embeddings"), \
             patch("src.evaluation.ragas_evaluator.evaluate", return_value=ragas_result):
            scores = evaluate_run(items, answers=["a", "b"], contexts=[["c"], ["c"]], include_context_metrics=True)

        assert scores.context_precision == pytest.approx(0.5)
        assert scores.faithfulness == pytest.approx(0.5)

    def test_falls_back_to_all_items_when_none_are_answerable(self):
        items = [_make_gold_item(1, expected_answerable=False)]
        ragas_result = self._mock_ragas_result({
            "answer_relevancy": [0.4], "answer_correctness": [0.2],
        })

        with patch("src.evaluation.ragas_evaluator.get_judge_llm"), \
             patch("src.evaluation.ragas_evaluator.get_embeddings"), \
             patch("src.evaluation.ragas_evaluator.evaluate", return_value=ragas_result):
            scores = evaluate_run(items, answers=["a"], contexts=[[]], include_context_metrics=False)

        assert scores.answer_relevancy == pytest.approx(0.4)
        assert scores.answer_metrics_n == 1


class TestMetricInstancesAreNotShared:
    """ragas' evaluate() binds llm/embeddings onto the metric objects it is
    handed and resets them when it returns. With the module-level singletons
    and four systems evaluated concurrently, the first call to finish reset
    the shared AnswerCorrectness under the others ("AnswerSimilarity must be
    set"), and the losing system's item dropped out of answer_correctness.
    Each call must therefore get its own instances."""

    def test_two_calls_get_distinct_instances(self):
        from src.evaluation.ragas_evaluator import build_metrics

        first = build_metrics(include_context_metrics=True)
        second = build_metrics(include_context_metrics=True)
        assert len(first) == 5
        for a, b in zip(first, second):
            assert type(a) is type(b)
            assert a is not b

    def test_the_answer_only_set_holds_no_context_metric(self):
        from src.evaluation.ragas_evaluator import build_metrics

        names = [m.name for m in build_metrics(include_context_metrics=False)]
        assert names == ["answer_relevancy", "answer_correctness"]

    def test_evaluate_run_hands_ragas_fresh_objects_each_time(self):
        from unittest.mock import MagicMock, patch

        from src.evaluation.ragas_evaluator import evaluate_run

        result = MagicMock()
        result.__getitem__.side_effect = lambda key: {"answer_relevancy": [0.5], "answer_correctness": [0.5]}[key]
        with patch("src.evaluation.ragas_evaluator.get_judge_llm"), \
             patch("src.evaluation.ragas_evaluator.get_embeddings"), \
             patch("src.evaluation.ragas_evaluator.evaluate", return_value=result) as mock_evaluate:
            evaluate_run([_make_gold_item()], ["a"], [[]], include_context_metrics=False)
            evaluate_run([_make_gold_item()], ["a"], [[]], include_context_metrics=False)
        first = mock_evaluate.call_args_list[0].kwargs["metrics"]
        second = mock_evaluate.call_args_list[1].kwargs["metrics"]
        assert all(a is not b for a, b in zip(first, second))
