"""
Unit tests for the evaluation orchestrator (src/evaluation/eval_runner.py).

The four metric pipelines (RAGAS, custom, citation, reasoning) are mocked
out — this file pins run_full_evaluation's own aggregation logic: failure
handling, the FA-4 cross-document success rate, the correction rate, the
citation-accuracy average over answerable items only, and the
include_context_metrics routing.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.eval_runner import SYSTEMS_WITHOUT_MEANINGFUL_CONTEXT_METRICS, run_full_evaluation
from src.evaluation.gold_standard_loader import GoldStandardItem


def _make_gold_item(item_id: int, fa_type: str = "FA-1", expected_answerable: bool = True) -> GoldStandardItem:
    return GoldStandardItem(
        id=item_id, question=f"q{item_id}", ground_truth="gt", doc_refs="AAPL",
        fa_type=fa_type, expected_answerable=expected_answerable,
    )


def _make_pipeline_result(answer: str = "answer", corrections: int = 0, cost: float = 0.01) -> SimpleNamespace:
    return SimpleNamespace(
        answer=answer,
        contexts=["ctx"],
        metrics=SimpleNamespace(
            estimated_cost_usd=cost,
            latency_seconds=1.0,
            token_usage=SimpleNamespace(total_tokens=100),
            num_steps=1,
            corrections=corrections,
        ),
    )


class FakePipeline:
    """Stands in for a system pipeline; `responses` maps question -> result or Exception."""

    def __init__(self, responses: dict):
        self.responses = responses

    def query(self, question: str):
        response = self.responses[question]
        if isinstance(response, Exception):
            raise response
        return response


def _patch_evaluators(custom_metrics_by_question=None, citation_by_question=None):
    """Common patch set for the four metric pipelines called by run_full_evaluation."""
    custom_metrics_by_question = custom_metrics_by_question or {}
    citation_by_question = citation_by_question or {}

    def fake_custom(item, answer):
        result = custom_metrics_by_question.get(item.question, MagicMock(exact_match=0.0, answer_recall=0.0))
        result.to_dict = MagicMock(return_value={"exact_match": result.exact_match, "answer_recall": result.answer_recall})
        return result

    def fake_citation(item, answer):
        result = citation_by_question.get(item.question, MagicMock(citation_accuracy=0.0))
        result.to_dict = MagicMock(return_value={"citation_accuracy": result.citation_accuracy})
        return result

    ragas_scores = MagicMock()
    ragas_scores.to_dict.return_value = {"composite_score": 0.5}

    reasoning_result = MagicMock()
    reasoning_result.to_dict.return_value = {"core": {}}

    return dict(
        ragas_evaluate_run=MagicMock(return_value=ragas_scores),
        reasoning_evaluate_batch=MagicMock(return_value=[reasoning_result] * 100),
        custom_metrics=MagicMock(side_effect=fake_custom),
        citation_metrics=MagicMock(side_effect=fake_citation),
        format_trajectory=MagicMock(return_value="trajectory"),
    )


class TestRunFullEvaluationFailureHandling:
    def test_single_query_failure_is_excluded_but_run_continues(self, tmp_path):
        items = [_make_gold_item(1), _make_gold_item(2)]
        pipeline = FakePipeline({
            "q1": RuntimeError("agent crashed"),
            "q2": _make_pipeline_result(answer="ok"),
        })
        patches = _patch_evaluators()
        patches["reasoning_evaluate_batch"].return_value = [MagicMock(to_dict=MagicMock(return_value={}))]

        with patch("src.evaluation.eval_runner.evaluate_run", patches["ragas_evaluate_run"]), \
             patch("src.evaluation.eval_runner.evaluate_reasoning_batch", patches["reasoning_evaluate_batch"]), \
             patch("src.evaluation.eval_runner.evaluate_custom_metrics", patches["custom_metrics"]), \
             patch("src.evaluation.eval_runner.evaluate_citation_accuracy", patches["citation_metrics"]), \
             patch("src.evaluation.eval_runner.format_trajectory", patches["format_trajectory"]), \
             patch("src.evaluation.eval_runner.calculate_cost_per_correct_answer", return_value=0.02):
            output = run_full_evaluation(pipeline, "rag_monolith", items, tmp_path)

        assert output["summary"]["total_queries"] == 2
        assert output["summary"]["successful_queries"] == 1
        assert len(output["detailed_results"]) == 1
        assert output["detailed_results"][0]["question"] == "q2"

    def test_all_queries_failing_returns_error_without_crashing(self, tmp_path):
        items = [_make_gold_item(1)]
        pipeline = FakePipeline({"q1": RuntimeError("boom")})

        output = run_full_evaluation(pipeline, "rag_monolith", items, tmp_path)

        assert output == {"error": "All queries failed"}


class TestRunFullEvaluationAggregation:
    def _run(self, tmp_path, items, pipeline, custom_metrics_by_question=None, citation_by_question=None):
        patches = _patch_evaluators(custom_metrics_by_question, citation_by_question)
        n = len(items)
        patches["reasoning_evaluate_batch"].return_value = [
            MagicMock(to_dict=MagicMock(return_value={})) for _ in range(n)
        ]

        with patch("src.evaluation.eval_runner.evaluate_run", patches["ragas_evaluate_run"]) as mock_ragas, \
             patch("src.evaluation.eval_runner.evaluate_reasoning_batch", patches["reasoning_evaluate_batch"]), \
             patch("src.evaluation.eval_runner.evaluate_custom_metrics", patches["custom_metrics"]), \
             patch("src.evaluation.eval_runner.evaluate_citation_accuracy", patches["citation_metrics"]), \
             patch("src.evaluation.eval_runner.format_trajectory", patches["format_trajectory"]), \
             patch("src.evaluation.eval_runner.calculate_cost_per_correct_answer", return_value=0.03) as mock_cost:
            output = run_full_evaluation(pipeline, "rag_monolith", items, tmp_path)

        return output, mock_ragas, mock_cost

    def test_cross_document_success_rate_only_considers_fa4_items(self, tmp_path):
        items = [
            _make_gold_item(1, fa_type="FA-1"),
            _make_gold_item(2, fa_type="FA-4"),
            _make_gold_item(3, fa_type="FA-4"),
        ]
        pipeline = FakePipeline({f"q{i}": _make_pipeline_result() for i in (1, 2, 3)})
        custom_by_q = {
            "q1": MagicMock(exact_match=0.0, answer_recall=0.0),  # FA-1, irrelevant to cross-doc rate
            "q2": MagicMock(exact_match=1.0, answer_recall=0.0),  # FA-4 success (exact_match)
            "q3": MagicMock(exact_match=0.0, answer_recall=0.2),  # FA-4 failure (below 0.5 recall)
        }

        output, _, _ = self._run(tmp_path, items, pipeline, custom_metrics_by_question=custom_by_q)

        assert output["summary"]["cross_document_success_rate"] == 0.5

    def test_correction_rate_averages_over_items_with_run_metrics(self, tmp_path):
        items = [_make_gold_item(1), _make_gold_item(2)]
        pipeline = FakePipeline({
            "q1": _make_pipeline_result(corrections=1),
            "q2": _make_pipeline_result(corrections=0),
        })

        output, _, _ = self._run(tmp_path, items, pipeline)

        assert output["summary"]["correction_rate"] == 0.5

    def test_citation_accuracy_averages_only_over_answerable_items(self, tmp_path):
        items = [
            _make_gold_item(1, expected_answerable=True),
            _make_gold_item(2, expected_answerable=False),
        ]
        pipeline = FakePipeline({f"q{i}": _make_pipeline_result() for i in (1, 2)})
        citation_by_q = {
            "q1": MagicMock(citation_accuracy=0.8),
            "q2": MagicMock(citation_accuracy=0.0),  # refusal item, should be excluded from the average
        }

        output, _, _ = self._run(tmp_path, items, pipeline, citation_by_question=citation_by_q)

        assert output["summary"]["citation_accuracy"] == 0.8

    def test_cost_tracking_sums_estimated_cost_across_queries(self, tmp_path):
        items = [_make_gold_item(1), _make_gold_item(2)]
        pipeline = FakePipeline({
            "q1": _make_pipeline_result(cost=0.01),
            "q2": _make_pipeline_result(cost=0.02),
        })

        output, _, mock_cost = self._run(tmp_path, items, pipeline)

        assert output["summary"]["total_cost_usd"] == pytest.approx(0.03)
        _, kwargs = mock_cost.call_args
        assert kwargs["total_cost_usd"] == pytest.approx(0.03)

    def test_score_to_track_prefers_exact_match_over_answer_recall(self, tmp_path):
        items = [_make_gold_item(1), _make_gold_item(2)]
        pipeline = FakePipeline({f"q{i}": _make_pipeline_result() for i in (1, 2)})
        custom_by_q = {
            "q1": MagicMock(exact_match=1.0, answer_recall=0.3),  # math question -> exact_match used
            "q2": MagicMock(exact_match=0.0, answer_recall=0.7),  # text question -> answer_recall used
        }

        _, _, mock_cost = self._run(tmp_path, items, pipeline, custom_metrics_by_question=custom_by_q)

        _, kwargs = mock_cost.call_args
        assert kwargs["correctness_scores"] == [1.0, 0.7]

    def test_include_context_metrics_false_for_systems_without_observable_retrieval(self, tmp_path):
        assert "long_context" in SYSTEMS_WITHOUT_MEANINGFUL_CONTEXT_METRICS
        items = [_make_gold_item(1)]
        pipeline = FakePipeline({"q1": _make_pipeline_result()})
        patches = _patch_evaluators()
        patches["reasoning_evaluate_batch"].return_value = [MagicMock(to_dict=MagicMock(return_value={}))]

        with patch("src.evaluation.eval_runner.evaluate_run", patches["ragas_evaluate_run"]) as mock_ragas, \
             patch("src.evaluation.eval_runner.evaluate_reasoning_batch", patches["reasoning_evaluate_batch"]), \
             patch("src.evaluation.eval_runner.evaluate_custom_metrics", patches["custom_metrics"]), \
             patch("src.evaluation.eval_runner.evaluate_citation_accuracy", patches["citation_metrics"]), \
             patch("src.evaluation.eval_runner.format_trajectory", patches["format_trajectory"]), \
             patch("src.evaluation.eval_runner.calculate_cost_per_correct_answer", return_value=0.0):
            run_full_evaluation(pipeline, "long_context", items, tmp_path)

        _, kwargs = mock_ragas.call_args
        assert kwargs["include_context_metrics"] is False

    def test_output_file_is_written_to_output_dir(self, tmp_path):
        items = [_make_gold_item(1)]
        pipeline = FakePipeline({"q1": _make_pipeline_result()})

        self._run(tmp_path, items, pipeline)

        written_files = list(tmp_path.glob("eval_rag_monolith_*.json"))
        assert len(written_files) == 1
        with open(written_files[0]) as f:
            data = json.load(f)
        assert "summary" in data and "detailed_results" in data
