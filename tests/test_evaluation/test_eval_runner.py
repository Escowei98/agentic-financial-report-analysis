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


def _make_gold_item(
    item_id: int,
    fa_type: str = "FA-1",
    expected_answerable: bool = True,
    gt_unit: str = "USD_billion",
) -> GoldStandardItem:
    return GoldStandardItem(
        id=item_id, question=f"q{item_id}", ground_truth="gt", doc_refs="AAPL",
        fa_type=fa_type, expected_answerable=expected_answerable, gt_unit=gt_unit,
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


def _patch_evaluators(custom_metrics_by_question=None, citation_by_question=None, ragas_per_sample=None):
    """Common patch set for the four metric pipelines called by run_full_evaluation."""
    custom_metrics_by_question = custom_metrics_by_question or {}
    citation_by_question = citation_by_question or {}

    def fake_custom(item, answer):
        result = custom_metrics_by_question.get(
            item.question, MagicMock(exact_match=0.0, answer_recall=0.0, refusal_accuracy=0.0)
        )
        result.to_dict = MagicMock(return_value={
            "exact_match": result.exact_match,
            "answer_recall": result.answer_recall,
            "refusal_accuracy": result.refusal_accuracy,
        })
        return result

    def fake_citation(item, answer):
        result = citation_by_question.get(item.question, MagicMock(citation_accuracy=0.0))
        result.to_dict = MagicMock(return_value={"citation_accuracy": result.citation_accuracy})
        return result

    ragas_scores = MagicMock()
    ragas_scores.to_dict.return_value = {"composite_score": 0.5}
    # Per-sample RAGAS scores, aligned with the items that actually ran.
    ragas_scores.per_sample = ragas_per_sample or {}

    reasoning_result = MagicMock()
    reasoning_result.to_dict.return_value = {"chain_emitted": False}

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
             patch("src.evaluation.eval_runner.evaluate_reasoning_chain_batch", patches["reasoning_evaluate_batch"]), \
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
             patch("src.evaluation.eval_runner.evaluate_reasoning_chain_batch", patches["reasoning_evaluate_batch"]), \
             patch("src.evaluation.eval_runner.evaluate_custom_metrics", patches["custom_metrics"]), \
             patch("src.evaluation.eval_runner.evaluate_citation_accuracy", patches["citation_metrics"]), \
             patch("src.evaluation.eval_runner.format_trajectory", patches["format_trajectory"]), \
             patch("src.evaluation.eval_runner.calculate_cost_per_correct_answer", return_value=0.03) as mock_cost:
            output = run_full_evaluation(pipeline, "rag_monolith", items, tmp_path)

        return output, mock_ragas, mock_cost

    def test_cross_document_rate_includes_fa3_cross_window_items(self, tmp_path):
        """FA-4 alone left out the 15 multi-year items whose span does not fit
        inside one report's comparative columns — precisely what
        `window_class` was added to identify.
        See EVAL_DECISION_LOG.md [2026-09-06].
        """
        items = [
            _make_gold_item(1, fa_type="FA-1"),
            _make_gold_item(2, fa_type="FA-4"),
            _make_gold_item(3, fa_type="FA-3"),
            _make_gold_item(4, fa_type="FA-3"),
        ]
        items[2].window_class = "cross_window"
        items[3].window_class = "within_window"
        pipeline = FakePipeline({f"q{i}": _make_pipeline_result() for i in (1, 2, 3, 4)})
        custom_by_q = {
            "q1": MagicMock(exact_match=1.0, answer_recall=1.0, refusal_accuracy=0.0),
            "q2": MagicMock(exact_match=1.0, answer_recall=1.0, refusal_accuracy=0.0),
            "q3": MagicMock(exact_match=0.0, answer_recall=1.0, refusal_accuracy=0.0),
            "q4": MagicMock(exact_match=0.0, answer_recall=0.0, refusal_accuracy=0.0),
        }

        output, _, _ = self._run(tmp_path, items, pipeline, custom_metrics_by_question=custom_by_q)

        # Population is q2 (FA-4) and q3 (FA-3 cross_window). q1 is single-doc,
        # q4 fits inside one filing's comparative columns.
        assert output["summary"]["cross_document_success_rate"] == 0.5

    def test_cross_document_rate_requires_a_full_exact_match(self, tmp_path):
        """The old bar also accepted `answer_recall >= 0.5`, and the metric
        saturated: S2, S3 and S4 all landed on exactly 28/30. A half-right
        cross-document synthesis is not a success.
        """
        items = [_make_gold_item(1, fa_type="FA-4"), _make_gold_item(2, fa_type="FA-4")]
        pipeline = FakePipeline({f"q{i}": _make_pipeline_result() for i in (1, 2)})
        custom_by_q = {
            "q1": MagicMock(exact_match=1.0, answer_recall=1.0, refusal_accuracy=0.0),
            # Would have counted as a success under the old rule.
            "q2": MagicMock(exact_match=0.0, answer_recall=0.9, refusal_accuracy=0.0),
        }

        output, _, _ = self._run(tmp_path, items, pipeline, custom_metrics_by_question=custom_by_q)

        assert output["summary"]["cross_document_success_rate"] == 0.5

    def test_failed_queries_are_reported_not_just_dropped(self, tmp_path):
        """A failed query has no result object to score, so it leaves the
        averages. That silently favours a system that crashes more often
        unless the count is on the record.
        """
        items = [_make_gold_item(1), _make_gold_item(2)]
        pipeline = FakePipeline({"q1": _make_pipeline_result()})  # q2 raises

        output, _, _ = self._run(tmp_path, items, pipeline)

        assert output["summary"]["failed_queries"] == 1
        assert output["summary"]["failed_query_ids"] == [2]
        assert output["summary"]["successful_queries"] == 1

    def test_single_document_items_stay_out_of_the_cross_document_rate(self, tmp_path):
        items = [
            _make_gold_item(1, fa_type="FA-1"),
            _make_gold_item(2, fa_type="FA-2"),
            _make_gold_item(3, fa_type="FA-4"),
            _make_gold_item(4, fa_type="FA-4"),
        ]
        pipeline = FakePipeline({f"q{i}": _make_pipeline_result() for i in (1, 2, 3, 4)})
        custom_by_q = {
            # FA-1 and FA-2 answer from a single filing — a perfect score here
            # must not lift the cross-document rate.
            "q1": MagicMock(exact_match=1.0, answer_recall=1.0, refusal_accuracy=0.0),
            "q2": MagicMock(exact_match=1.0, answer_recall=1.0, refusal_accuracy=0.0),
            "q3": MagicMock(exact_match=1.0, answer_recall=1.0, refusal_accuracy=0.0),
            "q4": MagicMock(exact_match=0.0, answer_recall=0.2, refusal_accuracy=0.0),
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

    def test_score_to_track_dispatches_on_ground_truth_unit(self, tmp_path):
        """exact_match for items whose ground truth is a figure it can rule
        on, answer_recall for the qualitative ones. The old rule read
        "exact_match unless it is 0, else answer_recall", which let a wrong
        figure with good coverage count as correct and contradicted the
        comment above it. See EVAL_DECISION_LOG.md [2026-09-06].
        """
        items = [
            _make_gold_item(1, gt_unit="USD_billion"),
            _make_gold_item(2, gt_unit="text"),
        ]
        pipeline = FakePipeline({f"q{i}": _make_pipeline_result() for i in (1, 2)})
        custom_by_q = {
            "q1": MagicMock(exact_match=1.0, answer_recall=0.3, refusal_accuracy=0.0),
            "q2": MagicMock(exact_match=0.0, answer_recall=0.7, refusal_accuracy=0.0),
        }

        _, _, mock_cost = self._run(tmp_path, items, pipeline, custom_metrics_by_question=custom_by_q)

        _, kwargs = mock_cost.call_args
        assert kwargs["correctness_scores"] == [1.0, 0.7]

    def test_wrong_figure_with_good_recall_is_not_counted_correct(self, tmp_path):
        items = [_make_gold_item(1, gt_unit="percent")]
        pipeline = FakePipeline({"q1": _make_pipeline_result()})
        custom_by_q = {
            "q1": MagicMock(exact_match=0.0, answer_recall=1.0, refusal_accuracy=0.0),
        }

        _, _, mock_cost = self._run(tmp_path, items, pipeline, custom_metrics_by_question=custom_by_q)

        _, kwargs = mock_cost.call_args
        assert kwargs["correctness_scores"] == [0.0]

    def test_score_to_track_uses_refusal_accuracy_for_unanswerable_items(self, tmp_path):
        """exact_match/answer_recall are never computed for refusal items — a
        correct refusal must still be able to count as a correct answer."""
        items = [_make_gold_item(1), _make_gold_item(2, expected_answerable=False)]
        pipeline = FakePipeline({f"q{i}": _make_pipeline_result() for i in (1, 2)})
        custom_by_q = {
            "q1": MagicMock(exact_match=1.0, answer_recall=0.0, refusal_accuracy=0.0),
            # Correct refusal: the two answerable-only metrics stay at 0.0.
            "q2": MagicMock(exact_match=0.0, answer_recall=0.0, refusal_accuracy=1.0),
        }

        _, _, mock_cost = self._run(tmp_path, items, pipeline, custom_metrics_by_question=custom_by_q)

        _, kwargs = mock_cost.call_args
        assert kwargs["correctness_scores"] == [1.0, 1.0]

    def test_answerable_items_ignore_refusal_accuracy(self, tmp_path):
        items = [_make_gold_item(1)]
        pipeline = FakePipeline({"q1": _make_pipeline_result()})
        custom_by_q = {"q1": MagicMock(exact_match=0.0, answer_recall=0.0, refusal_accuracy=1.0)}

        _, _, mock_cost = self._run(tmp_path, items, pipeline, custom_metrics_by_question=custom_by_q)

        _, kwargs = mock_cost.call_args
        assert kwargs["correctness_scores"] == [0.0]

    def test_include_context_metrics_false_for_systems_without_observable_retrieval(self, tmp_path):
        assert "long_context" in SYSTEMS_WITHOUT_MEANINGFUL_CONTEXT_METRICS
        items = [_make_gold_item(1)]
        pipeline = FakePipeline({"q1": _make_pipeline_result()})
        patches = _patch_evaluators()
        patches["reasoning_evaluate_batch"].return_value = [MagicMock(to_dict=MagicMock(return_value={}))]

        with patch("src.evaluation.eval_runner.evaluate_run", patches["ragas_evaluate_run"]) as mock_ragas, \
             patch("src.evaluation.eval_runner.evaluate_reasoning_chain_batch", patches["reasoning_evaluate_batch"]), \
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


class TestPerItemRagasScores:
    """Per-item RAGAS lands in detailed_results, aligned with the items that ran."""

    def test_per_sample_scores_are_attached_to_the_matching_item(self, tmp_path):
        items = [_make_gold_item(1), _make_gold_item(2)]
        pipeline = FakePipeline({
            "q1": _make_pipeline_result(answer="a1"),
            "q2": _make_pipeline_result(answer="a2"),
        })
        patches = _patch_evaluators(ragas_per_sample={
            "answer_relevancy": [0.9, 0.1],
            "answer_correctness": [0.8, None],
        })
        patches["reasoning_evaluate_batch"].return_value = [
            MagicMock(to_dict=MagicMock(return_value={})) for _ in items
        ]

        with patch("src.evaluation.eval_runner.evaluate_run", patches["ragas_evaluate_run"]), \
             patch("src.evaluation.eval_runner.evaluate_reasoning_chain_batch", patches["reasoning_evaluate_batch"]), \
             patch("src.evaluation.eval_runner.evaluate_custom_metrics", patches["custom_metrics"]), \
             patch("src.evaluation.eval_runner.evaluate_citation_accuracy", patches["citation_metrics"]), \
             patch("src.evaluation.eval_runner.format_trajectory", patches["format_trajectory"]), \
             patch("src.evaluation.eval_runner.calculate_cost_per_correct_answer", return_value=0.02):
            output = run_full_evaluation(pipeline, "rag_monolith", items, tmp_path)

        assert output["detailed_results"][0]["ragas_metrics"] == {
            "answer_relevancy": 0.9, "answer_correctness": 0.8,
        }
        assert output["detailed_results"][1]["ragas_metrics"] == {
            "answer_relevancy": 0.1, "answer_correctness": None,
        }

    def test_alignment_follows_the_successful_items_after_a_failure(self, tmp_path):
        """evaluate_run only sees the valid items, so index 0 is the item that ran."""
        items = [_make_gold_item(1), _make_gold_item(2)]
        pipeline = FakePipeline({
            "q1": RuntimeError("agent crashed"),
            "q2": _make_pipeline_result(answer="a2"),
        })
        patches = _patch_evaluators(ragas_per_sample={"answer_correctness": [0.42]})
        patches["reasoning_evaluate_batch"].return_value = [MagicMock(to_dict=MagicMock(return_value={}))]

        with patch("src.evaluation.eval_runner.evaluate_run", patches["ragas_evaluate_run"]), \
             patch("src.evaluation.eval_runner.evaluate_reasoning_chain_batch", patches["reasoning_evaluate_batch"]), \
             patch("src.evaluation.eval_runner.evaluate_custom_metrics", patches["custom_metrics"]), \
             patch("src.evaluation.eval_runner.evaluate_citation_accuracy", patches["citation_metrics"]), \
             patch("src.evaluation.eval_runner.format_trajectory", patches["format_trajectory"]), \
             patch("src.evaluation.eval_runner.calculate_cost_per_correct_answer", return_value=0.02):
            output = run_full_evaluation(pipeline, "rag_monolith", items, tmp_path)

        assert len(output["detailed_results"]) == 1
        assert output["detailed_results"][0]["question"] == "q2"
        assert output["detailed_results"][0]["ragas_metrics"] == {"answer_correctness": 0.42}


class TestReasoningChainSplitting:
    """The chain must be off the answer before any other evaluator sees it.

    The chain restates the answer's figures and repeats its citations. Left
    attached, it hands citation_accuracy a second copy of every citation and
    the recall judges a second copy of every number -- silently, and to a
    different degree per system, since the four do not write chains of equal
    length.
    """

    ANSWER_WITH_CHAIN = (
        "Revenue rose to $391,035M. (AAPL, FY2024, Income Statement)\n\n"
        "## Reasoning\n"
        "1. [E] FY2020 sales were $274,515M. (AAPL, FY2020, Income Statement)\n"
        "2. [I] So they rose.\n"
    )

    def _run(self, tmp_path, evidence_store=None):
        items = [_make_gold_item(1)]
        pipeline = FakePipeline({"q1": _make_pipeline_result(answer=self.ANSWER_WITH_CHAIN)})
        patches = _patch_evaluators()
        with patch("src.evaluation.eval_runner.evaluate_run", patches["ragas_evaluate_run"]), \
             patch("src.evaluation.eval_runner.evaluate_reasoning_chain_batch",
                   patches["reasoning_evaluate_batch"]), \
             patch("src.evaluation.eval_runner.evaluate_custom_metrics", patches["custom_metrics"]), \
             patch("src.evaluation.eval_runner.evaluate_citation_accuracy", patches["citation_metrics"]), \
             patch("src.evaluation.eval_runner.format_trajectory", patches["format_trajectory"]), \
             patch("src.evaluation.eval_runner.calculate_cost_per_correct_answer", return_value=0.02):
            result = run_full_evaluation(
                pipeline, "rag_monolith", items, tmp_path, evidence_store=evidence_store
            )
        return result, patches

    def test_downstream_evaluators_get_the_answer_without_the_chain(self, tmp_path):
        result, patches = self._run(tmp_path)
        seen = patches["custom_metrics"].call_args.args[1]
        assert "## Reasoning" not in seen
        assert "FY2020 sales" not in seen
        assert "Revenue rose to $391,035M." in seen

    def test_the_stored_answer_is_the_stripped_one(self, tmp_path):
        result, _ = self._run(tmp_path)
        assert "## Reasoning" not in result["detailed_results"][0]["answer"]

    def test_the_reasoning_evaluator_gets_the_raw_answer(self, tmp_path):
        store = MagicMock()
        _, patches = self._run(tmp_path, evidence_store=store)
        raw_answers = patches["reasoning_evaluate_batch"].call_args.args[1]
        assert "## Reasoning" in raw_answers[0]

    def test_without_a_store_the_chain_shape_is_still_recorded(self, tmp_path):
        """A run configured without corpus access still reports how many
        chains came back and how long they were -- chain_emission_rate is a
        covariate, not a by-product of scoring."""
        result, patches = self._run(tmp_path)
        patches["reasoning_evaluate_batch"].assert_not_called()
        reasoning = result["detailed_results"][0]["reasoning_metrics"]
        assert reasoning["chain_emitted"] is True
        assert reasoning["num_steps"] == 2
        assert reasoning["groundedness"] is None
