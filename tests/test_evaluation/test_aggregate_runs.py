"""
The cross-run aggregator in scripts/aggregate_runs.py.

Kapitel 5 reports each metric as the mean over three repetitions with the
run-to-run SD beside it, and reports the RAGAS metrics over the 120
answerable items rather than all 150. These tests pin the two properties
that make those numbers defensible:

  * the answerable-only restriction excludes exactly the 30 FA-Refusal
    items, on which a 0 is the correct outcome rather than a defect, and
    it is applied identically to every system, so it cannot move a
    ranking on its own; and
  * a composite averaged over 5 metrics is never presented as if it were
    the same quantity as one averaged over 2 — S3 and S4 retrieve no
    chunks and have no context metrics at all, so the naive composite
    rewards them for having fewer metrics.

Everything is built from small hand-made result dicts; no run directory
from an actual evaluation is read.
"""

from __future__ import annotations

import csv
import json
import math

import pytest

import scripts.aggregate_runs as agg


def _item(answerable: bool, ragas: dict, *, custom: dict | None = None) -> dict:
    return {
        "expected_answerable": answerable,
        "ragas_metrics": ragas,
        "custom_metrics": custom or {
            "exact_match": 1.0, "answer_recall": 1.0,
            "refusal_accuracy": 1.0, "refusal_quality": 1.0,
            "over_refusal": 0.0,
        },
        "reasoning_metrics": {},
        "run_metrics": {"total_tokens": 100, "latency_seconds": 1.0},
    }


def _result(detailed: list, ragas_summary: dict | None = None) -> dict:
    return {
        "summary": {
            "system_name": "x", "total_queries": len(detailed),
            "successful_queries": len(detailed), "failed_queries": 0,
            "retried_queries": 0,
            "ragas_summary": ragas_summary or {},
        },
        "detailed_results": detailed,
    }


class TestAnswerableOnlyRestriction:
    def test_refusal_items_are_excluded_from_context_metrics(self):
        """The 30 structural zeros must not enter the mean."""
        detailed = (
            [_item(True, {"context_precision": 0.8}) for _ in range(4)]
            + [_item(False, {"context_precision": 0.0}) for _ in range(4)]
        )
        out = agg.ragas_answerable_only(detailed)
        assert out["context_precision"] == pytest.approx(0.8)
        assert out["n_answerable"] == 4
        assert out["n_scored"]["context_precision"] == 4

    def test_matches_the_all150_mean_when_there_are_no_refusal_items(self):
        detailed = [_item(True, {"faithfulness": v}) for v in (0.2, 0.4, 0.6)]
        out = agg.ragas_answerable_only(detailed)
        assert out["faithfulness"] == pytest.approx(0.4)

    def test_missing_scores_leave_the_denominator_rather_than_deflate_it(self):
        """A judge that failed to score an item is not a 0 for that item."""
        detailed = [
            _item(True, {"context_recall": 1.0}),
            _item(True, {"context_recall": None}),
            _item(True, {}),
        ]
        out = agg.ragas_answerable_only(detailed)
        assert out["context_recall"] == pytest.approx(1.0)
        assert out["n_scored"]["context_recall"] == 1

    def test_nan_is_treated_as_missing_not_propagated(self):
        """A NaN would otherwise turn the whole mean into NaN."""
        detailed = [
            _item(True, {"answer_relevancy": 0.5}),
            _item(True, {"answer_relevancy": float("nan")}),
        ]
        out = agg.ragas_answerable_only(detailed)
        assert out["answer_relevancy"] == pytest.approx(0.5)
        assert not math.isnan(out["answer_relevancy"])

    def test_all_items_unscored_gives_none_not_zero(self):
        detailed = [_item(True, {}) for _ in range(3)]
        out = agg.ragas_answerable_only(detailed)
        assert out["context_precision"] is None
        assert out["composite_score"] is None

    def test_restriction_does_not_reorder_two_systems(self):
        """Applied identically, it shifts levels but cannot flip a ranking.

        Both systems get the same 2 refusal items scored 0; the better
        system on the answerable stratum stays the better one.
        """
        weak = [_item(True, {"context_precision": 0.2}) for _ in range(3)]
        strong = [_item(True, {"context_precision": 0.7}) for _ in range(3)]
        refusals = [_item(False, {"context_precision": 0.0}) for _ in range(2)]
        w = agg.ragas_answerable_only(weak + refusals)["context_precision"]
        s = agg.ragas_answerable_only(strong + refusals)["context_precision"]
        assert w < s


class TestCompositeComparability:
    def test_five_metric_and_two_metric_composites_are_distinguished(self):
        five = [_item(True, {
            "context_precision": 0.6, "context_recall": 0.6,
            "faithfulness": 0.6, "answer_relevancy": 0.6,
            "answer_correctness": 0.6,
        })]
        two = [_item(True, {"answer_relevancy": 0.6, "answer_correctness": 0.6})]
        assert agg.ragas_answerable_only(five)["composite_n_metrics"] == 5
        assert agg.ragas_answerable_only(two)["composite_n_metrics"] == 2

    def test_answer_only_composite_ignores_context_metrics(self):
        """The cross-system column must not move when context metrics do.

        This is the property that makes S2 comparable to S3/S4: S2 is
        penalised by three context metrics that S3 and S4 simply do not
        have, so only the answer metrics can be compared across all four.
        """
        with_ctx = agg.ragas_answerable_only([_item(True, {
            "context_precision": 0.0, "context_recall": 0.0,
            "faithfulness": 0.0, "answer_relevancy": 0.8,
            "answer_correctness": 0.6,
        })])
        without_ctx = agg.ragas_answerable_only([_item(True, {
            "answer_relevancy": 0.8, "answer_correctness": 0.6,
        })])
        assert with_ctx["composite_answer_only"] == pytest.approx(0.7)
        assert without_ctx["composite_answer_only"] == pytest.approx(0.7)
        # …while the naive composite would rank the context-less one higher.
        assert with_ctx["composite_score"] < without_ctx["composite_score"]


class TestSpread:
    def test_sd_needs_at_least_two_runs(self):
        assert agg._sd([0.5]) is None
        assert agg._sd([0.5, 0.7]) == pytest.approx(0.1414, abs=1e-3)

    def test_sd_is_the_sample_sd(self):
        """ddof=1: three repetitions are a sample of run-to-run variation."""
        assert agg._sd([1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_none_values_are_dropped_from_mean_and_sd(self):
        assert agg._mean([0.4, None, 0.6]) == pytest.approx(0.5)
        assert agg._sd([0.4, None]) is None


class TestLatencyExclusion:
    """Latency is averaged over run2 and run3 only.

    run1 ran against the `global` Vertex endpoint behind 633 quota errors
    and its S2 additionally spanned a machine suspend; `perf_counter`
    counts both as latency. Every other metric from run1 still counts —
    waiting for quota does not change which answer came back.
    """

    def _rows(self):
        return [
            {"run_id": "run1", "system": "S1", "source_file": "a.json",
             "latency_seconds": 100.0, "citation_accuracy": 0.5},
            {"run_id": "run2", "system": "S1", "source_file": "b.json",
             "latency_seconds": 10.0, "citation_accuracy": 0.7},
            {"run_id": "run3", "system": "S1", "source_file": "c.json",
             "latency_seconds": 12.0, "citation_accuracy": 0.9},
        ]

    def test_excluded_run_is_dropped_from_the_latency_mean(self):
        out = agg.aggregate(self._rows(), latency_excluded=("run1",))[0]
        assert out["latency_seconds_mean"] == pytest.approx(11.0)
        assert out["latency_runs"] == "run2+run3"
        assert out["n_latency_runs"] == 2

    def test_excluded_run_still_counts_for_every_other_metric(self):
        out = agg.aggregate(self._rows(), latency_excluded=("run1",))[0]
        assert out["citation_accuracy_mean"] == pytest.approx(0.7)
        assert out["n_runs"] == 3
        assert out["runs"] == "run1+run2+run3"

    def test_no_exclusion_averages_everything(self):
        out = agg.aggregate(self._rows())[0]
        assert out["latency_seconds_mean"] == pytest.approx(40.6667, abs=1e-3)
        assert out["n_latency_runs"] == 3

    def test_latency_sd_is_computed_over_the_kept_runs_only(self):
        out = agg.aggregate(self._rows(), latency_excluded=("run1",))[0]
        assert out["latency_seconds_sd"] == pytest.approx(1.4142, abs=1e-3)

    def test_excluding_every_run_leaves_latency_empty_not_wrong(self):
        out = agg.aggregate(
            self._rows(), latency_excluded=("run1", "run2", "run3")
        )[0]
        assert out["latency_seconds_mean"] is None
        assert out["n_latency_runs"] == 0
        assert out["latency_runs"] == "—"
        # …while the other metrics are untouched.
        assert out["citation_accuracy_mean"] == pytest.approx(0.7)


class TestEndToEnd:
    def _write_run(self, root, run_id, systems):
        run_dir = root / run_id
        run_dir.mkdir(parents=True)
        for system_name, result in systems.items():
            (run_dir / f"eval_{system_name}_20260101_000000.json").write_text(
                json.dumps(result), encoding="utf-8"
            )
        return run_dir

    def _systems(self, precision):
        answerable = [_item(True, {
            "context_precision": precision, "context_recall": 0.5,
            "faithfulness": 0.5, "answer_relevancy": 0.5,
            "answer_correctness": 0.5,
        }) for _ in range(4)]
        refusal = [_item(False, {"context_precision": 0.0}) for _ in range(2)]
        detailed = answerable + refusal
        return {
            "rag_monolith": _result(detailed),
            "rag_agent": _result(detailed),
            "long_context": _result(detailed),
            "multi_agent": _result(detailed),
        }

    def test_writes_three_files_and_averages_across_runs(self, tmp_path):
        root = tmp_path / "final_eval"
        self._write_run(root, "run1", self._systems(0.4))
        self._write_run(root, "run2", self._systems(0.6))
        out = tmp_path / "agg"

        agg.main(["run1", "run2", "--runs-root", str(root),
                  "--out-dir", str(out)])

        assert (out / "aggregate_per_run.csv").exists()
        assert (out / "aggregate_summary.csv").exists()
        assert (out / "aggregate_report.md").exists()

        rows = list(csv.DictReader(
            (out / "aggregate_summary.csv").open(encoding="utf-8")
        ))
        assert {r["system"] for r in rows} == {"S1", "S2", "S3", "S4"}
        s1 = next(r for r in rows if r["system"] == "S1")
        assert s1["n_runs"] == "2"
        # Mean of 0.4 and 0.6, with the refusal zeros excluded.
        assert float(s1["ragas_context_precision_mean"]) == pytest.approx(0.5)
        assert float(s1["ragas_context_precision_sd"]) == pytest.approx(
            0.1414, abs=1e-3
        )

    def test_per_run_csv_keeps_the_as_reported_value_for_traceability(
        self, tmp_path
    ):
        root = tmp_path / "final_eval"
        systems = self._systems(0.6)
        for result in systems.values():
            result["summary"]["ragas_summary"] = {"context_precision": 0.4}
        self._write_run(root, "run1", systems)
        out = tmp_path / "agg"

        agg.main(["run1", "--runs-root", str(root), "--out-dir", str(out)])

        rows = list(csv.DictReader(
            (out / "aggregate_per_run.csv").open(encoding="utf-8")
        ))
        s1 = next(r for r in rows if r["system"] == "S1")
        assert float(s1["ragas_context_precision"]) == pytest.approx(0.6)
        assert float(s1["ragas_context_precision_all150"]) == pytest.approx(0.4)

    def test_a_missing_run_is_skipped_rather_than_fatal(self, tmp_path):
        """Usable before run3 exists."""
        root = tmp_path / "final_eval"
        self._write_run(root, "run1", self._systems(0.5))
        out = tmp_path / "agg"

        agg.main(["run1", "run2", "run3", "--runs-root", str(root),
                  "--out-dir", str(out)])

        rows = list(csv.DictReader(
            (out / "aggregate_summary.csv").open(encoding="utf-8")
        ))
        assert all(r["n_runs"] == "1" for r in rows)
        assert all(r["ragas_context_precision_sd"] == "" for r in rows)

    def test_no_runs_at_all_is_an_error_not_an_empty_report(self, tmp_path):
        root = tmp_path / "final_eval"
        root.mkdir(parents=True)
        with pytest.raises(SystemExit):
            agg.main(["run1", "--runs-root", str(root),
                      "--out-dir", str(tmp_path / "agg")])

    def test_merged_file_wins_over_the_timestamped_one(self, tmp_path):
        """run1's S3 was re-measured and merged; the merge is authoritative."""
        root = tmp_path / "final_eval"
        run_dir = self._write_run(root, "run1", self._systems(0.1))
        merged = [_item(True, {"context_precision": 0.9}) for _ in range(4)]
        (run_dir / "eval_long_context_MERGED.json").write_text(
            json.dumps(_result(merged)), encoding="utf-8"
        )
        out = tmp_path / "agg"

        agg.main(["run1", "--runs-root", str(root), "--out-dir", str(out)])

        rows = list(csv.DictReader(
            (out / "aggregate_per_run.csv").open(encoding="utf-8")
        ))
        s3 = next(r for r in rows if r["system"] == "S3")
        assert s3["source_file"] == "eval_long_context_MERGED.json"
        assert float(s3["ragas_context_precision"]) == pytest.approx(0.9)

    def test_report_warns_when_composites_average_different_metric_counts(
        self, tmp_path
    ):
        root = tmp_path / "final_eval"
        systems = self._systems(0.5)
        # S3/S4 as they really are: answer metrics only, no retrieval.
        answer_only = [_item(True, {
            "answer_relevancy": 0.8, "answer_correctness": 0.6,
        }) for _ in range(4)]
        systems["long_context"] = _result(answer_only)
        systems["multi_agent"] = _result(answer_only)
        self._write_run(root, "run1", systems)
        out = tmp_path / "agg"

        agg.main(["run1", "--runs-root", str(root), "--out-dir", str(out)])

        report = (out / "aggregate_report.md").read_text(encoding="utf-8")
        assert "not comparable across systems" in report
        assert "composite (answer-only, comparable)" in report
