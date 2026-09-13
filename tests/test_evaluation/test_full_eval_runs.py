"""
The multi-run driver in scripts/run_full_eval.py.

Chapter 5 reports each item as the mean over three repetitions with the
run-to-run SD beside it, so a run is not a single invocation writing into a
single directory: three of them run one after the other, each with its
own output directory, its own `run_id` in both CSVs and its own provenance
record. These tests pin the properties that make those three files usable as
three separate measurements — and the guards that stop an expensive run from
overwriting a finished one or from silently continuing after a system died.

The systems themselves and run_full_evaluation are mocked out; what is under
test is the driver around them.
"""

from __future__ import annotations

import csv
import json
import time
from unittest.mock import MagicMock, patch

import pytest

import scripts.run_full_eval as full_eval


def _gold_items(n: int = 3) -> list:
    return [
        MagicMock(id=i, question=f"q{i}")
        for i in range(1, n + 1)
    ]


def _fake_result(system_name: str, gold_items: list) -> dict:
    """The shape write_summary_csv/write_per_query_csv/the report consume."""
    return {
        "summary": {
            "system_name": system_name,
            "total_queries": len(gold_items),
            "successful_queries": len(gold_items),
            "failed_queries": 0,
            "failed_query_ids": [],
            "ragas_summary": {"composite_score": 0.5},
            "cross_document_success_rate": 0.5,
            "citation_accuracy": 0.5,
            "correction_rate": 0.0,
            "cost_per_correct_answer_usd": 0.1,
            "total_cost_usd": 1.0,
        },
        "detailed_results": [
            {
                "query_id": item.id,
                "fa_type": "FA-1",
                "subtype": "",
                "refusal_evidence": "",
                "expected_answerable": True,
                "custom_metrics": {
                    "exact_match": 1.0, "answer_recall": 1.0,
                    "refusal_accuracy": 0.0, "refusal_quality": None,
                    "over_refusal": 0.0,
                },
                "citation_metrics": {"citation_accuracy": 1.0},
                "reasoning_metrics": {
                    "chain_emitted": True, "num_steps": 2, "num_evidential": 1,
                    "num_inferential": 1, "num_untagged": 0,
                    "groundedness": 1.0, "validity": 1.0, "completeness": 1.0,
                    "fully_grounded": True, "fully_valid": True,
                },
                "ragas_metrics": {"answer_correctness": 0.9},
                "run_metrics": {
                    "latency_seconds": 1.0, "total_tokens": 100,
                    "estimated_cost_usd": 0.01, "num_steps": 1, "corrections": 0,
                },
            }
            for item in gold_items
        ],
    }


@pytest.fixture
def driver(tmp_path, monkeypatch):
    """Run main() against mocked systems, returning the recorded eval calls."""
    monkeypatch.setattr(full_eval, "PROJECT_ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "base.yaml").write_text("llm: {}", encoding="utf-8")

    gold_items = _gold_items()
    calls: list[tuple] = []

    def fake_eval(pipeline, system_name, items, out_dir, **kwargs):
        calls.append((system_name, str(out_dir), len(items)))
        return _fake_result(system_name, list(items))

    builds: list = []

    def fake_build(filings):
        builds.append(filings)
        return [(label, MagicMock(), name) for label, name, _ in full_eval.SYSTEM_SPECS]

    patches = [
        patch.object(full_eval, "download_all_filings", return_value=[
            MagicMock(metadata=MagicMock(ticker="AAPL"))
        ]),
        patch.object(full_eval, "fiscal_year_from_metadata", return_value=2024),
        patch.object(full_eval, "load_gold_standard", return_value=gold_items),
        patch.object(full_eval, "EvidenceStore", MagicMock()),
        patch.object(full_eval, "_build_pipelines", side_effect=fake_build),
        patch.object(full_eval, "run_full_evaluation", side_effect=fake_eval),
        patch("src.evaluation.custom_evaluator.get_eval_llm", MagicMock()),
        patch("src.evaluation.citation_evaluator.get_eval_llm", MagicMock()),
        # PROJECT_ROOT points at tmp_path here, so the real repo is out of
        # reach; _git_provenance is covered on its own below.
        patch.object(full_eval, "_git_provenance", return_value={
            "commit": "abc1234", "branch": "dev", "dirty": False, "dirty_files": [],
        }),
    ]
    for p in patches:
        p.start()
    yield MagicMock(
        root=tmp_path / "data" / "results" / "final_eval",
        calls=calls, builds=builds, gold_items=gold_items,
    )
    for p in patches:
        p.stop()


class TestRunIdValidation:
    def test_duplicate_run_ids_are_rejected(self, driver):
        with pytest.raises(SystemExit, match="unique"):
            full_eval.main(["--run-ids", "run1", "run1"])

    @pytest.mark.parametrize("bad", ["a/b", "..", "."])
    def test_run_ids_must_be_plain_directory_names(self, driver, bad):
        with pytest.raises(SystemExit, match="plain directory names"):
            full_eval.main(["--run-ids", bad])


class TestOverwriteGuard:
    def _occupy(self, driver, run_id: str) -> None:
        run_dir = driver.root / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "eval_rag_monolith_20260909_120000.json").write_text("{}", encoding="utf-8")

    def test_existing_results_abort_before_any_query(self, driver):
        self._occupy(driver, "run1")
        with pytest.raises(SystemExit, match="already holds"):
            full_eval.main(["--run-ids", "run1"])
        assert driver.calls == []

    def test_a_later_run_directory_is_checked_up_front(self, driver):
        """run3 being occupied must abort before run1 spends a cent."""
        self._occupy(driver, "run3")
        with pytest.raises(SystemExit, match="already holds"):
            full_eval.main(["--run-ids", "run1", "run2", "run3"])
        assert driver.calls == []

    def test_force_writes_into_an_occupied_directory(self, driver):
        self._occupy(driver, "run1")
        full_eval.main(["--run-ids", "run1", "--force"])
        assert len(driver.calls) == 4


class TestThreeRuns:
    def test_each_run_gets_its_own_directory_and_full_output_set(self, driver):
        full_eval.main(["--run-ids", "run1", "run2", "run3"])

        for run_id in ("run1", "run2", "run3"):
            run_dir = driver.root / run_id
            for name in (
                "full_eval_summary.csv", "full_eval_per_query.csv",
                "full_eval_report.md", "run_provenance.json", "run.log",
            ):
                assert (run_dir / name).is_file(), f"{run_id}/{name} missing"

    def test_every_system_runs_once_per_run(self, driver):
        full_eval.main(["--run-ids", "run1", "run2", "run3"])

        assert len(driver.calls) == 12
        for run_id in ("run1", "run2", "run3"):
            systems = {c[0] for c in driver.calls if c[1].endswith(run_id)}
            assert systems == {"rag_monolith", "rag_agent", "long_context", "multi_agent"}

    def test_pipelines_are_built_once_for_all_runs(self, driver):
        """Building is run-invariant; query() carries no state between calls."""
        full_eval.main(["--run-ids", "run1", "run2", "run3"])
        assert len(driver.builds) == 1

    def test_both_csvs_carry_their_run_id(self, driver):
        full_eval.main(["--run-ids", "run1", "run2", "run3"])

        for run_id in ("run1", "run2", "run3"):
            run_dir = driver.root / run_id
            for name in ("full_eval_summary.csv", "full_eval_per_query.csv"):
                with open(run_dir / name, encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                assert rows, f"{run_id}/{name} is empty"
                assert {r["run_id"] for r in rows} == {run_id}

    def test_runs_do_not_share_a_directory(self, driver):
        """The defect --out-dir was introduced for, one level down."""
        full_eval.main(["--run-ids", "run1", "run2"])
        out_dirs = {c[1] for c in driver.calls}
        assert len(out_dirs) == 2


class TestParallelMode:
    def test_report_columns_stay_in_s1_to_s4_order(self, driver):
        """Completion order under --parallel is arbitrary; column order is not."""
        finish_delay = {
            "rag_monolith": 0.06, "rag_agent": 0.04,
            "long_context": 0.02, "multi_agent": 0.0,
        }

        def slow_eval(pipeline, system_name, items, out_dir, **kwargs):
            time.sleep(finish_delay[system_name])
            return _fake_result(system_name, list(items))

        with patch.object(full_eval, "run_full_evaluation", side_effect=slow_eval):
            full_eval.main(["--parallel", "--run-ids", "run1"])

        with open(driver.root / "run1" / "full_eval_summary.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert [r["system"] for r in rows] == ["S1", "S2", "S3", "S4"]

        report = (driver.root / "run1" / "full_eval_report.md").read_text(encoding="utf-8")
        assert "| Metric | S1 | S2 | S3 | S4 |" in report

    def test_concurrency_mode_is_recorded(self, driver):
        full_eval.main(["--parallel", "--run-ids", "run1"])
        provenance = json.loads((driver.root / "run1" / "run_provenance.json").read_text())
        assert provenance["concurrency"] == "parallel"
        assert provenance["max_workers"] == 4

    def test_sequential_is_the_default(self, driver):
        full_eval.main(["--run-ids", "run1"])
        provenance = json.loads((driver.root / "run1" / "run_provenance.json").read_text())
        assert provenance["concurrency"] == "sequential"


class TestFailureHandling:
    def _one_system_fails(self, driver, failing="long_context"):
        def failing_eval(pipeline, system_name, items, out_dir, **kwargs):
            driver.calls.append((system_name, str(out_dir), len(items)))
            if system_name == failing:
                raise RuntimeError("endpoint gone")
            return _fake_result(system_name, list(items))
        return patch.object(full_eval, "run_full_evaluation", side_effect=failing_eval)

    def test_remaining_runs_are_not_started(self, driver):
        with self._one_system_fails(driver), pytest.raises(RuntimeError, match="not started"):
            full_eval.main(["--run-ids", "run1", "run2", "run3"])

        assert not (driver.root / "run2").exists()
        assert {c[0] for c in driver.calls} == {
            "rag_monolith", "rag_agent", "long_context", "multi_agent"
        }

    def test_the_systems_that_finished_are_still_written(self, driver):
        with self._one_system_fails(driver), pytest.raises(RuntimeError):
            full_eval.main(["--run-ids", "run1"])

        with open(driver.root / "run1" / "full_eval_summary.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert [r["system"] for r in rows] == ["S1", "S2", "S4"]

    def test_the_failure_is_recorded_in_the_provenance(self, driver):
        with self._one_system_fails(driver), pytest.raises(RuntimeError):
            full_eval.main(["--run-ids", "run1"])

        provenance = json.loads((driver.root / "run1" / "run_provenance.json").read_text())
        assert provenance["systems"]["S3"]["status"] == "failed"
        assert "endpoint gone" in provenance["systems"]["S3"]["error"]
        assert provenance["systems"]["S1"]["status"] == "ok"

    def test_one_system_failing_does_not_stop_the_others_in_parallel_mode(self, driver):
        with self._one_system_fails(driver), pytest.raises(RuntimeError):
            full_eval.main(["--parallel", "--run-ids", "run1"])

        assert len([c for c in driver.calls if c[1].endswith("run1")]) == 4


class TestProvenance:
    def test_records_what_produced_the_run(self, driver):
        full_eval.main(["--run-ids", "run1", "run2"])
        provenance = json.loads((driver.root / "run2" / "run_provenance.json").read_text())

        assert provenance["run_id"] == "run2"
        assert provenance["run_index"] == 2
        assert provenance["of_runs"] == 2
        assert provenance["all_run_ids"] == ["run1", "run2"]
        assert provenance["gold_standard"]["sha256"]
        assert provenance["gold_standard"]["items_evaluated"] == 3
        assert provenance["corpus"]["n_filings"] == 1
        assert "commit" in provenance["git"]
        assert provenance["wall_seconds"] >= 0

    def test_a_full_run_is_not_marked_truncated(self, driver):
        full_eval.main(["--run-ids", "run1"])
        provenance = json.loads((driver.root / "run1" / "run_provenance.json").read_text())
        assert provenance["gold_standard"]["truncated"] is False
        assert provenance["gold_standard"]["limit"] is None

    def test_a_smoke_run_is_marked_truncated(self, driver):
        """--limit must never be mistakable for a real measurement."""
        full_eval.main(["--limit", "2", "--run-ids", "smoke"])
        provenance = json.loads((driver.root / "smoke" / "run_provenance.json").read_text())

        assert provenance["gold_standard"]["truncated"] is True
        assert provenance["gold_standard"]["items_evaluated"] == 2
        assert provenance["gold_standard"]["items_in_file"] == 3
        assert all(call[2] == 2 for call in driver.calls)


class TestRunLog:
    def test_each_run_logs_into_its_own_file_and_detaches_after(self, driver):
        import logging

        handlers_before = len(logging.getLogger().handlers)
        full_eval.main(["--run-ids", "run1", "run2"])

        assert len(logging.getLogger().handlers) == handlers_before, "log handler leaked"
        log1 = (driver.root / "run1" / "run.log").read_text(encoding="utf-8")
        log2 = (driver.root / "run2" / "run.log").read_text(encoding="utf-8")
        assert "id=run1" in log1 and "id=run2" not in log1
        assert "id=run2" in log2


class TestGitProvenance:
    """Against the real repository, unlike the rest of this file."""

    def test_names_the_code_that_produced_the_run(self):
        git = full_eval._git_provenance()
        assert git["commit"] and len(git["commit"]) == 40
        assert git["branch"]
        assert isinstance(git["dirty"], bool)

    def test_a_dirty_tree_lists_what_is_uncommitted(self):
        """A commit hash alone does not identify a dirty tree's code."""
        git = full_eval._git_provenance()
        assert git["dirty"] == bool(git["dirty_files"])
