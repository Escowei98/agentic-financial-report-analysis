"""
Runs the full evaluation pipeline for S1, S2, S3, S4 on the complete
n=150 gold standard (see GOLD_STANDARD_EN in
src/evaluation/gold_standard_loader.py).

One invocation can carry out SEVERAL complete runs of all four systems
(`--run-ids run1 run2 run3`): chapter 5 reports each item's score as the mean
over three repetitions with the run-to-run SD beside it, and the McNemar
tests operate on the item majority over those repetitions. The runs execute
strictly sequentially, never two at once, and each writes into its OWN
subdirectory of --out-dir, so two runs can never land in one directory and
be silently mixed. Within a run the four systems may be evaluated
concurrently (--parallel); doing it the same way in every run is what keeps
latency comparable across them.

Each run directory holds:
- eval_<system>_<timestamp>.json — per-system full detail (rationales, trajectories)
- full_eval_summary.csv — one row per system
- full_eval_per_query.csv — one row per (system, query)
- full_eval_report.md — the cross-system table
- run_provenance.json — git commit, gold-standard hash, corpus, concurrency
  mode, timings, per-system cost and failed ids: what a run has to record to
  stay reconstructible once three of them are being compared
- run.log — this run's log, in addition to the console

Both CSVs carry `run_id` as their first column, so the three runs' files can
be concatenated as they are for aggregation.
"""
import argparse
import contextlib
import csv
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from src.common.config import load_config
from src.common.ingestion import download_all_filings, fiscal_year_from_metadata
from src.evaluation.eval_runner import QUERY_MAX_ATTEMPTS, run_full_evaluation
from src.evaluation.evidence_store import EvidenceStore
from src.evaluation.gold_standard_loader import GOLD_STANDARD_EN, load_gold_standard
from src.evaluation.judge_validation_sample import JUDGE_VALIDATION_IDS
from src.systems.long_context.pipeline import LongContextPipeline
from src.systems.multi_agent.pipeline import MultiAgentPipeline
from src.systems.rag_agent.pipeline import AgentRAGPipeline
from src.systems.rag_monolith.pipeline import MonolithRAGPipeline

# Console only at import time. The file handler is attached per run inside
# main(): with several runs per invocation there is no single log file, and
# a module-level FileHandler would create one on mere import, which the
# test suite does.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Label, config name, pipeline class. The label orders the report columns;
# results are always assembled in this order, including after a parallel run
# where completion order is arbitrary.
# Typed as Any because the four pipeline classes share no base class;
# what they share is the build()/query() protocol run_full_evaluation uses.
SYSTEM_SPECS: list[tuple[str, str, Any]] = [
    ("S1", "rag_monolith", MonolithRAGPipeline),
    ("S2", "rag_agent", AgentRAGPipeline),
    ("S3", "long_context", LongContextPipeline),
    ("S4", "multi_agent", MultiAgentPipeline),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The planned thesis run:\n"
            "  uv run python scripts/run_full_eval.py --parallel "
            "--run-ids run1 run2 run3\n"
        ),
    )
    parser.add_argument(
        "--out-dir", default="data/results/final_eval",
        help="Root directory for the run(s). Each run id becomes a "
             "subdirectory of it, so two runs can never share a directory "
             "and mix their JSONs.",
    )
    parser.add_argument(
        "--run-ids", nargs="+", default=["run1"], metavar="ID",
        help="One id per repetition, e.g. `--run-ids run1 run2 run3`. Each "
             "gets its own subdirectory under --out-dir and is written into "
             "the `run_id` column of both CSVs. Runs execute in the order "
             "given, one after the other. Passing a subset re-runs just "
             "those (a repetition that crashed, say).",
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Evaluate the four systems of a run concurrently. Wall time then "
             "tracks the slowest system (S3) instead of the sum of all four. "
             "The pipelines are still BUILT sequentially: S1 and S2 share a "
             "vectorstore directory and would race on it. Off by default "
             "because it quadruples the request rate against both the model "
             "and the judge endpoints.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Evaluate only the first N gold-standard items. For smoke-testing "
             "the machinery before spending a full run — the result is NOT a "
             "valid n=150 measurement and is marked as truncated in "
             "run_provenance.json.",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=QUERY_MAX_ATTEMPTS, metavar="N",
        help=f"Attempts per query before it counts as failed (default "
             f"{QUERY_MAX_ATTEMPTS}, 1 disables retrying). Applied identically "
             f"to all four systems. Every attempt is counted: per item in the "
             f"`query_attempts` CSV column, per system in `retried_queries`.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Write into a run directory that already contains results. Off "
             "by default: overwriting a finished repetition costs a full run "
             "to recover.",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_to_project(path: Path) -> Path:
    """Repo-relative path where possible, absolute otherwise.

    Provenance must never be the thing that aborts a run, and a gold standard
    outside the project tree is a configuration to record, not to crash on.
    """
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def _git_provenance() -> dict:
    """Commit, branch and dirty flag of the working tree that produced a run.

    A run is only reconstructible if the code that produced it can be named.
    Failures are recorded rather than raised: no git, no repo, or a detached
    state must not abort an evaluation that costs three figures.
    """
    def _git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=PROJECT_ROOT, capture_output=True,
                text=True, check=True, timeout=10,
            ).stdout.strip()
        except (subprocess.SubprocessError, OSError) as exc:
            logger.warning("git %s failed: %s", " ".join(args), exc)
            return None

    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # True is the interesting case: a dirty tree means the commit alone
        # does not identify the code that ran.
        "dirty": None if status is None else bool(status),
        "dirty_files": status.splitlines() if status else [],
    }


def _config_hashes() -> dict:
    """Hash every config file a system resolves against.

    The four systems' settings are the other half of "what produced this
    run", and unlike the code they are not covered by the commit hash once
    the tree is dirty.
    """
    config_dir = PROJECT_ROOT / "configs"
    return {p.name: _sha256(p) for p in sorted(config_dir.glob("*.yaml"))}


@contextlib.contextmanager
def _run_log_file(run_dir: Path):
    """Tee this run's log into its own directory, then detach again.

    Attached to the ROOT logger so eval_runner, the pipelines and the
    evaluators land in it too. The thread name is in the format because under
    --parallel four systems write into one file interleaved, and without it
    the lines cannot be told apart.
    """
    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.addHandler(handler)
    # The root logger's level gates records before any handler sees them, so
    # a caller that left it above INFO would leave behind a run.log with
    # nothing in it but errors. Raised for the duration, restored after.
    previous_level = root.level
    if not root.isEnabledFor(logging.INFO):
        root.setLevel(logging.INFO)
    try:
        yield
    finally:
        root.setLevel(previous_level)
        root.removeHandler(handler)
        handler.close()


def _build_pipelines(filings: list) -> list[tuple[str, Any, str]]:
    """Instantiate and build all four systems once, for every run to reuse.

    Building is the expensive, run-invariant part (chunking, embedding, the
    vectorstore); `query()` holds no state across calls in any of the four
    pipelines, so three repetitions on one set of built pipelines are three
    independent repetitions. Built sequentially even under --parallel: S1 and
    S2 resolve to the same vectorstore directory and would otherwise have two
    writers on one ChromaDB.
    """
    logger.info("Initializing pipelines...")
    # load_config() takes a SYSTEM NAME, not a path: it resolves
    # configs/<name>.yaml itself and merges it over base.yaml. A Path would
    # silently fall back to base.yaml alone and leave S1 and S2 on diverging
    # built-in defaults, breaking the shared retrieval stack FP-3 requires.
    built = []
    for label, sys_name, cls in SYSTEM_SPECS:
        pipeline = cls(load_config(sys_name))
        pipeline.build(filings)
        built.append((label, pipeline, sys_name))
    return built


def _run_systems_sequential(systems, gold_items, run_dir, evidence_store, max_attempts):
    results, failures = {}, {}
    for label, pipeline, sys_name in systems:
        logger.info("=" * 70)
        logger.info("Running full eval for %s (%s) — %d items", label, sys_name, len(gold_items))
        logger.info("=" * 70)
        try:
            results[label] = run_full_evaluation(
                pipeline, sys_name, gold_items, run_dir,
                evidence_store=evidence_store, max_attempts=max_attempts,
            )
            logger.info("%s summary: %s", label, results[label].get("summary", {}))
        except BaseException as exc:  # noqa: BLE001 — reported by the caller
            failures[label] = exc
            logger.error("✗ %s failed: %s", label, exc, exc_info=True)
    return results, failures


def _run_systems_parallel(systems, gold_items, run_dir, evidence_store, max_attempts):
    """Evaluate the four systems of one run concurrently.

    Safe only after every pipeline has been built (see _build_pipelines).
    Querying and judging are pure I/O against remote endpoints, so threads are
    the right tool and the GIL is not in the way.

    The judge clients are constructed here, before the pool starts, so the
    lazy singletons in custom_evaluator and citation_evaluator are not raced
    into existence four times over. `evidence_store` is passed in already
    built for the same reason, and shared rather than copied: it is read-only,
    and one store for all four systems IS the fairness property the
    groundedness dimension rests on.

    A system that raises does not take the others down: its exception is
    collected and the systems that finished are still written to disk.
    """
    from src.evaluation.citation_evaluator import get_eval_llm as _warm_citation
    from src.evaluation.custom_evaluator import get_eval_llm as _warm_custom

    _warm_custom()
    _warm_citation()

    logger.info("=" * 70)
    logger.info("Running %d systems concurrently — %d items each", len(systems), len(gold_items))
    logger.info("=" * 70)

    results, failures = {}, {}
    with ThreadPoolExecutor(max_workers=len(systems), thread_name_prefix="sys") as pool:
        futures = {
            pool.submit(
                run_full_evaluation, pipeline, sys_name, gold_items, run_dir,
                evidence_store=evidence_store, max_attempts=max_attempts,
            ): label
            for label, pipeline, sys_name in systems
        }
        for future in as_completed(futures):
            label = futures[future]
            try:
                results[label] = future.result()
                logger.info("✓ %s finished", label)
            except BaseException as exc:  # noqa: BLE001 — reported by the caller
                failures[label] = exc
                logger.error("✗ %s failed: %s", label, exc, exc_info=True)
    return results, failures


def _write_provenance(
    path: Path, *, run_id: str, run_index: int, all_run_ids: list[str],
    args: argparse.Namespace, gs_path: Path, gold_items: list, n_total_items: int,
    filings: list, results: dict, failures: dict, started_at: float, finished_at: float,
) -> None:
    provenance = {
        "run_id": run_id,
        "run_index": run_index,
        "of_runs": len(all_run_ids),
        "all_run_ids": all_run_ids,
        "started_at": datetime.fromtimestamp(started_at).isoformat(timespec="seconds"),
        "finished_at": datetime.fromtimestamp(finished_at).isoformat(timespec="seconds"),
        "wall_seconds": round(finished_at - started_at, 1),
        "concurrency": "parallel" if args.parallel else "sequential",
        "max_workers": len(SYSTEM_SPECS) if args.parallel else 1,
        "query_max_attempts": args.max_attempts,
        "git": _git_provenance(),
        "config_sha256": _config_hashes(),
        "gold_standard": {
            "path": str(_relative_to_project(gs_path)),
            "sha256": _sha256(gs_path),
            "items_in_file": n_total_items,
            "items_evaluated": len(gold_items),
            # The one flag that decides whether this run may be reported at
            # all. A truncated run must never be mistaken for the real thing.
            "truncated": len(gold_items) != n_total_items,
            "limit": args.limit,
        },
        "corpus": {
            "n_filings": len(filings),
            "filings": sorted(
                f"{f.metadata.ticker} FY{fiscal_year_from_metadata(f)}" for f in filings
            ),
        },
        "systems": {
            label: {
                "system_name": sys_name,
                "status": "ok" if label in results else "failed",
                "error": None if label not in failures else repr(failures[label]),
                "successful_queries": results.get(label, {}).get("summary", {}).get("successful_queries"),
                "failed_queries": results.get(label, {}).get("summary", {}).get("failed_queries"),
                "failed_query_ids": results.get(label, {}).get("summary", {}).get("failed_query_ids"),
                "retried_queries": results.get(label, {}).get("summary", {}).get("retried_queries"),
                "retried_query_ids": results.get(label, {}).get("summary", {}).get("retried_query_ids"),
                "total_cost_usd": results.get(label, {}).get("summary", {}).get("total_cost_usd"),
            }
            for label, sys_name, _cls in SYSTEM_SPECS
        },
        "total_cost_usd": round(
            sum(
                res.get("summary", {}).get("total_cost_usd") or 0.0
                for res in results.values()
            ), 4,
        ),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
    }
    path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    logger.info("Provenance saved to %s", path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if len(set(args.run_ids)) != len(args.run_ids):
        raise SystemExit(f"--run-ids must be unique, got: {args.run_ids}")
    for run_id in args.run_ids:
        if "/" in run_id or run_id in {".", ".."}:
            raise SystemExit(f"--run-ids must be plain directory names, got: {run_id!r}")

    out_root = PROJECT_ROOT / args.out_dir
    # Checked for ALL runs before the first query is sent: discovering on run
    # 3 that its directory is occupied would waste two full runs' worth of
    # wall time and budget.
    for run_id in args.run_ids:
        existing = list((out_root / run_id).glob("eval_*.json"))
        if existing and not args.force:
            raise SystemExit(
                f"Run directory {out_root / run_id} already holds "
                f"{len(existing)} result JSON(s). Pick another --run-ids "
                f"value or pass --force to write into it anyway."
            )

    gs_path = GOLD_STANDARD_EN
    filings = download_all_filings()
    logger.info(
        "Corpus: %d filings — %s",
        len(filings),
        ", ".join(sorted(f"{f.metadata.ticker} FY{fiscal_year_from_metadata(f)}" for f in filings)),
    )

    all_gold_items = load_gold_standard(gs_path)
    gold_items = all_gold_items[:args.limit] if args.limit else all_gold_items
    logger.info("Loaded %d gold standard items from %s", len(all_gold_items), gs_path.name)
    if args.limit:
        logger.warning(
            "SMOKE MODE: only the first %d of %d items are evaluated. This run "
            "is not a valid n=%d measurement.",
            len(gold_items), len(all_gold_items), len(all_gold_items),
        )

    systems = _build_pipelines(filings)

    # Corpus access for the groundedness dimension. Built once, shared by all
    # four systems and all runs -- that sharing IS the fairness property:
    # every step is verified against passages fetched by one procedure from
    # one store, whatever the system's own retrieval did.
    evidence_store = EvidenceStore(filings)

    logger.info(
        "Plan: %d run(s) %s, %d systems %s, %d items each",
        len(args.run_ids), ", ".join(args.run_ids),
        len(systems), "in parallel" if args.parallel else "sequentially",
        len(gold_items),
    )

    for run_index, run_id in enumerate(args.run_ids, start=1):
        run_dir = out_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        started_at = time.time()

        with _run_log_file(run_dir):
            logger.info("#" * 70)
            logger.info("RUN %d/%d — id=%s → %s", run_index, len(args.run_ids), run_id, run_dir)
            logger.info("#" * 70)

            runner = _run_systems_parallel if args.parallel else _run_systems_sequential
            raw_results, failures = runner(
                systems, gold_items, run_dir, evidence_store, args.max_attempts
            )

            # Fixed S1..S4 order regardless of completion order, so the report
            # columns mean the same thing in every run.
            results = {
                label: raw_results[label]
                for label, _, _ in SYSTEM_SPECS if label in raw_results
            }

            if results:
                write_summary_csv(results, run_dir / "full_eval_summary.csv", run_id=run_id)
                write_per_query_csv(
                    results, gold_items, run_dir / "full_eval_per_query.csv", run_id=run_id
                )
                write_markdown_report(results, run_dir / "full_eval_report.md")

            _write_provenance(
                run_dir / "run_provenance.json",
                run_id=run_id, run_index=run_index, all_run_ids=args.run_ids,
                args=args, gs_path=gs_path, gold_items=gold_items,
                n_total_items=len(all_gold_items), filings=filings,
                results=results, failures=failures,
                started_at=started_at, finished_at=time.time(),
            )
            logger.info(
                "Run %s complete in %.1f min. Outputs in %s",
                run_id, (time.time() - started_at) / 60, run_dir,
            )

        if failures:
            # Remaining runs are abandoned on purpose. A run missing a system
            # is not a cross-system comparison, and three figures per run is
            # too much to spend on repeating a setup that just proved broken.
            # Everything that finished is already on disk; resume with
            # `--run-ids <the ids that are left>`.
            raise RuntimeError(
                f"Run {run_id}: {len(failures)} of {len(systems)} systems failed "
                f"({', '.join(sorted(failures))}). "
                f"{len(args.run_ids) - run_index} run(s) not started."
            )

    logger.info("All %d run(s) complete. Root: %s", len(args.run_ids), out_root)


def _avg(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _as_float(flag: bool | None) -> float | None:
    """Booleans into a form _avg can average, keeping None as missing."""
    return None if flag is None else float(flag)


def compute_extra_aggregates(detailed_results: list) -> dict:
    """
    Aggregate metrics not already present in eval_runner's summary dict.

    Custom metrics (exact_match/answer_recall/refusal_accuracy/
    refusal_quality/over_refusal) are only
    ever computed by evaluate_custom_metrics() for a subset of items —
    exact_match/answer_recall only when expected_answerable=True,
    refusal_accuracy only when expected_answerable=False (see
    src/evaluation/custom_evaluator.py::evaluate_custom_metrics). Averaging
    over all 150 items would dilute each score with 0.0-default values from
    items where the metric was never computed, so we filter to the subset
    it actually applies to — the same convention eval_runner.py already
    uses for citation_accuracy/correction_rate.

    The three reasoning dimensions follow the same rule for the same reason:
    they are defined on the answerable stratum only, and an item whose chain
    was never emitted carries None
    rather than 0.0 -- a format failure is not a reasoning defect and must not
    be averaged in as one. `_avg` drops Nones, so both cases fall out of the
    denominator instead of deflating it.
    """
    answerable = [d for d in detailed_results if d.get("expected_answerable")]
    refusal = [d for d in detailed_results if not d.get("expected_answerable")]
    reasoning = [d.get("reasoning_metrics", {}) for d in answerable]
    scored = [r for r in reasoning if r.get("chain_emitted")]

    return {
        "exact_match": _avg([d["custom_metrics"]["exact_match"] for d in answerable]),
        "answer_recall": _avg([d["custom_metrics"]["answer_recall"] for d in answerable]),
        "refusal_accuracy": _avg([d["custom_metrics"]["refusal_accuracy"] for d in refusal]),
        "refusal_quality": _avg(
            [d["custom_metrics"].get("refusal_quality") for d in refusal]
        ),
        # Share of ANSWERABLE items the system declined instead of attempting.
        # The counterweight to refusal_accuracy, which a system that declines
        # everything would otherwise score a perfect 1.0 on. Items scored fully
        # correct are recorded as 0.0 without a judge call, so the denominator
        # is the whole answerable stratum, not just the failures.
        "over_refusal_rate": _avg(
            [d["custom_metrics"].get("over_refusal") for d in answerable]
        ),
        # Share of answerable items where a parseable chain came back at all.
        # Reported per system because an uneven rate is itself a finding: it
        # would mean the three dimensions below are averaged over differently
        # sized, and possibly differently selected, sets of items.
        "chain_emission_rate": (
            round(len(scored) / len(answerable), 4) if answerable else None
        ),
        "groundedness": _avg([r.get("groundedness") for r in scored]),
        "validity": _avg([r.get("validity") for r in scored]),
        "completeness": _avg([r.get("completeness") for r in scored]),
        # Weakest-link aggregation (Jacovi et al. 2024): the share of chains
        # with NO violation at all. Reported next to the means because the two
        # respond differently to chain length, which is exactly what differs
        # between a monolith and a multi-agent system.
        "fully_grounded_rate": _avg(
            [_as_float(r.get("fully_grounded")) for r in scored]
        ),
        "fully_valid_rate": _avg([_as_float(r.get("fully_valid")) for r in scored]),
        # Faithfulness of the final answer against the corpus at its cited
        # loci (locus_faithfulness.py). Defined on the answerable stratum;
        # an answer citing no locus is None, not 0, so coverage is reported
        # next to the mean for the same reason chain_emission_rate is.
        "locus_faithfulness": _avg(
            [(d.get("locus_faithfulness") or {}).get("score") for d in answerable]
        ),
        "locus_faithfulness_coverage": (
            round(
                sum(
                    1 for d in answerable
                    if (d.get("locus_faithfulness") or {}).get("score") is not None
                ) / len(answerable), 4,
            ) if answerable else None
        ),
        "avg_chain_steps": _avg([r.get("num_steps") for r in scored]),
        "avg_evidential_steps": _avg([r.get("num_evidential") for r in scored]),
        "avg_inferential_steps": _avg([r.get("num_inferential") for r in scored]),
        "untagged_steps": sum(r.get("num_untagged") or 0 for r in scored),
        "total_tokens": _avg([d.get("run_metrics", {}).get("total_tokens") for d in detailed_results]),
        "latency_seconds": _avg([d.get("run_metrics", {}).get("latency_seconds") for d in detailed_results]),
    }


def write_summary_csv(results: dict, out_path: Path, run_id: str = "") -> None:
    # `run_id` is the first column of both CSVs so the repetitions can be
    # concatenated as they are: the aggregation step (mean over the three
    # runs, run-to-run SD, item majority for McNemar) needs to know which
    # repetition a row came from, and nothing else in the row says so.
    fieldnames = [
        "run_id",
        "system", "total_queries", "successful_queries", "failed_queries",
        # A system that reached 150/150 only after repeated retries is not
        # in the same state as one that never needed them.
        "retried_queries",
        "ragas_faithfulness", "ragas_answer_relevancy",
        "ragas_context_precision", "ragas_context_recall",
        "ragas_answer_correctness", "ragas_composite_score",
        "exact_match", "answer_recall", "refusal_accuracy",
        "refusal_quality", "over_refusal_rate",
        "chain_emission_rate", "groundedness", "validity", "completeness",
        "fully_grounded_rate", "fully_valid_rate",
        "avg_chain_steps", "avg_evidential_steps", "avg_inferential_steps",
        "untagged_steps",
        "locus_faithfulness", "locus_faithfulness_coverage",
        "cross_document_success_rate", "citation_accuracy",
        "correction_rate", "avg_total_tokens", "avg_latency_seconds",
        "cost_per_correct_answer_usd", "total_cost_usd",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sys_label, res in results.items():
            s = res.get("summary", {})
            ragas = s.get("ragas_summary", {})
            extra = compute_extra_aggregates(res.get("detailed_results", []))
            writer.writerow({
                "run_id": run_id,
                "system": sys_label,
                "total_queries": s.get("total_queries"),
                "successful_queries": s.get("successful_queries"),
                "failed_queries": s.get("failed_queries", 0),
                "retried_queries": s.get("retried_queries", 0),
                "ragas_faithfulness": ragas.get("faithfulness"),
                "ragas_answer_relevancy": ragas.get("answer_relevancy"),
                "ragas_context_precision": ragas.get("context_precision"),
                "ragas_context_recall": ragas.get("context_recall"),
                "ragas_answer_correctness": ragas.get("answer_correctness"),
                "ragas_composite_score": ragas.get("composite_score"),
                "exact_match": extra["exact_match"],
                "answer_recall": extra["answer_recall"],
                "refusal_accuracy": extra["refusal_accuracy"],
                "refusal_quality": extra["refusal_quality"],
                "over_refusal_rate": extra["over_refusal_rate"],
                "chain_emission_rate": extra["chain_emission_rate"],
                "groundedness": extra["groundedness"],
                "validity": extra["validity"],
                "completeness": extra["completeness"],
                "fully_grounded_rate": extra["fully_grounded_rate"],
                "fully_valid_rate": extra["fully_valid_rate"],
                "avg_chain_steps": extra["avg_chain_steps"],
                "avg_evidential_steps": extra["avg_evidential_steps"],
                "avg_inferential_steps": extra["avg_inferential_steps"],
                "untagged_steps": extra["untagged_steps"],
                "locus_faithfulness": extra["locus_faithfulness"],
                "locus_faithfulness_coverage": extra["locus_faithfulness_coverage"],
                "cross_document_success_rate": s.get("cross_document_success_rate"),
                "citation_accuracy": s.get("citation_accuracy"),
                "correction_rate": s.get("correction_rate"),
                "avg_total_tokens": extra["total_tokens"],
                "avg_latency_seconds": extra["latency_seconds"],
                "cost_per_correct_answer_usd": s.get("cost_per_correct_answer_usd"),
                "total_cost_usd": s.get("total_cost_usd"),
            })
    logger.info("Summary CSV saved to %s", out_path)


def write_per_query_csv(
    results: dict, gold_items: list, out_path: Path, run_id: str = ""
) -> None:
    fieldnames = [
        "run_id",
        "query_id", "fa_type", "subtype", "refusal_evidence",
        "expected_answerable",
        # True for the 34 ids the judge was validated and tuned on. Robustness
        # checks in chapter 5 filter on this column: a verdict that holds on
        # the remaining 116 items does not rest on items the judge was fitted
        # to. See src/evaluation/judge_validation_sample.py.
        "in_judge_validation_sample",
        "system",
        "exact_match", "answer_recall", "refusal_accuracy",
        "refusal_quality", "over_refusal",
        "citation_accuracy",
        "chain_emitted", "groundedness", "validity", "completeness",
        "fully_grounded", "fully_valid",
        "chain_steps", "evidential_steps", "inferential_steps",
        # Per-item RAGAS scores (see eval_runner.py). The three context
        # metrics stay empty for the systems without an observable
        # retrieval step; a NaN from RAGAS is written as an empty cell too.
        "ragas_answer_relevancy", "ragas_answer_correctness",
        "ragas_context_precision", "ragas_context_recall",
        "ragas_faithfulness",
        # Faithfulness against the corpus at the answer's cited loci, all
        # four systems; empty with the reason in the next column when the
        # item was not scored (refusal item, no locus cited, ...).
        "locus_faithfulness", "locus_faithfulness_excluded",
        "latency_seconds", "total_tokens", "estimated_cost_usd",
        "num_steps", "corrections", "query_attempts",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sys_label, res in results.items():
            for d in res.get("detailed_results", []):
                custom = d.get("custom_metrics", {})
                citation = d.get("citation_metrics", {})
                reasoning = d.get("reasoning_metrics", {})
                run_metrics = d.get("run_metrics", {})
                ragas = d.get("ragas_metrics", {})
                locus = d.get("locus_faithfulness") or {}
                writer.writerow({
                    "run_id": run_id,
                    "query_id": d.get("query_id"),
                    "fa_type": d.get("fa_type"),
                    "subtype": d.get("subtype"),
                    "refusal_evidence": d.get("refusal_evidence"),
                    "expected_answerable": d.get("expected_answerable"),
                    "in_judge_validation_sample": d.get("query_id") in JUDGE_VALIDATION_IDS,
                    "system": sys_label,
                    "exact_match": custom.get("exact_match"),
                    "answer_recall": custom.get("answer_recall"),
                    "refusal_accuracy": custom.get("refusal_accuracy"),
                    "refusal_quality": custom.get("refusal_quality"),
                    "over_refusal": custom.get("over_refusal"),
                    "citation_accuracy": citation.get("citation_accuracy"),
                    "chain_emitted": reasoning.get("chain_emitted"),
                    "groundedness": reasoning.get("groundedness"),
                    "validity": reasoning.get("validity"),
                    "completeness": reasoning.get("completeness"),
                    "fully_grounded": reasoning.get("fully_grounded"),
                    "fully_valid": reasoning.get("fully_valid"),
                    "chain_steps": reasoning.get("num_steps"),
                    "evidential_steps": reasoning.get("num_evidential"),
                    "inferential_steps": reasoning.get("num_inferential"),
                    "ragas_answer_relevancy": ragas.get("answer_relevancy"),
                    "ragas_answer_correctness": ragas.get("answer_correctness"),
                    "ragas_context_precision": ragas.get("context_precision"),
                    "ragas_context_recall": ragas.get("context_recall"),
                    "ragas_faithfulness": ragas.get("faithfulness"),
                    "locus_faithfulness": locus.get("score"),
                    "locus_faithfulness_excluded": locus.get("excluded"),
                    "latency_seconds": run_metrics.get("latency_seconds"),
                    "total_tokens": run_metrics.get("total_tokens"),
                    "estimated_cost_usd": run_metrics.get("estimated_cost_usd"),
                    "num_steps": run_metrics.get("num_steps"),
                    "corrections": run_metrics.get("corrections"),
                    "query_attempts": d.get("query_attempts"),
                })
    logger.info("Per-query CSV saved to %s", out_path)



_REFUSAL_SUBTYPES = ("not_in_corpus", "false_premise", "ambiguous_entity")


def _refusal_subtype_section(results: dict, labels: list) -> list[str]:
    """Break the FA-Refusal stratum down by subtype.

    The aggregate refusal_accuracy hides the finding this stratum was built
    for: the three subtypes are three different defect classes, and a system
    can be perfect at declining out-of-corpus questions while never noticing a
    false premise. Reported per subtype so that difference is visible.

    Ten items per subtype: descriptive only. The power analysis behind the
    design (thesis 3.6.1) justifies n=30 per stratum for the McNemar
    comparisons, not n=10 per subtype -- no significance test belongs on these
    cells.
    """
    lines = ["", "## FA-Refusal by subtype", ""]
    lines.append(
        "Refusal Accuracy = did the system avoid fabricating an answer "
        "(binary). Refusal Quality = did it diagnose WHY the question fails "
        "(0 = answered as though it were sound, 0.5 = declined without a "
        "diagnosis, 1.0 = named the defect). n=10 per subtype: descriptive, "
        "no significance tests."
    )
    lines.append("")
    lines.append("| Subtype | Metric | " + " | ".join(labels) + " |")
    lines.append("|---|---|" + "---|" * len(labels))
    for subtype in _REFUSAL_SUBTYPES:
        for metric, nice in (("refusal_accuracy", "Accuracy"), ("refusal_quality", "Quality")):
            cells = []
            for lbl in labels:
                items = [
                    d for d in results[lbl].get("detailed_results", [])
                    if d.get("subtype") == subtype
                ]
                value = _avg([d.get("custom_metrics", {}).get(metric) for d in items])
                cells.append(f"{value:.2f}" if isinstance(value, (int, float)) else "-")
            lines.append(f"| {subtype} | {nice} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def write_markdown_report(results: dict, out_path: Path) -> None:
    labels = list(results.keys())
    extras = {lbl: compute_extra_aggregates(results[lbl].get("detailed_results", [])) for lbl in labels}

    def fmt(x, nd=3):
        return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "-"

    def row(name: str, getter) -> str:
        vals = [getter(results[lbl].get("summary", {})) for lbl in labels]
        return f"| {name} | " + " | ".join(vals) + " |"

    def extra_row(name: str, key: str, nd: int = 3) -> str:
        vals = [fmt(extras[lbl][key], nd) for lbl in labels]
        return f"| {name} | " + " | ".join(vals) + " |"

    lines = ["# Full Evaluation Report (n=150)", ""]
    lines.append("| Metric | " + " | ".join(labels) + " |")
    lines.append("|---|" + "---|" * len(labels))
    lines.append(row("Successful Queries", lambda s: f"{s.get('successful_queries')}/{s.get('total_queries')}"))
    lines.append(row("Failed Queries", lambda s: str(s.get("failed_queries", 0))))
    lines.append(row("Retried Queries", lambda s: str(s.get("retried_queries", 0))))

    lines.append("| **RAGAS** | | | | |")
    lines.append(row("RAGAS Composite", lambda s: fmt(s.get("ragas_summary", {}).get("composite_score"))))
    lines.append(row("RAGAS Context Precision", lambda s: fmt(s.get("ragas_summary", {}).get("context_precision"))))
    lines.append(row("RAGAS Context Recall", lambda s: fmt(s.get("ragas_summary", {}).get("context_recall"))))
    lines.append(row("RAGAS Faithfulness", lambda s: fmt(s.get("ragas_summary", {}).get("faithfulness"))))
    lines.append(row("RAGAS Answer Relevancy", lambda s: fmt(s.get("ragas_summary", {}).get("answer_relevancy"))))
    lines.append(row("RAGAS Answer Correctness", lambda s: fmt(s.get("ragas_summary", {}).get("answer_correctness"))))

    lines.append("| **Custom Metrics** | | | | |")
    lines.append(extra_row("Exact Match (answerable items)", "exact_match"))
    lines.append(extra_row("Answer Recall (answerable items)", "answer_recall"))
    lines.append(extra_row("Refusal Accuracy (refusal items)", "refusal_accuracy"))
    lines.append(extra_row("Refusal Quality (refusal items, 0/0.5/1)", "refusal_quality"))
    lines.append(extra_row("Over-Refusal Rate (answerable items)", "over_refusal_rate"))

    # Three values, reported separately and never combined into one. The
    # systems are expected to differ in their PROFILE across the three, which
    # a composite would average away — and there is no defensible weighting
    # between grounding a claim, drawing a valid inference, and covering the
    # question.
    lines.append("| **Reasoning (answerable items, per-unit)** | | | | |")
    lines.append(extra_row("Chain Emission Rate", "chain_emission_rate"))
    lines.append(extra_row("Groundedness (mean over steps)", "groundedness"))
    lines.append(extra_row("Validity (mean over transitions)", "validity"))
    lines.append(extra_row("Completeness (sub-question recall)", "completeness"))
    # Weakest-link aggregation beside the means (Jacovi et al. 2024): a chain
    # is only as strong as its worst step. The two respond differently to
    # chain length, which is precisely what separates a monolith from a
    # multi-agent system, so reporting one without the other would hide the
    # effect rather than measure it.
    lines.append(extra_row("…chains with no groundedness defect", "fully_grounded_rate"))
    lines.append(extra_row("…chains with no validity defect", "fully_valid_rate"))
    lines.append("| **Reasoning covariates** | | | | |")
    lines.append(extra_row("Avg. Chain Steps", "avg_chain_steps", 2))
    lines.append(extra_row("Avg. Evidential Steps", "avg_evidential_steps", 2))
    lines.append(extra_row("Avg. Inferential Steps", "avg_inferential_steps", 2))
    lines.append(extra_row("Untagged Steps (total)", "untagged_steps", 0))

    lines.append("| **Process / Efficiency** | | | | |")
    lines.append(row("Cross-Doc Success (FA-4 + FA-3 cross_window)",
                     lambda s: fmt(s.get("cross_document_success_rate"))))
    lines.append(row("Citation Accuracy", lambda s: fmt(s.get("citation_accuracy"))))
    lines.append(row("Correction Rate", lambda s: fmt(s.get("correction_rate"))))
    lines.append(extra_row("Avg. Total Tokens / Query", "total_tokens", 0))
    lines.append(extra_row("Avg. Latency (s) / Query", "latency_seconds", 2))
    lines.append(row("Cost / Correct Answer (USD)", lambda s: fmt(s.get("cost_per_correct_answer_usd"), 4)))
    lines.append(row("Total Cost (USD)", lambda s: fmt(s.get("total_cost_usd"), 4)))
    lines.append("")

    lines.extend(_refusal_subtype_section(results, labels))

    if any(res.get("summary", {}).get("failed_queries", 0) for res in results.values()):
        lines.append(
            "> **Note (failed queries):** averages above are taken over the "
            "successful queries only — a failed query has no result object for "
            "the judge or RAGAS to score. Where the counts differ between "
            "systems, the averages are not on the same base; the failed ids "
            "are listed per system in the JSON under `summary.failed_query_ids`."
        )
        lines.append("")

    from src.evaluation.eval_runner import SYSTEMS_WITHOUT_MEANINGFUL_CONTEXT_METRICS
    context_metrics_skipped = any(
        res.get("summary", {}).get("system_name") in SYSTEMS_WITHOUT_MEANINGFUL_CONTEXT_METRICS
        for res in results.values()
    )
    if context_metrics_skipped:
        lines.append(
            "> **Note (Long-Context-family systems):** RAGAS Context Precision, "
            "Context Recall and Faithfulness are shown as `-` for systems that "
            "inline the full document set into the prompt ahead of time instead "
            "of retrieving it at query time (no observable retrieval step for "
            "RAGAS to score against — see `SYSTEMS_WITHOUT_MEANINGFUL_CONTEXT_METRICS` "
            "in `src/evaluation/eval_runner.py`). They are not computed at all "
            "for these systems, not merely hidden: an earlier run showed this "
            "produces a structurally deflated score for a single-agent long-context "
            "system (near-empty `contexts`) and a structurally inflated score "
            "for a multi-agent one (`contexts` = an upstream agent's own generated "
            "answer, checked for self-consistency rather than grounding in the "
            "primary source). Comparable work handles this the same way — "
            "FinanceBench (Islam et al., 2023), Li et al. (2024, \"RAG or "
            "Long-Context LLMs?\"), and Lithgow-Serrano et al. (2025, FinDoc-RAG) "
            "all compare retrieval- and non-retrieval-based conditions purely on "
            "final-answer metrics, not on retrieval-specific context metrics. "
            "RAGAS Answer Relevancy and Answer Correctness are unaffected (they "
            "score the answer against the question/ground truth, not against "
            "`contexts`) and remain comparable across all systems. RAGAS "
            "Composite is therefore a 5-metric average for retrieval-based "
            "systems and a 2-metric average for these systems (`ANSWER_METRICS` "
            "in `src/evaluation/ragas_evaluator.py`) — the two are "
            "not on the same scale."
        )
        lines.append("")

    lines.append(
        "> **Note (RAGAS answer metrics):** Answer Relevancy and Answer "
        "Correctness are averaged over the answerable items only "
        "(`expected_answerable=True`, 120 of 150 in the current gold "
        "standard). RAGAS scores a correct refusal as \"noncommittal\" "
        "(Answer Relevancy 0) and the refusal items carry an empty ground "
        "truth, so including them would penalise exactly the behaviour the "
        "refusal stratum rewards — `Refusal Accuracy` measures it instead. "
        "The per-item scores in `full_eval_per_query.csv` are unfiltered; "
        "`answer_metrics_n` in each system's JSON summary records how many "
        "items the average covers."
    )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Markdown report saved to %s", out_path)


if __name__ == "__main__":
    main()
