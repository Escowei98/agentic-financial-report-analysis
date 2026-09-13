"""
Aggregates the three repetitions into the numbers Kapitel 5 reports:
per-system means across runs, plus the run-to-run spread.

Reads only the eval_*.json files a run already produced and writes into a
separate directory. Nothing under a run directory is touched, so the raw
per-run results stay exactly as the runner left them and both RAGAS
variants below remain reconstructible.

RAGAS ON THE ANSWERABLE STRATUM
-------------------------------
eval_runner averages `context_precision` and `faithfulness` over all 150
items, the 30 FA-Refusal items included. On those, a 0.0 is not a defect
but the correct outcome: an item whose expected answer is "not in the
corpus" has no relevant context to retrieve, so precision against an empty
target is 0 by construction and the system is penalised for behaving
correctly.

Every other metric family in this codebase filters to the stratum it is
defined on: exact_match/answer_recall to the answerable items,
refusal_accuracy/refusal_quality to the refusal items (see
compute_extra_aggregates), citation_metrics skips refusal items outright
with `method="skipped"`, and the three reasoning dimensions are defined on
the answerable stratum. RAGAS is the one family that does not, and it is
not even internally consistent about it: `answer_relevancy` /
`answer_correctness` are restricted to the 120 (`answer_metrics_n`) while
the context metrics are not.

So this script reports BOTH, side by side and clearly labelled:

  ragas_<metric>            — restricted to the 120 answerable items
  ragas_<metric>_all150     — as eval_runner reported it, for traceability

The restriction is applied identically to all four systems and all runs,
which is what keeps the comparison fair; it changes the level of the
context metrics, not the ranking. Refusal behaviour is not lost by it —
it is carried by refusal_accuracy, refusal_quality and over_refusal_rate,
which cover the 30 refusal items and catch the case this exclusion might
otherwise hide, a system inventing a figure for a false-premise question.

What it does NOT fix: the absolute level stays well below the RAGAS
reference values usually quoted for a healthy pipeline, because the
fixed-size chunking splits the financial-statement tables so that the
fiscal-year column headers land in a different chunk than the value rows.
That is a genuine limitation of the retrieval corpus, not an artefact of
the averaging, and it is the reason S1 declines on 90 of 120 answerable
items (RAGAS scores a non-committal answer as answer_relevancy 0).

    uv run python scripts/aggregate_runs.py run1 run2 run3
    uv run python scripts/aggregate_runs.py --runs-root data/results/final_eval run1
"""
import argparse
import csv
import json
import logging
import math
import statistics
from pathlib import Path

from scripts.generate_report import SYSTEMS, latest_result_file
from scripts.run_full_eval import compute_extra_aggregates

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "data" / "results" / "final_eval"

RAGAS_METRICS = [
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
]

# Carried straight from each run's summary dict.
SUMMARY_PASSTHROUGH = [
    "total_queries",
    "successful_queries",
    "failed_queries",
    "retried_queries",
    "cross_document_success_rate",
    "citation_accuracy",
    "correction_rate",
    "cost_per_correct_answer_usd",
    "total_cost_usd",
]

# Latency is the one metric that is not comparable across all three runs.
# run1 executed against the `global` Vertex endpoint and spent much of its
# wall time in quota backoff, which `perf_counter` counts as latency; its
# S2 was additionally inflated by the machine suspending. Averaging that in
# would not measure the systems, so latency is reported over run2 and run3
# only. Every other metric from run1 stays in: backoff and suspend cost
# time, they do not change which answer came back.
LATENCY_METRICS = {"latency_seconds"}
DEFAULT_LATENCY_EXCLUDED_RUNS = ("run1",)

# Taken from compute_extra_aggregates, which already stratifies each of
# these correctly; reused rather than recomputed so a change to the
# per-run report cannot silently drift away from the aggregate.
EXTRA_METRICS = [
    "exact_match", "answer_recall", "refusal_accuracy", "refusal_quality",
    "over_refusal_rate", "chain_emission_rate", "groundedness", "validity",
    "completeness", "fully_grounded_rate", "fully_valid_rate",
    "locus_faithfulness", "locus_faithfulness_coverage",
    "avg_chain_steps", "avg_evidential_steps", "avg_inferential_steps",
    "untagged_steps", "total_tokens", "latency_seconds",
]


def _clean(value) -> float | None:
    """None for anything that is not a real number, NaN included.

    `_avg` in run_full_eval drops None but would propagate a NaN through
    the whole mean. RAGAS writes NaN when a judge call fails to parse, and
    although evaluate_run normalises those to None on the way out, a
    hand-patched or merged result file is not guaranteed to have gone
    through that path.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return None if math.isnan(value) or math.isinf(value) else float(value)


def _mean(values: list) -> float | None:
    vals = [v for v in (_clean(v) for v in values) if v is not None]
    return sum(vals) / len(vals) if vals else None


def _sd(values: list) -> float | None:
    """Sample SD across runs; None below two real values.

    ddof=1 because the three repetitions are a sample of the run-to-run
    variation, not the population of all possible runs.
    """
    vals = [v for v in (_clean(v) for v in values) if v is not None]
    return statistics.stdev(vals) if len(vals) > 1 else None


def _round(value, digits: int = 4):
    return None if value is None else round(value, digits)


def ragas_answerable_only(detailed_results: list) -> dict:
    """Re-average the five RAGAS metrics over the answerable stratum only.

    Returns the five means, their equal-weighted composite and the number
    of items each was actually averaged over. `n_scored` is per metric
    because the metrics do not all survive on the same items: a judge that
    fails to classify leaves a None behind, and context_recall in
    particular is frequently missing.
    """
    answerable = [d for d in detailed_results if d.get("expected_answerable")]
    means, n_scored = {}, {}
    for metric in RAGAS_METRICS:
        values = [
            _clean((d.get("ragas_metrics") or {}).get(metric)) for d in answerable
        ]
        present = [v for v in values if v is not None]
        means[metric] = sum(present) / len(present) if present else None
        n_scored[metric] = len(present)

    computed = [v for v in means.values() if v is not None]
    # Comparable across ALL FOUR systems, because these are the only two
    # RAGAS metrics every system has. S3 and S4 retrieve no chunks, so
    # context_precision/context_recall/faithfulness are absent for them
    # entirely and their `composite_score` is a mean of two metrics while
    # S1's and S2's is a mean of five. Reading those two numbers off the
    # same row would rank a system higher for having fewer metrics — the
    # trap EvalScores.composite_score's docstring warns about. This is the
    # column to use for a cross-system statement.
    answer_only = [
        v for v in (means["answer_relevancy"], means["answer_correctness"])
        if v is not None
    ]
    return {
        **means,
        # Equal-weighted over whichever metrics survived, matching
        # EvalScores.composite_score so the two are read the same way.
        "composite_score": sum(computed) / len(computed) if computed else None,
        "composite_n_metrics": len(computed),
        "composite_answer_only": (
            sum(answer_only) / len(answer_only) if answer_only else None
        ),
        "n_answerable": len(answerable),
        "n_scored": n_scored,
    }


def collect_run(run_dir: Path) -> dict:
    """One row per system for a single run directory."""
    rows = {}
    for label, system_name in SYSTEMS:
        try:
            path = latest_result_file(run_dir, system_name)
        except FileNotFoundError:
            logger.warning(
                "%s: no result file for %s — system skipped for this run.",
                run_dir.name, system_name,
            )
            continue
        result = json.loads(path.read_text())
        summary = result.get("summary", {})
        detailed = result.get("detailed_results", [])
        reported = summary.get("ragas_summary", {})
        restricted = ragas_answerable_only(detailed)
        extra = compute_extra_aggregates(detailed)

        row = {
            "run_id": run_dir.name,
            "system": label,
            "source_file": path.name,
            "n_items": len(detailed),
            "n_answerable": restricted["n_answerable"],
        }
        for metric in RAGAS_METRICS:
            row[f"ragas_{metric}"] = _round(restricted[metric])
            row[f"ragas_{metric}_n"] = restricted["n_scored"][metric]
            row[f"ragas_{metric}_all150"] = _round(_clean(reported.get(metric)))
        row["ragas_composite_score"] = _round(restricted["composite_score"])
        row["ragas_composite_n_metrics"] = restricted["composite_n_metrics"]
        row["ragas_composite_answer_only"] = _round(
            restricted["composite_answer_only"]
        )
        row["ragas_composite_score_all150"] = _round(
            _clean(reported.get("composite_score"))
        )
        for key in SUMMARY_PASSTHROUGH:
            row[key] = summary.get(key)
        for key in EXTRA_METRICS:
            row[key] = _round(_clean(extra.get(key)))
        rows[label] = row
    return rows


def aggregate(per_run_rows: list, latency_excluded: tuple = ()) -> list:
    """Mean and sample SD per system across the runs present.

    `latency_excluded` names runs whose latency is not a measurement of
    the system (see LATENCY_METRICS). Those runs still contribute every
    other metric, and their latency stays in aggregate_per_run.csv — only
    the latency mean and SD skip them.
    """
    numeric = [
        k for k in per_run_rows[0]
        if k not in ("run_id", "system", "source_file")
    ] if per_run_rows else []

    out = []
    for label, _ in SYSTEMS:
        rows = [r for r in per_run_rows if r["system"] == label]
        if not rows:
            continue
        lat_rows = [r for r in rows if r["run_id"] not in latency_excluded]
        agg = {
            "system": label,
            "n_runs": len(rows),
            "runs": "+".join(r["run_id"] for r in rows),
            "latency_runs": "+".join(r["run_id"] for r in lat_rows) or "—",
            "n_latency_runs": len(lat_rows),
        }
        for key in numeric:
            source = lat_rows if key in LATENCY_METRICS else rows
            values = [r.get(key) for r in source]
            agg[f"{key}_mean"] = _round(_mean(values))
            agg[f"{key}_sd"] = _round(_sd(values))
        out.append(agg)
    return out


def _write_csv(rows: list, out_path: Path) -> None:
    if not rows:
        logger.warning("Nothing to write to %s", out_path.name)
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %s (%d rows)", out_path.name, len(rows))


def _fmt(mean, sd, digits: int = 4) -> str:
    if mean is None:
        return "—"
    if sd is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


# Counts, not scores: four decimals on "5" reads as a measurement.
COUNT_METRICS = {"ragas_composite_n_metrics", "failed_queries", "untagged_steps"}


def write_markdown(
    agg_rows: list, per_run_rows: list, out_path: Path,
    latency_excluded: tuple = (),
) -> None:
    runs = sorted({r["run_id"] for r in per_run_rows})
    dropped = sorted(set(latency_excluded) & set(runs))
    lines = [
        "# Aggregated evaluation across runs",
        "",
        f"Runs: {', '.join(runs)} (n={len(runs)}). "
        "Cells are mean ± sample SD across runs; a single run has no SD.",
        "",
        "RAGAS metrics are averaged over the **120 answerable items**. The "
        "`_all150` columns in the CSVs carry the values as eval_runner "
        "reported them, over all 150 items including the 30 FA-Refusal "
        "items, where a 0 is the correct outcome rather than a defect. "
        "Refusal behaviour is reported separately by refusal_accuracy, "
        "refusal_quality and over_refusal_rate.",
        "",
    ]
    if dropped:
        lines += [
            f"**avg_latency_seconds excludes {', '.join(dropped)}.** Those "
            "runs executed against the `global` Vertex endpoint and spent "
            "much of their wall time in quota backoff, which `perf_counter` "
            "counts as latency; run1's S2 was additionally inflated by the "
            "machine suspending. Every other metric in this report uses all "
            f"{len(runs)} run(s) — see the `latency_runs` column in "
            "aggregate_summary.csv for which runs each latency figure "
            "actually covers.",
            "",
        ]

    # S3 and S4 retrieve no chunks, so they have no context metrics at all.
    # Spelling out the metric count next to the composite keeps the two
    # kinds of composite from being read off the same row as if they were
    # the same number.
    mixed_composite = len({
        r.get("ragas_composite_n_metrics_mean") for r in agg_rows
    }) > 1
    if mixed_composite:
        lines += [
            "> **The `composite` row is not comparable across systems.** S3 "
            "and S4 perform no chunk retrieval, so context_precision, "
            "context_recall and faithfulness do not exist for them and "
            "their composite averages 2 metrics where S1's and S2's "
            "averages 5. Fewer metrics is not a better system. For a "
            "cross-system statement use `composite (answer-only)`, which "
            "is the mean of answer_relevancy and answer_correctness for "
            "all four, or compare the individual metrics.",
            "",
        ]

    blocks = [
        ("RAGAS (answerable stratum, n=120)",
         [(f"ragas_{m}", m) for m in RAGAS_METRICS] +
         [("ragas_composite_answer_only", "composite (answer-only, comparable)"),
          ("ragas_composite_score", "composite (all available metrics)"),
          ("ragas_composite_n_metrics", "└ metrics averaged")]),
        ("Faithfulness (answer vs. corpus at cited loci, all four systems; "
         "post hoc, exploratory)",
         [("locus_faithfulness", "locus_faithfulness"),
          ("locus_faithfulness_coverage", "└ coverage (share of answerable items scored)")]),
        ("Answer and refusal quality",
         [("exact_match", "exact_match"), ("answer_recall", "answer_recall"),
          ("refusal_accuracy", "refusal_accuracy"),
          ("refusal_quality", "refusal_quality"),
          ("over_refusal_rate", "over_refusal_rate")]),
        ("Reasoning chains",
         [("chain_emission_rate", "chain_emission_rate"),
          ("groundedness", "groundedness"), ("validity", "validity"),
          ("completeness", "completeness"),
          ("fully_grounded_rate", "fully_grounded_rate"),
          ("fully_valid_rate", "fully_valid_rate"),
          ("avg_chain_steps", "avg_chain_steps")]),
        ("Task and cost",
         [("cross_document_success_rate", "cross_document_success_rate"),
          ("citation_accuracy", "citation_accuracy"),
          ("correction_rate", "correction_rate"),
          ("failed_queries", "failed_queries"),
          ("total_tokens", "avg_total_tokens"),
          ("latency_seconds",
           "avg_latency_seconds" + (f" (ohne {', '.join(dropped)})" if dropped else "")),
          ("total_cost_usd", "total_cost_usd")]),
    ]

    labels = [r["system"] for r in agg_rows]
    for title, metrics in blocks:
        lines += [f"## {title}", "",
                  "| metric | " + " | ".join(labels) + " |",
                  "|---" * (len(labels) + 1) + "|"]
        for key, display in metrics:
            digits = 1 if key in COUNT_METRICS else 4
            cells = [
                _fmt(r.get(f"{key}_mean"), r.get(f"{key}_sd"), digits)
                for r in agg_rows
            ]
            lines.append(f"| {display} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## Per-run detail", "",
              "| run | system | source file | items | answerable | "
              "ragas composite (120) | ragas composite (150) | failed |",
              "|---|---|---|---|---|---|---|---|"]
    for r in per_run_rows:
        lines.append(
            f"| {r['run_id']} | {r['system']} | {r['source_file']} | "
            f"{r['n_items']} | {r['n_answerable']} | "
            f"{r.get('ragas_composite_score')} | "
            f"{r.get('ragas_composite_score_all150')} | "
            f"{r.get('failed_queries')} |"
        )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", out_path.name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "run_ids", nargs="*", default=["run1", "run2", "run3"], metavar="RUN",
        help="Run directory names under --runs-root (default: run1 run2 "
             "run3). A run that does not exist yet is skipped with a "
             "warning, so this is usable before all three have finished — "
             "the aggregate then simply reports fewer runs and no SD.",
    )
    parser.add_argument(
        "--runs-root", default=str(DEFAULT_RUNS_ROOT),
        help="Directory holding the run directories.",
    )
    parser.add_argument(
        "--latency-exclude", nargs="*", default=list(
            DEFAULT_LATENCY_EXCLUDED_RUNS
        ), metavar="RUN",
        help="Runs whose latency is not a measurement of the system and is "
             "therefore left out of the latency mean and SD (default: "
             f"{' '.join(DEFAULT_LATENCY_EXCLUDED_RUNS)}). They still "
             "contribute every other metric. Pass with no value to average "
             "latency over all runs.",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Where the three aggregate files go (default: "
             "<runs-root>/aggregate). Deliberately not a run directory: "
             "the per-run results stay untouched.",
    )
    args = parser.parse_args(argv)

    runs_root = Path(args.runs_root)
    if not runs_root.is_absolute():
        runs_root = PROJECT_ROOT / runs_root
    out_dir = Path(args.out_dir) if args.out_dir else runs_root / "aggregate"
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    per_run_rows = []
    for run_id in args.run_ids:
        run_dir = runs_root / run_id
        if not run_dir.is_dir():
            logger.warning("Run directory missing, skipped: %s", run_dir)
            continue
        rows = collect_run(run_dir)
        logger.info("%s: %d system(s)", run_id, len(rows))
        per_run_rows.extend(rows[label] for label, _ in SYSTEMS if label in rows)

    if not per_run_rows:
        raise SystemExit(
            f"No run produced results under {runs_root}. Nothing aggregated."
        )

    found = sorted({r["run_id"] for r in per_run_rows})
    if len(found) < 3:
        logger.warning(
            "Aggregating %d of 3 planned runs (%s). The thesis figures need "
            "all three; treat this as provisional.",
            len(found), ", ".join(found),
        )

    # A system missing from some runs would be averaged over a different
    # set of runs than its neighbours, which is not a comparison.
    counts = {label: sum(1 for r in per_run_rows if r["system"] == label)
              for label, _ in SYSTEMS}
    present = {l: c for l, c in counts.items() if c}
    if len(set(present.values())) > 1:
        logger.warning(
            "Systems cover unequal numbers of runs: %s. The per-system means "
            "are then not over the same runs and must not be compared "
            "directly.",
            ", ".join(f"{l}={c}" for l, c in present.items()),
        )

    latency_excluded = tuple(args.latency_exclude)
    agg_rows = aggregate(per_run_rows, latency_excluded)

    dropped = sorted(set(latency_excluded) & set(found))
    if dropped:
        logger.info(
            "Latency averaged WITHOUT %s; every other metric uses all %d "
            "run(s).", ", ".join(dropped), len(found),
        )
    if any(r["n_latency_runs"] == 0 for r in agg_rows):
        logger.warning(
            "Every run present is excluded from latency — the latency "
            "columns are empty. Pass --latency-exclude with no value to "
            "average over all runs."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(per_run_rows, out_dir / "aggregate_per_run.csv")
    _write_csv(agg_rows, out_dir / "aggregate_summary.csv")
    write_markdown(
        agg_rows, per_run_rows, out_dir / "aggregate_report.md", latency_excluded
    )
    logger.info("Aggregate of %s written to %s", ", ".join(found), out_dir)


if __name__ == "__main__":
    main()
