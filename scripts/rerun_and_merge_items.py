"""
Re-runs SELECTED query ids for one system and merges them into that system's
existing result file in a run directory.

For a system that lost queries to quota exhaustion or to a defect fixed
afterwards, re-running the whole system would cost the same again and
change every intact measurement for no reason; re-running the affected ids
and merging keeps the rest as it was measured.

The merged system is then measured on a LATER CODE STATE than the other
systems of the same run. That breaks within-run code identity and is
recorded in `summary.merged_from`, to be declared wherever the run is
reported.

Everything about HOW a system is measured comes from run_full_eval.py, as in
run_systems_into_run.py. This file only chooses WHICH ids run again and how
the two result sets are stitched back together.

    # verify the summary recomputation reproduces the stored one, no API calls
    uv run python scripts/rerun_and_merge_items.py data/results/final_eval/run1 \
        --system long_context --verify-only

    # re-run the failed and the empty-answer ids and merge
    uv run python scripts/rerun_and_merge_items.py data/results/final_eval/run1 \
        --system long_context --failed --empty
"""
import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_full_eval as full_eval
from src.common.config import load_config
from src.common.ingestion import download_all_filings
from src.evaluation.eval_runner import _has_numeric_ground_truth, run_full_evaluation
from src.evaluation.evidence_store import EvidenceStore
from src.evaluation.gold_standard_loader import GOLD_STANDARD_EN, load_gold_standard
from src.evaluation.process_evaluator import calculate_cost_per_correct_answer

SPEC_BY_NAME = {name: (label, cls) for label, name, cls in full_eval.SYSTEM_SPECS}

logger = logging.getLogger(__name__)

# RAGAS metrics scored against the answer rather than the retrieved context.
# Averaged over answerable items only, mirroring ragas_evaluator.
ANSWER_METRICS = ("answer_relevancy", "answer_correctness")
CONTEXT_METRICS = ("context_precision", "context_recall", "faithfulness")


def recompute_summary(rows: list[dict], gold_by_id: dict, previous: dict) -> dict:
    """
    Rebuild a system summary from its per-item results.

    Every aggregate the runner stores is a plain function of the per-item
    values, so it can be rebuilt after a merge instead of being re-measured.
    `--verify-only` checks that claim against the stored summary before this
    is ever used to overwrite one.

    `previous` supplies only the fields that are not derivable from the rows:
    the attempt policy and the ids that never produced a row at all.
    """
    def mean(values):
        values = [v for v in values if isinstance(v, (int, float))]
        return sum(values) / len(values) if values else None

    answerable = [r for r in rows if r["expected_answerable"]]

    ragas = {}
    for key in CONTEXT_METRICS:
        ragas[key] = mean(r["ragas_metrics"].get(key) for r in rows)
    for key in ANSWER_METRICS:
        ragas[key] = mean(r["ragas_metrics"].get(key) for r in answerable)
    available = [ragas[k] for k in CONTEXT_METRICS + ANSWER_METRICS
                 if ragas[k] is not None]
    ragas["composite_score"] = sum(available) / len(available) if available else None
    ragas["answer_metrics_n"] = len(answerable)
    ragas = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ragas.items()}

    cross_doc_items = [
        r for r in rows
        if r["fa_type"] == "FA-4" or r.get("window_class") == "cross_window"
    ]
    cross_doc = (
        sum(1 for r in cross_doc_items if r["custom_metrics"]["exact_match"] == 1.0)
        / len(cross_doc_items)
    ) if cross_doc_items else 0.0

    citation = mean(r["citation_metrics"]["citation_accuracy"] for r in answerable) or 0.0

    with_run = [r for r in rows if "run_metrics" in r]
    correction_rate = (
        sum(r["run_metrics"]["corrections"] for r in with_run) / len(with_run)
    ) if with_run else 0.0

    total_cost = sum(r["run_metrics"]["estimated_cost_usd"] for r in with_run)

    # Same stratum-dependent correctness measure the runner uses.
    correctness = []
    for r in rows:
        item = gold_by_id[r["query_id"]]
        if not item.expected_answerable:
            correctness.append(r["custom_metrics"]["refusal_accuracy"])
        elif _has_numeric_ground_truth(item):
            correctness.append(r["custom_metrics"]["exact_match"])
        else:
            correctness.append(r["custom_metrics"]["answer_recall"])
    cost_per_correct = calculate_cost_per_correct_answer(
        total_cost_usd=total_cost, correctness_scores=correctness, threshold=0.8
    )

    return {
        "system_name": previous["system_name"],
        "total_queries": previous["total_queries"],
        "successful_queries": len(rows),
        "failed_queries": previous["total_queries"] - len(rows),
        "failed_query_ids": sorted(
            set(previous["total_query_ids"]) - {r["query_id"] for r in rows}
        ) if "total_query_ids" in previous else previous["failed_query_ids"],
        "query_max_attempts": previous["query_max_attempts"],
        "retried_queries": previous["retried_queries"],
        "retried_query_ids": previous["retried_query_ids"],
        "ragas_summary": ragas,
        "cross_document_success_rate": round(cross_doc, 4),
        "citation_accuracy": round(citation, 4),
        "correction_rate": round(correction_rate, 4),
        "cost_per_correct_answer_usd": (
            round(cost_per_correct, 4) if cost_per_correct is not None else None
        ),
        "total_cost_usd": round(total_cost, 4),
    }


def verify(existing: dict, gold_by_id: dict) -> bool:
    """Recompute the stored summary from the stored rows and diff the two."""
    rebuilt = recompute_summary(existing["detailed_results"], gold_by_id,
                                existing["summary"])
    stored = existing["summary"]
    ok = True
    for key in sorted(set(stored) | set(rebuilt)):
        a, b = stored.get(key), rebuilt.get(key)
        if isinstance(a, float) and isinstance(b, float):
            same = abs(a - b) < 5e-4
        else:
            same = a == b
        if not same:
            ok = False
            print(f"  ABWEICHUNG {key}:\n    gespeichert = {a}\n    neu berechnet = {b}")
    print("  Selbstpruefung:", "identisch" if ok else "ABWEICHUNGEN, siehe oben")
    return ok


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--system", required=True, choices=sorted(SPEC_BY_NAME))
    ap.add_argument("--ids", nargs="*", type=int, default=[],
                    help="Explicit query ids to re-run.")
    ap.add_argument("--failed", action="store_true",
                    help="Add the ids listed in summary.failed_query_ids.")
    ap.add_argument("--empty", action="store_true",
                    help="Add every id whose stored answer is empty.")
    ap.add_argument("--verify-only", action="store_true",
                    help="Only check that the summary can be rebuilt. No API calls.")
    ap.add_argument("--max-attempts", type=int, default=3)
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        print(f"Laufverzeichnis fehlt: {run_dir}", file=sys.stderr)
        return 2

    matches = sorted(run_dir.glob(f"eval_{args.system}_*.json"))
    matches = [m for m in matches if "MERGED" not in m.name]
    if len(matches) != 1:
        print(f"Erwartet genau eine Datei fuer {args.system}, gefunden: "
              f"{[m.name for m in matches]}", file=sys.stderr)
        return 2
    source = matches[0]
    existing = json.loads(source.read_text(encoding="utf-8"))

    gold = load_gold_standard(GOLD_STANDARD_EN)
    gold_by_id = {g.id: g for g in gold}

    print(f"Quelle: {source.name}  ({len(existing['detailed_results'])} Ergebnisse)")
    if not verify(existing, gold_by_id):
        print("Abbruch: die Zusammenfassung laesst sich nicht reproduzieren, "
              "ein Merge waere nicht vertrauenswuerdig.", file=sys.stderr)
        return 1
    if args.verify_only:
        return 0

    stored_by_id = {r["query_id"]: r for r in existing["detailed_results"]}
    targets = set(args.ids)
    if args.failed:
        targets |= set(existing["summary"]["failed_query_ids"])
    if args.empty:
        targets |= {i for i, r in stored_by_id.items()
                    if not (r.get("answer") or "").strip()}
    targets = sorted(targets)
    if not targets:
        print("Keine Ids ausgewaehlt.", file=sys.stderr)
        return 2
    print(f"Nachzuziehen: {len(targets)} Ids -> {targets}")

    items = [gold_by_id[i] for i in targets]
    label, cls = SPEC_BY_NAME[args.system]

    started = datetime.now()
    scratch = run_dir / "_rerun_tmp"
    scratch.mkdir(exist_ok=True)

    # Same construction path as run_full_eval.py, so a re-measured item is
    # produced exactly the way the original was.
    filings = download_all_filings()
    evidence_store = EvidenceStore(filings)
    with full_eval._run_log_file(run_dir):
        logger.info("#" * 70)
        logger.info("RERUN of %d ids for %s (%s) into %s",
                    len(items), label, args.system, run_dir)
        logger.info("#" * 70)
        pipeline = cls(load_config(args.system))
        pipeline.build(filings)
        fresh = run_full_evaluation(
            pipeline=pipeline,
            system_name=args.system,
            gold_items=items,
            output_dir=scratch,
            evidence_store=evidence_store,
            max_attempts=args.max_attempts,
        )
    minutes = (datetime.now() - started).total_seconds() / 60
    print(f"Nachlauf fertig in {minutes:.1f} min: "
          f"{fresh['summary']['successful_queries']}/{len(items)} erfolgreich")

    merged_by_id = dict(stored_by_id)
    for row in fresh["detailed_results"]:
        merged_by_id[row["query_id"]] = row
    rows = [merged_by_id[i] for i in sorted(merged_by_id)]

    previous = dict(existing["summary"])
    previous["total_query_ids"] = [g.id for g in gold]
    summary = recompute_summary(rows, gold_by_id, previous)
    summary["merged_from"] = {
        "source_file": source.name,
        "rerun_ids": targets,
        "rerun_at": started.isoformat(timespec="seconds"),
        "note": ("Diese Ids wurden nachtraeglich neu gemessen. Das System "
                 "laeuft damit auf einem spaeteren Codestand als die "
                 "uebrigen Systeme desselben Laufs."),
    }

    out = run_dir / f"eval_{args.system}_MERGED.json"
    backup = run_dir / f"{source.stem}.pre-merge.json"
    if not backup.exists():
        shutil.copy2(source, backup)
    out.write_text(json.dumps({"summary": summary, "detailed_results": rows},
                              indent=2), encoding="utf-8")
    print(f"Geschrieben: {out.name}   Sicherung: {backup.name}")
    print(f"Ergebnis: {summary['successful_queries']}/{summary['total_queries']} "
          f"erfolgreich, {summary['failed_queries']} Ausfaelle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
