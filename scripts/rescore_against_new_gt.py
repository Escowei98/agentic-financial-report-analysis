"""
Rescore the full_eval_n150 run's saved system answers against the corrected
gold standard (2026-08-20 audit), without re-querying any of the 4 systems.

Only the 25 `gt_value`-only-changed rows are rescored. 4 rows (27, 68, 69, 136)
also had their *question text* changed by the audit's fixes; the saved answer
was never generated for the new wording, so rescoring those against it would
not be meaningful -- they are carried through unchanged and flagged as skipped.

Never writes into any existing file under data/results/full_eval_n150/: only
reads eval_<system>_<timestamp>.json and writes two new, additively-named CSVs
so the original full_eval_report.md / full_eval_summary.csv /
full_eval_per_query.csv stay intact for before/after comparison.

Usage:
    uv run python scripts/rescore_against_new_gt.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.evaluation.custom_evaluator import evaluate_custom_metrics
from src.evaluation.gold_standard_loader import load_gold_standard
from src.evaluation.process_evaluator import calculate_cost_per_correct_answer

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "full_eval_n150"
GS_EN_CSV = PROJECT_ROOT / "data" / "gold_standard" / "gold_standard_v3_en.csv"

DIFF_OUT = RESULTS_DIR / "rescore_diff_per_query.csv"
SUMMARY_OUT = RESULTS_DIR / "rescore_summary.csv"

# ids whose gt_value changed but the question text did NOT -- these are
# cleanly rescorable against the saved answer.
RESCORABLE_IDS = {
    16, 17, 23, 28, 29, 32, 33, 36, 37, 45, 51, 55, 57, 60,
    61, 64, 67, 71, 72, 101, 104, 105, 109, 117, 118,
}
# ids whose question text also changed (Azure -> Intelligent Cloud rewording;
# MSFT acquisition threshold reworded $50bn -> $80bn). The saved answer
# answers a question that no longer exists in the gold standard, so these
# are excluded from rescoring and reported separately.
SKIPPED_IDS = {27, 68, 69, 136}

EVAL_FILES = {
    "rag_monolith": "eval_rag_monolith_20260815_184733.json",
    "rag_agent": "eval_rag_agent_20260815_192659.json",
    "multi_agent": "eval_multi_agent_20260815_215709.json",
    # eval_long_context_20260816_113846.json is only a 10-row partial rerun
    # (see scripts/rerun_s3.py / merge_s3_results.py); the MERGED file is the
    # complete 150-row long_context run and the one full_eval_report.md was
    # actually built from.
    "long_context": "eval_long_context_MERGED.json",
}


def main() -> None:
    gold_items = {item.id: item for item in load_gold_standard(GS_EN_CSV)}

    diff_rows: list[dict] = []
    summary_rows: list[dict] = []
    rescored_count = 0
    skipped_seen: set[int] = set()

    for system_name, filename in EVAL_FILES.items():
        path = RESULTS_DIR / filename
        data = json.loads(path.read_text(encoding="utf-8"))
        detailed = data["detailed_results"]
        old_summary = data["summary"]

        correctness_scores = []
        answerable_citation_scores = []
        fa4_flags = []

        for row in detailed:
            qid = row["query_id"]
            answer = row["answer"]
            old_custom = row["custom_metrics"]
            # citation_metrics is left untouched for every row: none of the
            # 29 corrected ids changed doc_ids/source_sections/
            # expected_answerable (verified against the pre-audit CSV), the
            # only inputs evaluate_citation_accuracy() depends on -- so
            # recomputing it would spend LLM-judge calls to reproduce a
            # number that's provably identical to what's already saved.
            citation = row["citation_metrics"]

            if qid in RESCORABLE_IDS:
                item = gold_items[qid]
                new_custom = evaluate_custom_metrics(item, answer).to_dict()

                old_correct = old_custom["exact_match"] >= 0.8 or old_custom["answer_recall"] >= 0.8
                new_correct = new_custom["exact_match"] >= 0.8 or new_custom["answer_recall"] >= 0.8

                diff_rows.append({
                    "system": system_name,
                    "query_id": qid,
                    "question": row["question"],
                    "old_ground_truth": row["ground_truth"],
                    "new_ground_truth": item.ground_truth,
                    "answer": answer,
                    "old_exact_match": old_custom["exact_match"],
                    "new_exact_match": new_custom["exact_match"],
                    "old_answer_recall": old_custom["answer_recall"],
                    "new_answer_recall": new_custom["answer_recall"],
                    "flipped": old_correct != new_correct,
                })
                rescored_count += 1
                effective_custom = new_custom
            else:
                if qid in SKIPPED_IDS:
                    skipped_seen.add(qid)
                effective_custom = old_custom

            score = (
                effective_custom["exact_match"]
                if effective_custom["exact_match"] > 0
                else effective_custom["answer_recall"]
            )
            correctness_scores.append(score)

            if row["expected_answerable"]:
                answerable_citation_scores.append(citation["citation_accuracy"])

            if row["fa_type"] == "FA-4":
                fa4_flags.append(
                    effective_custom["exact_match"] == 1.0
                    or effective_custom["answer_recall"] >= 0.5
                )

        cross_doc = sum(fa4_flags) / len(fa4_flags) if fa4_flags else 0.0
        avg_citation = (
            sum(answerable_citation_scores) / len(answerable_citation_scores)
            if answerable_citation_scores else 0.0
        )
        cost_per_correct = calculate_cost_per_correct_answer(
            total_cost_usd=old_summary["total_cost_usd"],
            correctness_scores=correctness_scores,
            threshold=0.8,
        )

        summary_rows.append({
            "system": system_name,
            "old_cross_document_success_rate": old_summary["cross_document_success_rate"],
            "new_cross_document_success_rate": round(cross_doc, 4),
            "old_citation_accuracy": old_summary["citation_accuracy"],
            "new_citation_accuracy": round(avg_citation, 4),
            "old_cost_per_correct_answer_usd": old_summary["cost_per_correct_answer_usd"],
            "new_cost_per_correct_answer_usd": round(cost_per_correct, 4),
            "total_cost_usd": old_summary["total_cost_usd"],
        })

        print(f"[{system_name}] rescored {sum(1 for r in diff_rows if r['system'] == system_name)} rows")

    with open(DIFF_OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(diff_rows[0].keys()))
        writer.writeheader()
        writer.writerows(diff_rows)

    with open(SUMMARY_OUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    flipped = [r for r in diff_rows if r["flipped"]]
    print()
    print(f"Rescored {rescored_count} (system, query_id) pairs across 4 systems "
          f"({len(RESCORABLE_IDS)} ids x 4 systems).")
    print(f"Skipped ids (question text changed, not rescorable): {sorted(SKIPPED_IDS)}")
    print(f"Score flipped (crossed the 0.8 correctness threshold): {len(flipped)}")
    for r in flipped:
        print(f"  [{r['system']}] id {r['query_id']}: "
              f"old exact_match={r['old_exact_match']} answer_recall={r['old_answer_recall']} -> "
              f"new exact_match={r['new_exact_match']} answer_recall={r['new_answer_recall']}")
    print()
    print(f"Wrote {DIFF_OUT.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {SUMMARY_OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
