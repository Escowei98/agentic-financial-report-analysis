"""
Rescore the full n=150 x 4-system evaluation run's saved answers with the
type-conditioned exact_match logic (numeric_atomic / comparison / qualitative,
see src/evaluation/custom_evaluator.py::_classify_answer_type).

Only the "comparison" (FA-4, 30 ids) and "qualitative" (gt_unit == "text",
7 ids) items are actually re-scored -- the "numeric_atomic" path (the other
~113 ids) is byte-identical to the pre-redesign code, so its saved
custom_metrics are carried through unchanged rather than spending judge
calls to reproduce the same result. No system is re-queried: answers are
read from the already-saved eval_<system>_*.json files.

Writes new, additively-named files only -- the original full_eval_summary.csv
/ full_eval_per_query.csv / full_eval_report.md and the raw eval_*.json files
are left untouched, so the pre-redesign numbers stay available for
before/after comparison.

Usage:
    uv run python scripts/rescore_typed_exact_match.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import scripts.run_full_eval as full_eval_script
from src.evaluation.custom_evaluator import _classify_answer_type, evaluate_custom_metrics
from src.evaluation.gold_standard_loader import load_gold_standard
from src.evaluation.process_evaluator import calculate_cost_per_correct_answer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "full_eval_n150"
GS_EN_CSV = PROJECT_ROOT / "data" / "gold_standard" / "gold_standard_v3_en.csv"

# (result label used in reports, system_name as used in eval_<system>_*.json)
SYSTEMS = [
    ("S1", "rag_monolith"),
    ("S2", "rag_agent"),
    ("S3", "long_context"),
    ("S4", "multi_agent"),
]

EVAL_FILES = {
    "rag_monolith": "eval_rag_monolith_20260815_184733.json",
    "rag_agent": "eval_rag_agent_20260815_192659.json",
    "multi_agent": "eval_multi_agent_20260815_215709.json",
    # eval_long_context_20260816_113846.json is only a 10-row partial rerun;
    # the MERGED file is the complete 150-row long_context run.
    "long_context": "eval_long_context_MERGED.json",
}


def main() -> None:
    gold_list = load_gold_standard(GS_EN_CSV)
    gold_by_id = {item.id: item for item in gold_list}

    results: dict[str, dict] = {}
    flips: list[dict] = []
    rescored_total = 0

    for label, system_name in SYSTEMS:
        path = RESULTS_DIR / EVAL_FILES[system_name]
        data = json.loads(path.read_text(encoding="utf-8"))
        old_summary = data["summary"]
        detailed = data["detailed_results"]

        new_detailed = []
        correctness_scores = []
        fa4_flags = []
        rescored_here = 0

        for row in detailed:
            qid = row["query_id"]
            item = gold_by_id[qid]
            old_custom = row["custom_metrics"]

            if _classify_answer_type(item) == "numeric_atomic":
                new_custom = old_custom
            else:
                new_custom = evaluate_custom_metrics(item, row["answer"]).to_dict()
                rescored_here += 1
                old_correct = old_custom["exact_match"] >= 0.8 or old_custom["answer_recall"] >= 0.8
                new_correct = new_custom["exact_match"] >= 0.8 or new_custom["answer_recall"] >= 0.8
                if old_correct != new_correct:
                    flips.append({
                        "system": system_name,
                        "query_id": qid,
                        "fa_type": item.fa_type,
                        "gt_unit": item.gt_unit,
                        "old_exact_match": old_custom["exact_match"],
                        "new_exact_match": new_custom["exact_match"],
                        "old_answer_recall": old_custom["answer_recall"],
                        "new_answer_recall": new_custom["answer_recall"],
                    })

            new_row = dict(row)
            new_row["custom_metrics"] = new_custom
            new_detailed.append(new_row)

            score = new_custom["exact_match"] if new_custom["exact_match"] > 0 else new_custom["answer_recall"]
            correctness_scores.append(score)
            if item.fa_type == "FA-4":
                fa4_flags.append(new_custom["exact_match"] == 1.0 or new_custom["answer_recall"] >= 0.5)

        new_summary = dict(old_summary)
        new_summary["cross_document_success_rate"] = round(
            sum(fa4_flags) / len(fa4_flags) if fa4_flags else 0.0, 4
        )
        new_summary["cost_per_correct_answer_usd"] = round(
            calculate_cost_per_correct_answer(
                total_cost_usd=old_summary["total_cost_usd"],
                correctness_scores=correctness_scores,
                threshold=0.8,
            ),
            4,
        )

        out_json = RESULTS_DIR / f"eval_{system_name}_typed_rescore.json"
        out_json.write_text(
            json.dumps({"summary": new_summary, "detailed_results": new_detailed}, indent=2),
            encoding="utf-8",
        )
        results[label] = {"summary": new_summary, "detailed_results": new_detailed}
        rescored_total += rescored_here
        print(f"[{system_name}] rescored {rescored_here} comparison/qualitative items "
              f"(of {len(detailed)} total) -> {out_json.relative_to(PROJECT_ROOT)}")

    summary_csv = RESULTS_DIR / "typed_rescore_summary.csv"
    per_query_csv = RESULTS_DIR / "typed_rescore_per_query.csv"
    report_md = RESULTS_DIR / "typed_rescore_report.md"
    full_eval_script.write_summary_csv(results, summary_csv)
    full_eval_script.write_per_query_csv(results, gold_list, per_query_csv)
    full_eval_script.write_markdown_report(results, report_md)

    print(f"\nRescored {rescored_total} (system, query_id) pairs total "
          f"(comparison + qualitative buckets only, x4 systems).")
    print(f"Score flipped (crossed the 0.8 correctness threshold): {len(flips)}")
    for f in flips:
        print(f"  [{f['system']}] id {f['query_id']} ({f['fa_type']}, gt_unit={f['gt_unit']}): "
              f"exact_match {f['old_exact_match']}->{f['new_exact_match']}, "
              f"answer_recall {f['old_answer_recall']}->{f['new_answer_recall']}")

    flips_csv = RESULTS_DIR / "typed_rescore_flips.csv"
    if flips:
        with open(flips_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flips[0].keys()))
            writer.writeheader()
            writer.writerows(flips)
        print(f"\nWrote {flips_csv.relative_to(PROJECT_ROOT)}")

    print(f"Wrote {summary_csv.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {per_query_csv.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {report_md.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
