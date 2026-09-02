"""
Merges the original 140-item S3 (long_context) result with the retry
batch covering the 10 previously-failed ids, into one complete n=150
result, then regenerates the combined full_eval_summary.csv /
full_eval_per_query.csv for all four systems.

Summary aggregates (cross_document_success_rate, citation_accuracy,
correction_rate, cost_per_correct_answer_usd) are recomputed directly
from the merged 150 detailed_results using the exact same logic as
src/evaluation/eval_runner.py, rather than naively averaging the two
partial summaries — several of those fields have denominators that are
NOT the raw item count (e.g. citation_accuracy only averages over
expected_answerable=True items, cross_document_success_rate only over
FA-4 items), so a naive count-weighted average of two summaries would
be wrong if the answerable/FA-type mix differs between the two batches.

The only field combined via weighted averaging of the two summaries is
the RAGAS block, since no per-item RAGAS scores are persisted — but
RAGAS's own aggregate is a plain arithmetic mean over ALL successfully-
run items (unfiltered), so a count-weighted average across the two
disjoint batches is mathematically identical to re-running RAGAS over
the union.
"""
import csv
import glob
import json
import logging
from pathlib import Path

from src.evaluation.process_evaluator import calculate_cost_per_correct_answer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "results" / "full_eval_n150"

ORIGINAL_S3 = OUT_DIR / "eval_long_context_20260815_203610.json"


def find_retry_file() -> Path:
    candidates = sorted(
        p for p in glob.glob(str(OUT_DIR / "eval_long_context_*.json"))
        if Path(p) != ORIGINAL_S3
    )
    if not candidates:
        raise FileNotFoundError("No S3 retry JSON found (expected a second eval_long_context_*.json)")
    return Path(candidates[-1])


def weighted_ragas(old_summary: dict, old_n: int, new_summary: dict, new_n: int) -> dict:
    old_r = old_summary["ragas_summary"]
    new_r = new_summary["ragas_summary"]
    total_n = old_n + new_n
    keys = ["context_precision", "context_recall", "faithfulness", "answer_relevancy", "answer_correctness"]
    merged = {
        k: round((old_r[k] * old_n + new_r[k] * new_n) / total_n, 4)
        for k in keys
    }
    merged["composite_score"] = round(sum(merged.values()) / len(keys), 4)
    return merged


def recompute_summary(detailed_results: list, total_cost_usd: float, old_summary: dict, new_summary: dict, old_n: int, new_n: int) -> dict:
    fa4_items = [d for d in detailed_results if d["fa_type"] == "FA-4"]
    cross_doc_success = 0.0
    if fa4_items:
        successes = sum(
            1 for d in fa4_items
            if d["custom_metrics"]["exact_match"] == 1.0 or d["custom_metrics"]["answer_recall"] >= 0.5
        )
        cross_doc_success = successes / len(fa4_items)

    items_with_run_metrics = [d for d in detailed_results if "run_metrics" in d]
    correction_rate = 0.0
    if items_with_run_metrics:
        correction_rate = sum(
            d["run_metrics"]["corrections"] for d in items_with_run_metrics
        ) / len(items_with_run_metrics)

    answerable_citation_scores = [
        d["citation_metrics"]["citation_accuracy"]
        for d in detailed_results
        if d["expected_answerable"]
    ]
    avg_citation_accuracy = (
        sum(answerable_citation_scores) / len(answerable_citation_scores)
        if answerable_citation_scores else 0.0
    )

    correctness_scores = [
        d["custom_metrics"]["exact_match"] if d["custom_metrics"]["exact_match"] > 0
        else d["custom_metrics"]["answer_recall"]
        for d in detailed_results
    ]
    cost_per_correct = calculate_cost_per_correct_answer(
        total_cost_usd=total_cost_usd,
        correctness_scores=correctness_scores,
        threshold=0.8,
    )

    return {
        "system_name": "long_context",
        "total_queries": 150,
        "successful_queries": len(detailed_results),
        "ragas_summary": weighted_ragas(old_summary, old_n, new_summary, new_n),
        "cross_document_success_rate": round(cross_doc_success, 4),
        "citation_accuracy": round(avg_citation_accuracy, 4),
        "correction_rate": round(correction_rate, 4),
        "cost_per_correct_answer_usd": round(cost_per_correct, 4),
        "total_cost_usd": round(total_cost_usd, 4),
    }


def main():
    retry_path = find_retry_file()
    logger.info("Original S3 file: %s", ORIGINAL_S3.name)
    logger.info("Retry S3 file: %s", retry_path.name)

    old = json.loads(ORIGINAL_S3.read_text())
    new = json.loads(retry_path.read_text())

    old_detailed = old["detailed_results"]
    new_detailed = new["detailed_results"]
    old_n = len(old_detailed)
    new_n = len(new_detailed)

    old_ids = {d["query_id"] for d in old_detailed}
    new_ids = {d["query_id"] for d in new_detailed}
    overlap = old_ids & new_ids
    if overlap:
        raise ValueError(f"Unexpected overlap between original and retry ids: {overlap}")

    merged_detailed = sorted(old_detailed + new_detailed, key=lambda d: d["query_id"])
    logger.info(
        "Merged %d (original) + %d (retry) = %d detailed results (expected 150)",
        old_n, new_n, len(merged_detailed),
    )
    still_missing = set(range(1, 151)) - {d["query_id"] for d in merged_detailed}
    if still_missing:
        logger.warning("Still missing ids after retry: %s", sorted(still_missing))

    total_cost_usd = old["summary"]["total_cost_usd"] + new["summary"]["total_cost_usd"]
    merged_summary = recompute_summary(
        merged_detailed, total_cost_usd, old["summary"], new["summary"], old_n, new_n,
    )
    logger.info("Merged S3 summary: %s", merged_summary)

    merged_out = {"summary": merged_summary, "detailed_results": merged_detailed}
    merged_path = OUT_DIR / "eval_long_context_MERGED.json"
    merged_path.write_text(json.dumps(merged_out, indent=2))
    logger.info("Wrote merged S3 result to %s", merged_path)

    # --- Regenerate the combined summary + per-query CSVs for all 4 systems ---
    results = {
        "S1": json.loads(sorted(glob.glob(str(OUT_DIR / "eval_rag_monolith_*.json")))[-1] and Path(sorted(glob.glob(str(OUT_DIR / "eval_rag_monolith_*.json")))[-1]).read_text()),
        "S2": json.loads(Path(sorted(glob.glob(str(OUT_DIR / "eval_rag_agent_*.json")))[-1]).read_text()),
        "S3": merged_out,
        "S4": json.loads(Path(sorted(glob.glob(str(OUT_DIR / "eval_multi_agent_*.json")))[-1]).read_text()),
    }

    import scripts.run_full_eval as full_eval_script
    from src.evaluation.gold_standard_loader import load_gold_standard

    gold_items = load_gold_standard(PROJECT_ROOT / "data" / "gold_standard" / "gold_standard_v3_en.csv")
    full_eval_script.write_summary_csv(results, OUT_DIR / "full_eval_summary.csv")
    full_eval_script.write_per_query_csv(results, gold_items, OUT_DIR / "full_eval_per_query.csv")
    logger.info("Regenerated full_eval_summary.csv and full_eval_per_query.csv with merged S3 data")


if __name__ == "__main__":
    main()
