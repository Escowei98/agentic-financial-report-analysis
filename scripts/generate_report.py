"""
Regenerates full_eval_summary.csv, full_eval_per_query.csv and
full_eval_report.md from the eval_*.json files already sitting in
data/results/full_eval_n150/, without re-running the evaluation.

Useful whenever a result file was patched/merged by hand (e.g. after a
targeted retry, see scripts/merge_s3_results.py) and the reports need to
catch up.
"""
import glob
import json
import logging
from pathlib import Path

import scripts.run_full_eval as full_eval_script
from src.evaluation.gold_standard_loader import GOLD_STANDARD_EN, load_gold_standard

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "results" / "full_eval_n150"

# (result label, system_name as used in eval_{system_name}_*.json filenames)
SYSTEMS = [
    ("S1", "rag_monolith"),
    ("S2", "rag_agent"),
    ("S3", "long_context"),
    ("S4", "multi_agent"),
]


def latest_result_file(system_name: str) -> Path:
    # Prefer a hand-merged file (e.g. eval_long_context_MERGED.json) over
    # the raw timestamped runs, since a merge means the original file was
    # incomplete (see scripts/merge_s3_results.py).
    merged = OUT_DIR / f"eval_{system_name}_MERGED.json"
    if merged.exists():
        return merged
    candidates = sorted(glob.glob(str(OUT_DIR / f"eval_{system_name}_*.json")))
    if not candidates:
        raise FileNotFoundError(f"No result file found for system '{system_name}' in {OUT_DIR}")
    return Path(candidates[-1])


def main():
    results = {}
    for label, system_name in SYSTEMS:
        path = latest_result_file(system_name)
        logger.info("%s (%s) <- %s", label, system_name, path.name)
        results[label] = json.loads(path.read_text())

    gold_items = load_gold_standard(GOLD_STANDARD_EN)

    full_eval_script.write_summary_csv(results, OUT_DIR / "full_eval_summary.csv")
    full_eval_script.write_per_query_csv(results, gold_items, OUT_DIR / "full_eval_per_query.csv")
    full_eval_script.write_markdown_report(results, OUT_DIR / "full_eval_report.md")


if __name__ == "__main__":
    main()
