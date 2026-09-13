"""
Regenerates full_eval_summary.csv, full_eval_per_query.csv and
full_eval_report.md from the eval_*.json files already sitting in a run
directory, without re-running the evaluation.

Useful whenever a report column changed, or a result file was patched by
hand, and the derived files need to catch up.

The `run_id` written into both CSVs is the directory's name, which is what
run_full_eval.py put there in the first place.

    uv run python scripts/generate_report.py data/results/final_eval/run1
"""
import argparse
import glob
import json
import logging
from pathlib import Path

import scripts.run_full_eval as full_eval_script
from src.evaluation.gold_standard_loader import GOLD_STANDARD_EN, load_gold_standard

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# (result label, system_name as used in eval_{system_name}_*.json filenames)
SYSTEMS = [
    ("S1", "rag_monolith"),
    ("S2", "rag_agent"),
    ("S3", "long_context"),
    ("S4", "multi_agent"),
]


def latest_result_file(run_dir: Path, system_name: str) -> Path:
    # Prefer a hand-merged file (eval_<system>_MERGED.json) over the raw
    # timestamped ones: a merge means the original was incomplete and the
    # merged file is the authoritative version of that system's run.
    merged = run_dir / f"eval_{system_name}_MERGED.json"
    if merged.exists():
        return merged
    candidates = sorted(glob.glob(str(run_dir / f"eval_{system_name}_*.json")))
    if not candidates:
        raise FileNotFoundError(
            f"No result file found for system '{system_name}' in {run_dir}"
        )
    if len(candidates) > 1:
        # Two runs in one directory is the failure --out-dir/--run-ids exist
        # to prevent; if it happened anyway, say so rather than silently
        # reporting whichever sorted last.
        logger.warning(
            "%s has %d result files in %s — using the newest (%s). Two runs in "
            "one directory are not one run.",
            system_name, len(candidates), run_dir, Path(candidates[-1]).name,
        )
    return Path(candidates[-1])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "run_dir",
        help="A run directory written by run_full_eval.py, e.g. "
             "data/results/final_eval/run1. Its name becomes the `run_id` "
             "column of both CSVs.",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    if not run_dir.is_dir():
        raise SystemExit(f"Not a directory: {run_dir}")

    run_id = run_dir.name

    results = {}
    for label, system_name in SYSTEMS:
        path = latest_result_file(run_dir, system_name)
        logger.info("%s (%s) <- %s", label, system_name, path.name)
        results[label] = json.loads(path.read_text())

    gold_items = load_gold_standard(GOLD_STANDARD_EN)

    full_eval_script.write_summary_csv(
        results, run_dir / "full_eval_summary.csv", run_id=run_id
    )
    full_eval_script.write_per_query_csv(
        results, gold_items, run_dir / "full_eval_per_query.csv", run_id=run_id
    )
    full_eval_script.write_markdown_report(results, run_dir / "full_eval_report.md")
    logger.info("Regenerated the three derived files in %s", run_dir)


if __name__ == "__main__":
    main()
