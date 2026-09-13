"""
Runs a SUBSET of the four systems into an existing run directory.

`run_full_eval.py` always evaluates all four systems of a run. When one of
them has to be produced separately -- a system that was interrupted, or one
held back until quota recovered -- this script writes the missing
`eval_<system>_*.json` into the same run directory, so that
`generate_report.py <run_dir>` can assemble the full report from the four
files afterwards.

Everything that decides how a system is measured is taken from
run_full_eval.py itself (SYSTEM_SPECS, the pipeline construction, the shared
EvidenceStore, the gold standard). This file only chooses WHICH of them run;
it must never grow its own copy of that setup, or a system produced here
would stop being comparable to one produced by the main script.

    uv run python scripts/run_systems_into_run.py \
        data/results/final_eval/run1 --systems multi_agent
"""
import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_full_eval as full_eval
from src.common.config import load_config
from src.common.ingestion import download_all_filings
from src.evaluation.eval_runner import run_full_evaluation
from src.evaluation.evidence_store import EvidenceStore
from src.evaluation.gold_standard_loader import GOLD_STANDARD_EN, load_gold_standard

logger = logging.getLogger(__name__)

VALID = [name for _, name, _ in full_eval.SYSTEM_SPECS]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path,
                    help="Existing run directory, e.g. data/results/final_eval/run1")
    ap.add_argument("--systems", nargs="+", required=True, metavar="NAME",
                    help=f"System names to run. One or more of: {', '.join(VALID)}")
    ap.add_argument("--max-attempts", type=int,
                    default=full_eval.QUERY_MAX_ATTEMPTS, metavar="N",
                    help="Attempts per query before it counts as failed "
                         f"(default {full_eval.QUERY_MAX_ATTEMPTS}).")
    ap.add_argument("--force", action="store_true",
                    help="Run even if a result file for that system already "
                         "exists in the run directory.")
    args = ap.parse_args(argv)

    unknown = [s for s in args.systems if s not in VALID]
    if unknown:
        raise SystemExit(f"Unknown system(s): {unknown}. Valid: {VALID}")
    if not args.run_dir.is_dir():
        raise SystemExit(f"Run directory does not exist: {args.run_dir}")

    # Refuse to silently produce a second result file for a system that
    # already has one: generate_report.py would then pick by filename order,
    # not by which run is the real one.
    for name in args.systems:
        existing = list(args.run_dir.glob(f"eval_{name}_*.json"))
        if existing and not args.force:
            raise SystemExit(
                f"{args.run_dir} already holds a result for '{name}' "
                f"({existing[0].name}). Pass --force to add another.")
    return args


def main(argv=None):
    args = parse_args(argv)

    filings = download_all_filings()
    gold_items = load_gold_standard(GOLD_STANDARD_EN)
    evidence_store = EvidenceStore(filings)

    wanted = set(args.systems)
    # Built one at a time: building all four would spend minutes of ingestion
    # on systems this invocation is not going to run.
    specs = [(l, n, c) for l, n, c in full_eval.SYSTEM_SPECS if n in wanted]

    with full_eval._run_log_file(args.run_dir):
        logger.info("#" * 70)
        logger.info("PARTIAL RUN into %s — systems: %s",
                    args.run_dir, ", ".join(args.systems))
        logger.info("#" * 70)

        for label, name, cls in specs:
            started = time.time()
            logger.info("=" * 70)
            logger.info("Running full eval for %s (%s) — %d items",
                        label, name, len(gold_items))
            logger.info("=" * 70)
            pipeline = cls(load_config(name))
            pipeline.build(filings)
            res = run_full_evaluation(
                pipeline, name, gold_items, args.run_dir,
                evidence_store=evidence_store, max_attempts=args.max_attempts,
            )
            logger.info("%s summary: %s", label, res.get("summary", {}))
            logger.info("%s done in %.1f min", label, (time.time() - started) / 60)

    print(f"\nDone. Now rebuild the report:\n"
          f"  uv run python scripts/generate_report.py {args.run_dir}")


if __name__ == "__main__":
    main()
