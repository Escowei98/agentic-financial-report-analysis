"""
Add `locus_faithfulness` to result files that were produced before it existed.

The metric (src/evaluation/locus_faithfulness.py) needs nothing a run did
not already store -- question, final answer, answerable flag -- plus the
corpus, so the stored answers are scored as they are: no system is queried
again, only the judge is called. The runs stay exactly the runs they were;
what changes is that one more column exists for every item.

Each result file is updated in place. Before the first write a copy of the
untouched file goes to `<run_dir>/_pre_locus_faithfulness/`, a subdirectory
rather than a sibling so the `eval_<system>_*.json` glob the report tooling
uses cannot pick the backup up as a second run. Items that already carry a
score are left alone unless --force is given, so an interrupted pass can be
resumed without paying for the judged items twice.

    uv run python scripts/rescore_locus_faithfulness.py run1 run2 run3
    uv run python scripts/rescore_locus_faithfulness.py run2 --systems long_context
    uv run python scripts/rescore_locus_faithfulness.py run1 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.aggregate_runs import DEFAULT_RUNS_ROOT  # noqa: E402
from scripts.generate_report import SYSTEMS, latest_result_file  # noqa: E402
from src.common.ingestion import download_all_filings  # noqa: E402
from src.evaluation.evidence_store import EvidenceStore  # noqa: E402
from src.evaluation.locus_faithfulness import (  # noqa: E402
    evaluate_locus_faithfulness_batch,
    summarise,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BACKUP_SUBDIR = "_pre_locus_faithfulness"


def _needs_scoring(item: dict, force: bool) -> bool:
    if force:
        return True
    existing = item.get("locus_faithfulness")
    if not existing:
        return True
    # Scored, or excluded for a documented reason: both are final.
    return existing.get("score") is None and not existing.get("excluded")


def rescore_file(
    path: Path, evidence_store: EvidenceStore, force: bool, dry_run: bool,
) -> dict:
    data = json.loads(path.read_text())
    detailed = data["detailed_results"]
    todo = [i for i, d in enumerate(detailed) if _needs_scoring(d, force)]
    logger.info("%s: %d of %d items to score", path.name, len(todo), len(detailed))
    if dry_run or not todo:
        return summarise([d.get("locus_faithfulness") or {} for d in detailed])

    results = evaluate_locus_faithfulness_batch(
        [detailed[i]["question"] for i in todo],
        [detailed[i]["answer"] for i in todo],
        [bool(detailed[i].get("expected_answerable")) for i in todo],
        evidence_store,
    )
    for i, result in zip(todo, results):
        detailed[i]["locus_faithfulness"] = result.to_dict()

    summary = summarise([d.get("locus_faithfulness") or {} for d in detailed])
    summary["rescored_at"] = datetime.now().isoformat(timespec="seconds")
    summary["rescored_post_hoc"] = True
    data["summary"]["locus_faithfulness_summary"] = summary

    backup_dir = path.parent / BACKUP_SUBDIR
    backup = backup_dir / path.name
    if not backup.exists():
        backup_dir.mkdir(exist_ok=True)
        shutil.copy2(path, backup)
        logger.info("backup written: %s", backup)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", help="run directory names, e.g. run1 run2")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument(
        "--systems", nargs="*", default=[name for _, name in SYSTEMS],
        help="subset of system names (default: all four)",
    )
    parser.add_argument("--force", action="store_true", help="re-judge scored items")
    parser.add_argument("--dry-run", action="store_true", help="count only, no judge calls")
    args = parser.parse_args(argv)

    evidence_store = EvidenceStore(download_all_filings())

    for run in args.runs:
        run_dir = args.runs_root / run
        for system in args.systems:
            try:
                path = latest_result_file(run_dir, system)
            except FileNotFoundError as exc:
                logger.warning("%s", exc)
                continue
            summary = rescore_file(path, evidence_store, args.force, args.dry_run)
            logger.info(
                "%s/%s: mean=%s scored=%s/%s coverage=%s excluded=%s",
                run, system, summary.get("mean"), summary.get("n_scored"),
                summary.get("n_answerable"), summary.get("coverage"),
                summary.get("excluded"),
            )


if __name__ == "__main__":
    main()
