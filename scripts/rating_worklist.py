"""
Rate the blind sample without the browser.

The HTML page is only one way to produce the JSON that
`merge_human_ratings.py` consumes -- that script takes the payload, not the
page. This gives the same job as a flat, readable file: one entry per rating
cell, carrying everything needed to decide it, with `"rating": null` waiting
to be filled in.

Two modes:

    # 1. write the worklist
    uv run python scripts/rating_worklist.py export --reserve

    # 2. fill in every "rating" (0 or 1), then convert and merge
    uv run python scripts/rating_worklist.py to-payload --reserve worklist.json
    uv run python scripts/merge_human_ratings.py --reserve payload.json

WHAT IS AND IS NOT IN HERE
--------------------------
The same blind protocol as the page: no system name, no judge verdict, no
reference answer. The system label stays the anonymous A-D. The evidence
passages are the ones THE JUDGE SAW, carried over from the run rather than
re-fetched -- a rater shown different evidence is not validating the judge.

Partial files are fine. A cell left at null is skipped, so the work can be
split across sessions or people; `merge_human_ratings.py` reports what is
still empty afterwards.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from scripts.build_human_rating_ui import (
    DIMENSION_LABELS,
    DIMENSION_RULES,
    REASONING_DIMENSIONS,
    _cell_state,
    _units_for,
    build_rubrics,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "judge_validation"


def _read_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _cell(row: dict, dimension: str, unit: dict) -> dict:
    """One rating cell, with the material needed to decide it."""
    entry = {
        "cell_id": f"{row['review_id']}.{dimension}.{unit['unit']}",
        "review_id": row["review_id"],
        "system_label": row.get("system_label", ""),
        "dimension": dimension,
        "unit": unit["unit"],
        "question": row.get("question", ""),
    }
    if dimension == "groundedness":
        entry["step"] = unit["step"]
        # The passages at the source the step itself cites. "Trägt diese
        # Passage die Tatsachenbehauptung des Schritts?" is the whole
        # question -- not whether the claim is true, and not whether a better
        # source existed.
        entry["evidence"] = unit["evidence"]
    elif dimension == "validity":
        entry["preceding_steps"] = unit["preceding"]
        entry["step"] = unit["step"]
    else:
        entry["subquestion"] = unit["subquestion"]
        entry["chain"] = [s["text"] for s in json.loads(row.get("chain_json") or "[]")]
    entry["rating"] = None
    return entry


def export(rows: list[dict], sample: str, out_path: Path) -> int:
    cells = []
    for row in rows:
        units = _units_for(row)
        for dimension in REASONING_DIMENSIONS:
            if _cell_state(row.get(f"{dimension}_human", "")) == "na":
                continue
            for unit in units[dimension]:
                cells.append(_cell(row, dimension, unit))

    payload = {
        "sample": sample,
        "source_csv": f"human_review_blind{'_reserve' if sample == 'reserve' else ''}.csv",
        "scale": {"1": "ja", "0": "nein"},
        "instructions": build_rubrics()["preamble"],
        "labels": DIMENSION_LABELS,
        "rules": DIMENSION_RULES,
        "cells": cells,
    }
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return len(cells)


def to_payload(worklist: dict, sample: str, out_path: Path) -> tuple[int, int]:
    """Fold a filled worklist into the shape merge_human_ratings.py expects."""
    if worklist.get("sample") != sample:
        raise SystemExit(
            f"Sample mismatch: worklist is for {worklist.get('sample')!r}, "
            f"target is {sample!r}."
        )

    ratings: dict[str, dict] = {}
    filled = skipped = 0
    for cell in worklist.get("cells", []):
        value = cell.get("rating")
        if value is None or str(value).strip() == "":
            skipped += 1
            continue
        value = str(value).strip()
        # Validated again in merge_human_ratings, but failing here names the
        # cell rather than the record, which is what a hand-edited file needs.
        if value not in ("0", "1"):
            raise SystemExit(
                f"{cell.get('cell_id')}: rating {cell.get('rating')!r} is not "
                "0 or 1. Every cell is a yes/no decision -- there is no scale."
            )
        by_dim = ratings.setdefault(cell["review_id"], {})
        by_dim.setdefault(cell["dimension"], {})[str(cell["unit"])] = value
        filled += 1

    out_path.write_text(
        json.dumps(
            {
                "sample": sample,
                "source_csv": worklist.get("source_csv", ""),
                "ratings": ratings,
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return filled, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["export", "to-payload"])
    parser.add_argument("worklist", nargs="?", type=Path,
                        help="to-payload: the filled worklist JSON")
    parser.add_argument("--reserve", action="store_true",
                        help="Work on the reserve sample instead of the primary one.")
    parser.add_argument("--out", type=Path, help="Override the output path.")
    args = parser.parse_args()

    sample = "reserve" if args.reserve else "primary"
    suffix = "_reserve" if args.reserve else ""

    if args.mode == "export":
        csv_path = RESULTS_DIR / f"human_review_blind{suffix}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Blind review CSV not found: {csv_path}")
        out = args.out or RESULTS_DIR / f"rating_worklist{suffix}.json"
        n = export(_read_rows(csv_path), sample, out)
        print(f"{n} offene Zellen geschrieben nach {out}")
        print('Jede "rating": null durch "1" (ja) oder "0" (nein) ersetzen, dann:')
        print(f"  uv run python scripts/rating_worklist.py to-payload"
              f"{' --reserve' if args.reserve else ''} {out}")
        return 0

    if args.worklist is None:
        raise SystemExit("to-payload braucht den Pfad der ausgefuellten Arbeitsliste.")
    worklist = json.loads(args.worklist.read_text(encoding="utf-8"))
    out = args.out or RESULTS_DIR / f"human_ratings{suffix}_from_worklist.json"
    filled, skipped = to_payload(worklist, sample, out)
    print(f"{filled} bewertete Zellen uebernommen, {skipped} noch offen -> {out}")
    print("Zusammenfuehren mit:")
    print(f"  uv run python scripts/merge_human_ratings.py"
          f"{' --reserve' if args.reserve else ''} {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
