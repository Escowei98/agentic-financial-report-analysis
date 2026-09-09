"""
Merge a rating JSON exported from rating_ui{,_reserve}.html back into the
blind CSV.

The rating page is deliberately read-only against the CSV: the trajectories
are multi-line quoted fields, and letting the browser rewrite them is
exactly how the file would silently corrupt. All CSV writes go through
Python's csv module here, from the JSON payload the page produced.

Rules for merging:
  - Only one pass's columns and the notes column are ever touched per run:
    the five reasoning-dimension columns by default, or (--pass2) the five
    custom-metric columns (exact_match/answer_recall/refusal_accuracy/
    refusal_quality/over_refusal). citation_accuracy is scored deterministically
    and is never human-rated, so it is never touched by either pass.
  - "N/A" cells are structural (a dimension that does not apply -- the
    agentic ones on rag_monolith, or a custom metric the automated evaluator
    never computed for that row) and are never overwritten. The rating page
    cannot produce a rating for them; if a JSON payload does, it's rejected
    with an error.
  - An existing rated value is not silently replaced. If the exported JSON
    disagrees with the CSV, the cell is skipped and reported unless
    --force is passed.
  - The original CSV is copied to <name>.bak.<timestamp> before writing.

Usage:
  uv run python scripts/merge_human_ratings.py \
      [--reserve] [--pass2] [--force] [--no-backup] <exported.json>
"""
import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "judge_validation"

PASS1_RATING_COLUMNS = [
    "groundedness_human",
    "validity_human",
    "completeness_human",
]

# The three pass-1 dimensions are rated PER UNIT -- one evidential step, one
# transition, one required sub-question -- so a record holds a variable number
# of cells and the CSV stores a JSON map {unit: 0|1} rather than a scalar.
# Everything else on the page stays scalar.
PER_UNIT_KEYS = {"groundedness", "validity", "completeness"}
PASS2_RATING_COLUMNS = [
    "exact_match_human",
    "answer_recall_human",
    "refusal_accuracy_human",
    "refusal_quality_human",
    "over_refusal_human",
]

# Allowed raw values per rating key. Each custom metric uses whatever scale
# its own judge prompt returns (binary, or the 0/0.5/1 partial-credit scale),
# see custom_evaluator.py's CustomEvalResult / DIMENSIONS in
# analyze_judge_validation.py. The per-unit reasoning dimensions are binary in
# every cell -- there is deliberately no scale left to disagree about.
_SCALE_BINARY = {"0", "1"}
_SCALE_TRI = {"0", "0.5", "1"}

RATING_SCALES = {
    "groundedness": _SCALE_BINARY,
    "validity": _SCALE_BINARY,
    "completeness": _SCALE_BINARY,
    "exact_match": _SCALE_BINARY,
    "answer_recall": _SCALE_TRI,
    "refusal_accuracy": _SCALE_BINARY,
    "refusal_quality": _SCALE_TRI,
    "over_refusal": _SCALE_BINARY,
}


def _sniff_delimiter(csv_path: Path) -> str:
    """Same delimiter-only sniffing used by the build script and analyzer."""
    with open(csv_path, encoding="utf-8-sig") as f:
        sample = f.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;").delimiter
    except csv.Error:
        return ","


def _read_rows(csv_path: Path, delimiter: str) -> tuple[list[str], list[dict]]:
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter, quotechar='"', doublequote=True)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def _write_rows(csv_path: Path, delimiter: str, fieldnames: list[str], rows: list[dict]) -> None:
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, delimiter=delimiter,
            quotechar='"', doublequote=True, quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writer.writerows(rows)


def _validate_rating(value, key: str, review_id: str) -> str:
    """Ensure a rating is on the allowed scale, and serialise it for the CSV.

    A per-unit dimension arrives as {unit: value} and leaves as compact JSON;
    everything else arrives and leaves as a scalar string. Both are validated
    cell by cell against the same scale, so an edited export cannot smuggle an
    off-scale value into the sample through the map form.
    """
    allowed = RATING_SCALES.get(key, set())

    def _check(raw) -> str:
        v = str(raw if raw is not None else "").strip()
        if v not in allowed:
            raise ValueError(
                f"{review_id}.{key}: rating {raw!r} is not one of "
                f"{sorted(allowed)} — the page shouldn't emit this. "
                "Refusing to merge."
            )
        return v

    if key in PER_UNIT_KEYS:
        if not isinstance(value, dict):
            raise ValueError(
                f"{review_id}.{key}: expected a per-unit map, got {type(value).__name__}. "
                "Refusing to merge."
            )
        # Sorted numerically where the units are step indices, so a merged
        # cell reads in chain order rather than lexicographically (10 before 2).
        def _sort_key(unit: str):
            return (0, int(unit)) if unit.isdigit() else (1, unit)

        return json.dumps(
            {unit: _check(value[unit]) for unit in sorted(value, key=_sort_key)},
            separators=(",", ":"),
        )

    return _check(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path, help="human_ratings_*.json exported from the page")
    parser.add_argument(
        "--sample", default=None,
        help="Merge into an arbitrary named sample (e.g. 'sensitivity') instead of the primary/reserve pair.",
    )
    parser.add_argument(
        "--reserve", action="store_true",
        help="Merge into human_review_blind_reserve.csv instead of the primary one.",
    )
    parser.add_argument(
        "--pass2", action="store_true",
        help="Merge the second-pass export (exact_match / answer_recall / "
             "refusal_accuracy / refusal_quality / over_refusal) instead of "
             "the first-pass reasoning-dimensions export.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing rated value if it disagrees with the JSON.",
    )
    parser.add_argument(
        "--no-backup", action="store_true",
        help="Skip the <name>.bak.<timestamp> backup (default: on).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would change without writing.",
    )
    args = parser.parse_args()

    RATING_COLUMNS = PASS2_RATING_COLUMNS if args.pass2 else PASS1_RATING_COLUMNS
    RATING_KEYS = [c[: -len("_human")] for c in RATING_COLUMNS]

    if args.sample and args.reserve:
        raise SystemExit("--sample und --reserve schliessen sich aus.")
    sample_name = args.sample or ("reserve" if args.reserve else "primary")
    suffix = f"_{args.sample}" if args.sample else ("_reserve" if args.reserve else "")
    csv_path = RESULTS_DIR / f"human_review_blind{suffix}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Blind review CSV not found: {csv_path}")
    if not args.payload.exists():
        raise FileNotFoundError(f"Exported rating JSON not found: {args.payload}")

    with open(args.payload, encoding="utf-8") as f:
        payload = json.load(f)

    expected_pass = "custom" if args.pass2 else None
    if payload.get("pass") != expected_pass:
        raise SystemExit(
            f"Pass mismatch: payload has pass={payload.get('pass')!r} but "
            f"{'--pass2 was' if args.pass2 else '--pass2 was not'} passed. "
            f"Use the matching JSON/flag combination."
        )

    expected_sample = sample_name
    if payload.get("sample") != expected_sample:
        raise SystemExit(
            f"Sample mismatch: payload is for {payload.get('sample')!r} but "
            f"the target CSV is the {expected_sample!r} sample. Pass "
            f"{'without ' if args.reserve else ''}--reserve or use the matching JSON."
        )
    if payload.get("source_csv") not in (None, csv_path.name):
        raise SystemExit(
            f"source_csv mismatch: payload was exported against "
            f"{payload.get('source_csv')!r}, target is {csv_path.name!r}."
        )

    ratings = payload.get("ratings") or {}
    if not isinstance(ratings, dict):
        raise SystemExit("Malformed payload: 'ratings' must be an object.")

    delimiter = _sniff_delimiter(csv_path)
    fieldnames, rows = _read_rows(csv_path, delimiter)
    row_by_id = {r["review_id"]: r for r in rows}

    missing_columns = [c for c in RATING_COLUMNS if c not in fieldnames]
    if missing_columns:
        raise SystemExit(
            f"CSV is missing expected rating columns: {missing_columns}. "
            f"The build script generates them; regenerate the CSV or check "
            f"you're pointing at the right file."
        )
    has_notes = "notes" in fieldnames

    updated = 0
    unchanged = 0   # JSON matches an already-rated cell — no-op.
    conflicts: list[str] = []
    na_attempts: list[str] = []
    unknown_ids: list[str] = []
    unknown_keys: list[str] = []
    notes_updated = 0

    for review_id, cell_map in ratings.items():
        row = row_by_id.get(review_id)
        if row is None:
            unknown_ids.append(review_id)
            continue

        for key, raw_value in cell_map.items():
            if key == "notes":
                if not has_notes:
                    continue
                new_notes = (raw_value or "").strip()
                if new_notes != (row.get("notes") or "").strip():
                    if not args.dry_run:
                        row["notes"] = new_notes
                    notes_updated += 1
                continue

            if key not in RATING_KEYS:
                unknown_keys.append(f"{review_id}.{key}")
                continue

            column = f"{key}_human"
            new_value = _validate_rating(raw_value, key, review_id)
            existing = (row.get(column) or "").strip()

            if existing.upper() == "N/A":
                # Structural — the page shouldn't have offered this cell. Flag
                # it so a bug in the build script or an edited JSON can't
                # silently poison the sample.
                na_attempts.append(f"{review_id}.{key} (would overwrite N/A with {new_value})")
                continue

            if existing == new_value:
                unchanged += 1
                continue

            if existing and not args.force:
                conflicts.append(f"{review_id}.{key}: CSV={existing!r} JSON={new_value!r}")
                continue

            if not args.dry_run:
                row[column] = new_value
            updated += 1

    still_open = 0
    for row in rows:
        for column in RATING_COLUMNS:
            v = (row.get(column) or "").strip()
            if not v:
                still_open += 1

    if unknown_ids:
        print(f"Ignored {len(unknown_ids)} unknown review_id(s): "
              f"{', '.join(unknown_ids[:5])}{'…' if len(unknown_ids) > 5 else ''}",
              file=sys.stderr)
    if unknown_keys:
        print(f"Ignored {len(unknown_keys)} unknown key(s): "
              f"{', '.join(unknown_keys[:5])}{'…' if len(unknown_keys) > 5 else ''}",
              file=sys.stderr)
    if na_attempts:
        print("Refused to overwrite N/A cells (structural — dimension does not "
              "apply to this system):", file=sys.stderr)
        for entry in na_attempts:
            print(f"  {entry}", file=sys.stderr)
    if conflicts:
        print("Rating conflicts — CSV already has a different value. Skipped "
              "(pass --force to overwrite):", file=sys.stderr)
        for entry in conflicts:
            print(f"  {entry}", file=sys.stderr)

    print()
    print(f"CSV:            {csv_path}")
    print(f"Payload:        {args.payload}")
    print(f"Cells updated:  {updated}")
    print(f"Notes updated:  {notes_updated}")
    print(f"Cells matched:  {unchanged}  (JSON agrees with CSV)")
    print(f"Conflicts:      {len(conflicts)}  ({'overwritten' if args.force else 'skipped, use --force'})")
    print(f"Cells still empty after merge: {still_open}")

    if args.dry_run:
        print("\nDry run — CSV not written.")
        return

    if updated == 0 and notes_updated == 0:
        print("\nNothing to write.")
        return

    if not args.no_backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = csv_path.with_suffix(csv_path.suffix + f".bak.{stamp}")
        shutil.copy2(csv_path, backup)
        print(f"Backup:         {backup.name}")

    _write_rows(csv_path, delimiter, fieldnames, rows)
    print("Merged.")


if __name__ == "__main__":
    main()
