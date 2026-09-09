"""
Merge the approved cross-window FA-3 items into the gold standard (Phase 1b).

Splits the multi-year stratum 15/15 into `within_window` and `cross_window`
and rewrites the 15 items listed in `fa3_cross_window_proposal.csv` to spans
that no single filing can serve.

Why: a 10-K reports the income statement over three fiscal years in
comparative columns, so with the previous consecutive-year corpus every FA-3
item was answerable from the FY2024 filing alone. The stratum measured table
lookup rather than cross-document synthesis. Each rewritten item mirrors a
retained one in company, metric and math_type and differs only in the span,
which makes the document boundary the single varying quantity inside the
stratum.

Operates on gold_standard_v4*.csv in place (v3 remains the frozen prior
version). Ground-truth values come from the proposal file, which carries the
file:line provenance and the arithmetic for each one.

Usage:
    uv run python scripts/merge_fa3_cross_window.py
"""

from __future__ import annotations

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GS_DIR = PROJECT_ROOT / "data" / "gold_standard"
PROPOSAL = GS_DIR / "fa3_cross_window_proposal.csv"


def load_proposal() -> dict[str, dict]:
    with open(PROPOSAL, encoding="utf-8") as fh:
        return {r["id"]: r for r in csv.DictReader(fh, delimiter=";")}


def main() -> int:
    proposal = load_proposal()
    print(f"{len(proposal)} freigegebene cross_window-Items: {sorted(proposal, key=int)}\n")

    # HISTORICAL — this is the generator that added the `window_class`
    # column to v4 on 2026-09-06. It edits those two files in place and
    # names them directly for that reason; it is not a consumer of the
    # canonical path and must not follow it to a future version.
    # See EVAL_DECISION_LOG.md [2026-09-06].
    for lang, name in (("de", "gold_standard_v4.csv"), ("en", "gold_standard_v4_en.csv")):
        path = GS_DIR / name
        with open(path, encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter=";")
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)

        if "window_class" not in fieldnames:
            fieldnames.append("window_class")

        rewritten = marked = 0
        for row in rows:
            if row["fa_type"] != "FA-3":
                row.setdefault("window_class", "")
                row["window_class"] = ""
                continue

            p = proposal.get(row["id"])
            if p is None:
                # Retained item: same question, now explicitly labelled as
                # answerable inside one filing's comparative window.
                row["window_class"] = "within_window"
                marked += 1
                continue

            row["query"] = p["query_de"] if lang == "de" else p["query_en"]
            row["gt_value"] = p["gt_value_de"] if lang == "de" else p["gt_value"]
            row["gt_unit"] = p["gt_unit"]
            row["subtype"] = p["subtype"]
            row["math_type"] = p["math_type"]
            row["doc_ids"] = p["doc_ids"]
            row["doc_id_groups"] = p["doc_id_groups"]
            row["source_sections"] = p["source_sections"]
            row["window_class"] = "cross_window"
            row["rationale"] = (
                f"Cross-window: {p['calculation']} — Quellen: {p['sources']}"
            )
            rewritten += 1

        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerows(rows)
        print(f"{name}: {rewritten} Items neu geschrieben, {marked} als within_window markiert")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
