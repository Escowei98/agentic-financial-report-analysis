"""
Consistency checks for the gold standard against the configured corpus.

This is the guard that makes the original defect impossible to repeat: the
first full n=150 run executed against 4 filings instead of 12 because nothing
ever asserted that the corpus the systems see actually contains the documents
the gold standard refers to.

Checks
------
1. corpus       every doc id referenced by the gold standard exists in the
                corpus built by `download_all_filings()`
2. alignment    the German and English files have identical ids, fa_types and
                doc ids (only free text may differ)
3. groups       `doc_id_groups` is consistent with the flat `doc_ids` union
4. reachability for items whose ground truth is a single reported figure,
                that figure is actually present in at least one of the mapped
                filings. Catches items that survived the doc-id remap
                structurally but whose content is not in the corpus (e.g. a
                headcount question when the mapped filing reports none).
                Derived values (margins, growth rates, ratios) are reported as
                "not directly checkable" rather than as failures -- they are
                computed, not quoted.
5. schema       columns added in v4/v5 are present and populated where required
6. refusal      every FA-Refusal row carries a `refusal_evidence` class and a
                `gt_correction`, `gt_value` stays empty, and a correction is
                only demanded where the corpus can actually support one

Exit code 0 = all checks passed, 1 = at least one hard failure.

Usage:
    uv run python scripts/validate_gold_standard.py [path/to/gold_standard.csv]
"""

from __future__ import annotations

import csv
import logging
import re
import sys
from pathlib import Path

from src.common.ingestion import download_all_filings, fiscal_year_from_metadata
from src.evaluation.gold_standard_loader import GOLD_STANDARD_EN

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GS_DIR = PROJECT_ROOT / "data" / "gold_standard"

# Filings report in millions; a ground truth of "$383.3 billion" must be
# matched against "383,285" in the text.
_SCALE_TO_MILLIONS = {"billion": 1000.0, "bn": 1000.0, "million": 1.0, "mn": 1.0}
_GT_NUMBER_RE = re.compile(
    r"\$?\s*(?P<num>\d[\d,]*\.?\d*)\s*(?P<unit>billion|bn|million|mn)?", re.IGNORECASE
)
_TEXT_NUMBER_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+\b")
_TOLERANCE = 0.02

# Subtypes whose ground truth is computed from reported figures rather than
# quoted from them. Searching the filing text for such a value is meaningless
# — it is not in there by construction — so they join the "derived" bucket
# instead of being reported as unreachable.
_DERIVED_SUBTYPES = {"ratio_compute", "segment_share", "segment_margin", "solvency"}


def _split(value: str, sep: str = "|") -> list[str]:
    return [p.strip() for p in (value or "").split(sep) if p.strip()]


def _gt_millions(gt_value: str, gt_unit: str) -> float | None:
    """Ground truth as a figure in millions, or None if not a plain amount."""
    if (gt_unit or "").strip().lower() not in ("usd_billion", "usd_million"):
        return None
    m = _GT_NUMBER_RE.search(gt_value or "")
    if not m:
        return None
    try:
        value = float(m.group("num").replace(",", ""))
    except ValueError:
        return None
    unit = (m.group("unit") or "").lower()
    if unit:
        return value * _SCALE_TO_MILLIONS[unit]
    return value * (1000.0 if gt_unit.strip().lower() == "usd_billion" else 1.0)


def _figure_present(target_millions: float, text: str) -> bool:
    lo, hi = target_millions * (1 - _TOLERANCE), target_millions * (1 + _TOLERANCE)
    for token in _TEXT_NUMBER_RE.findall(text):
        try:
            value = float(token.replace(",", ""))
        except ValueError:
            continue
        if lo <= value <= hi:
            return True
    return False


def main(argv: list[str]) -> int:
    gs_en = Path(argv[1]) if len(argv) > 1 else GOLD_STANDARD_EN
    gs_de = Path(str(gs_en).replace("_en.csv", ".csv"))

    filings = download_all_filings()
    corpus_text = {
        f"{f.metadata.ticker}_{fiscal_year_from_metadata(f)}": f.full_text for f in filings
    }
    corpus_ids = set(corpus_text)
    print(f"Korpus: {len(corpus_ids)} Berichte — {', '.join(sorted(corpus_ids))}\n")

    with open(gs_en, encoding="utf-8") as fh:
        rows_en = list(csv.DictReader(fh, delimiter=";"))
    with open(gs_de, encoding="utf-8") as fh:
        rows_de = list(csv.DictReader(fh, delimiter=";"))

    failures: list[str] = []
    warnings: list[str] = []

    # --- 1. corpus coverage -------------------------------------------------
    missing = sorted(
        {d for r in rows_en for d in _split(r.get("doc_ids", "")) if d not in corpus_ids}
    )
    if missing:
        failures.append(f"doc_ids ohne Bericht im Korpus: {missing}")
    print(f"[{'FAIL' if missing else ' OK '}] 1. Korpusabdeckung der doc_ids")

    # --- 2. DE/EN alignment -------------------------------------------------
    misaligned = []
    if len(rows_de) != len(rows_en):
        misaligned.append(f"Zeilenzahl {len(rows_de)} (de) vs. {len(rows_en)} (en)")
    else:
        # Every column except the three that legitimately differ by language
        # (`query`, `gt_value`, `rationale`) must be identical. The earlier
        # short list (id/fa_type/doc_ids/expected_answerable) let two real
        # divergences through: `gt_unit` was empty on all 30 refusal rows in
        # the English file where the German one carried `n/a`, and
        # `source_sections` could drift silently. See EVAL_DECISION_LOG.md
        # [2026-09-06].
        aligned_fields = [
            f for f in (rows_en[0].keys() if rows_en else [])
            if f not in ("query", "gt_value", "rationale", "gt_correction")
        ]
        for de, en in zip(rows_de, rows_en):
            for field in aligned_fields:
                # Case-insensitive: the files inherited "true"/"True" from v3.
                # Both are normalised to lowercase now, but the loader
                # lowercases anyway, so the tolerance stays as a guard rather
                # than as an accepted difference.
                if (de.get(field) or "").lower() != (en.get(field) or "").lower():
                    misaligned.append(f"id {en['id']}: {field} weicht ab")
    if misaligned:
        failures.append(f"DE/EN nicht synchron: {misaligned[:5]}")
    print(f"[{'FAIL' if misaligned else ' OK '}] 2. DE/EN-Synchronitaet")

    # --- 3. groups consistent with flat union -------------------------------
    if "doc_id_groups" in (rows_en[0] if rows_en else {}):
        bad_groups = []
        for r in rows_en:
            groups = [g.split("+") for g in _split(r.get("doc_id_groups", ""))]
            flat = sorted({d for g in groups for d in g})
            if flat != sorted(_split(r.get("doc_ids", ""))):
                bad_groups.append(r["id"])
        if bad_groups:
            failures.append(f"doc_id_groups inkonsistent zu doc_ids: ids {bad_groups}")
        print(f"[{'FAIL' if bad_groups else ' OK '}] 3. Gruppen konsistent zur flachen Menge")
    else:
        print("[SKIP] 3. Spalte doc_id_groups nicht vorhanden")

    # --- 4. reachability of the ground-truth figure -------------------------
    checked = unreachable = uncheckable = 0
    unreachable_ids = []
    for r in rows_en:
        if r.get("expected_answerable", "").strip().lower() != "true":
            continue
        target = _gt_millions(r.get("gt_value", ""), r.get("gt_unit", ""))
        if target is None or (r.get("subtype") or "").strip() in _DERIVED_SUBTYPES:
            uncheckable += 1
            continue
        docs = _split(r.get("doc_ids", ""))
        checked += 1
        if not any(_figure_present(target, corpus_text.get(d, "")) for d in docs):
            unreachable += 1
            unreachable_ids.append(f"{r['id']} ({r['fa_type']}, {r.get('gt_value','')})")
    if unreachable:
        warnings.append(
            f"{unreachable} von {checked} direkt berichteten Sollwerten nicht in den "
            f"zugeordneten Berichten gefunden: {unreachable_ids}"
        )
    print(
        f"[{'WARN' if unreachable else ' OK '}] 4. Sollwert im zugeordneten Bericht auffindbar "
        f"({checked - unreachable}/{checked} geprueft, {uncheckable} abgeleitet/nicht pruefbar)"
    )

    # --- 5. v4 schema -------------------------------------------------------
    header = list(rows_en[0].keys()) if rows_en else []
    schema_msgs = []
    # There is deliberately no `expected_tools` column: the deterministic
    # tool-selection metric was dropped (EVAL_DECISION_LOG 2026-08-15,
    # upheld 2026-09-06). Its reappearance would be a regression.
    if "expected_tools" in header:
        failures.append(
            "Spalte expected_tools ist wieder da — die deterministische "
            "Werkzeugauswahlgenauigkeit wurde bewusst verworfen"
        )
    fa3 = [r for r in rows_en if r.get("fa_type") == "FA-3"]
    windows = sorted({r.get("window_class", "") for r in fa3})
    if "window_class" in header:
        counts = {w: sum(1 for r in fa3 if r.get("window_class") == w) for w in windows}
        if counts.get("within_window") != 15 or counts.get("cross_window") != 15:
            schema_msgs.append(f"FA-3 window_class nicht 15/15: {counts}")
    else:
        schema_msgs.append("Spalte window_class fehlt noch (Phase 1b)")
    for msg in schema_msgs:
        print(f"[INFO] 5. {msg}")

    # --- 6. refusal stratum (v5) --------------------------------------------
    #
    # The scoring of this stratum branches on `refusal_evidence`, so an
    # unclassified or mis-classified row does not fail loudly at run time --
    # it silently falls into the default judge path and the subtype breakdown
    # in chapter 5 quietly loses an item. These invariants are the guard.
    valid_evidence = {"out_of_scope", "counter_evidence", "silent", "underspecified"}
    refusal_failures: list[str] = []
    if "refusal_evidence" in header:
        refusal_rows = [
            r for r in rows_en
            if (r.get("expected_answerable", "true").strip().lower() == "false")
        ]
        if len(refusal_rows) != 30:
            refusal_failures.append(
                f"FA-Refusal-Stratum hat {len(refusal_rows)} statt 30 Zeilen"
            )
        for r in refusal_rows:
            rid = r.get("id")
            evidence = (r.get("refusal_evidence") or "").strip()
            correction = (r.get("gt_correction") or "").strip()
            if evidence not in valid_evidence:
                refusal_failures.append(
                    f"id {rid}: refusal_evidence {evidence!r} nicht in {sorted(valid_evidence)}"
                )
            if not correction:
                refusal_failures.append(f"id {rid}: gt_correction ist leer")
            if (r.get("gt_value") or "").strip():
                refusal_failures.append(
                    f"id {rid}: gt_value ist befuellt — bei FA-Refusal muss es leer bleiben"
                )
            # An ambiguous question has no scope boundary and no counter-
            # evidence to name; a not_in_corpus question cannot be refuted by
            # a corpus that does not hold the data. Crossing these would make
            # the judge demand a statement the corpus cannot support.
            if r.get("subtype") == "ambiguous_entity" and evidence != "underspecified":
                refusal_failures.append(
                    f"id {rid}: subtype ambiguous_entity erwartet refusal_evidence "
                    f"'underspecified', hat {evidence!r}"
                )
            if r.get("subtype") == "not_in_corpus" and evidence == "counter_evidence":
                refusal_failures.append(
                    f"id {rid}: subtype not_in_corpus kann keine counter_evidence "
                    "haben — der Korpus enthaelt die Daten per Definition nicht"
                )
        # Any answerable row carrying refusal metadata means the annotation
        # leaked out of the stratum it belongs to.
        leaked = [
            r.get("id") for r in rows_en
            if r.get("expected_answerable", "true").strip().lower() == "true"
            and ((r.get("refusal_evidence") or "").strip()
                 or (r.get("gt_correction") or "").strip())
        ]
        if leaked:
            refusal_failures.append(
                f"beantwortbare Items tragen Refusal-Metadaten: {leaked}"
            )
        counts = {e: sum(1 for r in refusal_rows if r.get("refusal_evidence") == e)
                  for e in sorted(valid_evidence)}
        failures.extend(refusal_failures)
        print(
            f"[{'FAIL' if refusal_failures else ' OK '}] 6. Refusal-Stratum "
            f"klassifiziert: {counts}"
        )
    else:
        print("[INFO] 6. Spalte refusal_evidence fehlt (Datei aelter als v5)")

    # --- 7. reference decomposition (v6) ------------------------------------
    #
    # Dimension 3 of the reasoning metric scores coverage against this column.
    # An empty cell on an answerable item does not fail at run time: the judge
    # is simply handed an empty requirement list and the item silently drops
    # out of the completeness average. These invariants are the guard.
    if "reference_decomposition" in header:
        decomposition_failures: list[str] = []
        answerable_rows = [
            r for r in rows_en
            if r.get("expected_answerable", "true").strip().lower() == "true"
        ]
        for r in answerable_rows:
            rid = r.get("id")
            tfs = [t.strip() for t in (r.get("reference_decomposition") or "").split("|") if t.strip()]
            if not tfs:
                decomposition_failures.append(
                    f"id {rid}: reference_decomposition ist leer"
                )
                continue
            if len(tfs) > 12:
                decomposition_failures.append(
                    f"id {rid}: {len(tfs)} Teilfragen — vermutlich ein Parsefehler"
                )
            # The generator's fallback wordings. Both are legitimate DRAFT
            # values but must not survive into a file used for scoring: a
            # sub-question that does not name what to establish cannot be
            # judged for coverage. The second pattern is the ratio variant
            # ("the first component of Operating Margin"), corrected in v6.1
            # by scripts/fix_v6_component_names.py.
            unresolved = [
                t for t in tfs
                if "the figure the question asks for" in t
                or "component of" in t.lower()
            ]
            if unresolved:
                decomposition_failures.append(
                    f"id {rid}: {len(unresolved)} Teilfrage(n) noch mit "
                    "Platzhaltertext — Handpruefung ausstehend"
                )
        # Mirror of the refusal leak check: a decomposition on a refusal item
        # would mean the stratum boundary moved.
        leaked_decomposition = [
            r.get("id") for r in rows_en
            if r.get("expected_answerable", "true").strip().lower() == "false"
            and (r.get("reference_decomposition") or "").strip()
        ]
        if leaked_decomposition:
            decomposition_failures.append(
                f"FA-Refusal-Items tragen eine Referenzzerlegung: {leaked_decomposition}"
            )
        failures.extend(decomposition_failures)
        sizes = [
            len([t for t in (r.get("reference_decomposition") or "").split("|") if t.strip()])
            for r in answerable_rows
        ]
        span = f"{min(sizes)}-{max(sizes)}" if sizes else "n/a"
        print(
            f"[{'FAIL' if decomposition_failures else ' OK '}] 7. Referenzzerlegung "
            f"({len(answerable_rows)} beantwortbare Items, {span} Teilfragen)"
        )
    else:
        print("[INFO] 7. Spalte reference_decomposition fehlt (Datei aelter als v6)")

    print()
    for w in warnings:
        print(f"WARN: {w}")
    for f in failures:
        print(f"FAIL: {f}")
    if not failures and not warnings:
        print("Alle Pruefungen bestanden.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
