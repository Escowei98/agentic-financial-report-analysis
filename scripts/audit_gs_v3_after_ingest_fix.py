"""Audit Gold-Standard v3 entries affected by the 2026-07-25 ingestion fix.

For every GS entry whose `doc_ids` reference one of the three re-ingested filings
(MSFT_2023, MSFT_2024, GOOGL_2022) AND whose `source_sections` reference an
`item_8_*` sub-section AND which is `expected_answerable`, this script issues one
of three verdicts:

- AUTO_OK      -> ground-truth primary number is present (+/-1 %) in the
                  referenced filings' `## Financial Statements` section text.
                  Safe candidate for spot-check only.
- DERIVED      -> ground-truth is by construction a computed / derived value
                  (growth rate, CAGR, pp-diff, ratio, multi-step). Cannot be
                  auto-checked; must be verified by re-computing from filing
                  inputs.
- MANUAL_CHECK -> primary number could be extracted but not located in the
                  filing text at +/-1 %. Requires manual verification.

Output: `data/results/gs_v3_ingest_fix_audit.csv` with columns
`id;fa_type;subtype;doc_ids;query;gt_value;verdict;evidence`.

Usage:
    uv run python scripts/audit_gs_v3_after_ingest_fix.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GS_CSV = PROJECT_ROOT / "data" / "gold_standard" / "gold_standard_v3.csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT_CSV = PROJECT_ROOT / "data" / "results" / "gs_v3_ingest_fix_audit.csv"

AFFECTED_DOCS = {"MSFT_2023", "MSFT_2024", "GOOGL_2022"}

# Subtypes whose gt_value is by construction a derived / computed value that
# will never appear literally in the filing text. Always route to DERIVED.
DERIVED_SUBTYPES = {
    "ratio_compute",
    "growth_yoy",
    "growth_cagr",
    "pp_change",
    "multi_step_compute",
    "cross_firm_diff",
    "cross_firm_pairwise_growth",
    "trend_qualitative",
    "segment_margin",
    "segment_share",
    "synthesis",
    "cloud_comparison",
}

TOLERANCE = 0.01


# ---------------------------------------------------------------------------
#  doc_id -> filing path mapping via .meta.json sidecars
# ---------------------------------------------------------------------------

def build_doc_id_map() -> dict[str, Path]:
    """Return `{TICKER_YYYY: path/to/10K_YYYY-MM-DD.md}`.

    Uses the `fiscal_year_end` field from each `.meta.json` sidecar to derive
    the canonical doc_id, so the mapping stays correct even for tickers whose
    fiscal year does not align with the calendar year (AAPL: Sep, MSFT: Jun).
    """
    mapping: dict[str, Path] = {}
    for meta_path in DATA_PROCESSED.glob("*/10K_*.meta.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        ticker = meta.get("ticker", "").upper()
        fye = meta.get("fiscal_year_end", "")
        if not ticker or not fye:
            continue
        fy_year = fye[:4]
        doc_id = f"{ticker}_{fy_year}"
        md_path = meta_path.with_suffix("").with_suffix(".md")
        if md_path.exists():
            mapping[doc_id] = md_path
    return mapping


# ---------------------------------------------------------------------------
#  Financial Statements section extraction (mirrors validate_ingestion.py)
# ---------------------------------------------------------------------------

def extract_financial_statements(md_path: Path) -> str:
    """Return the raw content of `## Financial Statements`, or ''."""
    lines = md_path.read_text(encoding="utf-8").splitlines()
    inside = False
    collected: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if inside:
                break
            if line.strip() == "## Financial Statements":
                inside = True
                continue
        if inside:
            collected.append(line)
    return "\n".join(collected).strip()


# ---------------------------------------------------------------------------
#  gt_value primary-number extraction
# ---------------------------------------------------------------------------

# Order matters: check monetary before naked int, so "$391 Mrd" doesn't parse
# as "391" without magnitude.
_NUM_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("usd_billion", re.compile(r"\$\s*([+-]?\d+(?:[.,]\d+)?)\s*Mrd", re.IGNORECASE)),
    ("usd_million", re.compile(r"\$\s*([+-]?\d+(?:[.,]\d+)?)\s*Mio", re.IGNORECASE)),
    ("usd_per_share", re.compile(r"\$\s*([+-]?\d+(?:[.,]\d+)?)(?!\s*(?:Mrd|Mio))")),
    ("percent", re.compile(r"([+-]?\d+(?:[.,]\d+)?)\s*%")),
    ("pp_change", re.compile(r"([+-]?\d+(?:[.,]\d+)?)\s*pp")),
    ("count", re.compile(r"([+-]?\d+(?:[.,]\d+)?)")),
]


def parse_primary_number(gt_value: str) -> tuple[float | None, str | None]:
    """Return the first numeric value found in gt_value plus its unit kind."""
    if not gt_value or gt_value.strip() == "":
        return None, None
    for unit, pattern in _NUM_PATTERNS:
        m = pattern.search(gt_value)
        if m:
            raw = m.group(1).replace(",", ".").replace("+", "")
            try:
                return float(raw), unit
            except ValueError:
                continue
    return None, None


# ---------------------------------------------------------------------------
#  Numeric search in filing text with tolerance
# ---------------------------------------------------------------------------

# Numbers as they appear in filings: '245,122' (thousand separator, in millions)
# or '245.1' (billions with decimal) or '245' or '245.122'.
_FILING_NUMBER = re.compile(r"(?<![\d.])([+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[+-]?\d+(?:\.\d+)?)(?![\d,])")


def _iter_filing_numbers(text: str):
    """Yield floats extracted from filing-style numeric tokens."""
    for m in _FILING_NUMBER.finditer(text):
        raw = m.group(1).replace(",", "")
        try:
            yield float(raw)
        except ValueError:
            continue


def matches_within_tolerance(value: float, unit: str | None, section: str) -> tuple[bool, str]:
    """Look for `value` in `section` at +/-1 % tolerance.

    For monetary values (USD_billion) filings typically print millions, so also
    check the value * 1000. For percentages, both '44.6' and '44.6%' are
    candidates. Returns (matched, evidence-snippet) for logging.
    """
    if value == 0.0:
        return False, ""
    candidates: list[float] = [value]
    if unit == "usd_billion":
        candidates.append(value * 1000.0)  # filing prints millions
        candidates.append(value * 1_000_000.0)  # in case of full-dollar printing
    elif unit == "usd_million":
        candidates.append(value * 1000.0)  # in case value is thousands
    elif unit == "count":
        candidates.append(value * 1000.0)  # e.g. "+7,000" employees vs. thousand-scale
        candidates.append(value * 1_000_000.0)
        candidates.append(value / 1000.0)

    filing_numbers = list(_iter_filing_numbers(section))
    for cand in candidates:
        lower = abs(cand) * (1.0 - TOLERANCE)
        upper = abs(cand) * (1.0 + TOLERANCE)
        for fnum in filing_numbers:
            if lower <= abs(fnum) <= upper:
                return True, f"filing_hit={fnum} (candidate={cand}, tol={TOLERANCE*100:.0f}%)"
    return False, f"no filing number in [{value * (1-TOLERANCE):.4g}, {value * (1+TOLERANCE):.4g}] (or scaled)"


# ---------------------------------------------------------------------------
#  Main audit loop
# ---------------------------------------------------------------------------

def is_item8_source(source_sections: str) -> bool:
    return "item_8" in source_sections.lower()


def touches_affected_doc(doc_ids: str) -> tuple[bool, list[str]]:
    docs = [d.strip() for d in doc_ids.split("|") if d.strip()]
    hits = [d for d in docs if d in AFFECTED_DOCS]
    return (len(hits) > 0, hits)


def audit_row(row: dict, doc_map: dict[str, Path]) -> dict | None:
    """Audit a single GS row; return audit dict or None if not affected."""
    doc_ids = row["doc_ids"]
    source_sections = row["source_sections"]
    expected_answerable = row["expected_answerable"].strip().lower() == "true"
    subtype = row["subtype"].strip()

    if not expected_answerable:
        return None
    if not is_item8_source(source_sections):
        return None
    touches, affected = touches_affected_doc(doc_ids)
    if not touches:
        return None

    audit = {
        "id": row["id"],
        "fa_type": row["fa_type"],
        "subtype": subtype,
        "doc_ids": doc_ids,
        "query": row["query"],
        "gt_value": row["gt_value"],
        "affected_docs": "|".join(affected),
        "source_sections": source_sections,
    }

    if subtype in DERIVED_SUBTYPES:
        audit["verdict"] = "DERIVED"
        audit["evidence"] = f"subtype={subtype} is by construction a derived/computed value"
        return audit

    value, unit = parse_primary_number(row["gt_value"])
    if value is None:
        audit["verdict"] = "MANUAL_CHECK"
        audit["evidence"] = "could not parse a primary number from gt_value"
        return audit

    matched_any = False
    evidence_pieces: list[str] = []
    for doc_id in affected:
        md_path = doc_map.get(doc_id)
        if md_path is None:
            evidence_pieces.append(f"{doc_id}=NO_FILE")
            continue
        section = extract_financial_statements(md_path)
        if not section:
            evidence_pieces.append(f"{doc_id}=NO_FS_SECTION")
            continue
        found, note = matches_within_tolerance(value, unit, section)
        if found:
            matched_any = True
            evidence_pieces.append(f"{doc_id}=HIT ({note})")
        else:
            evidence_pieces.append(f"{doc_id}=MISS ({note})")

    audit["verdict"] = "AUTO_OK" if matched_any else "MANUAL_CHECK"
    audit["evidence"] = f"value={value} unit={unit} | " + " ; ".join(evidence_pieces)
    return audit


def main() -> int:
    if not GS_CSV.exists():
        print(f"ERROR: gold-standard CSV missing: {GS_CSV}", file=sys.stderr)
        return 2

    doc_map = build_doc_id_map()
    if not doc_map:
        print(f"ERROR: doc_id map empty (no .meta.json in {DATA_PROCESSED})", file=sys.stderr)
        return 2

    audits: list[dict] = []
    with GS_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            audit = audit_row(row, doc_map)
            if audit is not None:
                audits.append(audit)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "fa_type",
        "subtype",
        "doc_ids",
        "affected_docs",
        "source_sections",
        "gt_value",
        "verdict",
        "evidence",
        "query",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for a in audits:
            writer.writerow({k: a.get(k, "") for k in fieldnames})

    verdict_counts: dict[str, int] = {}
    for a in audits:
        verdict_counts[a["verdict"]] = verdict_counts.get(a["verdict"], 0) + 1

    print(f"Audited {len(audits)} affected GS entries")
    print(f"Output: {OUT_CSV.relative_to(PROJECT_ROOT)}")
    print("Verdict breakdown:")
    for verdict in ("AUTO_OK", "DERIVED", "MANUAL_CHECK"):
        print(f"  {verdict:>13}: {verdict_counts.get(verdict, 0)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
