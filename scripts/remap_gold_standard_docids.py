"""
Remap the gold standard's `doc_ids` from "the filing OF the asked fiscal year"
to "every filing that actually REPORTS that fiscal year".

Why this is needed
------------------
A Form 10-K reports prior fiscal years in comparative columns. A system that
answers "Apple's FY2023 revenue" by reading the comparative column of the
FY2024 filing and citing `(AAPL, FY2024, Income Statement)` has cited its
actual source correctly -- but the v3 gold standard expects `AAPL_2023` and
scores that as a citation error. The metric therefore measured *document
choice* rather than *ability to cite evidence* (FA-5 / NF-1).

The problem is independent of which fiscal years the corpus contains; it is
a property of how 10-K reporting works. It became visible when the corpus was
changed to alternating fiscal years (FY2020/2022/2024, see configs/base.yaml),
because then the filing of the asked year frequently is not in the corpus at
all.

Coverage windows
----------------
How many fiscal years back a given section reports. Regulation is the prior,
the probes below are the empirical confirmation on this corpus:

  item_8_income_stmt / _cash_flow / _segment   3 years   Reg S-X 3-02
      confirmed: FY2022 revenue is present in every company's FY2024 filing,
      FY2020 revenue is not; FY2020 revenue is present in the FY2022 filing.
  item_8_balance_sheet                          2 years   Reg S-X 3-01
      note: total equity IS recoverable 3 years back via the statement of
      stockholders' equity, but the balance sheet proper carries 2 years.
      The conservative window is used.
  item_7 (MD&A)                                 2 years   Item 303 Reg S-K
      confirmed: only Apple's MD&A carries a three-year net-sales table;
      MSFT/AMZN/GOOGL do not report FY2022 in their FY2024 MD&A.
  item_1 (Business)                             1 year    current year only

Output model: groups, not a flat set
------------------------------------
Each fact the item needs -- one (ticker, fiscal year) pair -- becomes one
GROUP of interchangeable filings. Citing ANY member of a group satisfies it.
A flat union would mis-score: an item needing AAPL FY2022 and FY2024 accepts
{AAPL_2022, AAPL_2024} for the first fact and {AAPL_2024} for the second, so
an answer citing only AAPL_2024 (both years read from one filing) is fully
correct, yet flat set-recall would score it 0.5. This mirrors the existing
group-aware section scoring in `citation_evaluator.py::_score_sections`.

`doc_ids` keeps the flat union for backwards compatibility and for the judge
prompt; the new `doc_id_groups` column carries the groups, `+` separating
alternatives within a group and `|` separating groups.

Writes additively: reads v3, writes v4 (DE and EN) plus a review report. The
v3 files are left untouched.

Usage:
    uv run python scripts/remap_gold_standard_docids.py
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from src.common.ingestion import download_all_filings, fiscal_year_from_metadata

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GS_DIR = PROJECT_ROOT / "data" / "gold_standard"

# Section -> how many fiscal years that section reports (including its own).
SECTION_WINDOW = {
    "item_8_income_stmt": 3,
    "item_8_cash_flow": 3,
    "item_8_segment": 3,
    "item_8_balance_sheet": 2,
    "item_7": 2,
    "item_1": 1,
}
# Issuer-specific override, empirically established (see module docstring).
SECTION_WINDOW_OVERRIDE = {("AAPL", "item_7"): 3}

# Used when an item carries no source_sections at all (the FA-Refusal stratum).
# Those items are skipped by the citation metric anyway; the mapping only keeps
# their doc_ids referring to filings that exist in the corpus.
DEFAULT_WINDOW = 3


def _window(ticker: str, section: str) -> int:
    if (ticker, section) in SECTION_WINDOW_OVERRIDE:
        return SECTION_WINDOW_OVERRIDE[(ticker, section)]
    return SECTION_WINDOW.get(section, DEFAULT_WINDOW)


def _split(value: str, sep: str = "|") -> list[str]:
    return [p.strip() for p in (value or "").split(sep) if p.strip()]


def acceptable_filings(
    ticker: str, asked_year: int, sections: list[str], corpus_years: set[int]
) -> list[str]:
    """Corpus filings of `ticker` that report `asked_year` in any of `sections`.

    Returns doc ids ("AAPL_2024"), nearest reporting year first, so the most
    natural citation is listed before the alternatives.
    """
    effective_sections = sections or [""]
    accepted = set()
    for filing_year in corpus_years:
        if filing_year < asked_year:
            continue
        for section in effective_sections:
            if filing_year - asked_year < _window(ticker, section):
                accepted.add(filing_year)
                break
    return [f"{ticker}_{y}" for y in sorted(accepted)]


def remap_row(row: dict, corpus: dict[str, set[int]]) -> tuple[list[list[str]], list[str]]:
    """Return (groups, unresolved) for one gold-standard row."""
    sections = _split(row.get("source_sections", ""))
    groups: list[list[str]] = []
    unresolved: list[str] = []

    for doc_id in _split(row.get("doc_ids", "")):
        ticker, _, year_str = doc_id.rpartition("_")
        if not ticker or not year_str.isdigit():
            unresolved.append(f"{doc_id} (unparsable)")
            continue
        candidates = acceptable_filings(
            ticker, int(year_str), sections, corpus.get(ticker, set())
        )
        if candidates:
            groups.append(candidates)
        else:
            unresolved.append(doc_id)
    return groups, unresolved


def main() -> int:
    filings = download_all_filings()
    corpus: dict[str, set[int]] = {}
    for f in filings:
        corpus.setdefault(f.metadata.ticker, set()).add(int(fiscal_year_from_metadata(f)))
    print("Korpus:", {t: sorted(y) for t, y in sorted(corpus.items())})

    report_rows = []
    stats: dict[str, int] = {"unveraendert": 0, "gemappt": 0, "klaerungsbeduerftig": 0}

    # HISTORICAL — this is the generator that built v4 FROM v3 on
    # 2026-09-05. Its input is v3 by definition. Re-running it would
    # overwrite v4 with a fresh derivation and discard every hand
    # correction since. See EVAL_DECISION_LOG.md [2026-09-06].
    for lang, src_name in (("de", "gold_standard_v3.csv"), ("en", "gold_standard_v3_en.csv")):
        src = GS_DIR / src_name
        dst = GS_DIR / src_name.replace("_v3", "_v4")
        with open(src, encoding="utf-8") as fh:
            reader = csv.DictReader(fh, delimiter=";")
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)

        if "doc_id_groups" not in fieldnames:
            fieldnames.insert(fieldnames.index("doc_ids") + 1, "doc_id_groups")

        for row in rows:
            before = _split(row.get("doc_ids", ""))
            groups, unresolved = remap_row(row, corpus)
            flat = sorted({d for g in groups for d in g})

            row["doc_id_groups"] = "|".join("+".join(g) for g in groups)
            row["doc_ids"] = "|".join(flat)

            if lang == "en":  # report once, the two files are row-aligned
                if unresolved:
                    status = "klaerungsbeduerftig"
                elif flat == sorted(before):
                    status = "unveraendert"
                else:
                    status = "gemappt"
                stats[status] += 1
                if status != "unveraendert":
                    report_rows.append({
                        "id": row["id"],
                        "fa_type": row["fa_type"],
                        "subtype": row.get("subtype", ""),
                        "status": status,
                        "source_sections": row.get("source_sections", ""),
                        "doc_ids_v3": "|".join(before),
                        "doc_ids_v4": row["doc_ids"],
                        "doc_id_groups_v4": row["doc_id_groups"],
                        "unresolved": "|".join(unresolved),
                        "query": row.get("query", "")[:100],
                    })

        with open(dst, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerows(rows)
        print(f"geschrieben: {dst.relative_to(PROJECT_ROOT)} ({len(rows)} Zeilen)")

    report = GS_DIR / "docid_remap_report.csv"
    if report_rows:
        with open(report, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(report_rows[0].keys()), delimiter=";")
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"geschrieben: {report.relative_to(PROJECT_ROOT)} ({len(report_rows)} Zeilen)")

    print("\nStatus:", {k: v for k, v in stats.items()})
    return 1 if stats["klaerungsbeduerftig"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
