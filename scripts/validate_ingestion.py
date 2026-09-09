"""Smoke-test for the ingestion pipeline output.

For every processed 10-K under `data/processed/`, verifies that:
  1. The `## Financial Statements` section exists and is >= MIN_ITEM8_CHARS
     characters long (guards against the TOC-stub failure mode seen in
     MSFT FY2023/FY2024 before the intra-word-whitespace fix).
  2. Its content contains at least MIN_MARKER_HITS of the canonical statement
     markers (INCOME STATEMENTS, BALANCE SHEETS, CASH FLOWS), with
     intra-word-whitespace tolerated.
  3. The raw file is not merely an iXBRL viewer wrapper (guards against the
     GOOGL FY2022 failure mode).
  4. The narrative sections (Business, Risk Factors, MD&A) are present and
     of plausible length. Item 8 alone is not enough: GOOGL FY2024 sat in
     the corpus for weeks with Risk Factors truncated to 504 chars and MD&A
     to 964 while its Financial Statements were intact at 128k, so every
     check above passed. See EVAL_DECISION_LOG.md [2026-09-06].

Exit code:
  0 -> all filings pass
  1 -> at least one filing failed a check

Usage:
    uv run python scripts/validate_ingestion.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

MIN_ITEM8_CHARS = 50_000
MIN_RAW_CHARS = 100_000
MIN_MARKER_HITS = 3

# Floors for the narrative sections, set well below the smallest value
# observed across the twelve corpus filings so that a real filing never
# trips them, but far above the few-hundred-char stubs a mis-parse leaves
# behind. Smallest observed: Business 12,084 (AMZN FY2020), Risk Factors
# 50,047 (AMZN FY2020), MD&A 15,358 (AAPL FY2024).
MIN_NARRATIVE_CHARS = {
    "Business": 8_000,
    "Risk Factors": 25_000,
    "MD&A": 10_000,
}

STATEMENT_MARKERS = {
    "INCOME/OPERATIONS": (
        r"(?:INCOME\s*STATE\s*MENTS?|"
        r"STATE\s*MENTS?\s*OF\s*INCOME|"
        r"STATE\s*MENTS?\s*OF\s*OPERATIONS)"
    ),
    "BALANCE SHEETS": r"BALANCE\s*SHEETS?",
    "CASH FLOWS": r"CASH\s*FLOWS?",
}


def _extract_section(md_text: str, heading: str) -> str:
    """Return the raw content of one `## <heading>` section, or ''."""
    inside = False
    collected: list[str] = []
    for line in md_text.splitlines():
        if line.startswith("## "):
            if inside:
                break
            if line.strip() == f"## {heading}":
                inside = True
                continue
        if inside:
            collected.append(line)
    return "\n".join(collected).strip()


def _count_markers(section_text: str) -> dict[str, int]:
    """Return a dict {marker_name: count_of_matches} on upper-cased text."""
    text_upper = section_text.upper()
    return {
        name: len(re.findall(pattern, text_upper))
        for name, pattern in STATEMENT_MARKERS.items()
    }


def _check_raw_wrapper(raw_path: Path) -> str | None:
    """Return a warning string if the raw file looks like an iXBRL viewer stub."""
    if not raw_path.exists():
        return f"raw file missing: {raw_path.relative_to(PROJECT_ROOT)}"
    size = raw_path.stat().st_size
    if size < MIN_RAW_CHARS:
        return (
            f"raw file suspiciously small ({size} chars, "
            f"expected >= {MIN_RAW_CHARS}): {raw_path.relative_to(PROJECT_ROOT)}"
        )
    head = raw_path.read_text(encoding="utf-8", errors="replace")[:2000].upper()
    if "XBRL VIEWER" in head or "IXVIEWER" in head or "PLEASE ENABLE JAVASCRIPT" in head:
        return f"raw file is an iXBRL viewer wrapper: {raw_path.relative_to(PROJECT_ROOT)}"
    return None


def _raw_path_for(md_path: Path) -> Path:
    """Map data/processed/TICKER/10K_YYYY-MM-DD.md -> data/raw/TICKER/10K_YYYY-MM-DD.txt."""
    ticker = md_path.parent.name
    stem = md_path.stem  # e.g. "10K_2023-07-27"
    return DATA_RAW / ticker / f"{stem}.txt"


def validate_filing(md_path: Path) -> list[str]:
    """Return a list of failure messages (empty list = all checks passed)."""
    failures: list[str] = []
    md_text = md_path.read_text(encoding="utf-8")
    item8 = _extract_section(md_text, "Financial Statements")

    if not item8:
        failures.append("no `## Financial Statements` section found")
    else:
        if len(item8) < MIN_ITEM8_CHARS:
            failures.append(
                f"Financial Statements section too small: {len(item8)} chars "
                f"(expected >= {MIN_ITEM8_CHARS})"
            )
        marker_hits = _count_markers(item8)
        hit_count = sum(1 for c in marker_hits.values() if c > 0)
        if hit_count < MIN_MARKER_HITS:
            details = ", ".join(f"{n}={c}" for n, c in marker_hits.items())
            failures.append(
                f"only {hit_count}/{len(STATEMENT_MARKERS)} statement markers found ({details})"
            )

    for heading, floor in MIN_NARRATIVE_CHARS.items():
        content = _extract_section(md_text, heading)
        if not content:
            failures.append(f"no `## {heading}` section found")
        elif len(content) < floor:
            failures.append(
                f"{heading} section too small: {len(content)} chars "
                f"(expected >= {floor}) — likely a mis-parsed section boundary"
            )

    raw_warn = _check_raw_wrapper(_raw_path_for(md_path))
    if raw_warn:
        failures.append(raw_warn)

    return failures


def main() -> int:
    md_files = sorted(DATA_PROCESSED.glob("*/10K_*.md"))
    if not md_files:
        print(f"No processed 10-K files found under {DATA_PROCESSED}")
        return 1

    total = len(md_files)
    ok = 0
    failed: list[tuple[Path, list[str]]] = []

    for md_path in md_files:
        rel = md_path.relative_to(PROJECT_ROOT)
        failures = validate_filing(md_path)
        if failures:
            failed.append((md_path, failures))
            print(f"[FAIL] {rel}")
            for f in failures:
                print(f"       - {f}")
        else:
            ok += 1
            print(f"[ OK ] {rel}")

    print(f"\nSummary: {ok}/{total} filings passed, {len(failed)} failed.")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
