"""
Which fiscal years a Form 10-K actually reports -- the corpus description
the systems are given, kept identical to the one the judge is given.

WHY THIS EXISTS
---------------
A 10-K reports its own fiscal year AND prior years in comparative columns:
three years for the income statement, cash-flow statement and segment note
(Reg S-X 3-02), two for the balance sheet (Reg S-X 3-01). The corpus is built
from alternating fiscal years (FY2020/2022/2024, see configs/base.yaml) on
purpose, so FY2023 exists in the corpus only as a comparative column of the
FY2024 filing -- and 22 answerable gold-standard items ask about such a year.

If only the judge knew this, a system that declines an FY2023 question as
out of scope would be scored against a corpus description it was never
given. That would be a property of the instrumentation, not of the
architectures. This module is the single description both sides use.

WHAT IS AND IS NOT SAID
-----------------------
The systems learn WHERE a fiscal year is reported, never what it contains.
The same sentence goes to every system in its own idiom: the filing overview
(S2/S3/S4 via `list_filings`, S3/S4 also in the inlined filing headers), the
`search_section` miss message (S2), the chunk-prefix explanation (S1), and
the non-answerability convention (all four).
"""

from __future__ import annotations

from typing import Iterable

# Coverage windows, in fiscal years including the filing's own, as
# prescribed by Regulation S-X; the conservative window is used where a
# statement's coverage varies by company (MD&A).
STATEMENT_WINDOW_YEARS = 3   # income statement, cash flow, segment note
BALANCE_SHEET_WINDOW_YEARS = 2


def comparative_years(filing_year: int | str) -> list[int]:
    """Fiscal years the filing carries as comparative columns, most recent first.

    Excludes the filing's own year. For a FY2024 filing: [2023, 2022].
    """
    fy = int(filing_year)
    return [fy - offset for offset in range(1, STATEMENT_WINDOW_YEARS)]


def balance_sheet_years(filing_year: int | str) -> list[int]:
    """Fiscal years the balance sheet covers, including the filing's own."""
    fy = int(filing_year)
    return [fy - offset for offset in range(BALANCE_SHEET_WINDOW_YEARS)]


def describe_coverage(filing_year: int | str) -> str:
    """One line for a filing overview, e.g.

        Reports FY2024; comparative columns: FY2023, FY2022 (balance sheet: FY2023 only)
    """
    fy = int(filing_year)
    comps = comparative_years(fy)
    bs_prior = [y for y in balance_sheet_years(fy) if y != fy]
    comp_text = ", ".join(f"FY{y}" for y in comps)
    bs_text = ", ".join(f"FY{y}" for y in bs_prior)
    return (
        f"Reports FY{fy}; comparative columns: {comp_text} "
        f"(balance sheet: {bs_text} only)"
    )


def filings_carrying_year(
    requested_year: int | str, available_filing_years: Iterable[int | str]
) -> list[int]:
    """Filing years in the corpus that report `requested_year`, own year first.

    Empty when no filing reaches the year -- that is the genuine
    out-of-corpus case.
    """
    target = int(requested_year)
    carrying = [
        int(fy) for fy in available_filing_years
        if target == int(fy) or target in comparative_years(fy)
    ]
    return sorted(set(carrying))


# The sentence every system gets, inside the shared non-answerability
# convention. Brace-free (see reasoning_chain_convention.py for why).
COMPARATIVE_COLUMNS_NOTE = (
    "A filing reports its own fiscal year and carries the two prior fiscal "
    "years in the comparative columns of its financial statements (the "
    "balance sheet: one prior year), so a fiscal year without a filing of its "
    "own may still be reported in a later filing."
)
