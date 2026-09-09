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

Until 2026-09-09 only the judge knew this: `custom_evaluator._corpus_scope_sentence`
tells it that comparative columns are in scope, so it would not score a
correct FY2023 answer as a hallucination. The systems were told the opposite,
implicitly: `list_filings` listed "AAPL (FY2024)", `search_section` filtered
on the filing year and returned "No chunks found" for FY2023, and the
non-answerability convention asked them to name "which fiscal years" are
available. All four systems then declined FY2023 questions as out of scope
(smoke test 2026-09-09, id 5: 4 of 4; pilot ids 20/24: S2 4 of 4, S4 3 of 4).

That is not a property of the architectures; it is a corpus description that
differs between the measuring instrument and the objects measured. This module
is the single description both sides use.

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

# Coverage windows, in fiscal years including the filing's own. Regulation is
# the prior, `scripts/remap_gold_standard_docids.py` holds the empirical
# confirmation on this corpus; the conservative window is used where a
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
