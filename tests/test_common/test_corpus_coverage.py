"""
Unit tests for src/common/corpus_coverage.py -- the one corpus description
the systems and the judge share.
"""

from src.common.corpus_coverage import (
    COMPARATIVE_COLUMNS_NOTE,
    balance_sheet_years,
    comparative_years,
    describe_coverage,
    filings_carrying_year,
)


class TestWindows:
    def test_statements_carry_two_prior_years(self):
        assert comparative_years(2024) == [2023, 2022]
        assert comparative_years("2020") == [2019, 2018]

    def test_balance_sheet_carries_one_prior_year(self):
        assert balance_sheet_years(2024) == [2024, 2023]

    def test_describe_coverage_reads_like_the_overview_line(self):
        assert describe_coverage(2024) == (
            "Reports FY2024; comparative columns: FY2023, FY2022 "
            "(balance sheet: FY2023 only)"
        )


class TestFilingsCarryingYear:
    """The alternating corpus FY2020/2022/2024, as in configs/base.yaml."""

    CORPUS = ("2020", "2022", "2024")

    def test_a_filing_year_is_carried_by_itself_and_the_next_filing(self):
        # FY2022 is the FY2022 filing's own year and a comparative column of FY2024.
        assert filings_carrying_year(2022, self.CORPUS) == [2022, 2024]

    def test_a_gap_year_is_carried_by_the_following_filing_only(self):
        assert filings_carrying_year(2023, self.CORPUS) == [2024]
        assert filings_carrying_year(2021, self.CORPUS) == [2022]

    def test_years_before_the_first_window_are_out_of_corpus(self):
        assert filings_carrying_year(2017, self.CORPUS) == []

    def test_years_after_the_last_filing_are_out_of_corpus(self):
        assert filings_carrying_year(2025, self.CORPUS) == []


class TestSharedNote:
    def test_note_is_brace_free_for_the_templated_prompts(self):
        assert "{" not in COMPARATIVE_COLUMNS_NOTE
        assert "}" not in COMPARATIVE_COLUMNS_NOTE
