"""Locus lookup for the groundedness dimension.

What is being pinned here is a fairness property, not a convenience: every
system's steps must be checked against passages fetched by the same
procedure, in the same quantity, from the same corpus. The moment the
evidence depends on how the system found it, groundedness starts measuring
retrieval again -- which is the coupling that produced the 2.75-point
differential bias in the retired instrument.
"""

from unittest.mock import patch

import pytest
from langchain_core.documents import Document

from src.evaluation.evidence_store import EvidenceStore


def _doc(ticker, year, section, text):
    return Document(
        page_content=text,
        metadata={"ticker": ticker, "fiscal_year": year, "section_name": section},
    )


CORPUS = [
    _doc("AAPL", "2024", "Financial Statements",
         "Total net sales 391,035 for fiscal 2024 compared with 383,285."),
    _doc("AAPL", "2024", "Financial Statements",
         "Cost of sales 210,352 and gross margin 180,683 for fiscal 2024."),
    _doc("AAPL", "2024", "Risk Factors",
         "The Company faces risks from supply chain concentration."),
    _doc("MSFT", "2024", "Financial Statements",
         "Total revenue 245,122 for the year ended June 30, 2024."),
]


@pytest.fixture
def store():
    with patch("src.evaluation.evidence_store.load_or_build_documents", return_value=CORPUS):
        yield EvidenceStore([])


class TestLookup:
    def test_a_filing_outside_the_corpus_is_reported_not_guessed(self, store):
        """This is the B6 case, settled without a judge call."""
        result = store.lookup("TSLA_2024", "Income Statement", "Revenue was 96,773.")
        assert not result.doc_exists
        assert result.passages == []

    def test_the_cited_section_scopes_the_search(self, store):
        result = store.lookup("AAPL_2024", "Risk Factors", "Supply chain concentration.")
        assert result.section_resolved
        assert all("supply chain" in p.lower() for p in result.passages)

    def test_the_figure_in_the_claim_ranks_its_passage_first(self, store):
        """Word overlap alone ranks on boilerplate; the number is the signal."""
        result = store.lookup(
            "AAPL_2024", "Income Statement", "Total net sales were 391,035 million."
        )
        assert "391,035" in result.passages[0]

    def test_an_unresolvable_section_widens_instead_of_failing(self, store):
        """Section-name drift is citation vocabulary, measured elsewhere.

        Failing the step here would count the same error twice and would
        penalise precisely the systems that cite more specifically.
        """
        result = store.lookup("AAPL_2024", "Schedule II", "Total net sales 391,035.")
        assert result.doc_exists
        assert not result.section_resolved
        assert result.passages

    def test_a_section_absent_from_that_filing_also_widens(self, store):
        result = store.lookup("MSFT_2024", "Risk Factors", "Total revenue 245,122.")
        assert result.doc_exists
        assert not result.section_resolved
        assert "245,122" in result.passages[0]

    def test_the_passage_budget_is_capped(self, store):
        result = store.lookup(
            "AAPL_2024", "Financial Statements", "Total net sales.", max_passages=1
        )
        assert len(result.passages) == 1


class TestSymmetry:
    def test_lookup_does_not_depend_on_who_is_asking(self, store):
        """There is no system parameter, and there must not be one."""
        first = store.lookup("AAPL_2024", "Income Statement", "Total net sales 391,035.")
        second = store.lookup("AAPL_2024", "Income Statement", "Total net sales 391,035.")
        assert first.passages == second.passages

    def test_every_corpus_filing_is_addressable(self, store):
        assert store.doc_ids == {"AAPL_2024", "MSFT_2024"}
