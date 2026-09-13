"""Faithfulness of the final answer against the corpus at its cited loci.

The property under test is symmetry: the evidence an answer is checked
against depends only on what the answer cites, never on which system wrote
it or how it found the passage. The second property is the exclusion rule:
an answer with nothing to check stays None with a stated reason, so that a
missing citation is not silently scored as an unfaithful answer.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.evaluation.evidence_store import EvidenceStore, LookupResult
from src.evaluation.locus_faithfulness import (
    EXCLUDED_JUDGE_FAILED,
    EXCLUDED_NO_LOCUS,
    EXCLUDED_NO_PASSAGES,
    EXCLUDED_NOT_ANSWERABLE,
    LocusFaithfulnessResult,
    evaluate_locus_faithfulness_batch,
    gather_passages,
    parse_loci,
    summarise,
)


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
    _doc("MSFT", "2024", "Financial Statements",
         "Total revenue 245,122 for the year ended June 30, 2024."),
]


@pytest.fixture
def store():
    with patch("src.evaluation.evidence_store.load_or_build_documents", return_value=CORPUS):
        yield EvidenceStore([])


class TestParseLoci:
    def test_round_brackets_per_the_convention(self):
        loci = parse_loci("Net sales were $391,035M (AAPL, FY2024, Income Statement).")
        assert [(l.doc_id, l.section_text) for l in loci] == [
            ("AAPL_2024", "Income Statement")
        ]

    def test_square_brackets_as_s1_sometimes_writes_them(self):
        """A locus in square brackets is still a locus; S1 must not lose
        a checkable answer to a punctuation choice."""
        loci = parse_loci("Per the [AAPL, FY2024, Financial Statements] chunk ...")
        assert [l.doc_id for l in loci] == ["AAPL_2024"]

    def test_duplicates_collapse_and_order_is_kept(self):
        text = ("(MSFT, FY2024, Income Statement) ... (AAPL, FY2024, MD&A) ... "
                "(msft, fy2024, income statement)")
        assert [l.doc_id for l in parse_loci(text)] == ["MSFT_2024", "AAPL_2024"]

    def test_no_citation_means_no_loci(self):
        assert parse_loci("Apple's revenue was $391 billion in 2024.") == []
        assert parse_loci("") == []


class TestGatherPassages:
    def test_passages_come_from_the_cited_locus_only(self, store):
        answer = "Total net sales were 391,035 (AAPL, FY2024, Income Statement)."
        passages, records = gather_passages(answer, parse_loci(answer), store)
        assert passages and all("245,122" not in p for p in passages)
        assert records[0]["doc_exists"] and records[0]["n_passages"] == len(passages)

    def test_a_fabricated_filing_yields_no_passages_but_a_record(self, store):
        """A year the corpus does not hold (the B6 case of groundedness)."""
        answer = "Revenue was 365,817 (AAPL, FY2021, Income Statement)."
        passages, records = gather_passages(answer, parse_loci(answer), store)
        assert passages == []
        assert records == [{
            "doc_id": "AAPL_2021", "section_text": "Income Statement",
            "doc_exists": False, "section_resolved": True, "n_passages": 0,
        }]

    def test_the_same_passage_cited_twice_is_shown_once(self, store):
        answer = ("391,035 (AAPL, FY2024, Income Statement) and again "
                  "(AAPL, FY2024, Financial Statements).")
        passages, _ = gather_passages(answer, parse_loci(answer), store)
        assert len(passages) == len(set(passages))


class TestBatch:
    """The judge is mocked; what is checked is who gets judged and why not."""

    def _run(self, answers, answerable, store, scores):
        with patch(
            "src.evaluation.locus_faithfulness._judge_faithfulness",
            return_value=scores,
        ) as judge:
            results = evaluate_locus_faithfulness_batch(
                ["q"] * len(answers), answers, answerable, store,
            )
        return results, judge

    def test_refusal_items_are_never_judged(self, store):
        results, judge = self._run(
            ["Not in the corpus (AAPL, FY2024, Income Statement)."], [False], store, [],
        )
        judge.assert_not_called()
        assert results[0].score is None
        assert results[0].excluded == EXCLUDED_NOT_ANSWERABLE

    def test_an_answer_without_a_locus_is_none_not_zero(self, store):
        results, judge = self._run(
            ["The provided context does not contain Apple's revenue."], [True], store, [],
        )
        judge.assert_not_called()
        assert results[0].score is None
        assert results[0].excluded == EXCLUDED_NO_LOCUS

    def test_a_locus_outside_the_corpus_is_none_with_its_own_reason(self, store):
        results, judge = self._run(
            ["Revenue 365,817 (AAPL, FY2021, Income Statement)."], [True], store, [],
        )
        judge.assert_not_called()
        assert results[0].excluded == EXCLUDED_NO_PASSAGES
        assert results[0].loci[0]["doc_exists"] is False

    def test_the_reasoning_block_is_split_off_before_judging(self, store):
        answer = (
            "Net sales were 391,035 (AAPL, FY2024, Income Statement).\n\n"
            "## Reasoning\n1. [E] (MSFT, FY2024, Income Statement) revenue 245,122\n"
        )
        results, judge = self._run([answer], [True], store, [0.5])
        _, judged_answers, judged_contexts = judge.call_args[0]
        assert "## Reasoning" not in judged_answers[0]
        # The chain's MSFT locus does not leak into the evidence set.
        assert all("245,122" not in p for p in judged_contexts[0])
        assert results[0].score == 0.5
        assert results[0].excluded is None

    def test_results_stay_aligned_when_some_items_are_skipped(self, store):
        answers = [
            "No citation here.",
            "391,035 (AAPL, FY2024, Income Statement).",
            "Refused.",
            "245,122 (MSFT, FY2024, Income Statement).",
        ]
        results, judge = self._run(answers, [True, True, False, True], store, [1.0, 0.25])
        assert [r.score for r in results] == [None, 1.0, None, 0.25]
        assert results[0].excluded == EXCLUDED_NO_LOCUS
        assert results[2].excluded == EXCLUDED_NOT_ANSWERABLE
        assert len(judge.call_args[0][1]) == 2

    def test_a_nan_from_the_judge_is_recorded_as_a_failure(self, store):
        results, _ = self._run(
            ["391,035 (AAPL, FY2024, Income Statement)."], [True], store, [None],
        )
        assert results[0].score is None
        assert results[0].excluded == EXCLUDED_JUDGE_FAILED

    def test_misaligned_inputs_are_rejected(self, store):
        with pytest.raises(ValueError):
            evaluate_locus_faithfulness_batch(["q"], ["a", "b"], [True], store)

    def test_identical_answers_get_identical_evidence_regardless_of_origin(self, store):
        """The symmetry property: evidence depends on the citation alone."""
        answer = "Total net sales were 391,035 (AAPL, FY2024, Income Statement)."
        _, judge = self._run([answer, answer], [True, True], store, [1.0, 1.0])
        _, _, contexts = judge.call_args[0]
        assert contexts[0] == contexts[1]


class TestSummarise:
    def test_mean_coverage_and_reasons_over_the_answerable_stratum(self):
        rows = [
            LocusFaithfulnessResult(score=1.0).to_dict(),
            LocusFaithfulnessResult(score=0.5).to_dict(),
            LocusFaithfulnessResult(excluded=EXCLUDED_NO_LOCUS).to_dict(),
            LocusFaithfulnessResult(excluded=EXCLUDED_NOT_ANSWERABLE).to_dict(),
        ]
        s = summarise(rows)
        assert s["mean"] == 0.75
        assert s["n_scored"] == 2
        assert s["n_answerable"] == 3
        assert s["coverage"] == round(2 / 3, 4)
        assert s["excluded"] == {EXCLUDED_NO_LOCUS: 1}

    def test_empty_input(self):
        assert summarise([])["mean"] is None
        assert summarise([])["coverage"] is None


class TestToDict:
    def test_shape_matches_what_the_report_tooling_reads(self):
        d = LocusFaithfulnessResult(score=0.33333, loci=[{"doc_id": "AAPL_2024"}], n_passages=3).to_dict()
        assert d == {
            "score": 0.3333, "excluded": None, "n_loci": 1, "n_passages": 3,
            "loci": [{"doc_id": "AAPL_2024"}],
        }


def test_store_mock_shape_matches_the_real_lookup_signature():
    """gather_passages calls lookup(doc_id, section_text, claim, max_passages=)."""
    store = MagicMock(spec=EvidenceStore)
    store.lookup.return_value = LookupResult(passages=["p"])
    passages, records = gather_passages(
        "x", parse_loci("(AAPL, FY2024, MD&A)"), store, max_passages_per_locus=2,
    )
    store.lookup.assert_called_once_with("AAPL_2024", "MD&A", "x", max_passages=2)
    assert passages == ["p"] and records[0]["n_passages"] == 1
