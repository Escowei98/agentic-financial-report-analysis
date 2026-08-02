import json
from unittest.mock import MagicMock, patch

import src.evaluation.citation_evaluator as citation_evaluator
from src.evaluation.citation_evaluator import (
    _map_section_text,
    _parse_freetext_citations,
    _parse_structured_citations,
    evaluate_citation_accuracy,
)
from src.evaluation.gold_standard_loader import GoldStandardItem


def setup_function():
    # get_eval_llm() caches its result in this module-level global, which
    # would otherwise leak the mock from one test into the next.
    citation_evaluator._EVAL_LLM = None


def _item(**overrides) -> GoldStandardItem:
    defaults = dict(
        id=1,
        question="What was AAPL's revenue in FY2024?",
        ground_truth="$391.0 billion",
        doc_refs="AAPL_2024",
        doc_ids=["AAPL_2024"],
        source_sections=["item_8_income_stmt"],
        expected_answerable=True,
    )
    defaults.update(overrides)
    return GoldStandardItem(**defaults)


# ---------------------------------------------------------------------------
#  Deterministic path (strict, parsable citation)
# ---------------------------------------------------------------------------

def test_strict_citation_exact_match_no_llm_call():
    item = _item()
    answer = (
        "AAPL's FY2024 total net sales were $391.0 billion "
        "(AAPL, FY2024, Income Statement)."
    )

    mock_llm = MagicMock()
    with patch("src.evaluation.citation_evaluator.get_judge_llm", return_value=mock_llm):
        result = evaluate_citation_accuracy(item, answer)

    assert result.method == "deterministic"
    assert result.citation_accuracy == 1.0
    assert result.doc_f1 == 1.0
    assert result.section_f1 == 1.0
    assert mock_llm.invoke.call_count == 0


def test_strict_citation_generic_financial_statements_matches_income_stmt():
    item = _item()
    answer = "AAPL's FY2024 total net sales were $391.0 billion (AAPL, FY2024, Financial Statements)."

    mock_llm = MagicMock()
    with patch("src.evaluation.citation_evaluator.get_judge_llm", return_value=mock_llm):
        result = evaluate_citation_accuracy(item, answer)

    assert result.method == "deterministic"
    assert result.section_f1 == 1.0
    assert mock_llm.invoke.call_count == 0


def test_strict_citation_wrong_doc_and_section_scores_zero():
    item = _item()
    answer = "Microsoft's FY2023 revenue was $211.9 billion (MSFT, FY2023, Business)."

    mock_llm = MagicMock()
    with patch("src.evaluation.citation_evaluator.get_judge_llm", return_value=mock_llm):
        result = evaluate_citation_accuracy(item, answer)

    assert result.method == "deterministic"
    assert result.doc_f1 == 0.0
    assert result.section_f1 == 0.0
    assert result.citation_accuracy == 0.0
    assert mock_llm.invoke.call_count == 0


def test_multi_doc_citation_partial_coverage():
    item = _item(
        doc_ids=["AAPL_2024", "MSFT_2024"],
        source_sections=["item_8_income_stmt"],
    )
    # Only cites one of the two expected companies.
    answer = "AAPL's FY2024 total net sales were $391.0 billion (AAPL, FY2024, Income Statement)."

    mock_llm = MagicMock()
    with patch("src.evaluation.citation_evaluator.get_judge_llm", return_value=mock_llm):
        result = evaluate_citation_accuracy(item, answer)

    assert result.method == "deterministic"
    assert result.doc_recall == 0.5
    assert result.doc_precision == 1.0
    assert mock_llm.invoke.call_count == 0


# ---------------------------------------------------------------------------
#  Free-text tier (second deterministic pass, no brackets needed)
# ---------------------------------------------------------------------------

def test_freetext_citation_matched_by_second_deterministic_tier_no_llm_call():
    # Prose citation (company name + fiscal year + section keyword, no
    # brackets) is now caught by the loose sentence-level parser instead of
    # falling back to the judge.
    item = _item()
    answer = "Based on Apple's fiscal 2024 income statement, total net sales were $391.0 billion."

    mock_llm = MagicMock()
    with patch("src.evaluation.citation_evaluator.get_judge_llm", return_value=mock_llm):
        result = evaluate_citation_accuracy(item, answer)

    assert result.method == "deterministic"
    assert result.citation_accuracy == 1.0
    assert mock_llm.invoke.call_count == 0


def test_freetext_citation_company_name_without_ticker():
    # Full company name (no ticker, no "FY" prefix) still resolves to the
    # right doc_id via the company-name -> ticker map.
    item = _item(doc_ids=["MSFT_2024"], source_sections=["item_1"])
    answer = "Microsoft reported strong headcount growth in 2024, per its business overview."

    mock_llm = MagicMock()
    with patch("src.evaluation.citation_evaluator.get_judge_llm", return_value=mock_llm):
        result = evaluate_citation_accuracy(item, answer)

    assert result.method == "deterministic"
    assert result.doc_f1 == 1.0
    assert result.section_f1 == 1.0
    assert mock_llm.invoke.call_count == 0


def test_freetext_citation_no_year_falls_through_to_judge():
    # Company is named but no fiscal year appears anywhere near it -> the
    # loose parser correctly declines to guess a doc_id, so this still
    # needs the judge.
    item = _item()
    answer = "Apple's income statement shows strong total net sales."

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content=json.dumps({
            "rationale": "Company is named but no fiscal year is specified.",
            "doc_match_score": 0.5,
            "section_match_score": 1.0,
        })
    )
    with patch("src.evaluation.citation_evaluator.get_judge_llm", return_value=mock_llm):
        result = evaluate_citation_accuracy(item, answer)

    assert result.method == "llm_judge"
    assert mock_llm.invoke.call_count == 1


# ---------------------------------------------------------------------------
#  LLM-judge fallback (source genuinely unidentifiable via keyword matching)
# ---------------------------------------------------------------------------

def test_freetext_citation_judge_partial_score():
    item = _item()
    answer = "Total net sales for the period were $391.0 billion."

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content=json.dumps({
            "rationale": "No company, year, or section is identifiable.",
            "doc_match_score": 0.0,
            "section_match_score": 0.0,
        })
    )

    with patch("src.evaluation.citation_evaluator.get_judge_llm", return_value=mock_llm):
        result = evaluate_citation_accuracy(item, answer)

    assert result.method == "llm_judge"
    assert result.citation_accuracy == 0.0
    assert mock_llm.invoke.call_count == 1


# ---------------------------------------------------------------------------
#  FA-Refusal: skipped, never errors, never calls the judge
# ---------------------------------------------------------------------------

def test_refusal_item_is_skipped():
    item = _item(
        question="What is the CEO's favorite color?",
        ground_truth="",
        doc_refs="",
        doc_ids=[],
        source_sections=[],
        expected_answerable=False,
    )
    answer = "I cannot answer this based on the provided filings."

    mock_llm = MagicMock()
    with patch("src.evaluation.citation_evaluator.get_judge_llm", return_value=mock_llm):
        result = evaluate_citation_accuracy(item, answer)

    assert result.method == "skipped"
    assert result.citation_accuracy == 0.0
    assert mock_llm.invoke.call_count == 0


# ---------------------------------------------------------------------------
#  Section-text mapping helper
# ---------------------------------------------------------------------------

def test_map_section_text_specific_wins_over_generic():
    assert _map_section_text("Financial Statements - Balance Sheet") == frozenset({"item_8_balance_sheet"})
    assert _map_section_text("MD&A") == frozenset({"item_7"})
    assert _map_section_text("Business") == frozenset({"item_1"})


def test_map_section_text_unrecognized_returns_empty():
    assert _map_section_text("Executive Compensation") == frozenset()


def test_parse_structured_citations_multiple():
    answer = "(AAPL, FY2024, Business) and also (MSFT, FY2023, MD&A) discuss this."
    parsed = _parse_structured_citations(answer)
    assert len(parsed) == 2
    assert parsed[0]["doc_id"] == "AAPL_2024"
    assert parsed[1]["doc_id"] == "MSFT_2023"


# ---------------------------------------------------------------------------
#  Free-text sentence-level parser (unit tests)
# ---------------------------------------------------------------------------

def test_parse_freetext_citations_two_sentences_two_companies():
    answer = (
        "Apple's Net Income for FY2024 was $93,736 million, per its income statement. "
        "Microsoft's Net Income for FY2024 was $88,136 million, per its income statement."
    )
    parsed = _parse_freetext_citations(answer)
    assert len(parsed) == 2
    assert {p["doc_id"] for p in parsed} == {"AAPL_2024", "MSFT_2024"}
    assert all(p["section_ids"] == frozenset({"item_8_income_stmt"}) for p in parsed)


def test_parse_freetext_citations_requires_both_company_and_year():
    assert _parse_freetext_citations("Apple makes phones.") == []
    assert _parse_freetext_citations("Revenue grew in 2024.") == []


def test_parse_freetext_citations_ticker_mention_without_full_name():
    parsed = _parse_freetext_citations("AMZN's FY2024 segment data shows AWS growth.")
    assert len(parsed) == 1
    assert parsed[0]["doc_id"] == "AMZN_2024"
    assert parsed[0]["section_ids"] == frozenset({"item_8_segment"})


def test_strict_citation_tolerates_stray_braces_around_ticker():
    # Observed in a real S1 (rag_monolith) run: Gemini echoes literal brace
    # punctuation around the ticker specifically, e.g. "({AAPL}, FY2024, MD&A)",
    # even though the rendered prompt instruction itself uses plain "{TICKER}"
    # text (single, unescaped braces). The parser must still recognize this
    # as a valid structured citation rather than falling back to the judge.
    item = _item(source_sections=["item_7"])
    answer = "Apple's Total net sales in FY2024 was $391,035 million ({AAPL}, FY2024, MD&A)."

    mock_llm = MagicMock()
    with patch("src.evaluation.citation_evaluator.get_judge_llm", return_value=mock_llm):
        result = evaluate_citation_accuracy(item, answer)

    assert result.method == "deterministic"
    assert result.citation_accuracy == 1.0
    assert mock_llm.invoke.call_count == 0
