import json
from unittest.mock import MagicMock, patch

import src.evaluation.custom_evaluator as custom_evaluator
from src.evaluation.custom_evaluator import _deterministic_numeric_match, evaluate_custom_metrics
from src.evaluation.gold_standard_loader import GoldStandardItem


def setup_function():
    # get_eval_llm() caches its result in this module-level global, which
    # would otherwise leak the mock from one test into the next.
    custom_evaluator._EVAL_LLM = None

def test_evaluate_custom_metrics_exact_match():
    item = GoldStandardItem(
        id=1,
        question="What was the revenue?",
        ground_truth="1.2",
        gt_unit="B",
        doc_refs="1",
        expected_answerable=True
    )
    answer = "The revenue was 1.2 billion."

    mock_llm = MagicMock()
    # First call is exact match, second is answer recall
    mock_llm.invoke.side_effect = [
        MagicMock(content=json.dumps({"score": 1.0, "rationale": "Matches perfectly."})),
        MagicMock(content=json.dumps({"score": 1.0, "rationale": "All info present."})),
    ]

    with patch("src.evaluation.custom_evaluator.get_judge_llm", return_value=mock_llm):
        result = evaluate_custom_metrics(item, answer)

    assert result.exact_match == 1.0
    assert result.answer_recall == 1.0
    assert result.refusal_accuracy == 0.0  # Not computed for answerable items
    assert mock_llm.invoke.call_count == 2

def test_evaluate_custom_metrics_refusal():
    item = GoldStandardItem(
        id=2,
        question="What is the CEO's favorite color?",
        ground_truth="",
        doc_refs="",
        expected_answerable=False
    )
    answer = "I cannot answer this based on the provided filings."

    mock_llm = MagicMock()
    # Call is refusal accuracy
    mock_llm.invoke.side_effect = [
        MagicMock(content=json.dumps({"score": 1.0, "rationale": "Correctly refused."})),
    ]

    with patch("src.evaluation.custom_evaluator.get_judge_llm", return_value=mock_llm):
        result = evaluate_custom_metrics(item, answer)

    assert result.exact_match == 0.0
    assert result.answer_recall == 0.0
    assert result.refusal_accuracy == 1.0
    assert mock_llm.invoke.call_count == 1


# ---------------------------------------------------------------------------
#  Deterministic numeric pre-check (regression tests for the R011/R031/R026
#  cases found during human judge-validation, see
#  EVAL_DECISION_LOG.md [2026-08-01])
# ---------------------------------------------------------------------------

def test_deterministic_match_million_vs_billion():
    # R011/R031: $112,390 million == $112.4 billion (same value, different unit word)
    assert _deterministic_numeric_match(
        "$112.4 billion", "GOOGL's Operating Income was $112,390 million.", "USD_billion"
    ) is True

def test_deterministic_match_percentage_point_rounding():
    # R026: 2.1pp accepted as a rounding match for GT 2.0pp
    assert _deterministic_numeric_match(
        "+2.0pp", "AAPL's gross margin changed by 2.1 percentage points.", "percent"
    ) is True

def test_deterministic_match_pp_change_gt_unit():
    # Gold standard v4 uses "pp_change" (distinct from "percent") for
    # percentage-point deltas (e.g. id 106) -- must be handled the same way.
    assert _deterministic_numeric_match(
        "+2.0pp", "AAPL's gross margin changed by 2.1 percentage points.", "pp_change"
    ) is True

def test_deterministic_match_pairwise_both_values():
    assert _deterministic_numeric_match(
        "AAPL ($93.7 billion > MSFT $88.1 billion)",
        "Apple's Net Income was $93,736 million, Microsoft's was $88,136 million.",
        "USD_billion",
    ) is True

def test_deterministic_match_inconclusive_on_mismatch():
    # A genuinely different number must NOT be asserted as a mismatch by the
    # deterministic layer -- it must return None (inconclusive), not False,
    # so the caller falls back to the LLM judge rather than auto-failing.
    result = _deterministic_numeric_match("$112.4 billion", "The value was $50 million.", "USD_billion")
    assert result is None

def test_deterministic_match_text_unit_always_inconclusive():
    assert _deterministic_numeric_match("Yes", "Yes, that is correct.", "text") is None
    assert _deterministic_numeric_match("", "N/A", "n/a") is None

def test_evaluate_custom_metrics_deterministic_shortcut_skips_llm():
    item = GoldStandardItem(
        id=3,
        question="What was the Operating Income of GOOGL in FY2024?",
        ground_truth="$112.4 billion",
        gt_unit="USD_billion",
        doc_refs="1",
        expected_answerable=True,
    )
    answer = "GOOGL's Operating Income for FY2024 was $112,390 million."

    mock_llm = MagicMock()
    with patch("src.evaluation.custom_evaluator.get_judge_llm", return_value=mock_llm):
        result = evaluate_custom_metrics(item, answer)

    assert result.exact_match == 1.0
    assert result.answer_recall == 1.0
    assert mock_llm.invoke.call_count == 0  # deterministic match short-circuits both LLM calls
