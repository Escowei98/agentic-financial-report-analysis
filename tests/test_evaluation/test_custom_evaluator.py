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


# ---------------------------------------------------------------------------
#  Type-conditioned exact_match (comparison / qualitative buckets)
# ---------------------------------------------------------------------------

def test_classify_answer_type_comparison():
    item = GoldStandardItem(
        id=10, question="q", ground_truth="gt", doc_refs="", fa_type="FA-4", gt_unit="percent"
    )
    assert custom_evaluator._classify_answer_type(item) == "comparison"

def test_classify_answer_type_qualitative():
    item = GoldStandardItem(
        id=11, question="q", ground_truth="gt", doc_refs="", fa_type="FA-3", gt_unit="text"
    )
    assert custom_evaluator._classify_answer_type(item) == "qualitative"

def test_classify_answer_type_numeric_atomic():
    item = GoldStandardItem(
        id=12, question="q", ground_truth="gt", doc_refs="", fa_type="FA-1", gt_unit="USD_billion"
    )
    assert custom_evaluator._classify_answer_type(item) == "numeric_atomic"

def test_classify_answer_type_fa4_wins_over_text_gt_unit():
    # FA-4 items whose gt_value embeds its own units (gt_unit="text", e.g.
    # gold standard ids 65/115) must still be classified "comparison", not
    # "qualitative" -- fa_type takes priority over gt_unit.
    item = GoldStandardItem(
        id=13, question="q", ground_truth="gt", doc_refs="", fa_type="FA-4", gt_unit="text"
    )
    assert custom_evaluator._classify_answer_type(item) == "comparison"

def test_evaluate_custom_metrics_comparison_reversed_verdict_not_deterministic():
    # Regression test for the entity-binding gap: the deterministic pre-check
    # matches numbers as an unordered set, so a reversed two-entity verdict
    # with the same two raw numbers present would be falsely confirmed if it
    # ran here. For "comparison" items it must never run -- both exact_match
    # and answer_recall must go through the LLM judge instead.
    item = GoldStandardItem(
        id=20,
        question="Which company had higher revenue growth?",
        ground_truth="AAPL (12% > AMZN 8%)",
        gt_unit="percent",
        doc_refs="1",
        fa_type="FA-4",
        expected_answerable=True,
    )
    # Same two raw numbers present, but the verdict is reversed.
    answer = "AMZN grew 12%, AAPL grew 8%, so AMZN had the higher growth rate."

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        MagicMock(content=json.dumps({"score": 0.0, "rationale": "Verdict reversed."})),
        MagicMock(content=json.dumps({"score": 0.5, "rationale": "Numbers present but misattributed."})),
    ]

    with patch("src.evaluation.custom_evaluator.get_judge_llm", return_value=mock_llm):
        result = evaluate_custom_metrics(item, answer)

    assert result.exact_match == 0.0
    assert mock_llm.invoke.call_count == 2  # no deterministic shortcut for comparison items
    comparison_prompt_used = mock_llm.invoke.call_args_list[0][0][0]
    assert "COMPARATIVE claim" in comparison_prompt_used

def test_evaluate_custom_metrics_qualitative_uses_qualitative_prompt():
    item = GoldStandardItem(
        id=21,
        question="What is the trend in operating margin?",
        ground_truth="Slightly rising",
        gt_unit="text",
        doc_refs="1",
        fa_type="FA-3",
        subtype="trend_qualitative",
        expected_answerable=True,
    )
    answer = "Operating margin increased modestly year over year."

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        MagicMock(content=json.dumps({"score": 1.0, "rationale": "Same trend."})),
        MagicMock(content=json.dumps({"score": 1.0, "rationale": "All info present."})),
    ]

    with patch("src.evaluation.custom_evaluator.get_judge_llm", return_value=mock_llm):
        result = evaluate_custom_metrics(item, answer)

    assert result.exact_match == 1.0
    qualitative_prompt_used = mock_llm.invoke.call_args_list[0][0][0]
    assert "QUALITATIVE or categorical claim" in qualitative_prompt_used

def test_evaluate_custom_metrics_comparison_with_text_gt_unit_no_leak():
    # e.g. gold standard ids 65/115: FA-4 with gt_unit="text" because the
    # gt_value embeds its own units. gt_unit must not be leaked into the
    # judge prompt as a literal "text" suffix.
    item = GoldStandardItem(
        id=22,
        question="Which company had higher operating cash flow?",
        ground_truth="AAPL (~$11.7B > MSFT ~$10.7B)",
        gt_unit="text",
        doc_refs="1",
        fa_type="FA-4",
        expected_answerable=True,
    )
    answer = "AAPL had higher operating cash flow at ~$11.7B vs MSFT's ~$10.7B."

    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [
        MagicMock(content=json.dumps({"score": 1.0, "rationale": "Verdict and figures match."})),
        MagicMock(content=json.dumps({"score": 1.0, "rationale": "All info present."})),
    ]

    with patch("src.evaluation.custom_evaluator.get_judge_llm", return_value=mock_llm):
        result = evaluate_custom_metrics(item, answer)

    assert result.exact_match == 1.0
    comparison_prompt_used = mock_llm.invoke.call_args_list[0][0][0]
    gt_line = next(
        l for l in comparison_prompt_used.splitlines() if l.startswith("**Ground Truth:**")
    )
    assert gt_line.strip() == "**Ground Truth:** AAPL (~$11.7B > MSFT ~$10.7B)"
