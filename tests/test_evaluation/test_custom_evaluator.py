import json
import re
from unittest.mock import MagicMock, patch

import pytest

import src.evaluation.custom_evaluator as custom_evaluator
from src.evaluation.custom_evaluator import (
    _deterministic_numeric_match,
    _extract_numbers,
    _normalize_to_base,
    evaluate_custom_metrics,
)
from src.evaluation.gold_standard_loader import GoldStandardItem


@pytest.fixture(autouse=True)
def _reset_judge_singleton():
    # get_eval_llm() caches its result in this module-level global, which would
    # otherwise leak the mock from one test into the next. An autouse fixture
    # rather than setup_function: the latter does not run for methods inside a
    # test class, so class-based tests silently reused an exhausted mock and
    # failed with StopIteration instead of the assertion they were written for.
    custom_evaluator._EVAL_LLM = None
    yield
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
        # Third call is over_refusal: the item is answerable and did not score
        # a full 1.0, so the control metric asks whether the system declined.
        MagicMock(content=json.dumps({"score": 0.0, "rationale": "Attempted an answer."})),
    ]

    with patch("src.evaluation.custom_evaluator.get_judge_llm", return_value=mock_llm):
        result = evaluate_custom_metrics(item, answer)

    assert result.exact_match == 0.0
    # exact_match + answer_recall both went to the judge (no deterministic
    # shortcut for comparison items), plus the over-refusal control call.
    assert mock_llm.invoke.call_count == 3
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


# ---------------------------------------------------------------------------
#  False-positive guard: a number in the *answer* only counts as a candidate
#  when it carries an explicit unit signal. Without this, every figure in a
#  long answer is a chance for an accidental confirmation, which favours the
#  systems that write longer answers.
# ---------------------------------------------------------------------------

def test_bare_number_in_answer_is_inconclusive_for_currency_units():
    # "391" here is a share count, not a dollar amount; under the old
    # lenient reading it would have been read as $391 billion and confirmed.
    assert _deterministic_numeric_match(
        "$391.0 billion",
        "Revenue rose over the period. The company also reported 391 in another line item.",
        "USD_billion",
    ) is None

def test_currency_marker_alone_is_enough_for_a_bare_number():
    assert _deterministic_numeric_match(
        "$391.0 billion", "Total net sales were $391.0 for the fiscal year.", "USD_billion"
    ) is True

def test_magnitude_word_still_matches_without_currency_marker():
    assert _deterministic_numeric_match(
        "$112.4 billion", "Operating income came to 112,390 million.", "USD_billion"
    ) is True

def test_bare_number_in_answer_is_inconclusive_for_percent_units():
    # A bare "24.3" (e.g. a dollar figure or an item number) must not
    # confirm a 24.3% ground truth.
    assert _deterministic_numeric_match(
        "24.3", "The segment contributed 24.3 to the consolidated total.", "percent"
    ) is None

def test_percent_marker_in_answer_still_matches():
    assert _deterministic_numeric_match(
        "24.3", "The gross margin was 24.3%.", "percent"
    ) is True

def test_count_unit_is_never_decided_deterministically():
    # gold standard v4 id 110 has gt_value "2" — a digit that shows up in
    # almost any long answer, so counts always go to the LLM judge.
    assert _deterministic_numeric_match(
        "2", "The company operates 2 reportable segments.", "count"
    ) is None

def test_long_answer_does_not_accumulate_accidental_candidates():
    """The point of the guard: answer length must not raise the hit rate."""
    long_answer = " ".join(
        f"Line item {i} shows a value of {i * 3}." for i in range(1, 60)
    )
    assert _deterministic_numeric_match(
        "$150.0 billion", long_answer, "USD_billion"
    ) is None


class TestExtractNumbers:
    def test_currency_prefix_is_recorded(self):
        assert _extract_numbers("$391.0")[0] == (391.0, None, True)

    def test_currency_suffix_is_recorded(self):
        assert _extract_numbers("391 dollars")[0] == (391.0, None, True)
        assert _extract_numbers("391 USD")[0] == (391.0, None, True)

    def test_bare_number_has_no_currency_signal(self):
        assert _extract_numbers("391")[0] == (391.0, None, False)

    def test_magnitude_word_is_still_extracted(self):
        assert _extract_numbers("$112.4 billion")[0] == (112.4, "billion", True)


class TestNormalizeToBaseStrictness:
    def test_ground_truth_side_keeps_the_lenient_reading(self):
        # strict=False (the default) is what ground-truth strings are parsed
        # with: they are authored at gt_unit's scale.
        assert _normalize_to_base(112.4, None, "usd_billion") == pytest.approx(112.4e9)
        assert _normalize_to_base(24.3, None, "percent") == pytest.approx(24.3)

    def test_answer_side_rejects_bare_numbers(self):
        assert _normalize_to_base(112.4, None, "usd_billion", strict=True) is None
        assert _normalize_to_base(24.3, None, "percent", strict=True) is None
        assert _normalize_to_base(2.4, None, "usd_per_share", strict=True) is None

    def test_answer_side_accepts_explicit_unit_signals(self):
        assert _normalize_to_base(112.4, None, "usd_billion", has_currency=True, strict=True) == pytest.approx(112.4e9)
        assert _normalize_to_base(112.4, "billion", "usd_billion", strict=True) == pytest.approx(112.4e9)
        assert _normalize_to_base(24.3, "%", "percent", strict=True) == pytest.approx(24.3)

    def test_incompatible_units_stay_rejected_in_both_modes(self):
        assert _normalize_to_base(24.3, "%", "usd_billion") is None
        assert _normalize_to_base(24.3, "%", "usd_billion", strict=True) is None
        assert _normalize_to_base(112.4, "billion", "percent", strict=True) is None


class TestRefusalPromptCorpusScope:
    """The refusal prompt used to hard-code "FY2022, FY2023 and FY2024 ONLY".
    The corpus moved to FY2020/FY2022/FY2024 on 2026-09-05 and the prompt did
    not follow, so the judge rewarded a refusal on a question the corpus could
    answer — across the whole 30-item refusal stratum. It is derived from
    configs/base.yaml now. See EVAL_DECISION_LOG.md [2026-09-06].
    """

    def test_scope_names_the_configured_fiscal_years(self):
        from src.common.config import load_config
        from src.evaluation.custom_evaluator import get_refusal_accuracy_prompt

        years = load_config().get("fiscal_years", [])
        prompt = get_refusal_accuracy_prompt()
        for year in years:
            assert f"FY{year}" in prompt

    def test_scope_names_the_configured_companies(self):
        from src.common.config import load_config
        from src.evaluation.custom_evaluator import get_refusal_accuracy_prompt

        prompt = get_refusal_accuracy_prompt()
        for company in load_config().get("companies", []):
            assert company["ticker"] in prompt

    def test_no_fiscal_year_outside_the_configured_set_is_named_as_covered(self):
        from src.common.config import load_config
        from src.evaluation.custom_evaluator import get_refusal_accuracy_prompt

        configured = {str(y) for y in load_config().get("fiscal_years", [])}
        scope = get_refusal_accuracy_prompt().split("**Question:**")[0]
        named = set(re.findall(r"FY(\d{4})", scope))
        assert named <= configured, f"prompt claims coverage of {named - configured}"

    def test_placeholders_survive_for_the_caller(self):
        from src.evaluation.custom_evaluator import get_refusal_accuracy_prompt

        prompt = get_refusal_accuracy_prompt()
        assert "{question}" in prompt and "{answer}" in prompt
        # The JSON example must not have been consumed by the format() call.
        assert '"score"' in prompt


# ---------------------------------------------------------------------------
#  Refusal Quality (secondary, subtype-conditioned) and Over-Refusal
#
#  The primary refusal_accuracy metric treats a bare "I cannot answer" and a
#  diagnosed refusal as equal. These tests pin the two additions that make the
#  difference visible: the per-subtype rubric, and the control metric that
#  stops "decline everything" from being a winning strategy.
# ---------------------------------------------------------------------------

def _refusal_item(**kwargs) -> GoldStandardItem:
    defaults = dict(
        id=131,
        question="How much was Apple's loss in FY2024?",
        ground_truth="",
        doc_refs="AAPL_2024",
        expected_answerable=False,
        fa_type="FA-Refusal",
        subtype="false_premise",
        refusal_evidence="counter_evidence",
        gt_correction="Apple reported a profit of $93,736 million, not a loss.",
    )
    defaults.update(kwargs)
    return GoldStandardItem(**defaults)


class TestRefusalQuality:
    def test_graded_score_is_recorded_from_a_second_judge_call(self):
        item = _refusal_item()
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            MagicMock(content=json.dumps({"score": 1.0, "rationale": "No fabrication."})),
            MagicMock(content=json.dumps({"score": 0.5, "rationale": "Declined, no diagnosis."})),
        ]
        with patch("src.evaluation.custom_evaluator.get_judge_llm", return_value=mock_llm):
            result = evaluate_custom_metrics(item, "I cannot answer that.")

        assert result.refusal_accuracy == 1.0
        assert result.refusal_quality == 0.5
        assert mock_llm.invoke.call_count == 2

    def test_stays_none_without_a_v5_reference(self):
        """A pre-v5 file has no gt_correction. Grading against a blank
        reference would be worse than not grading at all."""
        item = _refusal_item(gt_correction="", refusal_evidence="")
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            MagicMock(content=json.dumps({"score": 1.0, "rationale": "Refused."})),
        ]
        with patch("src.evaluation.custom_evaluator.get_judge_llm", return_value=mock_llm):
            result = evaluate_custom_metrics(item, "I cannot answer that.")

        assert result.refusal_quality is None
        assert mock_llm.invoke.call_count == 1

    def test_not_computed_for_answerable_items(self):
        item = GoldStandardItem(
            id=1, question="What was the revenue?", ground_truth="1.2",
            gt_unit="USD_billion", doc_refs="AAPL_2024", expected_answerable=True,
        )
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            MagicMock(content=json.dumps({"score": 1.0, "rationale": "ok"})),
            MagicMock(content=json.dumps({"score": 1.0, "rationale": "ok"})),
        ]
        with patch("src.evaluation.custom_evaluator.get_judge_llm", return_value=mock_llm):
            result = evaluate_custom_metrics(item, "Revenue was $1.2 billion.")

        assert result.refusal_quality is None


class TestRefusalQualityPromptSelection:
    """Each subtype gets its own rubric; an unknown one falls back to the
    least demanding rather than to whichever happens to be first."""

    def test_each_subtype_selects_its_own_rubric(self):
        from src.evaluation.custom_evaluator import get_refusal_quality_prompt

        scope = get_refusal_quality_prompt("not_in_corpus")
        premise = get_refusal_quality_prompt("false_premise")
        ambiguity = get_refusal_quality_prompt("ambiguous_entity")

        assert "does not contain" in scope
        assert "presupposes" in premise
        assert "several equally valid readings" in ambiguity
        assert len({scope, premise, ambiguity}) == 3

    def test_unknown_subtype_falls_back_to_the_scope_rubric(self):
        from src.evaluation.custom_evaluator import get_refusal_quality_prompt

        assert get_refusal_quality_prompt("") == get_refusal_quality_prompt("not_in_corpus")

    def test_every_rubric_carries_the_live_corpus_scope(self):
        from src.evaluation.custom_evaluator import get_refusal_quality_prompt

        for subtype in ("not_in_corpus", "false_premise", "ambiguous_entity"):
            prompt = get_refusal_quality_prompt(subtype)
            assert "AAPL" in prompt and "FY2024" in prompt


class TestEvidenceClause:
    """A correction is only legitimate where the corpus carries the
    counter-evidence. On a silent item the rubric must forbid the confident
    negative claim rather than reward it."""

    def test_counter_evidence_demands_the_actual_figure(self):
        from src.evaluation.custom_evaluator import get_evidence_clause

        assert "positively contradict" in get_evidence_clause("counter_evidence")

    def test_silent_forbids_asserting_the_fact_is_untrue(self):
        from src.evaluation.custom_evaluator import get_evidence_clause

        clause = get_evidence_clause("silent")
        assert "SILENT" in clause
        assert "0.0" in clause

    def test_unclassified_item_gets_the_conservative_clause(self):
        from src.evaluation.custom_evaluator import get_evidence_clause

        assert get_evidence_clause("") == get_evidence_clause("silent")


class TestOverRefusal:
    """refusal_accuracy is the correctness score of the refusal stratum, so a
    system that declines everything scores 30/30 on it. This is the
    counterweight that makes that visible."""

    def test_flags_a_refusal_on_an_answerable_item(self):
        item = GoldStandardItem(
            id=1, question="What was the revenue?", ground_truth="1.2",
            gt_unit="USD_billion", doc_refs="AAPL_2024", expected_answerable=True,
        )
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            MagicMock(content=json.dumps({"score": 0.0, "rationale": "No figure."})),
            MagicMock(content=json.dumps({"score": 0.0, "rationale": "Nothing present."})),
            MagicMock(content=json.dumps({"score": 1.0, "rationale": "Declined."})),
        ]
        with patch("src.evaluation.custom_evaluator.get_judge_llm", return_value=mock_llm):
            result = evaluate_custom_metrics(item, "I cannot answer this from the filings.")

        assert result.over_refusal == 1.0
        assert mock_llm.invoke.call_count == 3

    def test_a_fully_correct_answer_skips_the_judge(self):
        item = GoldStandardItem(
            id=1, question="What was the revenue?", ground_truth="1.2",
            gt_unit="USD_billion", doc_refs="AAPL_2024", expected_answerable=True,
        )
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            MagicMock(content=json.dumps({"score": 1.0, "rationale": "ok"})),
            MagicMock(content=json.dumps({"score": 1.0, "rationale": "ok"})),
        ]
        with patch("src.evaluation.custom_evaluator.get_judge_llm", return_value=mock_llm):
            result = evaluate_custom_metrics(item, "Revenue was $1.2 billion.")

        assert result.over_refusal == 0.0
        # Zero, not two: the deterministic numeric pre-check confirms this
        # match, so exact_match and answer_recall never reach the judge either.
        assert mock_llm.invoke.call_count == 0

    def test_not_computed_for_refusal_items(self):
        item = _refusal_item(gt_correction="", refusal_evidence="")
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            MagicMock(content=json.dumps({"score": 1.0, "rationale": "Refused."})),
        ]
        with patch("src.evaluation.custom_evaluator.get_judge_llm", return_value=mock_llm):
            result = evaluate_custom_metrics(item, "I cannot answer that.")

        assert result.over_refusal is None


class TestUndefinedMetricsSerializeAsNone:
    """A 0.0 default would be averaged in as a failure over the half of the
    dataset where these metrics have no meaning."""

    def test_to_dict_preserves_none(self):
        from src.evaluation.custom_evaluator import CustomEvalResult

        payload = CustomEvalResult().to_dict()
        assert payload["refusal_quality"] is None
        assert payload["over_refusal"] is None


class TestRefusalAccuracyPromptFix:
    """The 2026-09-07 differential defect, pinned so it cannot come back.

    On `false_premise` items the judge scored S3 and S4 0.0 for correctly
    rectifying the premise while S1 and S2 got 1.0 for a terser "not
    mentioned". The mechanism was that rule 5 asked the judge to decide
    groundedness without showing it anything to check against, so a longer
    answer simply offered more surface to suspect -- penalising the more
    informative response, and doing so differentially by architecture.
    """

    def test_the_prompt_carries_a_reference_to_check_against(self):
        from src.evaluation.custom_evaluator import get_refusal_accuracy_prompt
        assert "{gt_correction}" in get_refusal_accuracy_prompt()

    def test_the_prompt_forbids_penalising_detail(self):
        from src.evaluation.custom_evaluator import get_refusal_accuracy_prompt
        prompt = get_refusal_accuracy_prompt()
        assert "LENGTH AND DETAIL ARE NOT DEFECTS" in prompt

    def test_the_prompt_forces_a_scope_check(self):
        from src.evaluation.custom_evaluator import get_refusal_accuracy_prompt
        assert "CHECK THE SCOPE BEFORE CALLING SOMETHING OUT OF SCOPE" in prompt_of()

    def test_a_missing_reference_says_so_rather_than_going_blank(self):
        """An empty line under "Reference" reads as "nothing is supportable
        here", which is the opposite of the intended meaning."""
        from src.evaluation.custom_evaluator import NO_REFERENCE_AVAILABLE
        assert "no reference recorded" in NO_REFERENCE_AVAILABLE


def prompt_of() -> str:
    from src.evaluation.custom_evaluator import get_refusal_accuracy_prompt
    return get_refusal_accuracy_prompt()
