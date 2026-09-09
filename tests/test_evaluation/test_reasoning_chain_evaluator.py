"""The chain-based reasoning evaluator.

The judge is mocked throughout: what is under test is the scoring apparatus
around it -- what is settled without asking, what a missing verdict does to
the denominator, and which values stay None rather than becoming zero.

That last one carries most of the weight. A chain that was never emitted, or
a chain with nothing to infer from, must not be scored 0.0: a format failure
averaged in as a reasoning defect would let a prompt-following problem
masquerade as a reasoning result, and the four systems do not follow
instructions equally well.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation.evidence_store import EvidenceStore, LookupResult
from src.evaluation.reasoning_chain_evaluator import (
    B_LOCUS_NOT_IN_CORPUS,
    B_NO_LOCUS,
    ChainReasoningScores,
    evaluate_reasoning_chain,
    evaluate_reasoning_chain_batch,
)

CHAIN = """Revenue rose.

## Reasoning
1. [E] Total net sales for FY2020 were $274,515M. (AAPL, FY2020, Income Statement)
2. [E] Total net sales for FY2024 were $391,035M. (AAPL, FY2024, Income Statement)
3. [I] Sales therefore rose over the period.
"""

DECOMPOSITION = [
    "Establish revenue for Apple in FY2020.",
    "Establish revenue for Apple in FY2024.",
    "Compute the change between the two figures.",
]


def _store(**overrides):
    store = MagicMock(spec=EvidenceStore)
    store.lookup.return_value = overrides.get(
        "result", LookupResult(passages=["Total net sales 391,035."])
    )
    return store


def _judge(*payloads):
    """A judge returning the given JSON payloads in order."""
    llm = MagicMock()
    llm.invoke.side_effect = [
        SimpleNamespace(content=json.dumps(p)) for p in payloads
    ]
    return llm


ALL_GOOD = {
    "groundedness": [
        {"step": 1, "grounded": True, "code": None, "rationale": "in passage"},
        {"step": 2, "grounded": True, "code": None, "rationale": "in passage"},
    ],
    "validity": [
        {"step": 2, "valid": True, "code": None, "rationale": "follows"},
        {"step": 3, "valid": True, "code": None, "rationale": "follows"},
    ],
}
ALL_COVERED = {
    "coverage": [
        {"subquestion": i, "covered": True, "rationale": "covered"} for i in (1, 2, 3)
    ]
}


def _evaluate(answer=CHAIN, judge=None, store=None, decomposition=DECOMPOSITION):
    judge = judge or _judge(ALL_GOOD, ALL_COVERED)
    with patch(
        "src.evaluation.reasoning_chain_evaluator.get_judge_llm", return_value=judge
    ):
        return evaluate_reasoning_chain(
            question="How did Apple's revenue develop?",
            answer=answer,
            reference_decomposition=decomposition,
            store=store or _store(),
            system_name="rag_monolith",
            query_id=41,
        )


class TestCleanChain:
    def test_all_three_dimensions_are_scored(self):
        scores = _evaluate()
        assert scores.groundedness == 1.0
        assert scores.validity == 1.0
        assert scores.completeness == 1.0
        assert scores.fully_grounded and scores.fully_valid

    def test_step_types_are_counted_for_the_covariate_table(self):
        scores = _evaluate()
        assert (scores.num_steps, scores.num_evidential, scores.num_inferential) == (3, 2, 1)

    def test_completeness_is_asked_in_its_own_call(self):
        """Halo control: a judge that has just found a fabrication marks the
        chain down on coverage too if both are asked at once."""
        judge = _judge(ALL_GOOD, ALL_COVERED)
        _evaluate(judge=judge)
        assert judge.invoke.call_count == 2
        first, second = (c.args[0] for c in judge.invoke.call_args_list)
        assert "Groundedness" in first and "Required sub-questions" not in first
        assert "Required sub-questions" in second


class TestSettledWithoutTheJudge:
    def test_an_unsourced_claim_about_the_filings_is_B5(self):
        """No longer decided deterministically.

        Auto-B5 on every locus-less [E] step produced 7 of the 8 groundedness
        false alarms in the pilot: it fired on steps that assert nothing about
        the filings at all. The judge now separates the two cases, and only a
        step claiming filing content is B5.
        """
        answer = "x\n\n## Reasoning\n1. [E] Revenue was $391,035M.\n2. [I] So it rose.\n"
        scores = _evaluate(
            answer=answer,
            judge=_judge({
                "groundedness": [],
                "unsourced": [{"step": 1, "not_a_claim": False, "rationale": "states a figure"}],
                "validity": [{"step": 2, "valid": True, "code": None, "rationale": ""}],
            }, ALL_COVERED),
        )
        assert scores.groundedness == 0.0
        assert [v["code"] for v in scores.step_verdicts] == [B_NO_LOCUS]

    def test_a_step_that_only_narrates_the_run_is_no_unit_at_all(self):
        """"No context is provided for Microsoft", "The user is asking for X",
        "The calculated growth is 65.247%. (calculate)" -- these assert nothing
        about filing content, so there is nothing to ground. Neither passed
        nor failed: dropped."""
        answer = ("x\n\n## Reasoning\n"
                  "1. [E] No context is provided for Microsoft.\n"
                  "2. [I] So the comparison cannot be made.\n")
        scores = _evaluate(
            answer=answer,
            judge=_judge({
                "groundedness": [],
                "unsourced": [{"step": 1, "not_a_claim": True, "rationale": "describes the run"}],
                "validity": [{"step": 2, "valid": True, "code": None, "rationale": ""}],
            }, ALL_COVERED),
        )
        assert scores.step_verdicts == []
        assert scores.groundedness is None

    def test_a_locus_outside_the_corpus_is_B6(self):
        store = _store(result=LookupResult(doc_exists=False))
        scores = _evaluate(
            store=store,
            judge=_judge({"groundedness": [], "validity": [
                {"step": 2, "valid": True, "code": None, "rationale": ""},
                {"step": 3, "valid": True, "code": None, "rationale": ""}]}, ALL_COVERED),
        )
        assert {v["code"] for v in scores.step_verdicts} == {B_LOCUS_NOT_IN_CORPUS}
        assert scores.groundedness == 0.0

    def test_violation_codes_are_counted_for_the_diagnostic_table(self):
        scores = _evaluate(
            judge=_judge(
                {"groundedness": [
                    {"step": 1, "grounded": False, "code": "B1", "rationale": ""},
                    {"step": 2, "grounded": True, "code": None, "rationale": ""}],
                 "validity": [
                    {"step": 2, "valid": False, "code": "V1", "rationale": ""},
                    {"step": 3, "valid": True, "code": None, "rationale": ""}]},
                ALL_COVERED,
            )
        )
        counts = scores.violation_counts()
        assert counts["B1"] == 1 and counts["V1"] == 1
        assert scores.groundedness == 0.5 and scores.validity == 0.5
        assert scores.fully_grounded is False


class TestValidityScope:
    """Transitions into inferential steps only, plus the V5 exception."""

    def test_a_transition_into_an_evidential_step_is_not_a_unit(self):
        """CHAIN is [E] [E] [I]: only the step-3 transition is assessable."""
        scores = _evaluate(judge=_judge(
            {"groundedness": [
                {"step": 1, "grounded": True, "code": None, "rationale": ""},
                {"step": 2, "grounded": True, "code": None, "rationale": ""}],
             "validity": [
                {"step": 2, "valid": True, "code": None, "rationale": ""},
                {"step": 3, "valid": True, "code": None, "rationale": ""}]},
            ALL_COVERED))
        assert [v["step"] for v in scores.transition_verdicts] == [3]

    def test_a_confirming_verdict_on_a_premise_is_dropped_not_counted(self):
        """An unflagged premise is not evidence of coherence, it is simply not
        a unit. Scoring it 1 would inflate the denominator with units that
        cannot fail -- unevenly, since the evidential share differs per
        architecture."""
        scores = _evaluate(judge=_judge(
            {"groundedness": [
                {"step": 1, "grounded": True, "code": None, "rationale": ""},
                {"step": 2, "grounded": True, "code": None, "rationale": ""}],
             "validity": [
                {"step": 2, "valid": True, "code": None, "rationale": ""},
                {"step": 3, "valid": False, "code": "V1", "rationale": ""}]},
            ALL_COVERED))
        assert scores.validity == 0.0
        assert len(scores.transition_verdicts) == 1

    def test_V5_on_a_premise_is_counted(self):
        """An evidential step contradicting an earlier one is a coherence
        defect groundedness cannot see, so it counts against the chain."""
        scores = _evaluate(judge=_judge(
            {"groundedness": [
                {"step": 1, "grounded": True, "code": None, "rationale": ""},
                {"step": 2, "grounded": True, "code": None, "rationale": ""}],
             "validity": [
                {"step": 2, "valid": False, "code": "V5", "rationale": "contradicts step 1"},
                {"step": 3, "valid": True, "code": None, "rationale": ""}]},
            ALL_COVERED))
        steps = {v["step"]: v for v in scores.transition_verdicts}
        assert steps[2]["code"] == "V5" and steps[2]["premise_contradiction"]
        assert scores.validity == 0.5

    def test_a_chain_of_pure_lookups_has_no_validity(self):
        answer = ("x\n\n## Reasoning\n"
                  "1. [E] A. (AAPL, FY2024, Income Statement)\n"
                  "2. [E] B. (AAPL, FY2020, Income Statement)\n")
        scores = _evaluate(answer=answer, judge=_judge(
            {"groundedness": [
                {"step": 1, "grounded": True, "code": None, "rationale": ""},
                {"step": 2, "grounded": True, "code": None, "rationale": ""}],
             "validity": []},
            ALL_COVERED))
        assert scores.validity is None


class TestUnsupportedValueOverride:
    """A figure the judge could not trace settles the verdict mechanically.

    Measured, not defensive: in the sensitivity test of 2026-09-09 the judge
    caught 3 of 12 planted V1 defects, and its own rationale on one of them
    read "...and the Total Net Sales, which is assumed". It saw the gap and
    ruled valid anyway. Moving the decision off its judgement and onto its own
    bookkeeping took sensitivity from 25% to 75%.
    """

    def test_an_untraceable_input_overrides_a_valid_verdict(self):
        scores = _evaluate(judge=_judge(
            {"groundedness": [
                {"step": 1, "grounded": True, "code": None, "rationale": ""},
                {"step": 2, "grounded": True, "code": None, "rationale": ""}],
             "validity": [{
                "step": 3, "valid": True, "code": None,
                "values_used": ["274,515 (step 1)", "18% (nowhere)"],
                "unsupported_values": ["18%"],
                "rationale": "follows from the previous steps"}]},
            ALL_COVERED))
        verdict = scores.transition_verdicts[0]
        assert verdict["valid"] is False
        assert verdict["code"] == "V1"
        assert verdict["overruled_judge"] is True
        assert verdict["unsupported_values"] == ["18%"]

    def test_an_empty_list_leaves_the_judge_verdict_alone(self):
        scores = _evaluate(judge=_judge(
            {"groundedness": [
                {"step": 1, "grounded": True, "code": None, "rationale": ""},
                {"step": 2, "grounded": True, "code": None, "rationale": ""}],
             "validity": [{
                "step": 3, "valid": True, "code": None,
                "values_used": ["274,515 (step 1)"], "unsupported_values": [],
                "rationale": ""}]},
            ALL_COVERED))
        assert scores.transition_verdicts[0]["valid"] is True
        assert "overruled_judge" not in scores.transition_verdicts[0]

    def test_the_override_does_not_relabel_an_already_failing_verdict(self):
        """A judge that names V5 keeps V5; the override supplies V1 only when
        no code came back at all."""
        scores = _evaluate(judge=_judge(
            {"groundedness": [
                {"step": 1, "grounded": True, "code": None, "rationale": ""},
                {"step": 2, "grounded": True, "code": None, "rationale": ""}],
             "validity": [{
                "step": 3, "valid": False, "code": "V5",
                "unsupported_values": ["18%"], "rationale": ""}]},
            ALL_COVERED))
        assert scores.transition_verdicts[0]["code"] == "V5"


class TestFigureVerification:
    """A "grounded" verdict stands only if its figures are in the passages.

    Checked, not trusted: in the pilot the judge confirmed seven steps the
    rater rejected, each time asserting the passage "contains" a derived ratio
    no 10-K states. Verifying the figures took kappa from 0.21 to 0.68.
    """

    def test_a_figure_absent_from_the_passage_overrules_the_judge(self):
        scores = _evaluate(
            store=_store(result=LookupResult(passages=["Total net sales 391,035."])),
            judge=_judge({"groundedness": [
                {"step": 1, "grounded": True, "code": None,
                 "supporting_values": ["24.15%"], "rationale": "the passage contains it"},
                {"step": 2, "grounded": True, "code": None,
                 "supporting_values": ["391,035"], "rationale": ""}],
                "validity": [{"step": 3, "valid": True, "code": None, "rationale": ""}]},
                ALL_COVERED))
        by_step = {v["step"]: v for v in scores.step_verdicts}
        assert by_step[1]["grounded"] is False
        assert by_step[1]["code"] == "B1" and by_step[1]["overruled_judge"] is True
        assert by_step[2]["grounded"] is True

    def test_a_figure_written_with_units_still_matches(self):
        """The judge writes "$450,256 million"; the passage writes it inside a
        table. Normalising the whole string leaves "450256million" and matches
        nothing -- which is what produced 26 spurious failures before the
        number was extracted instead."""
        from src.evaluation.reasoning_chain_evaluator import _unsupported_figures
        passages = ["Total liabilities and equity\xa0$450,256\xa0"]
        assert _unsupported_figures(["$450,256 million"], passages) == []

    def test_short_figures_are_not_checked(self):
        """A bare "18" occurs in almost any financial passage, so testing it
        would wave through what the check exists for while adding noise."""
        from src.evaluation.reasoning_chain_evaluator import _unsupported_figures
        assert _unsupported_figures(["18%"], ["nothing relevant here"]) == []


class TestNoneRatherThanZero:
    def test_a_missing_chain_scores_nothing_at_all(self):
        scores = _evaluate(answer="Revenue rose to $391,035M.")
        assert not scores.chain_emitted
        assert scores.groundedness is None
        assert scores.validity is None
        assert scores.completeness is None

    def test_a_single_step_chain_has_no_validity(self):
        """A chain with nothing to get wrong has not shown valid inference."""
        answer = "x\n\n## Reasoning\n1. [E] Revenue was $391,035M. (AAPL, FY2024, Income Statement)\n"
        scores = _evaluate(
            answer=answer,
            judge=_judge({"groundedness": [
                {"step": 1, "grounded": True, "code": None, "rationale": ""}],
                "validity": []}, ALL_COVERED),
        )
        assert scores.groundedness == 1.0
        assert scores.validity is None

    def test_a_missing_judge_verdict_leaves_the_unit_out_of_the_denominator(self):
        """Counting an unanswered unit as a failure would make a judge parse
        error look like a system defect."""
        scores = _evaluate(
            judge=_judge(
                {"groundedness": [
                    {"step": 1, "grounded": True, "code": None, "rationale": ""}],
                 "validity": [
                    {"step": 2, "valid": True, "code": None, "rationale": ""},
                    {"step": 3, "valid": True, "code": None, "rationale": ""}]},
                ALL_COVERED,
            )
        )
        assert len(scores.step_verdicts) == 1
        assert scores.groundedness == 1.0

    def test_without_a_decomposition_completeness_stays_none(self):
        scores = _evaluate(decomposition=[], judge=_judge(ALL_GOOD))
        assert scores.completeness is None
        assert scores.groundedness == 1.0


class TestBatch:
    def test_refusal_items_are_skipped(self):
        """FA-Refusal has no reference decomposition -- the correct chain there
        is "the premise cannot be checked" (spec section 8.4)."""
        items = [
            SimpleNamespace(id=1, question="q", fa_type="FA-1", expected_answerable=True,
                            reference_decomposition=DECOMPOSITION),
            SimpleNamespace(id=2, question="q", fa_type="FA-Refusal",
                            expected_answerable=False, reference_decomposition=[]),
        ]
        with patch(
            "src.evaluation.reasoning_chain_evaluator.get_judge_llm",
            return_value=_judge(ALL_GOOD, ALL_COVERED),
        ):
            results = evaluate_reasoning_chain_batch(items, [CHAIN, CHAIN], _store(), "s1")
        assert results[0].groundedness == 1.0
        assert results[1].groundedness is None and not results[1].chain_emitted

    def test_alignment_survives_a_failing_item(self):
        items = [
            SimpleNamespace(id=i, question="q", fa_type="FA-1", expected_answerable=True,
                            reference_decomposition=DECOMPOSITION)
            for i in (1, 2)
        ]
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("judge down")
        with patch(
            "src.evaluation.reasoning_chain_evaluator.get_judge_llm", return_value=llm
        ):
            results = evaluate_reasoning_chain_batch(items, [CHAIN, CHAIN], _store(), "s1")
        assert [r.query_id for r in results] == [1, 2]

    def test_length_mismatch_is_refused(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            evaluate_reasoning_chain_batch([SimpleNamespace(id=1)], [], _store(), "s1")


class TestSerialisation:
    def test_evidence_is_only_carried_when_asked_for(self):
        """It multiplies a run's JSON several times over; only the human
        validation sample needs it."""
        default = _evaluate()
        assert default.evidence_shown == {}
        with patch(
            "src.evaluation.reasoning_chain_evaluator.get_judge_llm",
            return_value=_judge(ALL_GOOD, ALL_COVERED),
        ):
            kept = evaluate_reasoning_chain(
                question="q", answer=CHAIN, reference_decomposition=DECOMPOSITION,
                store=_store(), keep_evidence=True,
            )
        assert kept.evidence_shown

    def test_the_parsed_chain_is_persisted_with_the_verdicts(self):
        """`eval_runner` stores the answer with the chain already cut off, so
        this is the only place the measured artefact survives. Without it the
        run output holds verdicts about steps whose text is gone -- which is
        how the first judge-validation sample came out with zero groundedness
        and zero validity cells."""
        payload = _evaluate().to_dict()
        assert [s["text"] for s in payload["steps"]][:1] == [
            "Total net sales for FY2020 were $274,515M. (AAPL, FY2020, Income Statement)"
        ]
        assert payload["steps"][2]["type"] == "inferential"

    def test_the_chain_is_persisted_even_when_nothing_could_be_scored(self):
        """A chain the judge could not rule on is still the run's evidence of
        what the system produced."""
        answer = "x\n\n## Reasoning\n1. [E] Revenue was $391,035M.\n"
        scores = _evaluate(
            answer=answer,
            judge=_judge({"groundedness": [], "validity": []}, ALL_COVERED),
        )
        assert len(scores.to_dict()["steps"]) == 1

    def test_to_dict_is_json_serialisable(self):
        json.dumps(_evaluate().to_dict())

    def test_an_empty_result_serialises_too(self):
        json.dumps(ChainReasoningScores(query_id=1, system_name="s1").to_dict())
