"""End-to-end shape check of the judge-validation chain.

Run JSON -> blind CSV -> rating page -> export -> merge -> analysis. Five
scripts, each of which has at some point been changed without the next one
noticing; the 2026-09-08 audit found the rating page rendering five
dimensions the CSV had no columns for, and the CSV carrying columns no page
could fill.

Nothing here checks a coefficient. What it checks is that a rating made on
the page survives to the analysis attached to the right unit of the right
record -- and, in particular, that the per-unit maps do not degrade into
scalars anywhere along the way.
"""

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _locus(year: int) -> dict:
    return {"doc_id": f"AAPL_{year}", "section_text": "Income Statement",
            "section_ids": ["item_8_income_stmt"]}


# The loci matter: a step tagged [E] is excluded from validity only if it
# actually cites something checkable. One tagged [E] with no locus is the
# system narrating or carrying a computed result, and stays assessable.
CHAIN = [
    {"step_index": 1, "type": "evidential", "type_was_tagged": True,
     "loci": [_locus(2020)],
     "text": "Total net sales FY2020 were $274,515M. (AAPL, FY2020, Income Statement)"},
    {"step_index": 2, "type": "evidential", "type_was_tagged": True,
     "loci": [_locus(2024)],
     "text": "Total net sales FY2024 were $391,035M. (AAPL, FY2024, Income Statement)"},
    {"step_index": 3, "type": "inferential", "type_was_tagged": True, "loci": [],
     "text": "Sales therefore rose."},
]


def _eval_json(system_name: str) -> dict:
    return {
        "system_name": system_name,
        "detailed_results": [{
            "query_id": 41,
            "fa_type": "FA-3",
            "subtype": "growth_yoy",
            "expected_answerable": True,
            "refusal_evidence": "",
            "gt_correction": "",
            "gt_unit": "percent",
            "question": "How did Apple's revenue develop?",
            "ground_truth": "-0.8%",
            "answer": "It rose.",
            "trajectory": "## System 1 Trajectory (rag_monolith)\nSteps: 1",
            "custom_metrics": {"exact_match": 1.0, "answer_recall": 1.0,
                               "refusal_accuracy": 0.0, "refusal_quality": None,
                               "over_refusal": 0.0},
            "citation_metrics": {"citation_accuracy": 1.0},
            "reasoning_metrics": {
                "chain_emitted": True, "num_steps": 3, "num_evidential": 2,
                "num_inferential": 1, "num_untagged": 0,
                "groundedness": 0.5, "validity": 1.0, "completeness": 1.0,
                "fully_grounded": False, "fully_valid": True,
                "steps": CHAIN,
                "evidence_shown": {"1": ["Total net sales 274,515."],
                                   "2": ["Total net sales 391,035."]},
                "step_verdicts": [
                    {"step": 1, "grounded": True, "code": None, "rationale": ""},
                    {"step": 2, "grounded": False, "code": "B1", "rationale": ""},
                ],
                "transition_verdicts": [
                    {"step": 2, "valid": True, "code": None, "rationale": ""},
                    {"step": 3, "valid": True, "code": None, "rationale": ""},
                ],
                "subquestion_verdicts": [
                    {"subquestion": 1, "text": "Establish revenue FY2020.",
                     "covered": True, "rationale": ""},
                    {"subquestion": 2, "text": "Establish revenue FY2024.",
                     "covered": True, "rationale": ""},
                ],
            },
            "ragas_metrics": {},
            "run_metrics": {"latency_seconds": 1.0, "total_tokens": 10,
                            "estimated_cost_usd": 0.01, "num_steps": 1,
                            "corrections": 0},
        }],
    }


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A judge_validation directory laid out the way the scripts expect."""
    results = tmp_path / "judge_validation"
    raw = results / "raw_eval_reserve"
    raw.mkdir(parents=True)
    for system in ("rag_monolith", "rag_agent", "long_context", "multi_agent"):
        (raw / f"eval_{system}_20260908_120000.json").write_text(
            json.dumps(_eval_json(system)), encoding="utf-8"
        )
    return results


class TestSampleBuilding:
    def test_the_blind_csv_carries_the_chain_and_its_evidence(self, workspace):
        row = _build_rows(workspace)[0]
        assert json.loads(row["chain_json"])[0]["step_index"] == 1
        # The rater must see the passages the JUDGE saw, not a re-fetch.
        assert json.loads(row["evidence_json"])["2"] == ["Total net sales 391,035."]
        assert json.loads(row["subquestions_json"]) == [
            "Establish revenue FY2020.", "Establish revenue FY2024."
        ]

    def test_open_and_not_applicable_cells_are_distinguished(self, workspace):
        rows = _build_rows(workspace)
        row = rows[0]
        # Three units of work exist, so all three dimensions are open.
        assert row["groundedness_human"] == ""
        assert row["validity_human"] == ""
        assert row["completeness_human"] == ""


def _build_rows(results_dir: Path) -> list[dict]:
    """Build blind rows from the fixture eval JSONs.

    The builder's main() resolves the repo's own results directory, so rather
    than monkeypatching a path into it this mirrors its row construction and
    calls its real helpers (`_anonymize_trajectory`, `_open_if`) -- those are
    where the logic that matters lives.
    """
    import scripts.build_judge_validation_sample as builder

    rows: list[dict] = []
    raw_dir = results_dir / "raw_eval_reserve"
    for system in builder.SYSTEM_ORDER:
        data = json.loads(
            (raw_dir / f"eval_{system}_20260908_120000.json").read_text(encoding="utf-8")
        )
        for item in data["detailed_results"]:
            reasoning = item["reasoning_metrics"]
            rows.append({
                "review_id": f"R{len(rows) + 1:03d}",
                "fa_type": item["fa_type"],
                "system_label": "A",
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "answer": item["answer"],
                "trajectory": builder._anonymize_trajectory(item["trajectory"]),
                "subtype": item["subtype"],
                "refusal_evidence": item["refusal_evidence"],
                "gt_correction": item["gt_correction"],
                "gt_unit": item["gt_unit"],
                "chain_json": json.dumps(reasoning.get("steps", [])),
                "evidence_json": json.dumps(reasoning.get("evidence_shown", {})),
                "subquestions_json": json.dumps(
                    [v["text"] for v in reasoning["subquestion_verdicts"]]
                ),
                "groundedness_human": builder._open_if(
                    reasoning["chain_emitted"] and reasoning["num_evidential"] > 0),
                "validity_human": builder._open_if(
                    reasoning["chain_emitted"] and reasoning["num_steps"] > 1),
                "completeness_human": builder._open_if(
                    reasoning["chain_emitted"] and bool(reasoning["subquestion_verdicts"])),
                "exact_match_human": "",
                "answer_recall_human": "",
                "refusal_accuracy_human": "N/A",
                "refusal_quality_human": "N/A",
                "over_refusal_human": "",
                "citation_accuracy_human": "N/A",
                "notes": "",
            })
    return rows


class TestRatingPage:
    def test_one_cell_per_unit_reaches_the_page(self, workspace):
        from scripts.build_human_rating_ui import build_items
        items = build_items(_build_rows(workspace))
        dims = items[0]["dimensions"]
        # 2 evidential steps, 1 assessable transition, 2 sub-questions.
        # The chain is [E] [E] [I], so only the step-3 transition is offered:
        # a transition into an [E] step introduces a premise, which does not
        # follow from anything.
        assert len(dims["groundedness"]["units"]) == 2
        assert len(dims["validity"]["units"]) == 1
        assert len(dims["completeness"]["units"]) == 2

    def test_a_transition_shows_everything_that_precedes_it(self, workspace):
        """Validity is local, but "follows from the preceding steps" needs
        all of them -- a rater shown only step n-1 would mark V1 wherever the
        premise sat two steps back."""
        from scripts.build_human_rating_ui import build_items
        units = build_items(_build_rows(workspace))[0]["dimensions"]["validity"]["units"]
        assert [u["unit"] for u in units] == ["3"]
        assert len(units[0]["preceding"]) == 2

    def test_a_locus_less_evidential_step_stays_assessable(self, workspace):
        """A step tagged [E] whose "source" is (calculate), or which merely
        narrates, introduces no filing evidence -- so it is not a premise and
        a V1 can hide behind it. The sensitivity test found exactly that: in
        V022 the planted defect landed on such a step, no cell existed, and
        the rater could not have caught it."""
        from scripts.build_human_rating_ui import build_items
        rows = _build_rows(workspace)
        chain = json.loads(rows[0]["chain_json"])
        chain[1]["loci"] = []          # step 2 now cites nothing checkable
        rows[0]["chain_json"] = json.dumps(chain)
        units = build_items(rows)[0]["dimensions"]["validity"]["units"]
        assert [u["unit"] for u in units] == ["2", "3"]

    def test_transitions_into_sourced_evidential_steps_are_not_offered(self, workspace):
        """Settled empirically by the 2026-09-08 pilot: the judge answered 81
        of 81 transitions into an inferential step and 15 of 63 into an
        evidential one, without being told to distinguish them; the human
        rater declined the same ones. Counting them as trivially valid would
        fill the denominator with units that cannot discriminate -- and their
        share differs per architecture, so the dilution would too."""
        from scripts.build_human_rating_ui import build_items
        units = build_items(_build_rows(workspace))[0]["dimensions"]["validity"]["units"]
        assert "2" not in [u["unit"] for u in units]

    def test_inferential_steps_carry_no_groundedness_cell(self, workspace):
        """R2: an inferential step makes no factual claim, so grounding it
        would double-count what validity already judges."""
        from scripts.build_human_rating_ui import build_items
        units = build_items(_build_rows(workspace))[0]["dimensions"]["groundedness"]["units"]
        assert [u["unit"] for u in units] == ["1", "2"]


class TestMergeAndAnalyse:
    def test_a_per_unit_rating_survives_the_round_trip(self):
        from scripts.merge_human_ratings import _validate_rating

        merged = _validate_rating({"2": "0", "1": "1"}, "groundedness", "R001")
        assert json.loads(merged) == {"1": "1", "2": "0"}

    def test_the_analyser_pairs_each_unit_with_the_judges_verdict(self, workspace):
        from scripts.analyze_judge_validation import _collect_reasoning_pairs

        rows = [{
            "review_id": "R001",
            "groundedness_human": json.dumps({"1": "1", "2": "1"}),
        }]
        reference = {"R001": {
            "system_name": "rag_monolith",
            "judge_reasoning": _eval_json("rag_monolith")["detailed_results"][0]["reasoning_metrics"],
        }}
        pairs = _collect_reasoning_pairs(
            rows, reference, "groundedness_human", "step_verdicts", "grounded", "step"
        )
        # Two units, and the disagreement is on step 2 where the judge said
        # not grounded and the rater said grounded.
        assert len(pairs) == 2
        by_unit = {p[0]: (p[2], p[3]) for p in pairs}
        assert by_unit["R001#1"] == (1.0, 1.0)
        assert by_unit["R001#2"] == (1.0, 0.0)

    def test_an_unrated_unit_is_skipped_not_counted_as_disagreement(self, workspace):
        from scripts.analyze_judge_validation import _collect_reasoning_pairs

        rows = [{"review_id": "R001", "groundedness_human": json.dumps({"1": "1"})}]
        reference = {"R001": {
            "system_name": "rag_monolith",
            "judge_reasoning": _eval_json("rag_monolith")["detailed_results"][0]["reasoning_metrics"],
        }}
        pairs = _collect_reasoning_pairs(
            rows, reference, "groundedness_human", "step_verdicts", "grounded", "step"
        )
        assert len(pairs) == 1

    def test_na_dimensions_contribute_nothing(self, workspace):
        from scripts.analyze_judge_validation import _collect_reasoning_pairs

        rows = [{"review_id": "R001", "groundedness_human": "N/A"}]
        reference = {"R001": {"system_name": "s", "judge_reasoning": {}}}
        assert _collect_reasoning_pairs(
            rows, reference, "groundedness_human", "step_verdicts", "grounded", "step"
        ) == []


class TestCoefficients:
    def test_gwet_ac1_survives_a_skewed_marginal_where_kappa_collapses(self):
        """The refusal_accuracy case, reproduced.

        14-to-2 human marginal at 69% agreement gave kappa 0.13 and AC1 0.53.
        Both fail the 0.60 threshold -- which is why AC1 was adopted at this
        moment rather than a more convenient one -- but only one of them
        implies the raters were near-random.
        """
        from sklearn.metrics import cohen_kappa_score

        from scripts.analyze_judge_validation import _gwet_ac1

        # The actual 2x2 from the reserve sample, reconstructed from the
        # reported n=16, 69% agreement, 14-to-2 human and 11-to-5 judge
        # marginals: 10 both-1, 4 human-1/judge-0, 1 human-0/judge-1, 1 both-0.
        humans = ["1"] * 10 + ["1"] * 4 + ["0"] + ["0"]
        judges = ["1"] * 10 + ["0"] * 4 + ["1"] + ["0"]
        agreement = sum(h == j for h, j in zip(humans, judges)) / len(humans)
        kappa = cohen_kappa_score(humans, judges)
        ac1 = _gwet_ac1(humans, judges)

        assert round(agreement, 2) == 0.69
        assert round(kappa, 2) == 0.13
        assert round(ac1, 2) == 0.53
        # AC1 shows the defect is smaller than kappa suggests -- and still
        # short of the threshold, which is why adopting it changed no verdict.
        assert ac1 > kappa
        assert ac1 < 0.6

    def test_ac1_is_none_when_everyone_used_one_category(self):
        from scripts.analyze_judge_validation import _gwet_ac1
        assert _gwet_ac1(["1"] * 5, ["1"] * 5) is None


class TestDegenerateVerdict:
    """Kappa is identically 0 whenever ONE rater is constant. The verdict code
    used to treat only the both-constant case as degenerate, which turned the
    validity dimension (human 81:0, judge 78:3, 96% agreement) into "FAIL,
    kappa 0.00" -- a statement about the formula, not the judge."""

    @staticmethod
    def _pairs(humans, judges):
        return [(f"r{i}", "sys", h, j) for i, (h, j) in enumerate(zip(humans, judges))]

    def test_one_constant_rater_is_not_estimable_not_fail(self):
        from scripts.analyze_judge_validation import _analyze_dimension

        humans = [1.0] * 81
        judges = [1.0] * 78 + [0.0] * 3
        _, verdict, metrics = _analyze_dimension("validity", self._pairs(humans, judges), "custom")

        assert verdict.startswith("NOT ESTIMABLE")
        assert "the human rater" in verdict
        assert metrics["degenerate"] is True
        # The coefficient really is 0 here -- the point is that 0 is what it
        # always is against a constant, so it must not drive the verdict.
        assert round(metrics["kappa"], 2) == 0.0
        # AC1 stays reportable and is the number that carries the information.
        assert metrics["ac1"] is not None and metrics["ac1"] > 0.9

    def test_both_constant_stays_not_estimable(self):
        from scripts.analyze_judge_validation import _analyze_dimension

        _, verdict, metrics = _analyze_dimension(
            "validity", self._pairs([1.0] * 10, [1.0] * 10), "custom"
        )
        assert verdict.startswith("NOT ESTIMABLE")
        assert "both raters" in verdict
        assert metrics["kappa"] is None

    def test_a_real_disagreement_still_fails(self):
        """The change must not widen the escape hatch: with variance on both
        sides a low kappa is a low kappa."""
        from scripts.analyze_judge_validation import _analyze_dimension

        humans = [1.0] * 6 + [0.0] * 6
        judges = [1.0] * 3 + [0.0] * 3 + [1.0] * 3 + [0.0] * 3
        _, verdict, metrics = _analyze_dimension("x", self._pairs(humans, judges), "custom")
        assert verdict.startswith("FAIL")
        assert metrics["degenerate"] is False
