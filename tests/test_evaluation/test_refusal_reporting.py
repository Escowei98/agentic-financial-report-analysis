"""
Reporting of the FA-Refusal stratum.

The aggregate refusal_accuracy answers "did the system fabricate?" and nothing
else. These tests pin the two things the report has to add on top of it: the
per-subtype breakdown that makes the three defect classes visible, and the
over-refusal rate that stops "decline everything" from scoring a perfect
stratum unnoticed.
"""

from __future__ import annotations

import scripts.run_full_eval as full_eval


def _item(subtype: str, answerable: bool, **custom) -> dict:
    metrics = {
        "exact_match": 0.0,
        "answer_recall": 0.0,
        "refusal_accuracy": 0.0,
        "refusal_quality": None,
        "over_refusal": None,
    }
    metrics.update(custom)
    return {
        "expected_answerable": answerable,
        "subtype": subtype,
        "custom_metrics": metrics,
        "reasoning_metrics": {
            "chain_emitted": True,
            "num_steps": 3, "num_evidential": 2, "num_inferential": 1,
            "num_untagged": 0,
            "groundedness": 1.0, "validity": 1.0, "completeness": 1.0,
            "fully_grounded": True, "fully_valid": True,
        },
        "run_metrics": {"total_tokens": 10, "latency_seconds": 1.0},
    }


class TestExtraAggregates:
    def test_refusal_quality_averages_only_over_refusal_items(self):
        detailed = [
            _item("false_premise", False, refusal_accuracy=1.0, refusal_quality=1.0),
            _item("false_premise", False, refusal_accuracy=1.0, refusal_quality=0.5),
            _item("basis_kpi", True, exact_match=1.0, over_refusal=0.0),
        ]
        extra = full_eval.compute_extra_aggregates(detailed)
        assert extra["refusal_quality"] == 0.75

    def test_over_refusal_rate_averages_only_over_answerable_items(self):
        detailed = [
            _item("basis_kpi", True, over_refusal=1.0),
            _item("basis_kpi", True, exact_match=1.0, over_refusal=0.0),
            _item("false_premise", False, refusal_accuracy=1.0, refusal_quality=1.0),
        ]
        extra = full_eval.compute_extra_aggregates(detailed)
        assert extra["over_refusal_rate"] == 0.5

    def test_none_scores_are_skipped_not_counted_as_zero(self):
        """A pre-v5 run has no refusal_quality at all. That must read as
        'not measured', not as a system that scored zero."""
        detailed = [_item("false_premise", False, refusal_accuracy=1.0)]
        extra = full_eval.compute_extra_aggregates(detailed)
        assert extra["refusal_quality"] is None


class TestSubtypeBreakdown:
    def test_reports_every_subtype_for_every_system(self):
        results = {
            "S1": {"detailed_results": [
                _item("not_in_corpus", False, refusal_accuracy=1.0, refusal_quality=1.0),
                _item("false_premise", False, refusal_accuracy=1.0, refusal_quality=0.5),
                _item("ambiguous_entity", False, refusal_accuracy=0.0, refusal_quality=0.0),
            ]},
            "S2": {"detailed_results": [
                _item("not_in_corpus", False, refusal_accuracy=1.0, refusal_quality=0.5),
            ]},
        }
        lines = full_eval._refusal_subtype_section(results, ["S1", "S2"])
        body = "\n".join(lines)

        for subtype in ("not_in_corpus", "false_premise", "ambiguous_entity"):
            assert subtype in body
        assert "| not_in_corpus | Quality | 1.00 | 0.50 |" in body
        # S2 has no items of these subtypes -> "-" rather than a misleading 0.
        assert "| false_premise | Quality | 0.50 | - |" in body

    def test_carries_the_no_significance_caveat(self):
        """Ten items per subtype. The power analysis behind the design covers
        n=30 per stratum, not n=10 per subtype."""
        results = {"S1": {"detailed_results": []}}
        body = "\n".join(full_eval._refusal_subtype_section(results, ["S1"]))
        assert "no significance tests" in body
