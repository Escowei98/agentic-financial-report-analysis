"""Statistics core of scripts/analyze_final_eval.py.

These pin the pre-registered rules (majority collapse, Holm, McNemar cell
bookkeeping, Pratt zero handling) against hand-computed values so that a
refactor cannot silently change which system a hypothesis test favours.
"""
import numpy as np
import pytest

from scripts.analyze_final_eval import (
    Item,
    emission_intersection,
    holm,
    majority_correct,
    mcnemar_exact,
    paired_bootstrap_ci,
    wilcoxon_pratt,
)


class TestMajorityCollapse:
    def test_two_of_three_is_correct(self):
        assert majority_correct([1.0, 1.0, 0.0]) == 1.0

    def test_one_of_three_is_wrong(self):
        assert majority_correct([1.0, 0.0, 0.0]) == 0.0

    def test_off_schema_half_counts_as_wrong(self):
        # ids 64/71: judge returned 0.5 -- not a hit under the majority rule
        assert majority_correct([1.0, 0.5, 0.5]) == 0.0

    def test_missing_run_counts_as_wrong(self):
        assert majority_correct([1.0, None, 1.0]) == 1.0
        assert majority_correct([1.0, None, None]) == 0.0

    def test_empty(self):
        assert majority_correct([]) == 0.0


class TestMcNemar:
    def test_cells_and_direction(self):
        a = np.array([1, 1, 0, 0, 0, 1, 0, 0])
        b = np.array([1, 0, 1, 1, 1, 1, 0, 0])
        res = mcnemar_exact(a, b)
        assert res["both"] == 2 and res["neither"] == 2
        assert res["only_predicted"] == 3 and res["only_baseline"] == 1
        assert res["effect"] == pytest.approx((3 - 1) / 8)
        assert res["odds_ratio"] == pytest.approx(3.0)
        # exact two-sided binomial on 1 of 4 discordant
        assert res["p"] == pytest.approx(0.625)

    def test_no_discordant_pairs(self):
        a = np.array([1, 0, 1])
        b = a.copy()
        res = mcnemar_exact(a, b)
        assert res["p"] == 1.0 and res["effect"] == 0.0 and res["odds_ratio"] is None

    def test_h2_like_split(self):
        a = np.zeros(42)
        b = np.ones(42)
        a[:2] = 1
        b[:2] = 0
        res = mcnemar_exact(a, b)
        assert res["only_predicted"] == 40 and res["only_baseline"] == 2
        assert res["p"] < 1e-9


class TestWilcoxonPratt:
    def test_zeros_are_counted_and_flagged(self):
        a = np.array([1.0] * 12)
        b = np.array([1.0] * 8 + [0.5, 0.75, 0.8, 0.6])
        res = wilcoxon_pratt(a, b)
        assert res["n_zero"] == 8 and res["n_nonzero"] == 4
        assert res["approx_unreliable"] is True
        assert res["effect"] < 0 and res["rank_biserial"] == pytest.approx(-1.0)
        # exact Wilcox p for four same-signed differences is 2/16
        assert res["p_wilcox_exact"] == pytest.approx(0.125)

    def test_all_zero(self):
        a = np.array([0.5, 0.5])
        b = a.copy()
        res = wilcoxon_pratt(a, b)
        assert res["p"] == 1.0 and res["rank_biserial"] == 0.0

    def test_direction_of_effect(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 1, 40)
        b = np.clip(a + 0.2, 0, 1)
        res = wilcoxon_pratt(a, b)
        assert res["effect"] > 0 and res["rank_biserial"] > 0.9 and res["p"] < 0.001


class TestHolm:
    def test_step_down_and_monotone(self):
        adj = holm([0.01, 0.04, 0.03])
        assert adj == pytest.approx([0.03, 0.06, 0.06])

    def test_capped_at_one(self):
        assert holm([0.9, 0.95]) == [1.0, 1.0]

    def test_single(self):
        assert holm([0.02]) == [0.02]


class TestBootstrap:
    def test_seed_is_deterministic_and_brackets_point_estimate(self):
        a = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0] * 4)
        b = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0] * 4)
        def stat(x, y):
            return float(y.mean() - x.mean())
        lo1, hi1 = paired_bootstrap_ci(a, b, stat, reps=500)
        lo2, hi2 = paired_bootstrap_ci(a, b, stat, reps=500)
        assert (lo1, hi1) == (lo2, hi2)
        assert lo1 <= stat(a, b) <= hi1


def _item(qid, emitted_by_run: dict) -> Item:
    it = Item(query_id=qid, fa_type="FA-4", subtype="x", window_class="",
              math_type="", entity_form="", gt_unit="",
              refusal_evidence="", expected_answerable=True)
    for system, flags in emitted_by_run.items():
        for run, flag in zip(["run1", "run2", "run3"], flags):
            it.obs[system][run] = {"chain_emitted": float(flag)}
    return it


class TestEmissionIntersection:
    def test_majority_and_strict(self):
        items = {
            1: _item(1, {"S3": [1, 1, 1], "S4": [1, 1, 1]}),
            2: _item(2, {"S3": [1, 1, 0], "S4": [1, 1, 1]}),  # S3 2/3
            3: _item(3, {"S3": [1, 0, 0], "S4": [1, 1, 1]}),  # S3 1/3
        }
        runs = ["run1", "run2", "run3"]
        assert emission_intersection(items, "S3", "S4", runs) == {1, 2}
        assert emission_intersection(items, "S3", "S4", runs, strict=True) == {1}
