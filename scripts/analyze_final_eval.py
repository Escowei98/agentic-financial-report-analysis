"""
Statistical analysis of the three final_eval runs for Kapitel 5.

Implements the pre-registered design of Kap. 3.6.4 on top of the raw
eval_*.json files. `aggregate_runs.py`
reports means ± SD per system; this script does everything that needs the
item level: the three planned contrasts with their tests, the exploratory
pairs, stratum breakdowns, run-to-run stability, the emission-rule
intersection and the refusal/efficiency tables.

Design (fixed before the first test statistic was computed):

* Unit of analysis is the gold-standard item. Three runs are collapsed to
  one value per item and system: binary endpoints by MAJORITY
  (exact_match == 1.0 in >= 2 of 3 runs; a judge value of 0.5 counts as
  wrong), continuous endpoints by the item MEAN across runs.
* H1: S2 vs S3, H2: S1 vs S2, H3: S3 vs S4. Populations: H1/H3 on FA-4 ∪
  FA-3 cross_window (45), H2 on FA-2 ∪ FA-3 (60). Endpoints: exact_match
  (exact McNemar), answer_recall / groundedness / completeness (Wilcoxon
  signed-rank, zeros by Pratt). Two-sided, alpha 0.05, Holm within each
  hypothesis. Confirmed only if significant in the predicted direction.
* Reasoning endpoints (H3) are tested on the EMISSION INTERSECTION: items
  where both systems emitted a chain in >= 2 of 3 runs, value = mean over
  the runs with a chain. The strict variant (chain in all 3 runs for both)
  is reported as sensitivity.
* Failed queries would count as wrong (none occurred); empty answers are
  regular answers scored 0.
* Effect sizes: paired risk difference and conditional odds ratio for
  McNemar; matched-pairs rank-biserial correlation and mean difference for
  Wilcoxon. 95 % CIs from a paired item-level bootstrap, seed 20260911,
  10 000 resamples.
* Everything outside the seven confirmatory tests is exploratory and is
  reported with unadjusted p-values.

    uv run python scripts/analyze_final_eval.py run1 run2 run3
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import stats

from scripts.aggregate_runs import _clean
from scripts.generate_report import SYSTEMS, latest_result_file
from src.evaluation.gold_standard_loader import GOLD_STANDARD_EN

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "data" / "results" / "final_eval"
GOLD_STANDARD = GOLD_STANDARD_EN  # canonical path, see gold_standard_loader

BOOTSTRAP_SEED = 20260911
BOOTSTRAP_REPS = 10_000
ALPHA = 0.05
LATENCY_EXCLUDED_RUNS = ("run1",)  # see aggregate_runs.LATENCY_METRICS

SYSTEM_LABELS = dict(SYSTEMS)  # "S1" -> "rag_monolith"
LABEL_BY_NAME = {v: k for k, v in SYSTEMS}

# Planned contrasts. `predicted` is the system the hypothesis names as
# superior (B in Kap. 3.6.4); `baseline` is A.
HYPOTHESES: list[dict[str, Any]] = [
    {
        "id": "H1", "baseline": "S2", "predicted": "S3",
        "population": "FA-4 ∪ FA-3 cross_window",
        "endpoints": ["exact_match", "answer_recall"],
    },
    {
        "id": "H2", "baseline": "S1", "predicted": "S2",
        "population": "FA-2 ∪ FA-3",
        "endpoints": ["exact_match", "answer_recall"],
    },
    {
        "id": "H3", "baseline": "S3", "predicted": "S4",
        "population": "FA-4 ∪ FA-3 cross_window",
        "endpoints": ["exact_match", "groundedness", "completeness"],
    },
]

BINARY_ENDPOINTS = {"exact_match"}
REASONING_ENDPOINTS = {"groundedness", "completeness", "validity"}
CONTINUOUS_ENDPOINTS = {
    "answer_recall", "groundedness", "completeness", "validity",
    "citation_accuracy", "answer_correctness", "answer_relevancy",
    "locus_faithfulness",
}

EXPLORATORY_ENDPOINTS = [
    "exact_match", "answer_recall", "groundedness", "completeness",
    "citation_accuracy", "answer_correctness",
    # Post hoc, exploratory; pairs use the items scored for both systems,
    # i.e. where both answers cite a locus (see paired_arrays).
    "locus_faithfulness",
]

STRATUM_METRICS = [
    "exact_match", "answer_recall", "answer_correctness", "answer_relevancy",
    "citation_accuracy", "over_refusal", "chain_emitted", "groundedness",
    "validity", "completeness", "fully_grounded", "locus_faithfulness",
    "locus_faithfulness_scored", "total_tokens", "estimated_cost_usd",
]

B_CODES = ["B1", "B2", "B3", "B4", "B5", "B6"]
V_CODES = ["V1", "V2", "V3", "V4", "V5"]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

@dataclass
class Item:
    """One gold-standard item with its per-run, per-system observations."""
    query_id: int
    fa_type: str
    subtype: str
    window_class: str
    math_type: str
    entity_form: str
    gt_unit: str
    refusal_evidence: str
    expected_answerable: bool
    # obs[system_label][run_id] -> flat metric dict
    obs: dict = field(default_factory=lambda: defaultdict(dict))

    def population_membership(self) -> set[str]:
        pops = set()
        if self.expected_answerable:
            pops.add("answerable")
            pops.add(self.fa_type)
            if self.fa_type in ("FA-2", "FA-3"):
                pops.add("FA-2 ∪ FA-3")
            if self.fa_type == "FA-4" or (
                self.fa_type == "FA-3" and self.window_class == "cross_window"
            ):
                pops.add("FA-4 ∪ FA-3 cross_window")
        else:
            pops.add("FA-Refusal")
        return pops


def _flatten(record: dict) -> dict:
    cm = record.get("custom_metrics") or {}
    rm = record.get("reasoning_metrics") or {}
    rg = record.get("ragas_metrics") or {}
    ci = record.get("citation_metrics") or {}
    run = record.get("run_metrics") or {}
    lf = record.get("locus_faithfulness") or {}
    flat = {
        "exact_match": _clean(cm.get("exact_match")),
        "answer_recall": _clean(cm.get("answer_recall")),
        "refusal_accuracy": _clean(cm.get("refusal_accuracy")),
        "refusal_quality": _clean(cm.get("refusal_quality")),
        "over_refusal": _clean(cm.get("over_refusal")),
        "citation_accuracy": _clean(ci.get("citation_accuracy")),
        "citation_method": ci.get("method"),
        "chain_emitted": 1.0 if rm.get("chain_emitted") else 0.0,
        "groundedness": _clean(rm.get("groundedness")),
        "validity": _clean(rm.get("validity")),
        "completeness": _clean(rm.get("completeness")),
        "fully_grounded": (
            None if rm.get("fully_grounded") is None
            else float(bool(rm.get("fully_grounded")))
        ),
        "fully_valid": (
            None if rm.get("fully_valid") is None
            else float(bool(rm.get("fully_valid")))
        ),
        "num_chain_steps": _clean(rm.get("num_steps")),
        "violation_counts": rm.get("violation_counts") or {},
        "answer_relevancy": _clean(rg.get("answer_relevancy")),
        "answer_correctness": _clean(rg.get("answer_correctness")),
        "context_precision": _clean(rg.get("context_precision")),
        "context_recall": _clean(rg.get("context_recall")),
        "faithfulness": _clean(rg.get("faithfulness")),
        "locus_faithfulness": _clean(lf.get("score")),
        # Coverage flag on the answerable stratum: 1 scored, 0 excluded
        # (no locus cited / fabricated filing / judge NaN), None on refusal
        # items and on files without the metric.
        "locus_faithfulness_scored": (
            None if not lf or lf.get("excluded") == "not_answerable"
            else float(lf.get("score") is not None)
        ),
        "latency_seconds": _clean(run.get("latency_seconds")),
        "total_tokens": _clean(run.get("total_tokens")),
        "estimated_cost_usd": _clean(run.get("estimated_cost_usd")),
        "tool_calls": _clean(run.get("num_steps")),
        "corrections": _clean(run.get("corrections")),
        "query_attempts": _clean(record.get("query_attempts")),
        "empty_answer": 1.0 if not (record.get("answer") or "").strip() else 0.0,
    }
    return flat


def load_gold_standard(path: Path) -> dict[int, dict]:
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    return {int(r["id"]): r for r in rows}


def load_items(runs_root: Path, run_ids: list[str], gs_path: Path) -> dict[int, Item]:
    gs = load_gold_standard(gs_path)
    items: dict[int, Item] = {}
    for qid, row in gs.items():
        items[qid] = Item(
            query_id=qid,
            fa_type=row["fa_type"],
            subtype=row["subtype"],
            window_class=row.get("window_class", ""),
            math_type=row.get("math_type", ""),
            entity_form=row.get("entity_form", ""),
            gt_unit=row.get("gt_unit", ""),
            refusal_evidence=row.get("refusal_evidence", ""),
            expected_answerable=str(row["expected_answerable"]).strip().lower()
            in ("true", "1", "yes"),
        )
    for run_id in run_ids:
        run_dir = runs_root / run_id
        for label, name in SYSTEMS:
            path = latest_result_file(run_dir, name)
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            logger.info("%s %s <- %s", run_id, label, path.name)
            for rec in data["detailed_results"]:
                qid = int(rec["query_id"])
                if qid not in items:
                    raise KeyError(f"query_id {qid} in {path} not in gold standard")
                items[qid].obs[label][run_id] = _flatten(rec)
    return items


# --------------------------------------------------------------------------
# Item-level collapsing across runs
# --------------------------------------------------------------------------

def majority_correct(values: list[float | None]) -> float:
    """1.0 if exact_match == 1.0 in at least half of the runs (>= 2 of 3).

    A missing value (failed query) or an off-schema 0.5 counts as wrong.
    """
    n = len(values)
    hits = sum(1 for v in values if v is not None and v == 1.0)
    return 1.0 if hits * 2 >= n and n > 0 else 0.0


def item_mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def collapse(
    items: dict[int, Item], system: str, metric: str, run_ids: list[str],
) -> dict[int, float | None]:
    """One value per item for a system, across runs (majority or mean)."""
    out: dict[int, float | None] = {}
    for qid, it in items.items():
        vals = [it.obs[system].get(r, {}).get(metric) for r in run_ids]
        if metric in BINARY_ENDPOINTS:
            out[qid] = majority_correct(vals)
        else:
            out[qid] = item_mean(vals)
    return out


def emission_intersection(
    items: dict[int, Item], sys_a: str, sys_b: str, run_ids: list[str],
    strict: bool = False,
) -> set[int]:
    """Items where both systems emitted a chain (majority of runs, or all)."""
    need = len(run_ids) if strict else (len(run_ids) + 1) // 2
    keep = set()
    for qid, it in items.items():
        ok = True
        for s in (sys_a, sys_b):
            emitted = sum(
                1 for r in run_ids if it.obs[s].get(r, {}).get("chain_emitted") == 1.0
            )
            if emitted < need:
                ok = False
        if ok:
            keep.add(qid)
    return keep


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact McNemar on paired binaries. `b` is the predicted-superior system.

    b01 = A wrong, B right; b10 = A right, B wrong.
    """
    a = a.astype(int)
    b = b.astype(int)
    n = len(a)
    both = int(((a == 1) & (b == 1)).sum())
    neither = int(((a == 0) & (b == 0)).sum())
    only_b = int(((a == 0) & (b == 1)).sum())
    only_a = int(((a == 1) & (b == 0)).sum())
    discordant = only_a + only_b
    if discordant == 0:
        p = 1.0
    else:
        p = stats.binomtest(min(only_a, only_b), n=discordant, p=0.5,
                            alternative="two-sided").pvalue
    risk_diff = (only_b - only_a) / n
    if only_a == 0 and only_b == 0:
        odds = None
    elif only_a == 0:
        odds = float("inf")
    else:
        odds = only_b / only_a
    return {
        "test": "McNemar exact",
        "n": n, "both": both, "neither": neither,
        "only_predicted": only_b, "only_baseline": only_a,
        "discordant": discordant,
        "rate_baseline": float(a.mean()), "rate_predicted": float(b.mean()),
        "effect": risk_diff, "effect_name": "paired risk difference",
        "odds_ratio": odds, "p": float(p),
    }


def wilcoxon_pratt(a: np.ndarray, b: np.ndarray) -> dict:
    """Wilcoxon signed-rank on paired continuous values, zeros by Pratt.

    Rank-biserial r uses the Pratt ranks (zeros ranked, then dropped).
    """
    d = b - a
    n = len(d)
    n_zero = int((d == 0).sum())
    if np.all(d == 0):
        return {
            "test": "Wilcoxon signed-rank (Pratt)", "n": n, "n_zero": n_zero,
            "mean_baseline": float(a.mean()), "mean_predicted": float(b.mean()),
            "effect": 0.0, "effect_name": "mean difference",
            "rank_biserial": 0.0, "p": 1.0, "method": "all zero",
        }
    # scipy has no exact distribution under Pratt (method="exact" silently
    # enumerates as if every zero were a signed pair and returns nonsense),
    # so with zeros present this is the normal approximation with
    # continuity correction. Below ~10 non-zero pairs that approximation is
    # not trustworthy; the exact Wilcox p (zeros dropped) is carried along
    # as a control and the row is flagged.
    res = stats.wilcoxon(b, a, zero_method="pratt", alternative="two-sided",
                         method="approx")
    nonzero = d[d != 0]
    p_wilcox_exact = float(stats.wilcoxon(
        nonzero, alternative="two-sided", method="exact").pvalue)
    ranks = stats.rankdata(np.abs(d))
    t_plus = float(ranks[d > 0].sum())
    t_minus = float(ranks[d < 0].sum())
    rb = (t_plus - t_minus) / (t_plus + t_minus) if (t_plus + t_minus) else 0.0
    return {
        "test": "Wilcoxon signed-rank (Pratt, normal approx.)", "n": n, "n_zero": n_zero,
        "n_nonzero": int(len(nonzero)),
        "n_positive": int((d > 0).sum()), "n_negative": int((d < 0).sum()),
        "mean_baseline": float(a.mean()), "mean_predicted": float(b.mean()),
        "effect": float(d.mean()), "effect_name": "mean difference",
        "rank_biserial": rb, "statistic": float(res.statistic),
        "p": float(res.pvalue),
        "p_wilcox_exact": p_wilcox_exact,
        "approx_unreliable": bool(len(nonzero) < 10),
    }


def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, stat: Callable[[np.ndarray, np.ndarray], float],
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(reps, n))
    vals = np.array([stat(a[i], b[i]) for i in idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvalues[i])
        running = max(running, val)
        adjusted[i] = running
    return adjusted


def paired_arrays(
    items: dict[int, Item], qids: list[int], sys_a: str, sys_b: str,
    metric: str, run_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    ca = collapse(items, sys_a, metric, run_ids)
    cb = collapse(items, sys_b, metric, run_ids)
    keep = [q for q in qids if ca[q] is not None and cb[q] is not None]
    return (np.array([ca[q] for q in keep], dtype=float),
            np.array([cb[q] for q in keep], dtype=float), keep)


def run_contrast(
    items: dict[int, Item], population: list[int], sys_a: str, sys_b: str,
    metric: str, run_ids: list[str],
) -> dict:
    a, b, used = paired_arrays(items, population, sys_a, sys_b, metric, run_ids)
    if metric in BINARY_ENDPOINTS:
        res = mcnemar_exact(a, b)
        lo, hi = paired_bootstrap_ci(a, b, lambda x, y: float(y.mean() - x.mean()))
    else:
        res = wilcoxon_pratt(a, b)
        lo, hi = paired_bootstrap_ci(a, b, lambda x, y: float((y - x).mean()))
    res.update({
        "baseline": sys_a, "predicted": sys_b, "metric": metric,
        "n_population": len(population), "n_used": len(used),
        "ci_low": lo, "ci_high": hi,
        "direction_predicted": res["effect"] > 0,
    })
    return res


# --------------------------------------------------------------------------
# Confirmatory block
# --------------------------------------------------------------------------

def population_ids(items: dict[int, Item], name: str) -> list[int]:
    return sorted(q for q, it in items.items() if name in it.population_membership())


def confirmatory(items: dict[int, Item], run_ids: list[str]) -> list[dict]:
    rows = []
    for h in HYPOTHESES:
        pop = population_ids(items, h["population"])
        results = []
        for metric in h["endpoints"]:
            if metric in REASONING_ENDPOINTS:
                inter = emission_intersection(items, h["baseline"], h["predicted"], run_ids)
                pop_m = [q for q in pop if q in inter]
                note = f"emission intersection (>=2/3 runs): {len(pop_m)}/{len(pop)}"
            else:
                pop_m, note = pop, ""
            res = run_contrast(items, pop_m, h["baseline"], h["predicted"], metric, run_ids)
            res["note"] = note
            results.append(res)
        adj = holm([r["p"] for r in results])
        for r, pa in zip(results, adj):
            r["p_holm"] = pa
            r["significant"] = pa < ALPHA
            r["confirmed"] = bool(pa < ALPHA and r["direction_predicted"])
            r["hypothesis"] = h["id"]
            r["population"] = h["population"]
        n_conf = sum(r["confirmed"] for r in results)
        verdict = ("bestätigt" if n_conf == len(results)
                   else "teilweise bestätigt" if n_conf else "nicht bestätigt")
        for r in results:
            r["verdict"] = verdict
        rows.extend(results)
    return rows


def reasoning_sensitivity(items: dict[int, Item], run_ids: list[str]) -> list[dict]:
    """H3 reasoning endpoints under the strict intersection and on all items."""
    h = HYPOTHESES[2]
    pop = population_ids(items, h["population"])
    rows = []
    for metric in ("groundedness", "completeness", "validity"):
        for variant in ("majority", "strict", "no_rule"):
            if variant == "no_rule":
                pop_m = pop
            else:
                inter = emission_intersection(
                    items, h["baseline"], h["predicted"], run_ids,
                    strict=(variant == "strict"))
                pop_m = [q for q in pop if q in inter]
            res = run_contrast(items, pop_m, h["baseline"], h["predicted"], metric, run_ids)
            res.update({"variant": variant, "hypothesis": "H3"})
            rows.append(res)
    return rows


# --------------------------------------------------------------------------
# Exploratory blocks
# --------------------------------------------------------------------------

ALL_PAIRS = [("S1", "S2"), ("S2", "S3"), ("S3", "S4"),
             ("S1", "S3"), ("S2", "S4"), ("S1", "S4")]
PLANNED = {("S2", "S3"), ("S1", "S2"), ("S3", "S4")}
EXPLORATORY_POPULATIONS = [
    "answerable", "FA-1", "FA-2", "FA-3", "FA-4",
    "FA-2 ∪ FA-3", "FA-4 ∪ FA-3 cross_window",
]


def exploratory_pairs(items: dict[int, Item], run_ids: list[str]) -> list[dict]:
    rows = []
    for a, b in ALL_PAIRS:
        for popname in EXPLORATORY_POPULATIONS:
            pop = population_ids(items, popname)
            for metric in EXPLORATORY_ENDPOINTS:
                pop_m = pop
                if metric in REASONING_ENDPOINTS:
                    inter = emission_intersection(items, a, b, run_ids)
                    pop_m = [q for q in pop if q in inter]
                if len(pop_m) < 5:
                    continue
                res = run_contrast(items, pop_m, a, b, metric, run_ids)
                res["pair"] = f"{a} vs {b}"
                res["population"] = popname
                res["planned"] = (a, b) in PLANNED
                rows.append(res)
    return rows


def _per_run_stratum_mean(
    items: dict[int, Item], qids: list[int], system: str, metric: str, run_id: str,
) -> float | None:
    vals = [items[q].obs[system].get(run_id, {}).get(metric) for q in qids]
    return item_mean(vals)


def stratum_breakdown(
    items: dict[int, Item], run_ids: list[str], key: str,
    restrict_answerable: bool = True,
) -> list[dict]:
    """Mean ± SD across runs of the per-run stratum mean, per system."""
    groups: dict[str, list[int]] = defaultdict(list)
    for q, it in items.items():
        if restrict_answerable and not it.expected_answerable:
            continue
        val = getattr(it, key) or "(none)"
        groups[val].append(q)
    rows = []
    for gname, qids in sorted(groups.items()):
        for label, _ in SYSTEMS:
            row = {"grouping": key, "group": gname, "n_items": len(qids), "system": label}
            for metric in STRATUM_METRICS:
                per_run = [
                    _per_run_stratum_mean(items, qids, label, metric, r) for r in run_ids
                ]
                vals = [v for v in per_run if v is not None]
                row[f"{metric}_mean"] = (
                    round(sum(vals) / len(vals), 4) if vals else None)
                row[f"{metric}_sd"] = (
                    round(statistics.stdev(vals), 4) if len(vals) > 1 else None)
            # majority-based accuracy on the same group
            maj = collapse(items, label, "exact_match", run_ids)
            row["exact_match_majority"] = round(
                sum(maj[q] or 0.0 for q in qids) / len(qids), 4)
            rows.append(row)
    return rows


def refusal_breakdown(items: dict[int, Item], run_ids: list[str]) -> list[dict]:
    rows = []
    refusal = [q for q, it in items.items() if not it.expected_answerable]
    groupings: dict[str, dict[str, list[int]]] = {
        "all": {"all": refusal},
        "subtype": defaultdict(list),
        "refusal_evidence": defaultdict(list),
    }
    for q in refusal:
        groupings["subtype"][items[q].subtype].append(q)
        groupings["refusal_evidence"][items[q].refusal_evidence].append(q)
    for gkey, groups in groupings.items():
        for gname, qids in sorted(groups.items()):
            for label, _ in SYSTEMS:
                row = {"grouping": gkey, "group": gname, "n_items": len(qids),
                       "system": label}
                for metric in ("refusal_accuracy", "refusal_quality"):
                    per_run = [
                        _per_run_stratum_mean(items, qids, label, metric, r)
                        for r in run_ids]
                    vals = [v for v in per_run if v is not None]
                    row[f"{metric}_mean"] = (
                        round(sum(vals) / len(vals), 4) if vals else None)
                    row[f"{metric}_sd"] = (
                        round(statistics.stdev(vals), 4) if len(vals) > 1 else None)
                rows.append(row)
    return rows


def stability(items: dict[int, Item], run_ids: list[str]) -> list[dict]:
    """How often the three runs agree on exact_match, per system."""
    rows = []
    answerable = [q for q, it in items.items() if it.expected_answerable]
    n_runs = len(run_ids)
    for label, _ in SYSTEMS:
        hits_dist: Counter[int] = Counter()
        for q in answerable:
            vals = [items[q].obs[label].get(r, {}).get("exact_match") for r in run_ids]
            hits_dist[sum(1 for v in vals if v == 1.0)] += 1
        unanimous = hits_dist[0] + hits_dist[n_runs]
        majority_n = sum(c for h, c in hits_dist.items() if h * 2 >= n_runs)
        # continuous metrics: mean per-item SD across runs
        cont = {}
        for metric in ("answer_recall", "groundedness", "completeness",
                       "citation_accuracy", "answer_correctness"):
            sds = []
            for q in answerable:
                vals = [items[q].obs[label].get(r, {}).get(metric) for r in run_ids]
                vals = [v for v in vals if v is not None]
                if len(vals) > 1:
                    sds.append(statistics.stdev(vals))
            cont[f"{metric}_mean_item_sd"] = round(
                sum(sds) / len(sds), 4) if sds else None
        rows.append({
            "system": label, "n_items": len(answerable),
            **{f"hits_{h}_of_{n_runs}": hits_dist.get(h, 0) for h in range(n_runs + 1)},
            "unanimous_share": round(unanimous / len(answerable), 4),
            "majority_correct": majority_n,
            "majority_rate": round(majority_n / len(answerable), 4),
            "majority_from_split": hits_dist.get(n_runs - 1, 0) if n_runs == 3 else None,
            **cont,
        })
    return rows


def emission_check(items: dict[int, Item], run_ids: list[str]) -> list[dict]:
    """Selection check for the emission rule: what do non-emitting items look like."""
    rows = []
    answerable = [q for q, it in items.items() if it.expected_answerable]
    for label, _ in SYSTEMS:
        for r in run_ids:
            missing = [q for q in answerable
                       if items[q].obs[label].get(r, {}).get("chain_emitted") == 0.0]
            emitted = [q for q in answerable if q not in set(missing)]
            def _em(qs):
                v = [items[q].obs[label][r].get("exact_match") for q in qs]
                v = [x for x in v if x is not None]
                return round(sum(v) / len(v), 4) if v else None
            rows.append({
                "system": label, "run": r,
                "n_answerable": len(answerable), "n_no_chain": len(missing),
                "emission_rate": round(1 - len(missing) / len(answerable), 4),
                "no_chain_by_fa_type": dict(Counter(items[q].fa_type for q in missing)),
                "no_chain_empty_answers": sum(
                    1 for q in missing if items[q].obs[label][r].get("empty_answer") == 1.0),
                "exact_match_no_chain": _em(missing),
                "exact_match_with_chain": _em(emitted),
                "no_chain_ids": missing,
            })
    return rows


def violation_codes(items: dict[int, Item], run_ids: list[str]) -> list[dict]:
    rows = []
    answerable = [q for q, it in items.items() if it.expected_answerable]
    for label, _ in SYSTEMS:
        counts: Counter[str] = Counter()
        n_chains = 0
        for q in answerable:
            for r in run_ids:
                o = items[q].obs[label].get(r, {})
                if o.get("chain_emitted") != 1.0:
                    continue
                n_chains += 1
                for k, v in (o.get("violation_counts") or {}).items():
                    counts[k] += int(v or 0)
        row = {"system": label, "n_chains": n_chains}
        for code in B_CODES + V_CODES:
            row[code] = counts.get(code, 0)
        row["B_total"] = sum(counts.get(c, 0) for c in B_CODES)
        row["V_total"] = sum(counts.get(c, 0) for c in V_CODES)
        rows.append(row)
    return rows


def efficiency(items: dict[int, Item], run_ids: list[str]) -> list[dict]:
    rows = []
    latency_runs = [r for r in run_ids if r not in LATENCY_EXCLUDED_RUNS]
    for label, _ in SYSTEMS:
        tokens, cost, lat, calls, corr, attempts, empties = [], [], [], [], [], [], 0
        for q, it in items.items():
            for r in run_ids:
                o = it.obs[label].get(r, {})
                if not o:
                    continue
                for key, bucket in (("total_tokens", tokens), ("estimated_cost_usd", cost),
                                    ("tool_calls", calls), ("corrections", corr),
                                    ("query_attempts", attempts)):
                    if o.get(key) is not None:
                        bucket.append(o[key])
                empties += int(o.get("empty_answer") == 1.0)
                if r in latency_runs and o.get("latency_seconds") is not None:
                    lat.append(o["latency_seconds"])
        rows.append({
            "system": label, "n_obs": len(tokens),
            "tokens_mean": round(statistics.mean(tokens), 1),
            "tokens_median": round(statistics.median(tokens), 1),
            "tokens_p90": round(float(np.percentile(tokens, 90)), 1),
            "cost_per_query_mean_usd": round(statistics.mean(cost), 5),
            "cost_per_run_usd": round(sum(cost) / len(run_ids), 4),
            "latency_runs": "+".join(latency_runs),
            "latency_mean_s": round(statistics.mean(lat), 2) if lat else None,
            "latency_median_s": round(statistics.median(lat), 2) if lat else None,
            "latency_p90_s": round(float(np.percentile(lat, 90)), 2) if lat else None,
            "tool_calls_mean": round(statistics.mean(calls), 2) if calls else None,
            "tool_calls_median": statistics.median(calls) if calls else None,
            "tool_calls_zero_share": round(
                sum(1 for c in calls if c == 0) / len(calls), 4) if calls else None,
            "correction_rate": round(
                sum(1 for c in corr if c and c > 0) / len(corr), 4) if corr else None,
            "retried_share": round(
                sum(1 for a in attempts if a and a > 1) / len(attempts), 4) if attempts else None,
            "empty_answers": empties,
        })
    return rows


def cost_per_correct(items: dict[int, Item], run_ids: list[str]) -> list[dict]:
    """Kap. 3.6.2 definition, recomputed per run and averaged.

    Correct = refusal_accuracy >= 0.8 on refusal items; exact_match >= 0.8
    on answerable items with a numeric unit; answer_recall >= 0.8 otherwise.
    """
    rows = []
    for label, _ in SYSTEMS:
        per_run = []
        for r in run_ids:
            total_cost, n_correct = 0.0, 0
            for q, it in items.items():
                o = it.obs[label].get(r)
                if not o:
                    continue
                total_cost += o.get("estimated_cost_usd") or 0.0
                if not it.expected_answerable:
                    score = o.get("refusal_accuracy")
                elif it.gt_unit not in ("text", "n/a", ""):
                    score = o.get("exact_match")
                else:
                    score = o.get("answer_recall")
                if score is not None and score >= 0.8:
                    n_correct += 1
            per_run.append((total_cost, n_correct,
                            total_cost / n_correct if n_correct else None))
        cpc = [x[2] for x in per_run if x[2] is not None]
        rows.append({
            "system": label,
            "cost_per_run_usd": round(statistics.mean(x[0] for x in per_run), 4),
            "correct_per_run": round(statistics.mean(x[1] for x in per_run), 2),
            "cost_per_correct_usd_mean": round(statistics.mean(cpc), 5) if cpc else None,
            "cost_per_correct_usd_sd": round(statistics.stdev(cpc), 5) if len(cpc) > 1 else None,
        })
    return rows


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list))
                            else v) for k, v in r.items()})


def _f(x, d=3):
    if x is None:
        return "—"
    if isinstance(x, float) and x == float("inf"):
        return "∞"
    return f"{x:.{d}f}"


def _p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _conf_table(rows: list[dict]) -> str:
    out = ["| H | Paar | Endpunkt | n | A | B | Δ (B−A) | 95%-CI | Effekt | p | p_Holm | Richtung | Verdikt |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["metric"] in BINARY_ENDPOINTS:
            a, b = r["rate_baseline"], r["rate_predicted"]
            eff = f"b={r['only_predicted']}, c={r['only_baseline']}, OR={_f(r['odds_ratio'],2)}"
        else:
            a, b = r["mean_baseline"], r["mean_predicted"]
            eff = (f"r_rb={_f(r['rank_biserial'],2)}, nonzero={r['n_nonzero']}, "
                   f"p_exact(wilcox)={_p(r['p_wilcox_exact'])}"
                   + (" ⚠ <10 nonzero" if r["approx_unreliable"] else ""))
        out.append(
            f"| {r['hypothesis']} | {r['baseline']} vs {r['predicted']} | {r['metric']} | "
            f"{r['n_used']} | {_f(a)} | {_f(b)} | {_f(r['effect'])} | "
            f"[{_f(r['ci_low'])}, {_f(r['ci_high'])}] | {eff} | {_p(r['p'])} | "
            f"{_p(r['p_holm'])} | {'✓' if r['direction_predicted'] else '✗'} | "
            f"{'✓' if r['confirmed'] else '—'} ({r['verdict']}) |")
    return "\n".join(out)


def _expl_table(rows: list[dict]) -> str:
    out = ["| Paar | Population | Endpunkt | n | A | B | Δ | 95%-CI | p (unadj.) | geplant |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["metric"] in BINARY_ENDPOINTS:
            a, b = r["rate_baseline"], r["rate_predicted"]
        else:
            a, b = r["mean_baseline"], r["mean_predicted"]
        out.append(
            f"| {r['pair']} | {r['population']} | {r['metric']} | {r['n_used']} | "
            f"{_f(a)} | {_f(b)} | {_f(r['effect'])} | [{_f(r['ci_low'])}, {_f(r['ci_high'])}] | "
            f"{_p(r['p'])} | {'ja' if r['planned'] else 'nein'} |")
    return "\n".join(out)


def _stratum_table(rows: list[dict], metrics: list[str]) -> str:
    head = "| Gruppe | n | System | " + " | ".join(metrics) + " |"
    out = [head, "|" + "---|" * (3 + len(metrics))]
    for r in rows:
        cells = []
        for m in metrics:
            mean, sd = r.get(f"{m}_mean"), r.get(f"{m}_sd")
            cells.append("—" if mean is None else (
                f"{mean:.3f}" if sd is None else f"{mean:.3f} ± {sd:.3f}"))
        out.append(f"| {r['group']} | {r['n_items']} | {r['system']} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def _simple_table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(
            "—" if r.get(c) is None else (f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]))
            for c in cols) + " |")
    return "\n".join(out)


def write_report(out_dir: Path, run_ids: list[str], blocks: dict) -> Path:
    conf = blocks["confirmatory"]
    md = [
        "# Statistische Auswertung der drei Vollläufe",
        "",
        f"Läufe: {', '.join(run_ids)}. Item-Ebene; binäre Endpunkte über die Mehrheitsregel "
        f"(exact_match = 1,0 in ≥ 2 von {len(run_ids)} Läufen), stetige über das Item-Mittel. "
        f"Zweiseitig, α = {ALPHA}, Holm je Hypothese. 95%-CIs: gepaarter Item-Bootstrap, "
        f"{BOOTSTRAP_REPS} Wiederholungen, Seed {BOOTSTRAP_SEED}. Wilcoxon-Nullen nach Pratt. "
        "Reasoning-Endpunkte auf der Emissions-Schnittmenge (Kette bei beiden Systemen in ≥ 2 von 3 Läufen).",
        "",
        "## 1 Konfirmatorische Hypothesenprüfung",
        "",
        _conf_table(conf),
        "",
    ]
    for h in HYPOTHESES:
        rows = [r for r in conf if r["hypothesis"] == h["id"]]
        md.append(f"**{h['id']}** ({h['baseline']} vs {h['predicted']}, {h['population']}, "
                  f"n = {rows[0]['n_population']}): **{rows[0]['verdict']}**.")
        for r in rows:
            if r["metric"] in BINARY_ENDPOINTS:
                md.append(f"- {r['metric']}: Kontingenz beide richtig {r['both']}, beide falsch "
                          f"{r['neither']}, nur {r['predicted']} {r['only_predicted']}, nur "
                          f"{r['baseline']} {r['only_baseline']} (diskordant {r['discordant']}).")
            else:
                md.append(f"- {r['metric']}: n = {r['n_used']}, Differenzen +{r['n_positive']} / "
                          f"−{r['n_negative']} / 0: {r['n_zero']}"
                          + (f"; {r['note']}" if r.get("note") else "") + ".")
        md.append("")

    md += ["## 2 Sensitivität der Reasoning-Endpunkte (H3) gegenüber der Emissionsregel", "",
           "| Endpunkt | Variante | n | S3 | S4 | Δ | 95%-CI | p (unadj.) |",
           "|---|---|---|---|---|---|---|---|"]
    for r in blocks["reasoning_sensitivity"]:
        md.append(f"| {r['metric']} | {r['variant']} | {r['n_used']} | {_f(r['mean_baseline'])} | "
                  f"{_f(r['mean_predicted'])} | {_f(r['effect'])} | "
                  f"[{_f(r['ci_low'])}, {_f(r['ci_high'])}] | {_p(r['p'])} |")
    md += ["", "## 3 Emissionsregel: Selektionsprüfung", "",
           "| System | Lauf | Emissionsrate | ohne Kette | davon Leerantwort | nach FA | "
           "exact_match ohne / mit Kette |",
           "|---|---|---|---|---|---|---|"]
    for r in blocks["emission_check"]:
        md.append(f"| {r['system']} | {r['run']} | {r['emission_rate']:.3f} | {r['n_no_chain']} | "
                  f"{r['no_chain_empty_answers']} | {r['no_chain_by_fa_type']} | "
                  f"{_f(r['exact_match_no_chain'])} / "
                  f"{_f(r['exact_match_with_chain'])} |")

    md += ["", "## 4 Explorative Paarvergleiche (unadjustiert)", "",
           "Alle sechs Paare auf allen Populationen; die drei geplanten Paare auf ihrer "
           "Hypothesenpopulation sind bereits oben konfirmatorisch berichtet.", "",
           _expl_table(blocks["exploratory"]), ""]

    md += ["## 5 Aufschlüsselung nach Stratum (Mittel ± SD über Läufe)", ""]
    for key, rows in blocks["strata"].items():
        md += [f"### nach {key}", "",
               _stratum_table(rows, ["exact_match", "answer_recall", "answer_correctness",
                                     "citation_accuracy", "groundedness", "completeness",
                                     "over_refusal"]), ""]
    md += ["### exact_match nach Mehrheitsregel je Stratum", "",
           _simple_table(blocks["strata"]["fa_type"],
                         ["group", "n_items", "system", "exact_match_majority"]), ""]

    md += ["## 6 Refusal-Stratum (Judge-Werte, Nebenbefund)", "",
           _stratum_table(blocks["refusal"], ["refusal_accuracy", "refusal_quality"]), ""]

    md += ["## 7 Stabilität über die Läufe (beantwortbare Items)", "",
           _simple_table(blocks["stability"], [
               "system", "hits_0_of_3", "hits_1_of_3", "hits_2_of_3", "hits_3_of_3",
               "unanimous_share", "majority_rate", "majority_from_split",
               "answer_recall_mean_item_sd", "groundedness_mean_item_sd",
               "completeness_mean_item_sd"]), ""]

    md += ["## 8 Belegtheits- und Validitätscodes (Summe über Ketten, alle Läufe)", "",
           _simple_table(blocks["codes"],
                         ["system", "n_chains"] + B_CODES + ["B_total"] + V_CODES + ["V_total"]), ""]

    md += ["## 9 Effizienz und Prozess", "",
           _simple_table(blocks["efficiency"], [
               "system", "tokens_mean", "tokens_median", "tokens_p90", "cost_per_run_usd",
               "latency_runs", "latency_mean_s", "latency_median_s", "latency_p90_s",
               "tool_calls_mean", "tool_calls_median", "tool_calls_zero_share",
               "correction_rate", "retried_share", "empty_answers"]), "",
           "### Kosten je korrekter Antwort (Definition Kap. 3.6.2, je Lauf berechnet)", "",
           _simple_table(blocks["cost_per_correct"], [
               "system", "cost_per_run_usd", "correct_per_run",
               "cost_per_correct_usd_mean", "cost_per_correct_usd_sd"]), ""]

    path = out_dir / "statistical_analysis.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_ids", nargs="+")
    ap.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--gold-standard", type=Path, default=GOLD_STANDARD)
    args = ap.parse_args(argv)
    out_dir = args.out_dir or (args.runs_root / "aggregate")
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_items(args.runs_root, args.run_ids, args.gold_standard)
    run_ids = args.run_ids

    blocks: dict[str, Any] = {
        "confirmatory": confirmatory(items, run_ids),
        "reasoning_sensitivity": reasoning_sensitivity(items, run_ids),
        "emission_check": emission_check(items, run_ids),
        "exploratory": exploratory_pairs(items, run_ids),
        "strata": {
            key: stratum_breakdown(items, run_ids, key)
            for key in ("fa_type", "subtype", "window_class", "math_type",
                        "entity_form", "gt_unit")
        },
        "refusal": refusal_breakdown(items, run_ids),
        "stability": stability(items, run_ids),
        "codes": violation_codes(items, run_ids),
        "efficiency": efficiency(items, run_ids),
        "cost_per_correct": cost_per_correct(items, run_ids),
    }

    _write_csv(blocks["confirmatory"], out_dir / "hypothesis_tests.csv")
    _write_csv(blocks["reasoning_sensitivity"], out_dir / "h3_reasoning_sensitivity.csv")
    _write_csv(blocks["emission_check"], out_dir / "emission_check.csv")
    _write_csv(blocks["exploratory"], out_dir / "exploratory_pairs.csv")
    _write_csv([r for rows in blocks["strata"].values() for r in rows],  # type: ignore[attr-defined]
               out_dir / "stratum_breakdown.csv")
    _write_csv(blocks["refusal"], out_dir / "refusal_breakdown.csv")
    _write_csv(blocks["stability"], out_dir / "stability.csv")
    _write_csv(blocks["codes"], out_dir / "violation_codes.csv")
    _write_csv(blocks["efficiency"], out_dir / "efficiency.csv")
    _write_csv(blocks["cost_per_correct"], out_dir / "cost_per_correct.csv")
    # item-level collapsed values, for figures and spot checks
    item_rows = []
    for q, it in sorted(items.items()):
        for label, _ in SYSTEMS:
            row = {"query_id": q, "fa_type": it.fa_type, "subtype": it.subtype,
                   "window_class": it.window_class,
                   "expected_answerable": it.expected_answerable, "system": label}
            for m in ["exact_match", "answer_recall", "groundedness", "completeness",
                      "validity", "citation_accuracy", "answer_correctness",
                      "refusal_accuracy", "over_refusal", "chain_emitted"]:
                row[m] = collapse(items, label, m, run_ids)[q]
            item_rows.append(row)
    _write_csv(item_rows, out_dir / "item_level_collapsed.csv")

    path = write_report(out_dir, run_ids, blocks)
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
