"""
Measure the groundedness judgement against the human labels already collected.

WHY THIS EXISTS
---------------
The pilot put groundedness at kappa 0.21 with 86% raw agreement, and the
confusion matrix showed the problem is not prevalence alone:

    on the 97 steps the rater called grounded, the judge agreed on 89 (92%)
    on the 10 steps the rater called NOT grounded, it agreed on 3 (30%)

A larger sample cannot fix that. Kappa is a function of the proportions in
that matrix, not of n -- doubling the items at an unchanged detection rate
returns the same 0.21 with a narrower confidence interval around it. What
moves it is detection, so what is needed is a bench, not more data.

This replays the stored pilot chains against the CURRENT prompt and scores
the result against the human labels. The passages come from `evidence_shown`
in the run output, not from a fresh corpus lookup, so the judge sees exactly
what it saw the first time and a change in the numbers can only come from the
change under test.

Usage:
    uv run python scripts/groundedness_calibration.py
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "judge_validation"
RAW_DIR = RESULTS_DIR / "raw_eval_reserve"
SYSTEMS = ["rag_monolith", "rag_agent", "long_context", "multi_agent"]


class ReplayStore:
    """An EvidenceStore stand-in that serves the passages already recorded.

    Re-fetching from the corpus would let a store rebuild quietly change the
    input and make the before/after comparison meaningless.
    """

    def __init__(self, by_step: dict[int, list[str]]):
        self._by_step = by_step
        self._current: int | None = None

    def for_step(self, step_index: int) -> None:
        self._current = step_index

    def lookup(self, doc_id, section_text, claim_text, max_passages=6):
        from src.evaluation.evidence_store import LookupResult
        passages = self._by_step.get(self._current or -1, [])
        return LookupResult(passages=list(passages), doc_exists=bool(passages))


def _latest(system: str) -> Path:
    matches = sorted(glob.glob(str(RAW_DIR / f"eval_{system}_*.json")))
    if not matches:
        raise FileNotFoundError(f"No eval JSON for {system}")
    return Path(matches[-1])


def _human_labels() -> dict[tuple[str, int, int], int]:
    """(system, query_id, step_index) -> 0/1.

    The system belongs in the key: all four answer the same query_id, so a
    (query_id, step) key silently collapses four labels into whichever was
    read last -- which is how the first version of this bench "reproduced" a
    baseline of 81/2/17/7 against an actual 89/3/7/8.
    """
    ref = json.loads(
        (RESULTS_DIR / "judge_scores_reference_reserve.json").read_text(encoding="utf-8")
    )
    with (RESULTS_DIR / "human_review_blind_reserve.csv").open(encoding="utf-8-sig") as fh:
        rows = {r["review_id"]: r for r in csv.DictReader(fh)}
    out = {}
    for review_id, entry in ref.items():
        raw = rows[review_id]["groundedness_human"]
        if raw in ("", "N/A"):
            continue
        for step, value in json.loads(raw).items():
            out[(entry["system_name"], entry["query_id"], int(step))] = int(float(value))
    return out


def main() -> int:
    if "--rejudge" in sys.argv:
        return rejudge()

    from src.evaluation.reasoning_chain_evaluator import evaluate_reasoning_chain

    human = _human_labels()
    cm = {(h, j): 0 for h in (0, 1) for j in (0, 1)}
    mismatches = []

    for system in SYSTEMS:
        data = json.loads(_latest(system).read_text(encoding="utf-8"))
        for item in data["detailed_results"]:
            reasoning = item.get("reasoning_metrics", {})
            if not reasoning.get("chain_emitted"):
                continue
            evidence = {
                int(k): v for k, v in (reasoning.get("evidence_shown") or {}).items()
            }
            if not any((system, item["query_id"], s["step_index"]) in human
                       for s in reasoning["steps"]):
                continue

            chain_text = "\n".join(
                f"{s['step_index']}. [{'E' if s['type'] == 'evidential' else 'I'}] {s['text']}"
                for s in reasoning["steps"]
            )
            store = ReplayStore(evidence)

            # The evaluator asks the store once per locus; steer it to the
            # right step's recorded passages via the closure below.
            original_lookup = store.lookup

            def lookup(doc_id, section_text, claim_text, max_passages=6, _s=store):
                for step in reasoning["steps"]:
                    if step["text"] == claim_text:
                        _s.for_step(step["step_index"])
                        break
                return original_lookup(doc_id, section_text, claim_text, max_passages)

            store.lookup = lookup  # type: ignore[method-assign]

            scores = evaluate_reasoning_chain(
                question=item["question"],
                answer=f"(replay)\n\n## Reasoning\n{chain_text}\n",
                reference_decomposition=[],
                store=store,  # type: ignore[arg-type]
                system_name=system,
                query_id=item["query_id"],
            )
            for verdict in scores.step_verdicts:
                key = (system, item["query_id"], verdict["step"])
                if key not in human:
                    continue
                h, j = human[key], 1 if verdict["grounded"] else 0
                cm[(h, j)] += 1
                if h != j:
                    text = next(
                        (s["text"] for s in reasoning["steps"]
                         if s["step_index"] == verdict["step"]), ""
                    )
                    mismatches.append({
                        "system": system, "query_id": item["query_id"],
                        "step": verdict["step"], "human": h, "judge": j,
                        "code": verdict.get("code"), "text": text[:100],
                        "rationale": verdict.get("rationale", "")[:90],
                    })
            print(f"  {system} id={item['query_id']}: {len(scores.step_verdicts)} Schritte")

    from sklearn.metrics import cohen_kappa_score

    from scripts.analyze_judge_validation import _gwet_ac1

    n = sum(cm.values())
    h_labels = ["1"] * (cm[(1, 1)] + cm[(1, 0)]) + ["0"] * (cm[(0, 1)] + cm[(0, 0)])
    j_labels = (["1"] * cm[(1, 1)] + ["0"] * cm[(1, 0)]
                + ["1"] * cm[(0, 1)] + ["0"] * cm[(0, 0)])
    agree = (cm[(1, 1)] + cm[(0, 0)]) / n if n else 0

    print(f"\n{'':30} n={n}")
    print(f"  beide 'belegt'          {cm[(1, 1)]}")
    print(f"  beide 'nicht belegt'    {cm[(0, 0)]}")
    print(f"  Mensch nein / Judge ja  {cm[(0, 1)]}   (Defekt uebersehen)")
    print(f"  Mensch ja / Judge nein  {cm[(1, 0)]}   (Fehlalarm)")
    h_neg = cm[(0, 0)] + cm[(0, 1)]
    print(f"\n  Erkennung auf der Minderheitsklasse: "
          f"{cm[(0, 0)]}/{h_neg}" + (f" ({cm[(0, 0)] / h_neg:.0%})" if h_neg else ""))
    print(f"  Uebereinstimmung {agree:.0%}  kappa "
          f"{cohen_kappa_score(h_labels, j_labels):+.2f}  AC1 {_gwet_ac1(h_labels, j_labels):+.2f}")

    (RESULTS_DIR / "groundedness_calibration.json").write_text(
        json.dumps({"confusion": {f"{h}{j}": v for (h, j), v in cm.items()},
                    "mismatches": mismatches}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0




# ---------------------------------------------------------------------------
#  Re-judging the stored pilot with the current instrument
# ---------------------------------------------------------------------------

def rejudge() -> int:
    """Rewrite the reasoning verdicts in the reference file, judged afresh.

    The pilot ran at 23:41 on 2026-09-08; the groundedness repair landed the
    next day. So `validation_report_reserve.md` kept reporting kappa 0.21 for
    a prompt that no longer exists, while the bench measured 0.68 for the one
    that does -- two numbers for the same dimension, and the published one was
    the wrong one.

    Re-judging is legitimate here and would not be after a system change: the
    CHAINS do not move. They are stored verbatim in the run output, the human
    labels were given against them, and only the judge that reads them has
    changed. Re-running the systems instead would produce different answers,
    different chains, and would invalidate every human label collected.

    `judge_custom` and `judge_citation` are left alone -- the refusal prompt
    was already repaired before the pilot ran, so those verdicts are current.
    """
    import shutil

    from src.evaluation.gold_standard_loader import GOLD_STANDARD_EN, load_gold_standard
    from src.evaluation.reasoning_chain_evaluator import evaluate_reasoning_chain

    ref_path = RESULTS_DIR / "judge_scores_reference_reserve.json"
    backup = ref_path.with_suffix(".json.pre-groundedness-fix")
    if not backup.exists():
        shutil.copy(ref_path, backup)
        print(f"Sicherung: {backup.name}")

    reference = json.loads(ref_path.read_text(encoding="utf-8"))
    decompositions = {
        item.id: item.reference_decomposition
        for item in load_gold_standard(str(GOLD_STANDARD_EN))
    }
    by_key = {
        (entry["system_name"], entry["query_id"]): review_id
        for review_id, entry in reference.items()
    }

    updated = 0
    for system in SYSTEMS:
        data = json.loads(_latest(system).read_text(encoding="utf-8"))
        for item in data["detailed_results"]:
            review_id = by_key.get((system, item["query_id"]))
            reasoning = item.get("reasoning_metrics", {})
            if review_id is None or not reasoning.get("chain_emitted"):
                continue

            evidence = {
                int(k): v for k, v in (reasoning.get("evidence_shown") or {}).items()
            }
            chain_text = "\n".join(
                f"{s['step_index']}. [{'E' if s['type'] == 'evidential' else 'I'}] {s['text']}"
                for s in reasoning["steps"]
            )
            store = ReplayStore(evidence)
            original_lookup = store.lookup

            def lookup(doc_id, section_text, claim_text, max_passages=6,
                       _s=store, _steps=reasoning["steps"]):
                for step in _steps:
                    if step["text"] == claim_text:
                        _s.for_step(step["step_index"])
                        break
                return original_lookup(doc_id, section_text, claim_text, max_passages)

            store.lookup = lookup  # type: ignore[method-assign]

            scores = evaluate_reasoning_chain(
                question=item["question"],
                answer=f"(replay)\n\n## Reasoning\n{chain_text}\n",
                reference_decomposition=decompositions.get(item["query_id"], []),
                store=store,  # type: ignore[arg-type]
                system_name=system,
                query_id=item["query_id"],
                keep_evidence=True,
            )
            payload = scores.to_dict()
            # The chain is the object the human rated; carry it unchanged.
            payload["steps"] = reasoning["steps"]
            payload["evidence_shown"] = reasoning.get("evidence_shown", {})
            reference[review_id]["judge_reasoning"] = payload
            updated += 1
            print(f"  {system} id={item['query_id']}: "
                  f"g={payload['groundedness']} v={payload['validity']} "
                  f"c={payload['completeness']}")

    ref_path.write_text(json.dumps(reference, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{updated} Datensaetze neu beurteilt -> {ref_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
