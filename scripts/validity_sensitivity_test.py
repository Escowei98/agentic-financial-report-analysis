"""
Sensitivity check for the validity dimension.

WHY THIS EXISTS
---------------
In the pilot of 2026-09-08 the validity dimension came out at 1.00 for every
system, and both the judge and the human rater used a single category across
all 81 assessable transitions. Cohen's kappa and Gwet's AC1 are both
undefined on a constant, so the dimension could not be validated at all --
`NOT ESTIMABLE`, not PASS.

A larger field sample does not fix that. If the base rate of the defect is
near zero, 500 units carry exactly as much information about the instrument
as 81 do. The floor effect is the problem, not n.

Reading the chains explains the zero rate rather than excusing it: in all 15
incomplete chains, a system that lacked a premise SAID SO and declined to
conclude, instead of leaping. That is the behaviour V1 is meant to catch, and
they do not exhibit it -- plausibly because NON_ANSWERABILITY_CONVENTION ends
with "Never close such a gap with knowledge from outside the filings", which
targets exactly it.

So the open question is not "how often do the systems commit V1" (answer, in
this corpus: never observed) but "would the instrument notice if they did".
That is answerable on constructed cases.

WHAT IT DOES
------------
Takes real chains from the run, and in half of them deletes the premise the
final conclusion numerically depends on. The conclusion is left untouched, so
it now asserts a comparison whose second term the chain never states -- a
textbook V1. Controls pass through unchanged. Both are shuffled and judged
blind, by the deployed prompt and by a human on the usual page.

WHICH VARIANT OF V1 THIS TESTS
------------------------------
A pure deletion, with the conclusion left exactly as written. That yields the
BLATANT variant: the conclusion openly rests on a figure the chain never
states -- "comparing 24.15% < 30.29% < 31.51%" after the 24.15% premise is
gone. It is a genuine V1, and it is the easy one.

The subtle variant -- where the conclusion is phrased so the missing premise
leaves no trace -- is not covered, because producing it would mean rewriting
the conclusion, i.e. authoring chains rather than perturbing real ones, and
an instrument check on hand-written material tests the author's imagination
as much as the judge.

So the claim this test can support is one-directional and must be stated as
such: failing here means the instrument cannot catch V1 at all; passing means
it catches the detectable case, and says nothing about the subtle one.

WHAT IT DOES *NOT* LICENSE
--------------------------
This measures the INSTRUMENT, not the systems. A kappa from perturbed chains
must never be reported beside the field values of groundedness and
completeness as though it came from the same sample -- it is a manipulation
check and belongs in its own section. What it can establish is narrow and
worth stating plainly: whether a zero rate in the field is a property of the
systems or a blind spot of the measurement.

Usage:
    uv run python scripts/validity_sensitivity_test.py build
    uv run python scripts/build_human_rating_ui.py --sample sensitivity
    uv run python scripts/validity_sensitivity_test.py judge
    # rate the page, export, then:
    uv run python scripts/merge_human_ratings.py --sample sensitivity <export.json>
    uv run python scripts/validity_sensitivity_test.py analyze
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_judge_validation_sample import BLIND_FIELDS  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "judge_validation"
RAW_DIR = RESULTS_DIR / "raw_eval_reserve"
SYSTEM_ORDER = ["rag_monolith", "rag_agent", "long_context", "multi_agent"]

SEED = 42
N_PER_ARM = 12

_NUM_RE = re.compile(r"\d[\d,.]*")


def _numbers(text: str) -> set[str]:
    return {n.strip(".,").replace(",", "") for n in _NUM_RE.findall(text)} - {""}


def _latest_eval(system: str) -> Path:
    matches = sorted(glob.glob(str(RAW_DIR / f"eval_{system}_*.json")))
    if not matches:
        raise FileNotFoundError(f"No eval JSON for {system} in {RAW_DIR}")
    return Path(matches[-1])


def _candidates() -> list[dict]:
    """Chains that can carry a V1 perturbation at all.

    Requires at least two evidential steps (so one can go while the chain
    still reads as a chain) and a final conclusion that numerically depends
    on one of them -- otherwise deleting a premise changes nothing about
    whether the conclusion follows, and the "perturbed" arm would be full of
    chains that are still perfectly valid.
    """
    out = []
    for system in SYSTEM_ORDER:
        data = json.loads(_latest_eval(system).read_text(encoding="utf-8"))
        for item in data["detailed_results"]:
            reasoning = item.get("reasoning_metrics", {})
            if not reasoning.get("chain_emitted"):
                continue
            steps = reasoning.get("steps", [])
            evidential = [s for s in steps if s["type"] == "evidential"]
            inferential = [
                s for s in steps if s["type"] != "evidential" and s["step_index"] > 1
            ]
            if len(evidential) < 2 or not inferential:
                continue
            conclusion = inferential[-1]
            linked = [
                s for s in evidential
                if _numbers(s["text"]) & _numbers(conclusion["text"])
            ]
            if not linked:
                continue
            out.append({
                "system": system,
                "query_id": item["query_id"],
                "question": item["question"],
                "fa_type": item["fa_type"],
                "steps": steps,
                "target": linked[0]["step_index"],
            })
    return out


def _perturb(steps: list[dict], drop_index: int) -> list[dict]:
    """Remove one premise and renumber, leaving the conclusion untouched.

    Renumbering matters: a gap in the numbering would tell judge and rater
    which arm they are looking at, and the test would measure nothing.
    """
    kept = [dict(s) for s in steps if s["step_index"] != drop_index]
    for new_index, step in enumerate(kept, start=1):
        step["step_index"] = new_index
    return kept


def build(out_csv: Path, out_key: Path) -> tuple[int, int]:
    pool = _candidates()
    rng = random.Random(SEED)
    rng.shuffle(pool)

    n = min(N_PER_ARM, len(pool) // 2)
    perturbed_src, control_src = pool[:n], pool[n:2 * n]

    rows, key = [], {}
    entries = (
        [(c, True) for c in perturbed_src] + [(c, False) for c in control_src]
    )
    rng.shuffle(entries)

    for i, (cand, is_perturbed) in enumerate(entries, start=1):
        review_id = f"V{i:03d}"
        steps = (
            _perturb(cand["steps"], cand["target"]) if is_perturbed else cand["steps"]
        )
        row = {field: "" for field in BLIND_FIELDS}
        row.update({
            "review_id": review_id,
            "fa_type": cand["fa_type"],
            # Deliberately blank: the architecture is irrelevant here and
            # naming it would only invite the rater to theorise about it.
            "system_label": "?",
            "question": cand["question"],
            "answer": "(nicht gezeigt — bewertet wird allein die Kette)",
            "trajectory": "",
            "chain_json": json.dumps(steps, ensure_ascii=False),
            "evidence_json": "{}",
            "subquestions_json": "[]",
            # Only validity is under test. Offering the other two would make
            # the page longer without adding anything, and groundedness would
            # be unanswerable anyway -- the perturbed chains keep their
            # citations but the evidence is not shown.
            "groundedness_human": "N/A",
            "completeness_human": "N/A",
            "validity_human": "",
            "exact_match_human": "N/A",
            "answer_recall_human": "N/A",
            "refusal_accuracy_human": "N/A",
            "refusal_quality_human": "N/A",
            "over_refusal_human": "N/A",
            "citation_accuracy_human": "N/A",
        })
        rows.append(row)
        key[review_id] = {
            "perturbed": is_perturbed,
            "source_system": cand["system"],
            "source_query_id": cand["query_id"],
            "removed_step_index": cand["target"] if is_perturbed else None,
            "removed_step_text": (
                next(s["text"] for s in cand["steps"] if s["step_index"] == cand["target"])
                if is_perturbed else None
            ),
        }

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=BLIND_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    out_key.write_text(json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(perturbed_src), len(control_src)


def judge(in_csv: Path, out_json: Path) -> int:
    """Score every chain with the DEPLOYED validity judgement.

    The same prompt and the same evaluator as the real run -- a sensitivity
    test on a bespoke prompt would say nothing about the instrument in use.
    That is also why the evidence store is built: the deployed prompt asks
    for groundedness and validity in one call, and answering it with an
    empty evidence block would change the task the judge is doing.
    """
    from src.common.ingestion import download_all_filings
    from src.evaluation.evidence_store import EvidenceStore
    from src.evaluation.reasoning_chain_evaluator import evaluate_reasoning_chain

    store = EvidenceStore(download_all_filings())

    with in_csv.open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    verdicts = {}
    for row in rows:
        steps = json.loads(row["chain_json"])
        chain_text = "\n".join(
            f"{s['step_index']}. [{'E' if s['type'] == 'evidential' else 'I'}] {s['text']}"
            for s in steps
        )
        answer = f"(siehe Kette)\n\n## Reasoning\n{chain_text}\n"
        scores = evaluate_reasoning_chain(
            question=row["question"],
            answer=answer,
            reference_decomposition=[],   # completeness is not under test
            store=store,
            system_name="sensitivity",
            query_id=int(row["review_id"][1:]),
        )
        verdicts[row["review_id"]] = {
            "validity": scores.validity,
            "transition_verdicts": scores.transition_verdicts,
            "num_steps": scores.num_steps,
        }
        print(f"  {row['review_id']}: validity={scores.validity}")

    out_json.write_text(json.dumps(verdicts, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(verdicts)


def _flagged(entry: dict) -> bool:
    """Did this rater mark ANY transition in the chain invalid?

    Chain-level, not unit-level, on purpose: the perturbation invalidates one
    specific transition, but which index that lands on shifts with
    renumbering, and a rater who flags the neighbouring step has still caught
    the defect. Requiring the exact index would measure bookkeeping.
    """
    return any(not v.get("valid", True) for v in entry.get("transition_verdicts", []))


def analyze(in_csv: Path, key_path: Path, judge_path: Path) -> dict:
    key = json.loads(key_path.read_text(encoding="utf-8"))
    judged = json.loads(judge_path.read_text(encoding="utf-8")) if judge_path.exists() else {}

    with in_csv.open(encoding="utf-8-sig") as fh:
        rows = {r["review_id"]: r for r in csv.DictReader(fh)}

    def human_flagged(review_id: str) -> bool | None:
        raw = (rows[review_id].get("validity_human") or "").strip()
        if not raw or raw.upper() == "N/A":
            return None
        return any(str(v) == "0" for v in json.loads(raw).values())

    out = {}
    for who, flagged in (
        ("judge", lambda rid: _flagged(judged[rid]) if rid in judged else None),
        ("human", human_flagged),
    ):
        tp = fn = tn = fp = skipped = 0
        for rid, meta in key.items():
            got = flagged(rid)
            if got is None:
                skipped += 1
                continue
            if meta["perturbed"]:
                tp, fn = (tp + 1, fn) if got else (tp, fn + 1)
            else:
                fp, tn = (fp + 1, tn) if got else (fp, tn + 1)
        out[who] = {
            "sensitivity": tp / (tp + fn) if (tp + fn) else None,
            "specificity": tn / (tn + fp) if (tn + fp) else None,
            "true_positive": tp, "false_negative": fn,
            "true_negative": tn, "false_positive": fp,
            "not_rated": skipped,
        }

    both = [
        (_flagged(judged[rid]), human_flagged(rid))
        for rid in key
        if rid in judged and human_flagged(rid) is not None
    ]
    out["judge_human_agreement"] = (
        sum(1 for j, h in both if j == h) / len(both) if both else None
    )
    out["n_pairs"] = len(both)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["build", "judge", "analyze"])
    args = parser.parse_args()

    csv_path = RESULTS_DIR / "human_review_blind_sensitivity.csv"
    key_path = RESULTS_DIR / "sensitivity_key.json"
    judge_path = RESULTS_DIR / "sensitivity_judge_verdicts.json"

    if args.mode == "build":
        n_pert, n_ctrl = build(csv_path, key_path)
        print(f"{n_pert} manipulierte + {n_ctrl} Kontrollketten -> {csv_path.name}")
        print(f"Schluessel (NICHT oeffnen, bis bewertet): {key_path.name}")
        print("\nSeite bauen mit:")
        print("  uv run python scripts/build_human_rating_ui.py --sample sensitivity")
        return 0

    if args.mode == "judge":
        n = judge(csv_path, judge_path)
        print(f"{n} Ketten beurteilt -> {judge_path.name}")
        return 0

    report = analyze(csv_path, key_path, judge_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    for who in ("judge", "human"):
        r = report[who]
        s, sp = r["sensitivity"], r["specificity"]
        print(
            f"\n{who}: Sensitivitaet "
            f"{'—' if s is None else f'{s:.0%}'} ({r['true_positive']}/{r['true_positive'] + r['false_negative']}), "
            f"Spezifitaet {'—' if sp is None else f'{sp:.0%}'} "
            f"({r['true_negative']}/{r['true_negative'] + r['false_positive']})"
        )
    if report["judge_human_agreement"] is not None:
        print(f"\nJudge/Mensch einig: {report['judge_human_agreement']:.0%} (n={report['n_pairs']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
