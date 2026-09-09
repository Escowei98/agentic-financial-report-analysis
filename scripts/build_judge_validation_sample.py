"""
Builds the blind human-review sample from the raw eval JSONs produced by
run_judge_validation_batch.py.

Produces two files under data/results/judge_validation/:
  - human_review_blind.csv: question/answer/trajectory/ground_truth, system
    identity anonymized (label A-D, trajectory headers stripped), judge
    scores hidden, empty columns for the human rater to fill in.
  - judge_scores_reference.json: the same items keyed by review_id, with the
    real system name and the judge's scores + rationales. Do NOT open this
    until the blind CSV has been fully rated.

See docs/decisions/EVAL_DECISION_LOG.md [2026-08-01] for the full protocol
(metric scope, blind protocol, trust thresholds, remediation plan).

REASONING RATINGS ARE PER UNIT, NOT PER DIMENSION
-------------------------------------------------
The retired instrument asked for five 1-5 scores per record. It reached
weighted kappa 0.03-0.17, and the diagnosis was not that raters disagreed
about the chains but that an unanchored five-point scale gives them nothing to
agree on. The rating unit is now a single step, a single transition or a
single required sub-question, and the answer is yes or no.

A record therefore no longer carries a fixed number of rating cells, so the
three reasoning columns hold JSON maps from unit index to 0/1 rather than a
scalar. `chain_json` and `evidence_json` carry what the rater needs to decide
them -- the evidence being the passages THE JUDGE SAW, carried over from the
run rather than re-fetched, so that rater and judge are answering the same
question about the same text.
"""
import argparse
import csv
import glob
import json
import random
import re
from pathlib import Path

SYSTEM_ORDER = ["rag_monolith", "rag_agent", "long_context", "multi_agent"]
ANON_SEED = 42
SHUFFLE_SEED = 42

TRAJECTORY_HEADER_RE = re.compile(r"^## System \d Trajectory \(.*?\)\s*$", re.MULTILINE)

BLIND_FIELDS = [
    "review_id",
    "fa_type",
    "system_label",
    "question",
    "ground_truth",
    "answer",
    "trajectory",
    # Item metadata, not a rating target. Needed by build_human_rating_ui.py
    # to pick the same subtype/evidence-conditioned rubric the judge used for
    # THIS item (see get_refusal_quality_prompt / get_evidence_clause /
    # _classify_answer_type in custom_evaluator.py) -- none of it reveals a
    # judge score.
    "subtype",
    "refusal_evidence",
    "gt_correction",
    "gt_unit",
    # Reasoning material shown to the rater (not rating targets).
    "chain_json",
    "evidence_json",
    "subquestions_json",
    # Per-unit reasoning ratings: JSON maps {unit index: 0|1}, or "N/A" where
    # the dimension does not apply (no chain, no transition, refusal item).
    "groundedness_human",
    "validity_human",
    "completeness_human",
    "exact_match_human",
    "answer_recall_human",
    "refusal_accuracy_human",
    "refusal_quality_human",
    "over_refusal_human",
    "citation_accuracy_human",
    "notes",
]


def _latest_eval_json(raw_dir: Path, system_name: str) -> Path:
    matches = sorted(glob.glob(str(raw_dir / f"eval_{system_name}_*.json")))
    if not matches:
        raise FileNotFoundError(f"No eval JSON found for {system_name} in {raw_dir}")
    return Path(matches[-1])


def _anonymize_trajectory(trajectory: str) -> str:
    return TRAJECTORY_HEADER_RE.sub("## Trajectory", trajectory)


def _open_if(applicable: object) -> str:
    """Empty cell where the rater has something to decide, 'N/A' where not."""
    return "" if applicable else "N/A"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reserve", action="store_true",
        help="Build from the reserve-sample raw eval JSONs instead of the "
             "primary sample; writes to separate *_reserve output files.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    raw_subdir = "raw_eval_reserve" if args.reserve else "raw_eval"
    raw_dir = project_root / "data" / "results" / "judge_validation" / raw_subdir
    out_dir = project_root / "data" / "results" / "judge_validation"
    suffix = "_reserve" if args.reserve else ""

    # Fixed, reproducible label assignment (kept only in the hidden reference file).
    labels = ["A", "B", "C", "D"]
    rng_anon = random.Random(ANON_SEED)
    shuffled_systems = SYSTEM_ORDER[:]
    rng_anon.shuffle(shuffled_systems)
    system_to_label = dict(zip(shuffled_systems, labels))

    blind_rows = []
    reference = {}
    review_counter = 0

    for system_name in SYSTEM_ORDER:
        eval_path = _latest_eval_json(raw_dir, system_name)
        with open(eval_path, encoding="utf-8") as f:
            data = json.load(f)

        label = system_to_label[system_name]

        for item in data["detailed_results"]:
            review_counter += 1
            review_id = f"R{review_counter:03d}"

            expected_answerable = item.get("expected_answerable", True)
            reasoning = item.get("reasoning_metrics", {})

            # Fail loudly rather than emit a record with no rateable cells.
            # The first build of this sample produced 0 groundedness and 0
            # validity cells because the evaluator was not yet writing the
            # parsed chain into its output -- silently, since an empty chain
            # simply yields an empty unit list further down the line.
            if reasoning.get("chain_emitted") and not reasoning.get("steps"):
                raise SystemExit(
                    f"{eval_path.name}, query {item.get('query_id')}: "
                    "reasoning_metrics reports an emitted chain but carries no "
                    "'steps'. The run predates the fix that persists the parsed "
                    "chain; re-run it rather than building an unrateable sample."
                )
            custom = item["custom_metrics"]
            citation = item.get("citation_metrics")

            blind_rows.append({
                "review_id": review_id,
                "fa_type": item["fa_type"],
                "system_label": label,
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "answer": item["answer"],
                "trajectory": _anonymize_trajectory(item.get("trajectory", "")),
                "subtype": item.get("subtype", ""),
                "refusal_evidence": item.get("refusal_evidence", ""),
                "gt_correction": item.get("gt_correction", ""),
                "gt_unit": item.get("gt_unit", ""),
                "chain_json": json.dumps(reasoning.get("steps", [])),
                "evidence_json": json.dumps(reasoning.get("evidence_shown", {})),
                "subquestions_json": json.dumps(
                    [v.get("text", "") for v in reasoning.get("subquestion_verdicts", [])]
                ),
                # Open where there is something to rate, "N/A" where there is
                # not. A chain that was never emitted has no units at all --
                # that is a format failure, tracked as chain_emission_rate,
                # and must not be handed to a rater as if it were a defect
                # they could score.
                "groundedness_human": _open_if(
                    reasoning.get("chain_emitted")
                    and reasoning.get("num_evidential", 0) > 0
                ),
                # Only transitions into an inferential step are assessable
                # (see EVAL_DECISION_LOG.md [2026-09-09]); a chain that is all
                # lookups has none.
                "validity_human": _open_if(
                    reasoning.get("chain_emitted")
                    and any(
                        s.get("step_index", 1) > 1
                        and not (s.get("type") == "evidential" and s.get("loci"))
                        for s in reasoning.get("steps", [])
                    )
                ),
                "completeness_human": _open_if(
                    reasoning.get("chain_emitted")
                    and bool(reasoning.get("subquestion_verdicts"))
                ),
                "exact_match_human": "" if expected_answerable else "N/A",
                "answer_recall_human": "" if expected_answerable else "N/A",
                "refusal_accuracy_human": "" if not expected_answerable else "N/A",
                # Graded 0 / 0.5 / 1.0, unlike the binary refusal_accuracy
                # beside it: 1.0 = the defect was named, 0.5 = declined
                # without a diagnosis, 0.0 = answered as though the question
                # were sound. Only defined on the FA-Refusal stratum.
                "refusal_quality_human": "" if not expected_answerable else "N/A",
                # over_refusal is itself conditional on the automated score
                # (custom_evaluator only asks the judge when the item was
                # marked wrong -- a correct answer can't be a refusal), so
                # its applicability is per-row, not per fa_type like the
                # others. Gating on whether the judge computed a value at
                # all does not reveal what that value is.
                "over_refusal_human": "" if custom.get("over_refusal") is not None else "N/A",
                "citation_accuracy_human": "" if expected_answerable else "N/A",
                "notes": "",
            })

            reference[review_id] = {
                "system_name": system_name,
                "system_label": label,
                "query_id": item["query_id"],
                "fa_type": item["fa_type"],
                "expected_answerable": expected_answerable,
                "judge_reasoning": reasoning,
                "judge_custom": custom,
                "judge_citation": citation,
            }

    rng_shuffle = random.Random(SHUFFLE_SEED)
    rng_shuffle.shuffle(blind_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    blind_path = out_dir / f"human_review_blind{suffix}.csv"
    with open(blind_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BLIND_FIELDS)
        writer.writeheader()
        writer.writerows(blind_rows)

    reference_path = out_dir / f"judge_scores_reference{suffix}.json"
    with open(reference_path, "w", encoding="utf-8") as f:
        json.dump(reference, f, indent=2)

    print(f"Blind review file ({len(blind_rows)} rows): {blind_path}")
    print(f"Hidden judge reference (do not open yet): {reference_path}")


if __name__ == "__main__":
    main()
