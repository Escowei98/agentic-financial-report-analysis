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
"""
import csv
import glob
import json
import random
import re
from pathlib import Path

from src.evaluation.reasoning_evaluator import AGENTIC_SYSTEMS

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
    "logical_soundness_human",
    "synthesis_quality_human",
    "evidence_faithfulness_human",
    "tool_selection_human",
    "error_recovery_human",
    "exact_match_human",
    "answer_recall_human",
    "refusal_accuracy_human",
    "notes",
]


def _latest_eval_json(raw_dir: Path, system_name: str) -> Path:
    matches = sorted(glob.glob(str(raw_dir / f"eval_{system_name}_*.json")))
    if not matches:
        raise FileNotFoundError(f"No eval JSON found for {system_name} in {raw_dir}")
    return Path(matches[-1])


def _anonymize_trajectory(trajectory: str) -> str:
    return TRAJECTORY_HEADER_RE.sub("## Trajectory", trajectory)


def main():
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "results" / "judge_validation" / "raw_eval"
    out_dir = project_root / "data" / "results" / "judge_validation"

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

        is_agentic = system_name in AGENTIC_SYSTEMS
        label = system_to_label[system_name]

        for item in data["detailed_results"]:
            review_counter += 1
            review_id = f"R{review_counter:03d}"

            expected_answerable = item.get("expected_answerable", True)
            core = item["reasoning_metrics"]["core"]
            agentic = item["reasoning_metrics"].get("agentic")
            custom = item["custom_metrics"]

            blind_rows.append({
                "review_id": review_id,
                "fa_type": item["fa_type"],
                "system_label": label,
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "answer": item["answer"],
                "trajectory": _anonymize_trajectory(item.get("trajectory", "")),
                "logical_soundness_human": "",
                "synthesis_quality_human": "",
                "evidence_faithfulness_human": "",
                "tool_selection_human": "" if is_agentic else "N/A",
                "error_recovery_human": "" if is_agentic else "N/A",
                "exact_match_human": "" if expected_answerable else "N/A",
                "answer_recall_human": "" if expected_answerable else "N/A",
                "refusal_accuracy_human": "" if not expected_answerable else "N/A",
                "notes": "",
            })

            reference[review_id] = {
                "system_name": system_name,
                "system_label": label,
                "query_id": item["query_id"],
                "fa_type": item["fa_type"],
                "expected_answerable": expected_answerable,
                "judge_core": core,
                "judge_agentic": agentic,
                "judge_custom": custom,
            }

    rng_shuffle = random.Random(SHUFFLE_SEED)
    rng_shuffle.shuffle(blind_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    blind_path = out_dir / "human_review_blind.csv"
    with open(blind_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BLIND_FIELDS)
        writer.writeheader()
        writer.writerows(blind_rows)

    reference_path = out_dir / "judge_scores_reference.json"
    with open(reference_path, "w", encoding="utf-8") as f:
        json.dump(reference, f, indent=2)

    print(f"Blind review file ({len(blind_rows)} rows): {blind_path}")
    print(f"Hidden judge reference (do not open yet): {reference_path}")


if __name__ == "__main__":
    main()
