"""
Compares the human ratings in human_review_blind.csv (filled in by hand)
against the hidden judge scores in judge_scores_reference.json.

Run this ONLY after the blind CSV has been fully rated.

Implements the measurement/threshold protocol from
docs/decisions/EVAL_DECISION_LOG.md [2026-08-01]:
  - 1-5 scales: exact-match rate, +/-1-tolerance rate, quadratic-weighted
    Cohen's kappa, Spearman rho, mean signed difference (judge - human).
  - binary/tri-level custom metrics: exact-match rate, unweighted kappa,
    mean signed difference.
  - all of the above broken down per system, to catch differential bias
    (a dimension fails regardless of its pooled kappa if the judge's bias
    differs meaningfully between systems).

Trust thresholds (pre-registered, see decision log): pooled weighted
kappa >= 0.60, +/-1-tolerance >= 80%, |mean signed diff| <= 0.5,
no visible per-system bias spread > 0.5.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "judge_validation"

KAPPA_THRESHOLD = 0.60
TOLERANCE_THRESHOLD = 0.80
BIAS_THRESHOLD = 0.5
SYSTEM_SPREAD_THRESHOLD = 0.5

# (dimension name, human CSV column, path into reference[review_id], scale)
DIMENSIONS = [
    ("logical_soundness", "logical_soundness_human", ("judge_core", "logical_soundness"), "1-5"),
    ("synthesis_quality", "synthesis_quality_human", ("judge_core", "synthesis_quality"), "1-5"),
    ("evidence_faithfulness", "evidence_faithfulness_human", ("judge_core", "evidence_faithfulness"), "1-5"),
    ("tool_selection", "tool_selection_human", ("judge_agentic", "tool_selection"), "1-5"),
    ("error_recovery", "error_recovery_human", ("judge_agentic", "error_recovery"), "1-5"),
    ("exact_match", "exact_match_human", ("judge_custom", "exact_match"), "custom"),
    ("answer_recall", "answer_recall_human", ("judge_custom", "answer_recall"), "custom"),
    ("refusal_accuracy", "refusal_accuracy_human", ("judge_custom", "refusal_accuracy"), "custom"),
    ("citation_accuracy", "citation_accuracy_human", ("judge_citation", "citation_accuracy"), "custom"),
]


def _parse_value(raw: str):
    raw = (raw or "").strip()
    if not raw or raw.upper() == "N/A":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _collect_pairs(rows, reference, human_col, judge_path):
    pairs = []  # (review_id, system_name, human, judge)
    for row in rows:
        review_id = row["review_id"]
        human_val = _parse_value(row.get(human_col, ""))
        if human_val is None:
            continue
        ref = reference.get(review_id)
        if ref is None:
            continue
        judge_block = ref.get(judge_path[0])
        if judge_block is None:
            continue
        judge_val = judge_block.get(judge_path[1])
        if judge_val is None:
            continue
        pairs.append((review_id, ref["system_name"], human_val, float(judge_val)))
    return pairs


def _analyze_dimension(name, pairs, scale):
    n = len(pairs)
    lines = [f"### {name} (n={n})"]
    if n == 0:
        lines.append("No rated items yet — skipped.")
        return lines, None

    humans = [p[2] for p in pairs]
    judges = [p[3] for p in pairs]

    exact_rate = sum(1 for h, j in zip(humans, judges) if abs(h - j) < 1e-9) / n
    mean_signed_diff = sum(j - h for h, j in zip(humans, judges)) / n

    lines.append(f"- Exact-match rate: {exact_rate:.0%}")
    lines.append(f"- Mean signed difference (judge - human): {mean_signed_diff:+.2f}")

    kappa = None
    tolerance_rate = None
    rho = None

    if len(set(humans)) < 2 and len(set(judges)) < 2:
        lines.append("- Kappa/Spearman: not computable (no variance in ratings)")
    else:
        try:
            weights = "quadratic" if scale == "1-5" else None
            # sklearn infers "continuous" for float labels like 0.0/0.5/1.0 and
            # refuses to compute kappa on them; these are categorical (3 discrete
            # levels), not a continuous scale, so cast to strings to force that.
            kappa_humans = humans if scale == "1-5" else [str(h) for h in humans]
            kappa_judges = judges if scale == "1-5" else [str(j) for j in judges]
            kappa = cohen_kappa_score(kappa_humans, kappa_judges, weights=weights)
            lines.append(f"- Cohen's kappa ({'quadratic-weighted' if weights else 'unweighted'}): {kappa:.2f}")
        except ValueError as e:
            lines.append(f"- Kappa: not computable ({e})")

        if scale == "1-5":
            tolerance_rate = sum(1 for h, j in zip(humans, judges) if abs(h - j) <= 1) / n
            lines.append(f"- +/-1-tolerance rate: {tolerance_rate:.0%}")
            try:
                rho, _ = spearmanr(humans, judges)
                lines.append(f"- Spearman rho: {rho:.2f}")
            except ValueError:
                lines.append("- Spearman rho: not computable")

    # Per-system breakdown (differential-bias check)
    by_system = defaultdict(list)
    for _, system_name, h, j in pairs:
        by_system[system_name].append(j - h)

    lines.append("- Per-system mean signed difference:")
    system_means = {}
    for system_name, diffs in sorted(by_system.items()):
        m = sum(diffs) / len(diffs)
        system_means[system_name] = m
        lines.append(f"    - {system_name} (n={len(diffs)}): {m:+.2f}")

    spread = (max(system_means.values()) - min(system_means.values())) if len(system_means) > 1 else 0.0

    # Verdict
    verdict_flags = []
    if kappa is not None and kappa < KAPPA_THRESHOLD:
        verdict_flags.append(f"kappa {kappa:.2f} < {KAPPA_THRESHOLD}")
    if tolerance_rate is not None and tolerance_rate < TOLERANCE_THRESHOLD:
        verdict_flags.append(f"+/-1-tolerance {tolerance_rate:.0%} < {TOLERANCE_THRESHOLD:.0%}")
    if abs(mean_signed_diff) > BIAS_THRESHOLD:
        verdict_flags.append(f"|bias| {abs(mean_signed_diff):.2f} > {BIAS_THRESHOLD}")
    if spread > SYSTEM_SPREAD_THRESHOLD:
        verdict_flags.append(f"per-system spread {spread:.2f} > {SYSTEM_SPREAD_THRESHOLD} (differential bias)")

    if verdict_flags:
        verdict = "FAIL — " + "; ".join(verdict_flags)
    else:
        verdict = "PASS"
    lines.append(f"- **Verdict: {verdict}**")

    return lines, verdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reserve", action="store_true",
        help="Analyze the reserve-sample files (human_review_blind_reserve.csv / "
             "judge_scores_reference_reserve.json) instead of the primary sample.",
    )
    args = parser.parse_args()

    suffix = "_reserve" if args.reserve else ""
    blind_csv = RESULTS_DIR / f"human_review_blind{suffix}.csv"
    reference_json = RESULTS_DIR / f"judge_scores_reference{suffix}.json"
    report_md = RESULTS_DIR / f"validation_report{suffix}.md"

    if not blind_csv.exists():
        raise FileNotFoundError(f"Blind review CSV not found: {blind_csv}")
    if not reference_json.exists():
        raise FileNotFoundError(f"Judge reference JSON not found: {reference_json}")

    with open(blind_csv, encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        # Only trust the sniffer for the delimiter -- its guessed quoting
        # attributes (e.g. doublequote) are unreliable and can silently
        # mis-split quoted multi-line fields (trajectory/answer) into
        # garbage rows. Force standard, safe quoting explicitly instead.
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;").delimiter
        except csv.Error:
            delimiter = ","
        rows = list(csv.DictReader(f, delimiter=delimiter, quotechar='"', doublequote=True))
    with open(reference_json, encoding="utf-8") as f:
        reference = json.load(f)

    report_lines = ["# Judge Validation Report", ""]
    verdicts = {}

    for name, human_col, judge_path, scale in DIMENSIONS:
        pairs = _collect_pairs(rows, reference, human_col, judge_path)
        lines, verdict = _analyze_dimension(name, pairs, scale)
        report_lines.extend(lines)
        report_lines.append("")
        if verdict is not None:
            verdicts[name] = verdict

    report_lines.append("## Summary")
    for name, verdict in verdicts.items():
        report_lines.append(f"- **{name}**: {verdict}")

    report_text = "\n".join(report_lines)
    print(report_text)

    with open(report_md, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nReport written to {report_md}")


if __name__ == "__main__":
    main()
