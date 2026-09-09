"""
Compares the human ratings in human_review_blind.csv (filled in by hand)
against the hidden judge scores in judge_scores_reference.json.

Run this ONLY after the blind CSV has been fully rated.

Implements the measurement/threshold protocol from
docs/decisions/EVAL_DECISION_LOG.md [2026-08-01]:
  - binary/tri-level metrics: agreement rate, unweighted Cohen's kappa,
    Gwet's AC1, mean signed difference.
  - 1-5 scales, if any survive: additionally +/-1-tolerance and Spearman rho,
    with quadratic weighting on kappa.
  - all of the above broken down per system, to catch differential bias
    (a dimension fails regardless of its pooled kappa if the judge's bias
    differs meaningfully between systems).

Trust thresholds (pre-registered, see decision log): pooled kappa AND AC1
>= 0.60, +/-1-tolerance >= 80% where applicable, |mean signed diff| <= 0.5,
no per-system bias spread > 0.5.

WHY GWET'S AC1 IS REPORTED ALONGSIDE KAPPA
------------------------------------------
Cohen's kappa collapses when one category dominates the marginal, even at
high raw agreement -- the prevalence paradox. It happened here: at 69%
agreement, `refusal_accuracy` scored kappa 0.13 on a 14-to-2 human marginal.
Prevalence-adjusted, the same data give AC1 0.53. Reporting only kappa
overstates the defect; reporting only agreement hides it.

AC1 was introduced at the point where it changed NO verdict -- 0.53 and 0.58
still fall short of the 0.60 threshold, exactly as 0.13 and 0.43 did. That is
deliberate: a coefficient adopted while it cannot rescue anything is adopted
on its properties rather than on its result. The threshold is unchanged and
now applies to both coefficients; a metric must clear it on each.

The marginal distributions are printed with every dimension so a low
coefficient can be read for what it is, and so a scale level that never
occurs in the sample is visible rather than silently absorbed.
"""
import argparse
import csv
import json
import sys
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
    ("exact_match", "exact_match_human", ("judge_custom", "exact_match"), "custom"),
    ("answer_recall", "answer_recall_human", ("judge_custom", "answer_recall"), "custom"),
    ("refusal_accuracy", "refusal_accuracy_human", ("judge_custom", "refusal_accuracy"), "custom"),
    ("refusal_quality", "refusal_quality_human", ("judge_custom", "refusal_quality"), "custom"),
    ("over_refusal", "over_refusal_human", ("judge_custom", "over_refusal"), "custom"),
    ("citation_accuracy", "citation_accuracy_human", ("judge_citation", "citation_accuracy"), "custom"),
]

# Reasoning dimensions. The rating unit is one step / one transition / one
# required sub-question, so a single record contributes as many pairs as its
# chain has units. Both sides are JSON maps from unit index to 0/1.
#
# (dimension name, human CSV column, judge verdict list, verdict key, unit key)
REASONING_DIMENSIONS = [
    ("groundedness", "groundedness_human", "step_verdicts", "grounded", "step"),
    ("validity", "validity_human", "transition_verdicts", "valid", "step"),
    ("completeness", "completeness_human", "subquestion_verdicts", "covered", "subquestion"),
]

SYSTEM_ORDER = ["rag_monolith", "rag_agent", "long_context", "multi_agent"]


def _gwet_ac1(humans: list, judges: list) -> float | None:
    """Gwet's AC1: chance agreement estimated without the prevalence trap.

    Kappa estimates chance agreement from the product of the two raters'
    marginals, which grows towards 1 exactly when one category dominates --
    so the coefficient falls even though the raters agree. AC1 instead
    estimates it from how evenly the categories are used overall, which does
    not blow up on a skewed marginal.
    """
    n = len(humans)
    if n == 0:
        return None
    categories = sorted({str(v) for v in humans} | {str(v) for v in judges})
    q = len(categories)
    if q < 2:
        return None
    p_observed = sum(1 for h, j in zip(humans, judges) if h == j) / n
    pi = [
        ([str(v) for v in humans].count(c) + [str(v) for v in judges].count(c)) / (2 * n)
        for c in categories
    ]
    p_chance = sum(p * (1 - p) for p in pi) / (q - 1)
    if abs(1 - p_chance) < 1e-12:
        return None
    return (p_observed - p_chance) / (1 - p_chance)


def _marginal(values: list) -> str:
    counts: dict[str, int] = {}
    for v in values:
        key = f"{v:g}" if isinstance(v, float) else str(v)
        counts[key] = counts.get(key, 0) + 1
    return ", ".join(f"{k}:{counts[k]}" for k in sorted(counts))


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


def _collect_reasoning_pairs(rows, reference, human_col, verdict_list, verdict_key, unit_key):
    """Explode per-unit ratings into (review_id, system, human, judge) pairs.

    One record contributes one pair per rated unit, so a chain of six steps
    yields six groundedness pairs. That is the point of the rewrite: the unit
    of agreement is a decision about one step, not a summary impression of a
    whole chain.

    Units the rater left blank are skipped rather than counted as disagreement
    -- a partially rated record still contributes what it does carry. Units the
    judge did not return a verdict for are skipped for the same reason, and
    logged by their absence from n.
    """
    pairs = []
    for row in rows:
        raw = (row.get(human_col) or "").strip()
        if not raw or raw.upper() == "N/A":
            continue
        try:
            human_map = json.loads(raw)
        except json.JSONDecodeError:
            continue
        ref = reference.get(row["review_id"])
        if ref is None:
            continue
        verdicts = (ref.get("judge_reasoning") or {}).get(verdict_list) or []
        judge_map = {
            str(v.get(unit_key)): (1.0 if v.get(verdict_key) else 0.0)
            for v in verdicts
        }
        for unit, human_value in human_map.items():
            if human_value is None or str(human_value).strip() == "":
                continue
            judge_value = judge_map.get(str(unit))
            if judge_value is None:
                continue
            pairs.append(
                (f"{row['review_id']}#{unit}", ref["system_name"],
                 float(human_value), judge_value)
            )
    return pairs


def _analyze_dimension(name, pairs, scale):
    n = len(pairs)
    lines = [f"### {name} (n={n})"]
    if n == 0:
        lines.append("No rated items yet — skipped.")
        return lines, None, {"n": 0, "kappa": None, "ac1": None,
                             "exact_rate": None, "tolerance_rate": None,
                             "bias": None, "spread": None, "system_means": {}}

    humans = [p[2] for p in pairs]
    judges = [p[3] for p in pairs]

    exact_rate = sum(1 for h, j in zip(humans, judges) if abs(h - j) < 1e-9) / n
    mean_signed_diff = sum(j - h for h, j in zip(humans, judges)) / n

    lines.append(f"- Exact-match rate: {exact_rate:.0%}")
    lines.append(f"- Mean signed difference (judge - human): {mean_signed_diff:+.2f}")
    # Printed unconditionally: a coefficient cannot be read without the
    # marginal it was computed on, and an unused scale level is a finding.
    lines.append(f"- Human marginal: {_marginal(humans)}")
    lines.append(f"- Judge marginal: {_marginal(judges)}")

    kappa = None
    ac1 = None
    tolerance_rate = None
    rho = None

    # Cohen's kappa is identically 0 whenever ONE rater is constant: with
    # p(human=1)=1, expected agreement equals the judge's own rate of 1s,
    # which is also the observed agreement, so the numerator is 0 whatever
    # the raters did. Until 2026-09-09 only the both-sides-constant case was
    # treated as degenerate, and the validity dimension -- human 81:0, judge
    # 78:3, 96% agreement -- came out as "FAIL, kappa 0.00", a verdict about
    # the formula rather than about the judge. See EVAL_DECISION_LOG.md
    # [2026-09-09].
    human_constant = len(set(humans)) < 2
    judge_constant = len(set(judges)) < 2
    degenerate = human_constant or judge_constant
    if human_constant and judge_constant:
        constant_side = "both raters"
    elif human_constant:
        constant_side = "the human rater"
    else:
        constant_side = "the judge"

    if human_constant and judge_constant:
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
            label = f"Cohen's kappa ({'quadratic-weighted' if weights else 'unweighted'})"
            if degenerate:
                lines.append(
                    f"- {label}: {kappa:.2f} — identically 0 against a constant rater; "
                    "an arithmetic property of the coefficient, not a measurement"
                )
            else:
                lines.append(f"- {label}: {kappa:.2f}")
        except ValueError as e:
            lines.append(f"- Kappa: not computable ({e})")

        ac1 = _gwet_ac1(humans, judges)
        if ac1 is not None:
            lines.append(f"- Gwet's AC1: {ac1:.2f}")

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
    if ac1 is not None and ac1 < KAPPA_THRESHOLD:
        verdict_flags.append(f"AC1 {ac1:.2f} < {KAPPA_THRESHOLD}")
    if tolerance_rate is not None and tolerance_rate < TOLERANCE_THRESHOLD:
        verdict_flags.append(f"+/-1-tolerance {tolerance_rate:.0%} < {TOLERANCE_THRESHOLD:.0%}")
    if abs(mean_signed_diff) > BIAS_THRESHOLD:
        verdict_flags.append(f"|bias| {abs(mean_signed_diff):.2f} > {BIAS_THRESHOLD}")
    if spread > SYSTEM_SPREAD_THRESHOLD:
        verdict_flags.append(f"per-system spread {spread:.2f} > {SYSTEM_SPREAD_THRESHOLD} (differential bias)")

    if degenerate:
        # Neither PASS nor FAIL. At least one rater used a single category,
        # so kappa is identically 0 and there is nothing for it to measure:
        # perfect agreement here is the arithmetic consequence of a constant,
        # not evidence that the judge tracks the humans -- and a kappa of 0
        # is not evidence that it does not. Reporting either as a verdict
        # would let an unmeasured dimension into chapter 5 wearing a badge it
        # did not earn. The sample simply contains no case where the
        # dimension could discriminate; the agreement rate, AC1 and the
        # marginals are what remains to report, and a deliberately harder
        # sample (see scripts/validity_sensitivity_test.py) is the only thing
        # that settles whether the judge can discriminate at all.
        ac1_note = f", AC1 {ac1:.2f}" if ac1 is not None else ""
        verdict = (
            f"NOT ESTIMABLE — {constant_side} used one category only "
            f"(human {_marginal(humans)}, judge {_marginal(judges)}); kappa is "
            f"identically 0 against a constant rater, so agreement "
            f"{exact_rate:.0%}{ac1_note} and the marginals carry the information"
        )
    elif verdict_flags:
        verdict = "FAIL — " + "; ".join(verdict_flags)
    else:
        verdict = "PASS"
    lines.append(f"- **Verdict: {verdict}**")

    metrics = {
        "n": n,
        "kappa": kappa,
        "ac1": ac1,
        "degenerate": degenerate,
        "exact_rate": exact_rate,
        "tolerance_rate": tolerance_rate,
        "bias": mean_signed_diff,
        "spread": spread,
        "system_means": system_means,
    }
    return lines, verdict, metrics


def _sensitivity_section() -> list[str]:
    """Read the validity manipulation check, if it has been run.

    Belongs in this report rather than beside it: for a dimension whose field
    base rate is zero, "no violations found" is uninterpretable on its own.
    Whether that is a property of the systems or a blind spot of the
    instrument is exactly what a manipulation check answers, and a reader
    needs both numbers in one place.
    """
    key_path = RESULTS_DIR / "sensitivity_key.json"
    judge_path = RESULTS_DIR / "sensitivity_judge_verdicts.json"
    csv_path = RESULTS_DIR / "human_review_blind_sensitivity.csv"
    if not (key_path.exists() and judge_path.exists()):
        return []

    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.validity_sensitivity_test import analyze as _sensitivity_analyze

    report = _sensitivity_analyze(csv_path, key_path, judge_path)
    lines = [
        "",
        "---",
        "",
        "## Sensitivity check \u2014 validity",
        "",
        "Validity found no violation in the field, so its coefficients are "
        "undefined: there is nothing for two raters to agree or disagree "
        "about. A larger sample does not help, because the limit is the base "
        "rate, not n.",
        "",
        "This check answers what a zero rate leaves open. Real chains were "
        "taken from the run and in half of them the premise the conclusion "
        "numerically depends on was deleted, leaving the conclusion untouched "
        "\u2014 a textbook V1. Controls passed through unchanged; both arms were "
        "shuffled and judged blind.",
        "",
        "| | sensitivity | specificity |",
        "|---|---|---|",
    ]
    for who, label in (("human", "Human rater"), ("judge", "LLM judge")):
        r = report.get(who, {})
        sens, spec = r.get("sensitivity"), r.get("specificity")
        tp, fn = r.get("true_positive", 0), r.get("false_negative", 0)
        tn, fp = r.get("true_negative", 0), r.get("false_positive", 0)
        lines.append(
            f"| {label} | "
            + (f"{sens:.0%} ({tp}/{tp + fn})" if sens is not None else "\u2014")
            + " | "
            + (f"{spec:.0%} ({tn}/{tn + fp})" if spec is not None else "\u2014")
            + " |"
        )
    agreement = report.get("judge_human_agreement")
    lines += [
        "",
        (f"Judge and rater agree on {agreement:.0%} of the "
         f"{report.get('n_pairs', 0)} chains."
         if agreement is not None else "Not yet rated by a human."),
        "",
        "**What this does and does not license.** It measures the INSTRUMENT, "
        "not the systems: these figures must not be reported beside the field "
        "values above as though they came from the same sample. Two further "
        "limits are structural. The deletion produces the BLATANT variant of "
        "V1 \u2014 the conclusion openly rests on a figure the chain never states "
        "\u2014 so passing says nothing about the subtle variant, where the "
        "missing premise leaves no trace. And the instrument was tuned against "
        "this same constructed set, the candidate pool being too small to hold "
        "out a clean half, so the figures are optimistic. An unbiased estimate "
        "needs perturbations drawn from the full run.",
        "",
        "What it does establish: a rater applying the codebook finds the "
        "planted defect reliably, so the dimension is decidable rather than "
        "ill-defined, and the zero rate in the field is readable as a property "
        "of the systems.",
    ]
    return lines


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
    pooled_metrics = {}
    all_dimension_names = (
        [d[0] for d in REASONING_DIMENSIONS] + [d[0] for d in DIMENSIONS]
    )
    per_system_means: dict[str, dict[str, float]] = {}

    def _run(name, pairs):
        lines, verdict, metrics = _analyze_dimension(name, pairs, "custom")
        report_lines.extend(lines)
        report_lines.append("")
        pooled_metrics[name] = metrics
        per_system_means[name] = metrics.get("system_means") or {}
        if verdict is not None:
            verdicts[name] = verdict

    report_lines += ["## Reasoning dimensions (per-unit, binary)", ""]
    for name, human_col, verdict_list, verdict_key, unit_key in REASONING_DIMENSIONS:
        _run(name, _collect_reasoning_pairs(
            rows, reference, human_col, verdict_list, verdict_key, unit_key,
        ))

    report_lines += ["## Outcome and refusal metrics (per record)", ""]
    for name, human_col, judge_path, scale in DIMENSIONS:
        pairs = _collect_pairs(rows, reference, human_col, judge_path)
        lines, verdict, metrics = _analyze_dimension(name, pairs, scale)
        report_lines.extend(lines)
        report_lines.append("")
        pooled_metrics[name] = metrics
        per_system_means[name] = metrics.get("system_means") or {}
        if verdict is not None:
            verdicts[name] = verdict

    report_lines.append("## Summary")
    for name, verdict in verdicts.items():
        report_lines.append(f"- **{name}**: {verdict}")

    # --- Fairness: differential bias per system ----------------------------
    #
    # This replaces a robustness check that compared the two systems untouched
    # by the 2026-09-08 reflection defect. That defect lived in outputs which
    # no longer exist -- the verifier was repaired before the current pilot
    # ran -- so the subset isolated nothing and the section measured nothing.
    #
    # What it stood in for is the question that actually matters and is asked
    # directly here: does the judge treat one architecture differently from
    # another? The retired Core Score failed on exactly this, at 2.33 to 2.75
    # scale points, while its pooled agreement looked merely mediocre.
    report_lines += [
        "",
        "---",
        "",
        "## Differential bias per system",
        "",
        "A metric can agree well overall and still be unusable for comparing "
        "architectures, if it is systematically harder on one of them. That is "
        "how the retired Core Score failed: pooled kappa around 0.1, but a "
        "spread of 2.33 to 2.75 scale points between systems. The threshold "
        f"below is {SYSTEM_SPREAD_THRESHOLD}.",
        "",
        "| Metric | " + " | ".join(SYSTEM_ORDER) + " | spread | within threshold |",
        "|---" * (len(SYSTEM_ORDER) + 3) + "|",
    ]
    for name in all_dimension_names:
        means = per_system_means.get(name)
        if not means:
            continue
        cells = [
            f"{means[s]:+.2f}" if s in means else "—" for s in SYSTEM_ORDER
        ]
        spread = (max(means.values()) - min(means.values())) if len(means) > 1 else 0.0
        ok = "yes" if spread <= SYSTEM_SPREAD_THRESHOLD else "**NO**"
        report_lines.append(
            f"| {name} | " + " | ".join(cells) + f" | {spread:.2f} | {ok} |"
        )
    report_lines += [
        "",
        "Values are the mean signed difference (judge − human): a negative "
        "number means the judge scores that system below the rater.",
    ]
    # --- Sensitivity check, where one has been run -------------------------
    #
    # Belongs in this report and not beside it: for a dimension whose field
    # base rate is zero, "no violations found" is uninterpretable on its own.
    # Whether that is a property of the systems or a blind spot of the
    # instrument is exactly what a manipulation check answers, and a reader
    # of the validation report needs both numbers in one place.
    report_lines += _sensitivity_section()

    report_lines += [
        "",
        "---",
        "",
        "## Results at a glance",
        "",
        "| Metric | n | agreement | kappa | AC1 | verdict |",
        "|---|---|---|---|---|---|",
    ]

    def _fmt(v, pct=False):
        if v is None:
            return "\u2014"
        return f"{v:.0%}" if pct else f"{v:.2f}"

    for name in all_dimension_names:
        m = pooled_metrics.get(name)
        if m is None or not m["n"]:
            continue
        short = verdicts.get(name, "\u2014").split(" \u2014 ")[0]
        report_lines.append(
            f"| {name} | {m['n']} | {_fmt(m['exact_rate'], True)} | "
            f"{_fmt(m['kappa'])} | {_fmt(m.get('ac1'))} | {short} |"
        )

    report_lines += [
        "",
        f"Both coefficients must clear {KAPPA_THRESHOLD} for a PASS, and the "
        f"per-system spread must stay within {SYSTEM_SPREAD_THRESHOLD}.",
        "",
        "Where kappa and AC1 diverge sharply, read the marginals under the "
        "dimension: a low kappa beside a high AC1 and a high agreement rate is "
        "the prevalence paradox \u2014 the minority class is too small for kappa to "
        "be estimated, not evidence that the raters are near-random. It fails "
        "the threshold either way, but it points at a different remedy (more "
        "items in the minority class) than a genuinely low agreement rate does "
        "(a defective prompt).",
        "",
        "`NOT ESTIMABLE` is neither PASS nor FAIL: at least one side used a "
        "single category. Cohen's kappa is identically 0 against a constant "
        "rater whatever the other rater did, so it can neither pass nor fail "
        "such a dimension; the agreement rate, Gwet's AC1 and the marginals "
        "are reported instead, and a deliberately constructed sample is the "
        "only way to tell whether the judge can discriminate at all.",
        "",
    ]

    report_text = "\n".join(report_lines)
    print(report_text)

    with open(report_md, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nReport written to {report_md}")


if __name__ == "__main__":
    main()
