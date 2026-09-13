# Judge Validation Report

## Reasoning dimensions (per-unit, binary)

### groundedness (n=99)
- Exact-match rate: 96%
- Mean signed difference (judge - human): +0.02
- Human marginal: 0:9, 1:90
- Judge marginal: 0:7, 1:92
- Cohen's kappa (unweighted): 0.73
- Gwet's AC1: 0.95
- Per-system mean signed difference:
    - long_context (n=28): +0.04
    - multi_agent (n=18): +0.00
    - rag_agent (n=29): +0.00
    - rag_monolith (n=24): +0.04
- **Verdict: PASS**

### validity (n=81)
- Exact-match rate: 96%
- Mean signed difference (judge - human): -0.04
- Human marginal: 1:81
- Judge marginal: 0:3, 1:78
- Cohen's kappa (unweighted): 0.00 — identically 0 against a constant rater; an arithmetic property of the coefficient, not a measurement
- Gwet's AC1: 0.96
- Per-system mean signed difference:
    - long_context (n=21): +0.00
    - multi_agent (n=11): +0.00
    - rag_agent (n=21): +0.00
    - rag_monolith (n=28): -0.11
- **Verdict: NOT ESTIMABLE — the human rater used one category only (human 1:81, judge 0:3, 1:78); kappa is identically 0 against a constant rater, so agreement 96%, AC1 0.96 and the marginals carry the information**

### completeness (n=130)
- Exact-match rate: 99%
- Mean signed difference (judge - human): +0.01
- Human marginal: 0:36, 1:94
- Judge marginal: 0:35, 1:95
- Cohen's kappa (unweighted): 0.98
- Gwet's AC1: 0.99
- Per-system mean signed difference:
    - long_context (n=28): +0.04
    - multi_agent (n=34): +0.00
    - rag_agent (n=34): +0.00
    - rag_monolith (n=34): +0.00
- **Verdict: PASS**

## Outcome and refusal metrics (per record)

### exact_match (n=48)
- Exact-match rate: 96%
- Mean signed difference (judge - human): -0.01
- Human marginal: 0:17, 1:31
- Judge marginal: 0:17, 0.5:1, 1:30
- Cohen's kappa (unweighted): 0.91
- Gwet's AC1: 0.95
- Per-system mean signed difference:
    - long_context (n=12): -0.08
    - multi_agent (n=12): +0.00
    - rag_agent (n=12): +0.04
    - rag_monolith (n=12): +0.00
- **Verdict: PASS**

### answer_recall (n=48)
- Exact-match rate: 100%
- Mean signed difference (judge - human): +0.00
- Human marginal: 0:15, 0.5:2, 1:31
- Judge marginal: 0:15, 0.5:2, 1:31
- Cohen's kappa (unweighted): 1.00
- Gwet's AC1: 1.00
- Per-system mean signed difference:
    - long_context (n=12): +0.00
    - multi_agent (n=12): +0.00
    - rag_agent (n=12): +0.00
    - rag_monolith (n=12): +0.00
- **Verdict: PASS**

### refusal_accuracy (n=48)
- Exact-match rate: 79%
- Mean signed difference (judge - human): -0.04
- Human marginal: 0:6, 1:42
- Judge marginal: 0:8, 1:40
- Cohen's kappa (unweighted): 0.17
- Gwet's AC1: 0.72
- Per-system mean signed difference:
    - long_context (n=12): -0.08
    - multi_agent (n=12): -0.08
    - rag_agent (n=12): -0.08
    - rag_monolith (n=12): +0.08
- **Verdict: FAIL — kappa 0.17 < 0.6**

### refusal_quality (n=48)
- Exact-match rate: 73%
- Mean signed difference (judge - human): -0.04
- Human marginal: 0:3, 0.5:17, 1:28
- Judge marginal: 0:4, 0.5:19, 1:25
- Cohen's kappa (unweighted): 0.51
- Gwet's AC1: 0.63
- Per-system mean signed difference:
    - long_context (n=12): -0.12
    - multi_agent (n=12): -0.04
    - rag_agent (n=12): +0.00
    - rag_monolith (n=12): +0.00
- **Verdict: FAIL — kappa 0.51 < 0.6**

### over_refusal (n=48)
- Exact-match rate: 94%
- Mean signed difference (judge - human): -0.06
- Human marginal: 0:34, 1:14
- Judge marginal: 0:37, 1:11
- Cohen's kappa (unweighted): 0.84
- Gwet's AC1: 0.90
- Per-system mean signed difference:
    - long_context (n=12): +0.00
    - multi_agent (n=12): +0.00
    - rag_agent (n=12): -0.08
    - rag_monolith (n=12): -0.17
- **Verdict: PASS**

### citation_accuracy (n=0)
No rated items yet — skipped.

## Summary
- **groundedness**: PASS
- **validity**: NOT ESTIMABLE — the human rater used one category only (human 1:81, judge 0:3, 1:78); kappa is identically 0 against a constant rater, so agreement 96%, AC1 0.96 and the marginals carry the information
- **completeness**: PASS
- **exact_match**: PASS
- **answer_recall**: PASS
- **refusal_accuracy**: FAIL — kappa 0.17 < 0.6
- **refusal_quality**: FAIL — kappa 0.51 < 0.6
- **over_refusal**: PASS

---

## Differential bias per system

A metric can agree well overall and still be unusable for comparing architectures, if it is systematically harder on one of them. That is how the retired Core Score failed: pooled kappa around 0.1, but a spread of 2.33 to 2.75 scale points between systems. The threshold below is 0.5.

| Metric | rag_monolith | rag_agent | long_context | multi_agent | spread | within threshold |
|---|---|---|---|---|---|---|
| groundedness | +0.04 | +0.00 | +0.04 | +0.00 | 0.04 | yes |
| validity | -0.11 | +0.00 | +0.00 | +0.00 | 0.11 | yes |
| completeness | +0.00 | +0.00 | +0.04 | +0.00 | 0.04 | yes |
| exact_match | +0.00 | +0.04 | -0.08 | +0.00 | 0.12 | yes |
| answer_recall | +0.00 | +0.00 | +0.00 | +0.00 | 0.00 | yes |
| refusal_accuracy | +0.08 | -0.08 | -0.08 | -0.08 | 0.17 | yes |
| refusal_quality | +0.00 | +0.00 | -0.12 | -0.04 | 0.12 | yes |
| over_refusal | -0.17 | -0.08 | +0.00 | +0.00 | 0.17 | yes |

Values are the mean signed difference (judge − human): a negative number means the judge scores that system below the rater.

---

## Sensitivity check — validity

Validity found no violation in the field, so its coefficients are undefined: there is nothing for two raters to agree or disagree about. A larger sample does not help, because the limit is the base rate, not n.

This check answers what a zero rate leaves open. Real chains were taken from the run and in half of them the premise the conclusion numerically depends on was deleted, leaving the conclusion untouched — a textbook V1. Controls passed through unchanged; both arms were shuffled and judged blind.

| | sensitivity | specificity |
|---|---|---|
| Human rater | 92% (11/12) | 100% (12/12) |
| LLM judge | 83% (10/12) | 100% (12/12) |

Judge and rater agree on 88% of the 24 chains.

**What this does and does not license.** It measures the INSTRUMENT, not the systems: these figures must not be reported beside the field values above as though they came from the same sample. Two further limits are structural. The deletion produces the BLATANT variant of V1 — the conclusion openly rests on a figure the chain never states — so passing says nothing about the subtle variant, where the missing premise leaves no trace. And the instrument was tuned against this same constructed set, the candidate pool being too small to hold out a clean half, so the figures are optimistic. An unbiased estimate needs perturbations drawn from the full run.

What it does establish: a rater applying the codebook finds the planted defect reliably, so the dimension is decidable rather than ill-defined, and the zero rate in the field is readable as a property of the systems.

---

## Results at a glance

| Metric | n | agreement | kappa | AC1 | verdict |
|---|---|---|---|---|---|
| groundedness | 99 | 96% | 0.73 | 0.95 | PASS |
| validity | 81 | 96% | 0.00 | 0.96 | NOT ESTIMABLE |
| completeness | 130 | 99% | 0.98 | 0.99 | PASS |
| exact_match | 48 | 96% | 0.91 | 0.95 | PASS |
| answer_recall | 48 | 100% | 1.00 | 1.00 | PASS |
| refusal_accuracy | 48 | 79% | 0.17 | 0.72 | FAIL |
| refusal_quality | 48 | 73% | 0.51 | 0.63 | FAIL |
| over_refusal | 48 | 94% | 0.84 | 0.90 | PASS |

Both coefficients must clear 0.6 for a PASS, and the per-system spread must stay within 0.5.

Where kappa and AC1 diverge sharply, read the marginals under the dimension: a low kappa beside a high AC1 and a high agreement rate is the prevalence paradox — the minority class is too small for kappa to be estimated, not evidence that the raters are near-random. It fails the threshold either way, but it points at a different remedy (more items in the minority class) than a genuinely low agreement rate does (a defective prompt).

`NOT ESTIMABLE` is neither PASS nor FAIL: at least one side used a single category. Cohen's kappa is identically 0 against a constant rater whatever the other rater did, so it can neither pass nor fail such a dimension; the agreement rate, Gwet's AC1 and the marginals are reported instead, and a deliberately constructed sample is the only way to tell whether the judge can discriminate at all.
