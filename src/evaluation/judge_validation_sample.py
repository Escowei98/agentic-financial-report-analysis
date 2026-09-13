"""
The gold-standard ids the LLM judge was validated and tuned on.

Lives in `src/evaluation/` rather than in the batch script because two
consumers need it: `scripts/run_judge_validation_batch.py` runs the systems
on exactly these ids, and `scripts/run_full_eval.py` marks them in the n=150
output. The judge prompts were revised against the human ratings of these
items, so any robustness check of the full run can exclude them: a metric
that holds on the other items does not rest on items the judge was fitted to.

Primary sample: 10 stratified ids (2x per fa_type).

Reserve sample: disjoint, same stratification, seed=42, reserved for
re-validating a judge fix without reusing the items that motivated it.
It is deliberately unbalanced: 12 FA-Refusal items against 3 per answerable
stratum, because `refusal_accuracy` and `refusal_quality` are defined on the
refusal stratum alone and a small n there leaves Cohen's kappa at the mercy
of the prevalence paradox. The 12 cover every `refusal_evidence` class three
times, except `out_of_scope`, where the free pool holds only three items.
The answerable items carry the three reasoning dimensions, whose rating unit
is a step, a transition or a sub-question, so 12 items yield several hundred
binary judgements.
"""

from __future__ import annotations

PRIMARY_SAMPLE_IDS: list[int] = [85, 16, 21, 26, 45, 106, 59, 147, 124, 76]

RESERVE_SAMPLE_IDS: list[int] = [
    # answerable (12): reasoning dimensions, exact_match, answer_recall,
    # over_refusal
    18, 20, 24, 44, 52, 74, 91, 119,
    41, 27, 81, 117,
    # FA-Refusal (12): refusal_accuracy, refusal_quality
    123, 122, 126,                   # out_of_scope
    136, 134, 140,                   # counter_evidence
    137, 125, 138,                   # silent
    144, 141, 149,                   # underspecified
]

# Every id the judge was ever validated or tuned on. This is the set the full
# run marks with `in_judge_validation_sample`.
JUDGE_VALIDATION_IDS: frozenset[int] = frozenset(PRIMARY_SAMPLE_IDS + RESERVE_SAMPLE_IDS)
