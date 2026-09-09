"""
The gold-standard ids that were used to validate -- and, through the
remediation rounds, to tune -- the LLM judge.

Lives in `src/evaluation/` rather than in the batch script because two very
different consumers need it: `scripts/run_judge_validation_batch.py` runs the
systems on exactly these ids, and `scripts/run_full_eval.py` has to MARK them
in the n=150 output. The judge prompts for exact_match, refusal_accuracy,
groundedness and validity were revised against the human ratings of these
items (EVAL_DECISION_LOG.md [2026-08-01], [2026-09-08], [2026-09-09]), so any
robustness check of the full run should be able to exclude them: a metric
that holds on the other 116 items does not rest on items the judge was fitted
to.

Primary sample: 10 stratified ids (2x per fa_type), drawn 2026-08-01.

Reserve sample: disjoint, same stratification, seed=42 -- reserved in
EVAL_DECISION_LOG.md [2026-08-01] specifically for re-validating a judge
prompt/logic fix WITHOUT reusing the items that motivated the fix.

Grown from 12 to 24 items on 2026-09-08, in two independent moves.

(a) FA-Refusal 4 -> 12. `refusal_accuracy` and `refusal_quality` are defined
    on this stratum alone, so 4 items capped them at n=16 records -- and of
    those 16, the human marginal came out 14-to-2. Cohen's kappa collapses
    on a marginal like that whatever the raters do (the prevalence paradox):
    69% agreement scored kappa 0.13, while prevalence-adjusted the same data
    give AC1 0.53. Neither clears the 0.60 threshold, so the defect is real
    and not merely an artefact of n -- but n=16 badly overstates it, and
    12 items lift both metrics to n=48, the range in which `exact_match`
    reached kappa 0.87.

    The eight additions bring every `refusal_evidence` class to 3. The
    `out_of_scope` class stops at 3 because the free pool holds only two
    more; report that imbalance rather than padding it from another class.
    Ids 123 and 136 (added 2026-09-08 earlier the same day) already closed
    the ZERO coverage those two classes had: `counter_evidence` is the
    variant that demands a grounded counter-fact and carries the
    `refusal_quality` rubric for false_premise, and `out_of_scope` is the
    not_in_corpus subtype.

(b) Answerable 8 -> 12, one more per FA type. These carry the three
    reasoning dimensions, whose rating unit is a step, a transition or a
    sub-question rather than a record -- so 12 items yield several hundred
    binary judgements. The four additions are drawn with seed 42 from the
    items in neither sample.

The FA-Refusal stratum is therefore 12 items against 3 for each other
stratum: an intentional, documented deviation from uniform stratification,
because that stratum is where the two failing metrics live. Report it
separately if the imbalance matters for how the results are read.
"""

from __future__ import annotations

PRIMARY_SAMPLE_IDS: list[int] = [85, 16, 21, 26, 45, 106, 59, 147, 124, 76]

RESERVE_SAMPLE_IDS: list[int] = [
    # answerable (12): reasoning dimensions, exact_match, answer_recall,
    # over_refusal
    18, 20, 24, 44, 52, 74, 91, 119,
    41, 27, 81, 117,
    # FA-Refusal (12): refusal_accuracy, refusal_quality
    123, 122, 126,                   # out_of_scope (pool exhausted at 3)
    136, 134, 140,                   # counter_evidence
    137, 125, 138,                   # silent
    144, 141, 149,                   # underspecified
]

# Every id the judge was ever validated or tuned on. This is the set the full
# run marks with `in_judge_validation_sample`.
JUDGE_VALIDATION_IDS: frozenset[int] = frozenset(PRIMARY_SAMPLE_IDS + RESERVE_SAMPLE_IDS)
