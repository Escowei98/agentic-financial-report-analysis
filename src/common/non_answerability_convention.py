"""
Shared non-answerability instruction, injected symmetrically into all four
systems' prompts (S1-S4) so no architecture gets an unfair advantage on the
FA-Refusal stratum.

Separate constant, separate module, same discipline as
`answer_format_convention.py`: defined once and imported everywhere, never
copy-pasted, so the four systems receive byte-identical wording. The two
conventions stay apart because they answer different questions -- one governs
how a delivered answer is formatted, this one governs what to do when there is
no answer to deliver -- and because either may need to change without the
other.

WHY THIS EXISTS
---------------
The systems used to be told only "if the data is not in the filings, say so".
Under that instruction a bare "I cannot answer" was the complete expected
behaviour, and the FA-Refusal stratum could not distinguish the three defect
classes it is built from: data outside the corpus, a false premise, and an
underspecified question. For the latter two a bare abstention conflates "your
question is faulty" with "I could not find it" -- only one of those shows the
system recognised the defect.

The four cases below are stated in a fixed order, but the systems are never
told which case a given question falls into. Classifying it is the work being
measured (see refusal_quality in src/evaluation/custom_evaluator.py).

Case (c) is the load-bearing one. The corpus can support "the filings do not
report X" but not "X did not happen" -- for several gold-standard items
(e.g. the GOOGL stock split, the AMZN dividend) the filings are simply silent,
and a confident negative claim there is parametric knowledge, i.e. exactly the
NF-1 violation this stratum is meant to detect.

This instruction also reaches the 120 answerable items, where it pushes
towards declining. That cost is measured by the `over_refusal` control metric
in the same run; it must not be left unmeasured.

Case (a) names comparative columns since 2026-09-09: the corpus holds
alternating fiscal years, so FY2023 exists only inside the FY2024 filing, and
the systems were treating it as out of scope while the judge treated it as in
scope. See src/common/corpus_coverage.py.
"""

from src.common.corpus_coverage import COMPARATIVE_COLUMNS_NOTE

NON_ANSWERABILITY_CONVENTION = (
    "\nWhen a question cannot be answered from the filings available to you, "
    "do not produce a figure. Instead state which of the following applies, "
    "using only what the filings themselves support:\n"
    "(a) The data lies outside the available filings -- name what IS "
    "available (which companies, which fiscal years including those carried "
    "in a filing's comparative columns, which report type). "
    + COMPARATIVE_COLUMNS_NOTE + "\n"
    "(b) The question assumes something the filings contradict -- say so "
    "explicitly and give the actual figure with its citation.\n"
    "(c) The filings do not report the requested item at all -- say that they "
    "do not report it. Do NOT claim the underlying fact is untrue: a filing "
    "that is silent on a matter is not evidence that the matter did not "
    "occur.\n"
    "(d) The question does not identify a single company and/or fiscal year "
    "-- name what is missing, and lay out the possible readings where that "
    "helps.\n"
    "Never close such a gap with knowledge from outside the filings."
)
