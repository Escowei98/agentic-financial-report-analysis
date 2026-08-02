"""
Shared answer-format instruction, injected symmetrically into all four
systems' prompts (S1-S4) so no architecture gets an unfair clarity
advantage over another.

Single source of truth: this text must match the "Answer Format
Convention" section in data/gold_standard/gold_standard_README.md.
Defined once and imported everywhere it's needed, rather than
copy-pasted, so the four systems are guaranteed to receive byte-identical
wording -- see EVAL_DECISION_LOG.md [2026-08-01].
"""

ANSWER_FORMAT_CONVENTION = (
    "When answering ratio or percentage-based questions (e.g. Debt/Equity "
    "Ratio, margins), always express the result as a percentage, not a "
    "decimal fraction. For Debt/Equity Ratio specifically, use Total "
    "Liabilities / Total Stockholders' Equity as the definition unless the "
    "question states a different basis. When asked about a trend or "
    "development across multiple periods, always state the overall "
    "directional trend (e.g. rising, falling, roughly flat, or mixed) "
    "explicitly in your answer -- a year-by-year breakdown may be included "
    "as supporting detail but does not replace the overall statement."
)
