"""
Custom Evaluators for the Financial RAG/Agent Pipeline.

Implements domain-specific metrics:
1. Exact Match (Numerical Fidelity)
2. Answer Recall (Information Completeness)
3. Refusal Accuracy (Over-Reliance / Hallucination on Unanswerable Queries)
4. Refusal Quality (subtype-conditioned diagnosis of WHY a query fails)
5. Over-Refusal (declining a question that was answerable — the control that
   keeps 3. from being satisfiable by declining everything)
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import NamedTuple

from src.common.config import load_config
from src.common.llm_client import get_judge_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Lazy LLM singleton for evaluation
# ---------------------------------------------------------------------------
_EVAL_LLM = None

def get_eval_llm():
    """Lazily load the judge LLM instance for evaluation judging."""
    global _EVAL_LLM
    if _EVAL_LLM is None:
        _EVAL_LLM = get_judge_llm()
    return _EVAL_LLM

# ---------------------------------------------------------------------------
#  Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CustomEvalResult:
    exact_match: float = 0.0
    answer_recall: float = 0.0
    refusal_accuracy: float = 0.0
    rationales: dict = field(default_factory=dict)

    # --- metrics that are undefined for part of the dataset -----------------
    #
    # None, not 0.0: these two are each computed on only one half of the gold
    # standard, and a 0.0 default would be averaged in as "the system failed"
    # over the half where the metric has no meaning at all. The three metrics
    # above predate this and keep their 0.0 default because eval_runner
    # already branches on expected_answerable before reading them.
    refusal_quality: float | None = None
    """0/0.5/1.0 on FA-Refusal items, None on answerable ones."""

    over_refusal: float | None = None
    """1.0 if the system declined an ANSWERABLE question, 0.0 if it attempted
    one. None on FA-Refusal items and on answerable items the system got
    right (a correct answer cannot be a refusal, so the judge is not asked)."""

    def to_dict(self):
        return {
            "exact_match": round(self.exact_match, 2),
            "answer_recall": round(self.answer_recall, 2),
            "refusal_accuracy": round(self.refusal_accuracy, 2),
            "refusal_quality": (
                None if self.refusal_quality is None else round(self.refusal_quality, 2)
            ),
            "over_refusal": (
                None if self.over_refusal is None else round(self.over_refusal, 2)
            ),
            "rationales": self.rationales,
        }

# ---------------------------------------------------------------------------
#  Deterministic numeric tolerance pre-check (exact_match / answer_recall)
#
#  Human-validation on the judge-validation sample (see
#  EVAL_DECISION_LOG.md [2026-08-01]) found kappa=0.08 for exact_match and
#  kappa=0.41 for answer_recall: the LLM judge does not reliably apply its
#  own stated rounding/unit-equivalence rule (e.g. "$112,390 million" vs.
#  GT "$112.4 billion" — mathematically equal, scored as a mismatch in
#  roughly half of the disagreement cases). This pre-check only ever
#  CONFIRMS a match (never asserts a mismatch) — anything it can't
#  confidently verify still falls through to the LLM judge, so it cannot
#  introduce new false negatives of its own.
#
#  It can, however, produce false POSITIVES, and those are not symmetric
#  across systems: a bare number in the system answer used to be assumed
#  to already sit on the ground truth's scale, so the longer an answer is,
#  the more numbers it offers as accidental matches — which would favour
#  the systems that write longer answers (S3/S4). Numbers taken from the
#  *answer* are therefore only accepted when they carry an explicit unit
#  signal (magnitude word, currency marker or percent marker); a bare
#  number is treated as unusable and the item falls through to the LLM
#  judge. Numbers taken from the *ground truth* keep the old lenient
#  reading: those strings are short, curated, and authored at gt_unit's
#  scale by construction.
# ---------------------------------------------------------------------------

_UNIT_MULTIPLIERS = {
    "billion": 1e9, "bn": 1e9,
    "million": 1e6, "mn": 1e6,
    "thousand": 1e3, "k": 1e3,
}

_NUMBER_RE = re.compile(
    r"(?P<sign>[+-])?\s*(?P<cur_pre>\$|usd\b|eur\b|€)?\s*(?P<number>\d[\d,]*\.?\d*)\s*"
    r"(?P<unit>billion|bn|million|mn|thousand(?!\s+points)|k\b|"
    r"percentage\s+points?|pp\b|%|percent)?"
    r"(?P<cur_post>\s*(?:usd\b|eur\b|dollars?\b))?",
    re.IGNORECASE,
)


class ExtractedNumber(NamedTuple):
    """One number found in a text, with whatever unit signals surround it.

    `unit` is the lowercased magnitude/percent word attached to the
    occurrence (None when there is none); `has_currency` records whether a
    currency marker ($, USD, EUR, "dollars") sits directly before or after
    it. Both are needed to tell a number that names an amount from one
    that just happens to appear in the text — see _normalize_to_base.
    """

    value: float
    unit: str | None
    has_currency: bool


def _extract_numbers(text: str) -> list[ExtractedNumber]:
    """Extract every number in `text` together with its unit signals."""
    results = []
    for m in _NUMBER_RE.finditer(text or ""):
        raw = m.group("number")
        if not raw:
            continue
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        if m.group("sign") == "-":
            value = -value
        unit = m.group("unit")
        has_currency = bool(m.group("cur_pre") or m.group("cur_post"))
        results.append(ExtractedNumber(value, unit.lower() if unit else None, has_currency))
    return results


_PERCENT_UNIT_WORDS = {"%", "percent", "percentage point", "percentage points", "pp"}


def _normalize_to_base(
    value: float,
    unit_word: str | None,
    gt_unit: str,
    has_currency: bool = False,
    strict: bool = False,
) -> float | None:
    """Convert an extracted number to a common base, given the item's gt_unit.

    Returns None if the number's unit is incompatible with gt_unit (e.g. a
    percentage found where a dollar amount was expected) — such numbers are
    not usable for comparison, not a signal of anything.

    Args:
        value: The number as extracted.
        unit_word: Its magnitude/percent word, or None if it carried none.
        gt_unit: The gold-standard item's unit.
        has_currency: Whether a currency marker sits next to the number.
        strict: Set for numbers taken from a system answer. A bare number
            (no magnitude word, no currency marker, no percent marker) is
            then rejected as unusable instead of being read as already
            sitting on gt_unit's scale. Free-form answers — especially the
            long ones the long-context systems produce — contain plenty of
            numbers that have nothing to do with the asked-for figure, and
            the lenient reading turns each of them into a chance for an
            accidental confirmation. Ground-truth strings are short and
            authored at gt_unit's scale, so they are parsed with
            strict=False.
    """
    gt_unit = (gt_unit or "").strip().lower()

    if gt_unit in ("usd_billion", "usd_million"):
        if unit_word in _PERCENT_UNIT_WORDS:
            return None
        if unit_word in _UNIT_MULTIPLIERS:
            return value * _UNIT_MULTIPLIERS[unit_word]
        if unit_word is None:
            if strict and not has_currency:
                return None
            # No explicit magnitude word on this occurrence — assume it's
            # already expressed at gt_unit's scale.
            return value * (1e9 if gt_unit == "usd_billion" else 1e6)
        return None

    if gt_unit in ("percent", "pp_change"):
        if unit_word in _PERCENT_UNIT_WORDS:
            return value
        if unit_word is None:
            # A bare number in an answer is not evidence of a percentage:
            # the tolerance here is absolute (0.15pp), so any nearby
            # figure of the right magnitude would confirm the match.
            return None if strict else value
        return None

    if gt_unit == "usd_per_share":
        if unit_word is None:
            return None if (strict and not has_currency) else value
        return None

    # count / text / n/a / unrecognized gt_unit: not a numeric comparison
    # this function can make — see _UNITS_WITHOUT_DETERMINISTIC_CHECK.
    return None


# gt_units the pre-check never rules on. "text"/"n/a"/"" aren't numeric
# comparisons at all; "count" is numeric but carries no unit signal that
# could tie a number in the answer to the asked-for quantity (gold standard
# v4 has counts such as "2" — a digit that appears in almost any long
# answer by chance). All of them fall through to the LLM judge.
_UNITS_WITHOUT_DETERMINISTIC_CHECK = {"", "text", "n/a", "count"}


def _deterministic_numeric_match(
    ground_truth: str, answer: str, gt_unit: str,
    relative_tolerance: float = 0.02, pp_tolerance: float = 0.15,
) -> bool | None:
    """
    Returns True if every number in the Ground Truth has a unit-normalized,
    tolerance-matching counterpart in the System Answer. Returns None
    (inconclusive — caller should fall back to the LLM judge) whenever
    parsing is incomplete or any GT number can't be confidently matched;
    never returns False, by design (see module-level comment above).

    Only numbers that carry an explicit unit signal in the answer (a
    magnitude word, a currency marker, or a percent marker) are eligible
    counterparts — a bare number is not read as an amount on gt_unit's
    scale, since that would let any figure in a long answer confirm the
    match by coincidence.
    """
    gt_unit_clean = (gt_unit or "").strip().lower()
    if gt_unit_clean in _UNITS_WITHOUT_DETERMINISTIC_CHECK:
        return None

    gt_numbers = [
        n for e in _extract_numbers(ground_truth)
        if (n := _normalize_to_base(e.value, e.unit, gt_unit_clean, e.has_currency)) is not None
    ]
    if not gt_numbers:
        return None

    # Answer-side numbers are parsed strictly: only those carrying an
    # explicit unit signal count as candidates (see _normalize_to_base).
    ans_numbers = [
        n for e in _extract_numbers(answer)
        if (n := _normalize_to_base(
            e.value, e.unit, gt_unit_clean, e.has_currency, strict=True,
        )) is not None
    ]
    if not ans_numbers:
        return None

    def _close(gt_val: float, ans_val: float) -> bool:
        if gt_unit_clean in ("percent", "pp_change"):
            return abs(gt_val - ans_val) <= pp_tolerance
        if gt_val == 0:
            return abs(ans_val) <= pp_tolerance
        return abs(gt_val - ans_val) / abs(gt_val) <= relative_tolerance

    return all(any(_close(g, a) for a in ans_numbers) for g in gt_numbers) or None


# ---------------------------------------------------------------------------
#  Prompts
#
#  exact_match is dispatched by answer type (see _classify_answer_type):
#  atomic numeric values, cross-entity comparison verdicts, and qualitative/
#  categorical claims each get a purpose-built prompt rather than one
#  generic numeric-match prompt for all three. See EVAL_DECISION_LOG.md
#  [2026-09] for the rationale (a single numeric-match prompt was found to
#  near-always fail on qualitative items via literal wording, and the
#  deterministic pre-check is unsound for comparison items -- it matches
#  numbers as an unordered set with no entity binding, so a reversed verdict
#  with the same two raw numbers present would be falsely confirmed).
# ---------------------------------------------------------------------------

EXACT_MATCH_PROMPT = """\
You are a strict financial evaluator. Your task is to determine if the System Answer contains the EXACT numerical value specified in the Ground Truth.

**Question:** {question}
**Ground Truth Value:** {ground_truth} {gt_unit}
**System Answer:** {answer}

Rules:
1. The numbers must match exactly, but you must allow for reasonable rounding differences and derived calculations (e.g., 2.08 or 2.1 is an acceptable match for 2.0 if it clearly stems from the same underlying calculation).
2. Units must be semantically equivalent (e.g., "$1.2B", "1.2 billion dollars", "1,200 million" are equivalent).
3. If the Ground Truth is empty or not a number (e.g. NA), and the system also didn't provide a number, it's a match.
4. Both the Ground Truth and System Answer use standard English number formatting (e.g., '1,234.56' is one thousand two hundred thirty-four point five six). Do NOT interpret periods as thousands separators.
5. The system answer can contain extra text, as long as the correct number is clearly stated as the answer to the question.

Output ONLY a JSON object:
{{
  "rationale": "<1-2 sentences explaining if the number matches>",
  "score": <1.0 for match, 0.0 for mismatch>
}}
"""

COMPARISON_PROMPT = """\
You are a strict financial evaluator. The Ground Truth is a COMPARATIVE claim between two entities (e.g. "WINNER (X vs LOSER Y)") -- it names which entity is greater/leads on a specific measure, along with supporting figures for both.

**Question:** {question}
**Ground Truth:** {ground_truth} {gt_unit}
**System Answer:** {answer}

Rules:
1. The PRIMARY criterion is the verdict: the System Answer must identify the same winning/leading entity as the Ground Truth. If the winner is wrong or reversed, this is a MISMATCH regardless of how accurate any numbers in the answer are.
2. The SECONDARY criterion is the supporting figures: numbers for each entity should match the Ground Truth's figures for that same entity, allowing for reasonable rounding and derived-calculation differences (e.g., 2.08 or 2.1 is an acceptable match for 2.0 if it clearly stems from the same underlying calculation). Do not fail a correct verdict purely for imprecise secondary figures.
3. Units must be semantically equivalent (e.g., "$1.2B", "1.2 billion dollars", "1,200 million" are equivalent).
4. The system answer can contain extra text, as long as the correct verdict is clearly and unambiguously stated.

Output ONLY a JSON object:
{{
  "rationale": "<1-2 sentences explaining if the verdict (and figures) match>",
  "score": <1.0 for correct verdict with acceptable figures, 0.0 for a wrong or reversed verdict>
}}
"""

QUALITATIVE_PROMPT = """\
You are an expert evaluator. The Ground Truth is a QUALITATIVE or categorical claim (a trend, direction, category, period/date, or yes/no answer) without one specific number to match.

**Question:** {question}
**Ground Truth:** {ground_truth}
**System Answer:** {answer}

Rules:
1. Judge semantic equivalence, not literal wording. E.g. "Slightly rising" and "increased modestly year over year" convey the same claim; "FY2024" and "the fiscal year 2024" are the same category.
2. The System Answer's core qualitative/categorical claim must match the Ground Truth's core claim.
3. Additional supporting detail (e.g. a year-by-year breakdown before stating the overall trend) is fine and must not be penalized, as long as the overall claim is consistent with the Ground Truth.
4. If the Ground Truth and System Answer disagree on the claim (e.g. GT says "increasing", answer says "decreasing" or "flat"; or GT says "FY2024" and the answer says "FY2023"), it is a MISMATCH.

Output ONLY a JSON object:
{{
  "rationale": "<1-2 sentences explaining if the qualitative claim matches>",
  "score": <1.0 for match, 0.0 for mismatch>
}}
"""

ANSWER_RECALL_PROMPT = """\
You are an expert evaluator. Your task is to determine if ALL the key facts from the Ground Truth are present in the System Answer.

**Question:** {question}
**Ground Truth:** {ground_truth}
**System Answer:** {answer}

Rules:
1. The System Answer must contain all the essential information present in the Ground Truth.
2. If the Ground Truth mentions 3 companies, the System Answer must mention all 3.
3. If the System Answer misses any crucial part of the Ground Truth, it is incomplete.
4. Numerical variance: Accept answers that represent the same underlying data but differ slightly due to rounding or independent calculation (e.g., accepting 2.08 or 2.1 for a Ground Truth of 2.0). Do NOT penalize these minor recalculation differences.
5. Thoroughness is not a defect. If the System Answer provides additional supporting detail beyond the Ground Truth's phrasing (e.g., a year-by-year breakdown when the Ground Truth is a one-line trend statement like "Slightly rising"), and the Ground Truth's core claim is clearly and correctly stated among that detail (even if implicit in the numbers shown), this counts as completely present. Only penalize for MISSING information, never for additional correct information.
6. Score 1.0 if completely present, 0.5 if partially present, 0.0 if entirely missing.

Output ONLY a JSON object:
{{
  "rationale": "<1-2 sentences explaining what is present or missing>",
  "score": <1.0, 0.5, or 0.0>
}}
"""

def _corpus_scope_sentence() -> str:
    """Describe the corpus the systems actually see, from configs/base.yaml.

    Hard-coding this was a live defect: the prompt claimed "FY2022, FY2023 and
    FY2024 ONLY" long after the corpus had moved to FY2020/FY2022/FY2024, so
    the judge rewarded a refusal on a question the corpus could answer and
    penalised the correct answer — across the whole 30-item refusal stratum.
    Deriving it from the same file the systems are built from makes that
    divergence impossible. See EVAL_DECISION_LOG.md [2026-09-06].
    """
    config = load_config()
    tickers = [c["ticker"] for c in config.get("companies", [])]
    years = sorted(config.get("fiscal_years", []))
    company_list = ", ".join(tickers) if tickers else "the configured companies"
    if years:
        year_list = ", ".join(f"FY{y}" for y in years)
        # A 10-K carries prior years in comparative columns, so a fiscal year
        # can be answerable without its own filing being present. Saying only
        # which filings exist would make the judge reject those.
        span = (
            f"filings for fiscal years {year_list} ONLY. Note that a 10-K "
            f"reports the two preceding fiscal years in comparative columns, "
            f"so a figure for a year between {years[0]} and {years[-1]} may "
            f"well be answerable even when that year has no filing of its own"
        )
    else:
        span = "the configured fiscal years ONLY"
    return (
        f"The system's corpus contains SEC 10-K filings for exactly "
        f"{len(tickers)} companies ({company_list}), covering {span}."
    )


REFUSAL_ACCURACY_PROMPT_TEMPLATE = """\
You are an expert evaluator checking for "Over-Reliance" (hallucination). The user asked an unanswerable question (e.g., data not in context, false premise, or ambiguous).

**Knowledge base scope:** {corpus_scope} A question about a fiscal year \
outside that range, or about a company outside this list, is by construction \
NOT answerable from this corpus — refusing such a question is the CORRECT \
behavior, even though that data might exist in the real world outside this \
corpus. Do not judge answerability against your own general knowledge of \
these companies.

**Question:** {{question}}
**System Answer:** {{answer}}

**Reference — what this corpus can legitimately support here:** {{gt_correction}}

The reference is what a well-grounded response could say. It is NOT a required
wording, and an answer that says less than the reference can still be correct.
Use it for one purpose only: to tell a grounded statement from an invented one,
instead of having to guess.

Rules:
1. The system MUST NOT answer with a fabricated number or fact.
2. A correct refusal looks like: "I cannot answer this based on the provided
filings", "This information is not available", or correcting a false premise
("Company X did not report a loss, but a profit of...").
3. If the system fabricates a direct answer (e.g., "The revenue for 2035 was
$50B" when 2035 is not in the filings), score it 0.0.
4. CHECK THE SCOPE BEFORE CALLING SOMETHING OUT OF SCOPE. Read the fiscal year
in the question against the list above. A year named in that list is IN scope,
and so is any year carried in a filing's comparative columns. Do not score an
answer down on the ground that the corpus does not reach a year that it does.
5. A correction is only credited when it is grounded. Judge that against the
reference, not against your own knowledge of these companies. A statement the
reference supports is grounded; a figure or fact that contradicts it, or that
goes materially beyond it, is a fabrication like any other and scores 0.0.
In particular, a confident negative claim about the world ("no stock split took
place", "the company pays no dividend") is NOT grounded by a filing that simply
does not mention the matter; only "the filings do not report it" is. Absence of
a statement in a filing is not evidence that the underlying fact is untrue.
6. LENGTH AND DETAIL ARE NOT DEFECTS. An answer that refuses AND explains what
the filings do say, or names the corpus boundary, or gives the correct figure
from the reference, is a better refusal than a bare one — never a worse one.
Score it 0.0 only if something in it is actually fabricated, never for
volunteering grounded detail. Two answers that both avoid fabrication score the
same whether one is a single sentence and the other a paragraph.
7. Score 1.0 for a correct, grounded refusal/clarification, 0.0 for a
hallucinated/fabricated answer or an ungrounded correction.

Output ONLY a JSON object:
{{{{
  "rationale": "<1-2 sentences explaining if the model correctly refused or hallucinated>",
  "score": <1.0 or 0.0>
}}}}
"""


# ---------------------------------------------------------------------------
#  Refusal Quality (secondary, subtype-conditioned)
#
#  refusal_accuracy answers one question: did the system fabricate an answer?
#  That is the primary, FA-6/NF-1-bearing measurement and it deliberately
#  treats a bare "I cannot answer" and a diagnosed refusal as equal, so it
#  stays comparable across all four architectures and all 30 items.
#
#  It is, however, blind to the difference between the three defect classes
#  the stratum is built from. A bare abstention conflates "the premise of your
#  question is false" with "I could not find the data" -- for false_premise and
#  ambiguous_entity items those are different findings, and only one of them
#  shows that the system actually recognised the defect. refusal_quality grades
#  that recognition on 0 / 0.5 / 1.0.
#
#  Deliberately a SEPARATE judge call rather than a compound output of the
#  primary prompt: the headline number must not move when this rubric is
#  tuned, and the secondary metric has to remain droppable without touching
#  the primary one if human validation does not support it.
#
#  The systems are never told which subtype an item belongs to. The rubric
#  therefore measures whether a system DIAGNOSES the defect, not whether it
#  follows a per-subtype instruction.
# ---------------------------------------------------------------------------

_REFUSAL_QUALITY_HEADER = """\
You are grading HOW WELL a system handled a question it could not answer from
its knowledge base. Whether the system avoided fabricating an answer is graded
elsewhere — here you grade the diagnostic quality of what it said instead.

**Knowledge base scope:** {corpus_scope} Judge only against this corpus, never \
against your own general knowledge of these companies.

"""

_REFUSAL_QUALITY_FOOTER = """\

Output ONLY a JSON object:
{{{{
  "rationale": "<1-2 sentences on what the system did or failed to diagnose>",
  "score": <1.0, 0.5, or 0.0>
}}}}
"""

# not_in_corpus: the request falls outside the corpus boundary. The added
# value over a bare refusal is naming that boundary.
REFUSAL_QUALITY_SCOPE_PROMPT_TEMPLATE = _REFUSAL_QUALITY_HEADER + """\
The question asks for data the knowledge base does not contain.

**Question:** {{question}}
**Reference (what the corpus can legitimately support):** {{gt_correction}}
**System Answer:** {{answer}}

Rules:
1. Score 1.0 if the system declined AND named the reason in a way that matches \
the reference — which companies, fiscal years or report types are available, \
or that this particular item is not reported in a 10-K. The wording need not \
match; the substance must.
2. Score 0.5 if the system declined but gave no usable reason ("I cannot \
answer this", "the information is not available", full stop).
3. Score 0.0 if the system produced an answer to the question as though it \
were answerable, or invented a reason that the corpus does not support.
4. Do not reward length. A short, correct scope statement scores 1.0.
""" + _REFUSAL_QUALITY_FOOTER

# false_premise: the question presupposes something untrue. Whether a
# correction is legitimate depends on whether the corpus carries the
# counter-evidence -- `evidence_clause` injects that per item.
REFUSAL_QUALITY_PREMISE_PROMPT_TEMPLATE = _REFUSAL_QUALITY_HEADER + """\
The question presupposes something that is not the case.

**Question:** {{question}}
**Reference (what the corpus can legitimately support):** {{gt_correction}}
**System Answer:** {{answer}}

{{evidence_clause}}

Rules:
1. Score 1.0 if the system identified the false assumption AND said what the \
filings actually show, consistent with the reference above.
2. Score 0.5 if the system declined or said the data was unavailable without \
identifying that the question's assumption was itself wrong.
3. Score 0.0 if the system answered as though the premise held (e.g. reported \
a loss figure for a profitable year), OR if it "corrected" the premise with a \
claim the corpus does not support.
4. A system that merely repeats the question's framing back without \
challenging it has not identified the false assumption.
""" + _REFUSAL_QUALITY_FOOTER

_EVIDENCE_CLAUSE_COUNTER = (
    "Evidence basis for this item: the filings positively contradict the "
    "premise, so a correction naming the actual figure IS supported and is "
    "what a 1.0 requires."
)
_EVIDENCE_CLAUSE_SILENT = (
    "Evidence basis for this item: the filings are simply SILENT on the "
    "matter — they contain no counter-figure. The only grounded statement is "
    "that the filings do not report it. An answer that confidently asserts "
    "the underlying fact is untrue (e.g. \"no such split happened\", \"the "
    "company pays no dividend\") goes beyond the corpus and must score 0.0, "
    "not 1.0, however plausible it sounds."
)

# ambiguous_entity: the question does not identify a single company and/or
# fiscal year. The added value is naming what is missing.
REFUSAL_QUALITY_AMBIGUITY_PROMPT_TEMPLATE = _REFUSAL_QUALITY_HEADER + """\
The question does not identify a single company and/or fiscal year, so it has
several equally valid readings.

**Question:** {{question}}
**Reference (what is actually underspecified):** {{gt_correction}}
**System Answer:** {{answer}}

Rules:
1. Score 1.0 if the system named what is missing (company, fiscal year, or \
the metric), consistent with the reference — either by asking for it or by \
laying out the possible readings.
2. Score 0.5 if the system declined without naming what was missing.
3. Score 0.0 if the system silently picked one reading and answered it as \
though the question had been specific. Answering for one company when four \
were possible is exactly this case, even if the figure it reports is correct \
for that company.
4. Enumerating the readings and answering ALL of them, while making the \
ambiguity explicit, counts as naming what is missing — score 1.0.
""" + _REFUSAL_QUALITY_FOOTER


# ---------------------------------------------------------------------------
#  Over-Refusal (control metric, answerable items only)
#
#  refusal_accuracy is the correctness score of the FA-Refusal stratum, so a
#  system that declines everything scores a perfect 30/30 on it. Without a
#  counterweight the metric is formally degenerate. The counterweight is the
#  answerable stratum: over_refusal measures how often a system declined a
#  question it was supposed to answer.
#
#  This matters most once the non-answerability convention is in the system
#  prompts: that instruction pushes toward declining, and its cost has to be
#  visible in the same run. Only evaluated on answerable items the system got
#  wrong -- a correct answer cannot be a refusal.
# ---------------------------------------------------------------------------

OVER_REFUSAL_PROMPT = """\
You are checking whether a system DECLINED to answer a question that it was
supposed to answer, as opposed to attempting an answer and getting it wrong.

**Question:** {question}
**System Answer:** {answer}

Rules:
1. Score 1.0 if the answer is a refusal, a non-answer, or a request for \
clarification — e.g. "I cannot answer this from the provided filings", "the \
information is not available", "please specify which company you mean".
2. Score 0.0 if the system attempted a substantive answer, even if that \
answer is incomplete or wrong. A wrong figure is an attempt, not a refusal.
3. An answer that gives the requested figure and additionally notes a caveat \
or limitation is an attempt — score 0.0.
4. Partial refusals (answers one part of a two-part question, declines the \
other) score 0.0: the system did engage with the question.

Output ONLY a JSON object:
{{
  "rationale": "<1 sentence: did the system decline, or did it attempt an answer?>",
  "score": <1.0 or 0.0>
}}
"""


def get_refusal_quality_prompt(subtype: str) -> str:
    """The refusal-quality template for one subtype, corpus scope filled in.

    Falls back to the scope template for an unknown subtype: that is the
    least demanding of the three (it only asks for the boundary to be named),
    so a mis-annotated item cannot be scored against a rubric that expects
    more than the corpus can support.
    """
    templates = {
        "not_in_corpus": REFUSAL_QUALITY_SCOPE_PROMPT_TEMPLATE,
        "false_premise": REFUSAL_QUALITY_PREMISE_PROMPT_TEMPLATE,
        "ambiguous_entity": REFUSAL_QUALITY_AMBIGUITY_PROMPT_TEMPLATE,
    }
    template = templates.get(subtype, REFUSAL_QUALITY_SCOPE_PROMPT_TEMPLATE)
    return template.format(corpus_scope=_corpus_scope_sentence())


def get_evidence_clause(refusal_evidence: str) -> str:
    """The per-item evidence clause for the false-premise rubric.

    An unclassified item (pre-v5 file) gets the conservative clause: demanding
    a grounded correction that the corpus may not support would manufacture
    false negatives, so silence is assumed until the annotation says otherwise.
    """
    if refusal_evidence == "counter_evidence":
        return _EVIDENCE_CLAUSE_COUNTER
    return _EVIDENCE_CLAUSE_SILENT


def get_refusal_accuracy_prompt() -> str:
    """The refusal prompt with the live corpus scope filled in."""
    return REFUSAL_ACCURACY_PROMPT_TEMPLATE.format(
        corpus_scope=_corpus_scope_sentence()
    )


# Fallback reference for a pre-v5 file, where `gt_correction` is empty. Stating
# that no reference is available is not the same as passing an empty string:
# a blank line under a "Reference" heading reads to the judge as "nothing is
# supportable here", which is the opposite of the intended meaning.
NO_REFERENCE_AVAILABLE = (
    "(no reference recorded for this item — judge grounding on the filings "
    "alone, and do not treat additional detail as evidence of fabrication)"
)

# ---------------------------------------------------------------------------
#  Evaluation Logic
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    try:
        parsed: dict = json.loads(cleaned)
        return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            return parsed
        except json.JSONDecodeError:
            pass
    logger.warning("Failed to parse JSON from judge output: %s", text[:200])
    return {}


def _classify_answer_type(item) -> str:
    """Dispatch key for exact_match scoring strategy.

    "comparison" (FA-4, cross-entity verdicts) and "qualitative"
    (gt_unit == "text", no single number to match) each need a scoring
    approach that a generic atomic-numeric-match prompt can't provide --
    see the module comment above the prompt definitions.
    """
    if item.fa_type == "FA-4":
        return "comparison"
    if (item.gt_unit or "").strip().lower() == "text":
        return "qualitative"
    return "numeric_atomic"


def evaluate_custom_metrics(item, answer: str) -> CustomEvalResult:
    """
    Evaluate a single item against the custom domain metrics.
    Uses LLM-as-a-judge for robustness on formatting and units.
    
    Args:
        item: GoldStandardItem
        answer: The string answer from the system
    """
    # Coerce to string to avoid Gemini part issues
    if not isinstance(answer, str):
        answer = str(answer)

    llm = get_eval_llm()
    result = CustomEvalResult()

    answer_type = _classify_answer_type(item)

    # Deterministic numeric pre-check: only ever confirms a match (True) or
    # is inconclusive (None); never asserts a mismatch on its own. See
    # module-level comment above _deterministic_numeric_match. Restricted to
    # "numeric_atomic" items -- it compares numbers as an unordered set with
    # no entity binding, which is unsound for "comparison" items (a reversed
    # two-entity verdict with the same two raw numbers present would be
    # falsely confirmed as a match).
    det_match = None
    if item.expected_answerable and item.ground_truth and answer_type == "numeric_atomic":
        det_match = _deterministic_numeric_match(item.ground_truth, answer, item.gt_unit)

    # gt_unit is only meaningful for numeric_atomic/comparison prompts, and
    # even there may itself be "text"/"n/a" for a handful of items (e.g. a
    # comparison whose gt_value embeds its own units) -- don't leak that
    # literally into the judge prompt.
    gt_unit_display = (
        item.gt_unit if (item.gt_unit or "").strip().lower() not in ("", "text", "n/a") else ""
    )

    # 1. Exact Match (Only if there is a Ground Truth value and it's answerable)
    if item.expected_answerable and item.ground_truth:
        if det_match:
            result.exact_match = 1.0
            result.rationales["exact_match"] = (
                "Deterministic numeric match (unit-normalized, within tolerance); LLM judge skipped."
            )
        else:
            if answer_type == "comparison":
                prompt = COMPARISON_PROMPT.format(
                    question=item.question,
                    ground_truth=item.ground_truth,
                    gt_unit=gt_unit_display,
                    answer=answer,
                )
            elif answer_type == "qualitative":
                prompt = QUALITATIVE_PROMPT.format(
                    question=item.question,
                    ground_truth=item.ground_truth,
                    answer=answer,
                )
            else:
                prompt = EXACT_MATCH_PROMPT.format(
                    question=item.question,
                    ground_truth=item.ground_truth,
                    gt_unit=gt_unit_display,
                    answer=answer,
                )
            response = llm.invoke(prompt)
            data = _extract_json(response.content if hasattr(response, "content") else str(response))
            result.exact_match = float(data.get("score", 0.0))
            result.rationales["exact_match"] = data.get("rationale", "")

    # 2. Answer Recall (Completeness - Only if answerable)
    if item.expected_answerable and item.ground_truth:
        if det_match:
            result.answer_recall = 1.0
            result.rationales["answer_recall"] = (
                "Deterministic numeric match (unit-normalized, within tolerance); LLM judge skipped."
            )
        else:
            prompt = ANSWER_RECALL_PROMPT.format(
                question=item.question,
                ground_truth=item.ground_truth,
                answer=answer
            )
            response = llm.invoke(prompt)
            data = _extract_json(response.content if hasattr(response, "content") else str(response))
            result.answer_recall = float(data.get("score", 0.0))
            result.rationales["answer_recall"] = data.get("rationale", "")

    # 3. Refusal Accuracy (Only for unanswerable/refusal questions)
    if not item.expected_answerable:
        # `gt_correction` is passed for one narrowly defined job: to let the
        # judge check a correction against a known-grounded reference instead
        # of guessing. Guessing was the documented defect — on id 137 the
        # judge scored S3 and S4 0.0 for correctly rectifying a false premise
        # while S1 and S2 got 1.0 for a terser "not mentioned", and the
        # mechanism was that a longer answer offers more surface to suspect.
        # That penalised the more informative answer, differentially by
        # architecture. See EVAL_DECISION_LOG.md [2026-09-07], status OPEN.
        prompt = get_refusal_accuracy_prompt().format(
            question=item.question,
            answer=answer,
            gt_correction=item.gt_correction or NO_REFERENCE_AVAILABLE,
        )
        response = llm.invoke(prompt)
        data = _extract_json(response.content if hasattr(response, "content") else str(response))
        result.refusal_accuracy = float(data.get("score", 0.0))
        result.rationales["refusal_accuracy"] = data.get("rationale", "")

        # 4. Refusal Quality (secondary, subtype-conditioned). Separate judge
        # call on purpose — see the rubric block above. Needs the v5 reference
        # text; on a pre-v5 file gt_correction is empty and the metric stays
        # None rather than grading against a blank reference.
        if item.gt_correction:
            quality_prompt = get_refusal_quality_prompt(item.subtype).format(
                question=item.question,
                gt_correction=item.gt_correction,
                answer=answer,
                evidence_clause=get_evidence_clause(item.refusal_evidence),
            )
            response = llm.invoke(quality_prompt)
            data = _extract_json(
                response.content if hasattr(response, "content") else str(response)
            )
            result.refusal_quality = float(data.get("score", 0.0))
            result.rationales["refusal_quality"] = data.get("rationale", "")

    # 5. Over-Refusal (control metric, answerable items only). Only asked when
    # the item was scored wrong: a correct answer cannot be a refusal, so
    # spending a judge call on it would buy nothing.
    else:
        # Gate on the better of the two correctness scores rather than on the
        # gt_unit rule eval_runner uses to pick between them: this module
        # cannot import that helper (eval_runner imports this one), and the
        # looser gate only ever asks the judge more often, never less. A fully
        # correct answer under either reading cannot be a refusal.
        if max(result.exact_match, result.answer_recall) < 1.0:
            prompt = OVER_REFUSAL_PROMPT.format(
                question=item.question,
                answer=answer,
            )
            response = llm.invoke(prompt)
            data = _extract_json(
                response.content if hasattr(response, "content") else str(response)
            )
            result.over_refusal = float(data.get("score", 0.0))
            result.rationales["over_refusal"] = data.get("rationale", "")
        else:
            result.over_refusal = 0.0

    return result
