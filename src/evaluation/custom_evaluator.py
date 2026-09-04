"""
Custom Evaluators for the Financial RAG/Agent Pipeline.

Implements domain-specific metrics:
1. Exact Match (Numerical Fidelity)
2. Answer Recall (Information Completeness)
3. Refusal Accuracy (Over-Reliance / Hallucination on Unanswerable Queries)
"""

import json
import logging
import re
from dataclasses import dataclass

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
    rationales: dict = None

    def __post_init__(self):
        if self.rationales is None:
            self.rationales = {}

    def to_dict(self):
        return {
            "exact_match": round(self.exact_match, 2),
            "answer_recall": round(self.answer_recall, 2),
            "refusal_accuracy": round(self.refusal_accuracy, 2),
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
# ---------------------------------------------------------------------------

_UNIT_MULTIPLIERS = {
    "billion": 1e9, "bn": 1e9,
    "million": 1e6, "mn": 1e6,
    "thousand": 1e3, "k": 1e3,
}

_NUMBER_RE = re.compile(
    r"(?P<sign>[+-])?\s*\$?\s*(?P<number>\d[\d,]*\.?\d*)\s*"
    r"(?P<unit>billion|bn|million|mn|thousand(?!\s+points)|k\b|"
    r"percentage\s+points?|pp\b|%|percent)?",
    re.IGNORECASE,
)


def _extract_numbers(text: str) -> list[tuple[float, str | None]]:
    """Extract (signed value, lowercase unit-word-or-None) pairs from text."""
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
        results.append((value, unit.lower() if unit else None))
    return results


_PERCENT_UNIT_WORDS = {"%", "percent", "percentage point", "percentage points", "pp"}


def _normalize_to_base(value: float, unit_word: str | None, gt_unit: str) -> float | None:
    """Convert an extracted number to a common base, given the item's gt_unit.

    Returns None if the number's unit is incompatible with gt_unit (e.g. a
    percentage found where a dollar amount was expected) — such numbers are
    not usable for comparison, not a signal of anything.
    """
    gt_unit = (gt_unit or "").strip().lower()

    if gt_unit in ("usd_billion", "usd_million"):
        if unit_word in _PERCENT_UNIT_WORDS:
            return None
        if unit_word in _UNIT_MULTIPLIERS:
            return value * _UNIT_MULTIPLIERS[unit_word]
        if unit_word is None:
            # No explicit magnitude word on this occurrence — assume it's
            # already expressed at gt_unit's scale.
            return value * (1e9 if gt_unit == "usd_billion" else 1e6)
        return None

    if gt_unit in ("percent", "pp_change"):
        if unit_word in _PERCENT_UNIT_WORDS or unit_word is None:
            return value
        return None

    if gt_unit in ("usd_per_share", "count"):
        if unit_word is None:
            return value
        return None

    return None  # text / n/a / unrecognized gt_unit: not a numeric comparison


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
    """
    gt_unit_clean = (gt_unit or "").strip().lower()
    if gt_unit_clean in ("", "text", "n/a"):
        return None

    gt_numbers = [
        n for v, u in _extract_numbers(ground_truth)
        if (n := _normalize_to_base(v, u, gt_unit_clean)) is not None
    ]
    if not gt_numbers:
        return None

    ans_numbers = [
        n for v, u in _extract_numbers(answer)
        if (n := _normalize_to_base(v, u, gt_unit_clean)) is not None
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

REFUSAL_ACCURACY_PROMPT = """\
You are an expert evaluator checking for "Over-Reliance" (hallucination). The user asked an unanswerable question (e.g., data not in context, false premise, or ambiguous).

**Knowledge base scope:** The system's corpus contains SEC 10-K filings for \
exactly four companies (AAPL, MSFT, AMZN, GOOGL), covering fiscal years \
FY2022, FY2023, and FY2024 ONLY. A question about any other fiscal year, or \
about a company outside this list, is by construction NOT answerable from \
this corpus — refusing such a question is the CORRECT behavior, even though \
that data might exist in the real world outside this corpus. Do not judge \
answerability against your own general knowledge of these companies.

**Question:** {question}
**System Answer:** {answer}

Rules:
1. The system MUST refuse to answer with a fabricated number or fact.
2. A correct refusal looks like: "I cannot answer this based on the provided filings", "This information is not available", or correcting a false premise ("Company X did not report a loss, but a profit of...").
3. If the system fabricates a direct answer (e.g., "The revenue for 2025 was $50B" when 2025 is not in the filings), score it 0.0.
4. A refusal that correctly explains the corpus is scoped to FY2022-FY2024 (e.g. "MSFT FY2020 is not available; only FY2022-FY2024 filings are in this corpus") is a CORRECT refusal — score it 1.0, do not penalize it for not producing FY2020 data.
5. Score 1.0 for a correct refusal/clarification, 0.0 for a hallucinated/fabricated answer.

Output ONLY a JSON object:
{{
  "rationale": "<1-2 sentences explaining if the model correctly refused or hallucinated>",
  "score": <1.0 or 0.0>
}}
"""

# ---------------------------------------------------------------------------
#  Evaluation Logic
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
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
        prompt = REFUSAL_ACCURACY_PROMPT.format(
            question=item.question,
            answer=answer
        )
        response = llm.invoke(prompt)
        data = _extract_json(response.content if hasattr(response, "content") else str(response))
        result.refusal_accuracy = float(data.get("score", 0.0))
        result.rationales["refusal_accuracy"] = data.get("rationale", "")

    return result
