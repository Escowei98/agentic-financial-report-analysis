"""
Citation Accuracy Evaluator for the Financial RAG/Agent Pipeline.

Checks whether a system answer correctly attributes its content to the
expected source(s) (company/fiscal-year and 10-K section), independent of
whether the answer's factual content is correct (see custom_evaluator.py
for that).

Three-tier approach, mirroring the deterministic-pre-check + LLM-judge-fallback
pattern in custom_evaluator.py:
  1. Strict deterministic: parse a structured "(TICKER, FYYEAR, SECTION_NAME)"
     citation out of the answer with a regex.
  2. Loose deterministic: if no structured citation is found (e.g. a system
     whose prompt only loosely instructs it to cite a source), scan the
     answer sentence-by-sentence for company name/ticker + fiscal-year +
     section-keyword mentions in prose. This directly targets the same
     question the LLM judge would otherwise be asked ("does the wording
     make clear which company/year/section this comes from?"), just via
     keyword matching instead of a judge call — shrinking how often that
     (measurably noisy, see docs/decisions/EVAL_DECISION_LOG.md) fallback
     is needed at all.
  Both deterministic tiers score against the gold-standard doc_ids/
  source_sections with set-based Precision/Recall/F1 (doc ids reuse
  evaluate_tool_selection from process_evaluator.py; sections use a
  citation-group-aware variant, see _score_sections).
  3. LLM judge fallback: only when neither deterministic tier finds any
     company/ticker mention at all (i.e. the source is genuinely
     unidentifiable via keyword matching, not just unformatted).
"""

import logging
import re
from dataclasses import dataclass

from src.common.llm_client import get_judge_llm
from src.evaluation.process_evaluator import evaluate_tool_selection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Lazy LLM singleton for evaluation (mirrors custom_evaluator.py)
# ---------------------------------------------------------------------------
_EVAL_LLM = None


def get_eval_llm():
    """Lazily load the judge LLM instance for citation judging."""
    global _EVAL_LLM
    if _EVAL_LLM is None:
        _EVAL_LLM = get_judge_llm()
    return _EVAL_LLM


# ---------------------------------------------------------------------------
#  Dataclass
# ---------------------------------------------------------------------------

@dataclass
class CitationEvalResult:
    citation_accuracy: float = 0.0
    doc_precision: float = 0.0
    doc_recall: float = 0.0
    doc_f1: float = 0.0
    section_precision: float = 0.0
    section_recall: float = 0.0
    section_f1: float = 0.0
    method: str = "skipped"  # "deterministic" | "llm_judge" | "skipped"
    rationales: dict = None

    def __post_init__(self):
        if self.rationales is None:
            self.rationales = {}

    def to_dict(self):
        return {
            "citation_accuracy": round(self.citation_accuracy, 2),
            "doc_precision": round(self.doc_precision, 2),
            "doc_recall": round(self.doc_recall, 2),
            "doc_f1": round(self.doc_f1, 2),
            "section_precision": round(self.section_precision, 2),
            "section_recall": round(self.section_recall, 2),
            "section_f1": round(self.section_f1, 2),
            "method": self.method,
            "rationales": self.rationales,
        }


# ---------------------------------------------------------------------------
#  Deterministic structured-citation parser
#
#  Matches the unified "(TICKER, FYYEAR, SECTION_NAME)" format now enforced
#  by all four systems' prompts (S1: src/systems/rag_monolith/pipeline.py,
#  S2: src/systems/rag_agent/agent.py, S3: src/systems/long_context/prompt.py,
#  S4: src/systems/multi_agent/prompts.py).
# ---------------------------------------------------------------------------

#  Ticker/year are each allowed one optional stray '{'/'}' wrapping them —
#  observed in practice with S1 (Gemini echoes literal brace punctuation
#  from the "{TICKER}" placeholder in its instruction around the ticker
#  token specifically, e.g. "({AAPL}, FY2024, MD&A)") even though the
#  rendered prompt itself contains plain, unescaped "{TICKER}" text.
_CITATION_RE = re.compile(
    r"\(\s*\{?(?P<ticker>AAPL|MSFT|AMZN|GOOGL)\}?\s*,\s*\{?(?:FY\s?)?(?P<year>20\d{2})\}?\s*,\s*(?P<section>[^)]+?)\s*\)",
    re.IGNORECASE,
)

# Free-text section name -> canonical source_section id(s) it satisfies.
# Grounded in the actual section vocabulary the systems can cite (see
# src/common/ingestion.py SECTION_PATTERNS / TENK_DIRECT_ATTRS) and the
# gold standard's controlled vocabulary (data/gold_standard/gold_standard_v3.csv:
# item_1, item_7, item_8_income_stmt, item_8_balance_sheet, item_8_cash_flow,
# item_8_segment). More specific aliases are listed before the "financial
# statements"/"item 8" catch-alls so a specific sub-section match wins.
_SECTION_TEXT_TO_IDS: dict[str, frozenset] = {
    "business": frozenset({"item_1"}),
    "item 1": frozenset({"item_1"}),
    "md&a": frozenset({"item_7"}),
    "managements discussion and analysis": frozenset({"item_7"}),
    "item 7": frozenset({"item_7"}),
    "income statement": frozenset({"item_8_income_stmt"}),
    "statement of income": frozenset({"item_8_income_stmt"}),
    "balance sheet": frozenset({"item_8_balance_sheet"}),
    "cash flow statement": frozenset({"item_8_cash_flow"}),
    "cash flow": frozenset({"item_8_cash_flow"}),
    "segment information": frozenset({"item_8_segment"}),
    "segment": frozenset({"item_8_segment"}),
    "financial statements": frozenset({
        "item_8_income_stmt", "item_8_balance_sheet", "item_8_cash_flow", "item_8_segment",
    }),
    "item 8": frozenset({
        "item_8_income_stmt", "item_8_balance_sheet", "item_8_cash_flow", "item_8_segment",
    }),
}

# Human-readable labels for the LLM-judge prompt.
_ID_TO_LABEL = {
    "item_1": "Business (Item 1)",
    "item_7": "MD&A (Item 7)",
    "item_8_income_stmt": "Financial Statements - Income Statement (Item 8)",
    "item_8_balance_sheet": "Financial Statements - Balance Sheet (Item 8)",
    "item_8_cash_flow": "Financial Statements - Cash Flow Statement (Item 8)",
    "item_8_segment": "Financial Statements - Segment Information (Item 8)",
}


def _map_section_text(text: str) -> frozenset:
    """Map a free-text section reference to canonical source_section id(s)."""
    normalized = re.sub(r"[^a-z0-9&\s]", "", text.lower()).strip()
    for alias, ids in _SECTION_TEXT_TO_IDS.items():
        if alias in normalized:
            return ids
    return frozenset()


def _parse_structured_citations(answer: str) -> list[dict]:
    """Extract all structured "(TICKER, FYYEAR, SECTION)" citations from an answer."""
    parsed = []
    for m in _CITATION_RE.finditer(answer):
        ticker = m.group("ticker").upper()
        year = m.group("year")
        section_text = m.group("section").strip()
        parsed.append({
            "doc_id": f"{ticker}_{year}",
            "section_text": section_text,
            "section_ids": _map_section_text(section_text),
        })
    return parsed


# ---------------------------------------------------------------------------
#  Loose (free-text) sentence-level parser — second deterministic tier
# ---------------------------------------------------------------------------

_COMPANY_NAME_TO_TICKER = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "amazon": "AMZN",
    "alphabet": "GOOGL",
    "google": "GOOGL",
}

_TICKER_OR_NAME_RE = re.compile(
    r"\b(AAPL|MSFT|AMZN|GOOGL|Apple|Microsoft|Amazon|Alphabet|Google)\b",
    re.IGNORECASE,
)

# Deliberately loose: the "FY" prefix is optional, so a bare "in 2024" also
# counts — in this corpus (SEC 10-K answers for 4 companies, FY2022-2024
# only) a standalone 4-digit "20xx" token is overwhelmingly a fiscal-year
# reference, not a dollar figure (those are always much larger and/or
# comma-formatted, e.g. "$391,035 million").
_YEAR_RE = re.compile(r"\b(?:FY\s?)?(20\d{2})\b", re.IGNORECASE)

# Sentence boundary: punctuation + whitespace, followed by a capital letter
# or opening paren — avoids splitting inside decimal numbers like "391.0".
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _split_sentences(answer: str) -> list[str]:
    sentences = []
    for line in answer.split("\n"):
        sentences.extend(_SENTENCE_SPLIT_RE.split(line))
    return sentences


def _parse_freetext_citations(answer: str) -> list[dict]:
    """
    Sentence-level fallback parser for prose citations without brackets,
    e.g. "Apple's fiscal 2024 income statement shows total net sales of
    $391.0 billion." A sentence only yields a citation if it names BOTH a
    company and a year; the section (if any) is whatever section-keyword
    alias is found in that same sentence (may be empty — an unqualified
    citation still counts toward doc scoring, same as the strict tier).
    """
    parsed = []
    for sentence in _split_sentences(answer):
        if not sentence.strip():
            continue
        company_matches = _TICKER_OR_NAME_RE.findall(sentence)
        if not company_matches:
            continue
        year_match = _YEAR_RE.search(sentence)
        if not year_match:
            continue
        year = year_match.group(1)
        section_ids = _map_section_text(sentence)
        tickers = {_COMPANY_NAME_TO_TICKER.get(m.lower(), m.upper()) for m in company_matches}
        for ticker in tickers:
            parsed.append({
                "doc_id": f"{ticker}_{year}",
                "section_text": sentence.strip(),
                "section_ids": section_ids,
            })
    return parsed


def _format_doc_ids_for_judge(doc_ids: list[str]) -> str:
    parts = []
    for d in doc_ids:
        ticker, _, year = d.partition("_")
        parts.append(f"{ticker} FY{year}")
    return "; ".join(parts) if parts else "(none)"


def _format_sections_for_judge(section_ids: list[str]) -> str:
    labels = [_ID_TO_LABEL.get(s, s) for s in section_ids]
    return "; ".join(labels) if labels else "(none)"


# ---------------------------------------------------------------------------
#  Deterministic scoring
# ---------------------------------------------------------------------------

def _score_sections(
    expected_ids: set, section_id_sets: list
) -> tuple[float, float, float]:
    """
    Precision/Recall/F1 for section citations, at citation-group granularity.

    Section text maps to *groups* of ids, not single ids (e.g. a generic
    "Financial Statements" citation maps to all four item_8_* sub-ids —
    see _SECTION_TEXT_TO_IDS). Scoring plain set-intersection over the
    expanded ids would penalize that generic-but-correct citation as three
    false positives. Instead: a citation is "correct" (counts toward
    precision) if its id group overlaps the expected ids at all; an
    expected id is "recalled" if any citation's group contains it.
    """
    if not expected_ids:
        return (1.0, 1.0, 1.0) if not section_id_sets else (0.0, 0.0, 0.0)
    if not section_id_sets:
        return 0.0, 0.0, 0.0

    recalled = sum(1 for eid in expected_ids if any(eid in ids for ids in section_id_sets))
    recall = recalled / len(expected_ids)

    correct_citations = sum(1 for ids in section_id_sets if ids & expected_ids)
    precision = correct_citations / len(section_id_sets)

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _score_deterministic(item, parsed: list[dict], tier: str = "structured") -> CitationEvalResult:
    expected_doc_ids = list(set(item.doc_ids))
    expected_section_ids = set(item.source_sections)

    parsed_doc_ids = list({c["doc_id"] for c in parsed})
    section_id_sets = [c["section_ids"] for c in parsed]
    parsed_section_ids = sorted({sid for ids in section_id_sets for sid in ids})

    # Doc ids are unambiguous 1:1 (ticker_year), so the generic set-based
    # Precision/Recall/F1 already used for tool selection applies directly.
    doc_res = evaluate_tool_selection(expected_doc_ids, parsed_doc_ids)
    section_precision, section_recall, section_f1 = _score_sections(
        expected_section_ids, section_id_sets
    )

    citation_accuracy = (doc_res.tool_f1 + section_f1) / 2

    return CitationEvalResult(
        citation_accuracy=citation_accuracy,
        doc_precision=doc_res.tool_precision,
        doc_recall=doc_res.tool_recall,
        doc_f1=doc_res.tool_f1,
        section_precision=section_precision,
        section_recall=section_recall,
        section_f1=section_f1,
        method="deterministic",
        rationales={
            "citation_accuracy": (
                f"Parsed {len(parsed)} {tier} citation(s): "
                f"doc_ids={sorted(parsed_doc_ids)}, sections={parsed_section_ids}."
            )
        },
    )


# ---------------------------------------------------------------------------
#  LLM-judge fallback (free-text citations)
# ---------------------------------------------------------------------------

CITATION_JUDGE_PROMPT = """\
You are evaluating whether a financial analyst's answer correctly identifies \
its source, even if only loosely or in prose (not a strict structured citation).

**Question:** {question}
**System Answer:** {answer}

**Expected source(s):**
- Company/fiscal year(s): {expected_docs}
- Section(s): {expected_sections}

Rules:
1. The answer does not need a formal "(TICKER, FYYEAR, SECTION)" citation. Judge whether the answer's wording makes clear which company, fiscal year, and filing section the information comes from (e.g. naming the company and year in prose, or referring to "the balance sheet", "the risk factors section", etc.).
2. If the answer references additional companies/years/sections beyond the expected ones, do not penalize as long as the expected ones are also correctly covered.
3. Score doc_match_score 1.0 if all expected companies/fiscal years are correctly identifiable in the answer, 0.5 if some but not all, 0.0 if none/wrong.
4. Score section_match_score 1.0 if all expected filing sections are correctly identifiable, 0.5 if some but not all, 0.0 if none/wrong/not mentioned.

Output ONLY a JSON object:
{{
  "rationale": "<1-2 sentences>",
  "doc_match_score": <0.0, 0.5, or 1.0>,
  "section_match_score": <0.0, 0.5, or 1.0>
}}
"""


def _extract_json(text: str) -> dict:
    import json

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
    logger.warning("Failed to parse JSON from citation judge output: %s", text[:200])
    return {}


def _score_with_judge(item, answer: str) -> CitationEvalResult:
    llm = get_eval_llm()
    prompt = CITATION_JUDGE_PROMPT.format(
        question=item.question,
        answer=answer,
        expected_docs=_format_doc_ids_for_judge(item.doc_ids),
        expected_sections=_format_sections_for_judge(item.source_sections),
    )
    response = llm.invoke(prompt)
    data = _extract_json(response.content if hasattr(response, "content") else str(response))

    doc_score = float(data.get("doc_match_score", 0.0))
    section_score = float(data.get("section_match_score", 0.0))
    citation_accuracy = (doc_score + section_score) / 2

    return CitationEvalResult(
        citation_accuracy=citation_accuracy,
        doc_precision=doc_score,
        doc_recall=doc_score,
        doc_f1=doc_score,
        section_precision=section_score,
        section_recall=section_score,
        section_f1=section_score,
        method="llm_judge",
        rationales={"citation_accuracy": data.get("rationale", "")},
    )


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def evaluate_citation_accuracy(item, answer: str) -> CitationEvalResult:
    """
    Evaluate whether `answer` correctly attributes its content to the
    gold-standard source(s) (item.doc_ids / item.source_sections).

    Args:
        item: GoldStandardItem.
        answer: The string answer from the system under test.
    """
    if not isinstance(answer, str):
        answer = str(answer)

    if not item.expected_answerable:
        return CitationEvalResult(
            method="skipped",
            rationales={
                "citation_accuracy": (
                    "Item is FA-Refusal (expected_answerable=False); no source to cite."
                )
            },
        )

    if not item.doc_ids and not item.source_sections:
        return CitationEvalResult(
            method="skipped",
            rationales={
                "citation_accuracy": "No ground-truth doc_ids/source_sections for this item."
            },
        )

    parsed = _parse_structured_citations(answer)
    if parsed:
        return _score_deterministic(item, parsed, tier="structured")

    parsed = _parse_freetext_citations(answer)
    if parsed:
        return _score_deterministic(item, parsed, tier="free-text")

    return _score_with_judge(item, answer)
