"""
Corpus lookup for Dimension 1 (Belegtheit).

Given a locus a reasoning step cites -- (TICKER, FYYEAR, SECTION) -- this
returns the passages at that place in the corpus, so the judge can check
whether they carry the step's claim.

WHY A LOOKUP AND NOT THE SYSTEM'S CONTEXT
-----------------------------------------
The obvious design is to show the judge the context the system actually had.
It cannot be done symmetrically here. S3 and S4 hold ~609k tokens of corpus,
which is not promptable; S1 and S2 hold a handful of retrieved chunks, which
is. Feeding the judge whatever each architecture happens to expose is exactly
what produced the differential bias the 2026-09-08 validation measured --
`evidence_faithfulness` ran 2.75 points below the human raters for
`long_context` and 0.42 for `rag_monolith`. `_WITHHELD_NOTE` in the retired
reasoning_evaluator.py withheld retrieved text for that reason, which kept the
comparison fair at the price of leaving the judge nothing to check against.

Anchoring on the cited locus resolves both problems at once: every system is
verified against passages fetched by the same procedure, from the same corpus,
in the same quantity, regardless of how it found them.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
No vector search, no reranking, no embedding. Retrieval quality is a separate
construct with its own metric (`citation_evaluator.py`); if evidence for the
groundedness check were itself retrieved, a retrieval miss would show up as a
reasoning defect. Passage selection inside a section is lexical and
deterministic (`_score_passage`), so the same step against the same corpus
always yields the same evidence.

See docs/decisions/REASONING_QUALITY_SPEC.md section 6.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from langchain_core.documents import Document

from src.common.ingestion import ProcessedFiling
from src.common.retrieval import load_or_build_documents

logger = logging.getLogger(__name__)

# Chunking for the evidence store is PINNED here, not read from any system's
# config. S1 and S2 run different chunk sizes; verifying their steps against
# differently-cut passages would make groundedness partly a function of the
# system's own retrieval configuration -- the very coupling this module
# exists to remove.
EVIDENCE_CHUNK_SIZE = 1000
EVIDENCE_OVERLAP_PCT = 0.20
EVIDENCE_CACHE_DIR = Path("data/vectorstores/evidence_store")

# How many passages one locus contributes to the judge prompt.
MAX_PASSAGES_PER_LOCUS = 6

# Cited section text -> section_name as it appears in chunk metadata (see
# SECTION_PATTERNS in src/common/ingestion.py). Broader than the gold
# standard's vocabulary on purpose: the corpus holds "Risk Factors", which no
# gold-standard source_section names, and a step citing it must still be
# checkable.
_SECTION_ALIASES: dict[str, str] = {
    "business": "Business",
    "item 1": "Business",
    "risk factor": "Risk Factors",
    "item 1a": "Risk Factors",
    "md&a": "MD&A",
    "mda": "MD&A",
    "managements discussion": "MD&A",
    "management discussion": "MD&A",
    "item 7": "MD&A",
    "income statement": "Financial Statements",
    "statement of income": "Financial Statements",
    "statements of operations": "Financial Statements",
    "balance sheet": "Financial Statements",
    "cash flow": "Financial Statements",
    "segment": "Financial Statements",
    "financial statement": "Financial Statements",
    "item 8": "Financial Statements",
    "director": "Directors and Corporate Governance",
    "item 10": "Directors and Corporate Governance",
}

_NUM_RE = re.compile(r"\d[\d,.]*")
_WORD_RE = re.compile(r"[a-z][a-z&]{2,}")

# Words too common in 10-K prose to discriminate between passages.
_STOPWORDS = frozenset({
    "the", "and", "for", "was", "were", "with", "that", "this", "from", "its",
    "has", "had", "are", "not", "but", "all", "any", "have", "been", "which",
    "their", "these", "those", "than", "then", "also", "such", "into", "per",
    "total", "year", "fiscal", "company", "million", "billion", "reported",
})


@dataclass
class LookupResult:
    """Passages found at one cited locus."""

    passages: list[str] = field(default_factory=list)
    doc_exists: bool = True
    section_resolved: bool = True

    @property
    def found(self) -> bool:
        return bool(self.passages)


def _normalise_section(text: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9&\s]", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for alias, section_name in _SECTION_ALIASES.items():
        if alias in normalized:
            return section_name
    return None


def _tokenise(text: str) -> tuple[set[str], set[str]]:
    """Return (numeric tokens, content words) of a claim or passage."""
    numbers = {n.strip(".,").replace(",", "") for n in _NUM_RE.findall(text)}
    numbers = {n for n in numbers if n}
    words = {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}
    return numbers, words


def _score_passage(claim_numbers: set[str], claim_words: set[str], passage: str) -> float:
    """Lexical overlap score, numbers weighted far above words.

    In this corpus a factual step almost always turns on a figure, and the
    figure is what makes one passage in a 40k-character section the right one.
    Word overlap alone would rank on boilerplate.
    """
    numbers, words = _tokenise(passage)
    return 5.0 * len(claim_numbers & numbers) + float(len(claim_words & words))


class EvidenceStore:
    """Metadata-filtered access to corpus passages by cited locus."""

    def __init__(
        self,
        filings: Sequence[ProcessedFiling],
        cache_dir: Path | str = EVIDENCE_CACHE_DIR,
    ) -> None:
        documents = load_or_build_documents(
            filings,
            persist_directory=str(cache_dir),
            chunk_size=EVIDENCE_CHUNK_SIZE,
            overlap_pct=EVIDENCE_OVERLAP_PCT,
        )
        self._by_doc: dict[str, list[Document]] = {}
        for doc in documents:
            meta = doc.metadata or {}
            doc_id = f"{meta.get('ticker', '?')}_{meta.get('fiscal_year', '?')}"
            self._by_doc.setdefault(doc_id, []).append(doc)
        logger.info(
            "Evidence store ready: %d passages across %d filings",
            len(documents), len(self._by_doc),
        )

    @property
    def doc_ids(self) -> set[str]:
        return set(self._by_doc)

    def lookup(
        self,
        doc_id: str,
        section_text: str,
        claim_text: str,
        max_passages: int = MAX_PASSAGES_PER_LOCUS,
    ) -> LookupResult:
        """Fetch the passages at one locus, ranked against the claim.

        A locus whose filing is not in the corpus returns `doc_exists=False`
        -- that is a B6 (fabricated locus) and needs no judge call.

        A locus whose section name does not resolve widens to the whole
        filing rather than failing. Section-name drift ("Income Statement"
        vs "Financial Statements") is a citation-vocabulary matter that
        `citation_accuracy` already measures; scoring it as a reasoning
        defect here would count the same error twice and punish the systems
        that cite more specifically.
        """
        candidates = self._by_doc.get(doc_id)
        if not candidates:
            return LookupResult(doc_exists=False)

        section_name = _normalise_section(section_text)
        section_resolved = section_name is not None
        if section_resolved:
            scoped = [
                d for d in candidates
                if (d.metadata or {}).get("section_name") == section_name
            ]
            if not scoped:
                # The filing is in the corpus but that section was never
                # parsed out of it. Widening keeps the step checkable.
                scoped = candidates
                section_resolved = False
        else:
            scoped = candidates

        claim_numbers, claim_words = _tokenise(claim_text)
        ranked = sorted(
            scoped,
            key=lambda d: _score_passage(claim_numbers, claim_words, d.page_content),
            reverse=True,
        )
        return LookupResult(
            passages=[d.page_content for d in ranked[:max_passages]],
            doc_exists=True,
            section_resolved=section_resolved,
        )
