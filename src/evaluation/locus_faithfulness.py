"""
Faithfulness of the final answer against the corpus at the loci it cites.

WHY A SECOND FAITHFULNESS
-------------------------
RAGAS `faithfulness` scores an answer against `retrieved_contexts`, which
only exists for the two RAG systems: S3 inlines the whole corpus into its
prompt and S4 hands its synthesizer specialist paraphrases, so neither has a
context artefact that could stand in for "the evidence the answer was
grounded in" (see SYSTEMS_WITHOUT_MEANINGFUL_CONTEXT_METRICS in
eval_runner.py). That left NF-1, Faktentreue, measurable for half the
comparison.

This module anchors the check on the citations instead. Every system emits
`(TICKER, FYYEAR, SECTION)` loci in its answer under the shared Answer Format
Convention; the passages at those loci are fetched from the corpus by the
same `EvidenceStore` the groundedness dimension uses -- same pinned
chunking, same section widening, same lexical ranking -- and become the
`retrieved_contexts` RAGAS Faithfulness is run over. Every system is thereby
verified against passages produced by one procedure from one corpus,
regardless of how it found them.

WHAT IT MEASURES AND WHAT IT DOES NOT
-------------------------------------
The score is the share of the answer's statements supported by the corpus
AT THE PLACES THE ANSWER SAYS THEY COME FROM. A correct figure attributed to
the wrong filing scores as unsupported; that coupling to citation behaviour
is deliberate and is the price of a procedure that treats S3 identically to
S1. Whether the system actually read those passages is not measured and,
for S3/S4, not observable.

It is a different construct from RAGAS `faithfulness` on S1/S2 (answer vs.
what retrieval handed the generator) and the two must never be pooled; the
distinct key `locus_faithfulness` exists so they cannot be confused in any
file. It differs from `groundedness` in level (whole answer, not one [E]
step) and in not depending on a chain being emitted at all, and from
`citation_accuracy` in scoring the passage behind the citation rather than
the citation itself.

POPULATION AND EXCLUSIONS
-------------------------
Only answerable items are scored: a correct refusal carries no claim to
verify. An answerable item whose answer cites no locus stays None with
`excluded="no_locus"` -- not 0.0. Citation discipline is `citation_accuracy`'s
construct, and folding it in here would double-count it; almost all such
answers are over-refusals ("the context does not contain ...") that carry no
factual claim either. Coverage is reported per system so the gap is visible.

STATUS
------
Not pre-registered: computed on the stored answers of the final runs
(scripts/rescore_locus_faithfulness.py, judge calls only) and reported
exploratively, never as a hypothesis endpoint.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Sequence, cast

from ragas import evaluate
from ragas.dataset_schema import (
    EvaluationDataset,
    EvaluationResult,
    MultiTurnSample,
    SingleTurnSample,
)
from ragas.metrics._faithfulness import Faithfulness
from ragas.run_config import RunConfig

from src.common.llm_client import get_judge_llm
from src.evaluation.evidence_store import MAX_PASSAGES_PER_LOCUS, EvidenceStore
from src.evaluation.reasoning_chain_parser import split_answer

logger = logging.getLogger(__name__)

# The citation evaluator's regex accepts round brackets only, which is what
# the convention prescribes. S1 nevertheless writes `[AAPL, FY2024, ...]` on
# a few answers; a locus in square brackets is still a locus, and dropping
# it would exclude a checkable answer for a punctuation choice.
_LOCUS_RE = re.compile(
    r"[\(\[]\s*\{?(?P<ticker>AAPL|MSFT|AMZN|GOOGL)\}?\s*,\s*"
    r"\{?(?:FY\s?)?(?P<year>20\d{2})\}?\s*,\s*(?P<section>[^\)\]]+?)\s*[\)\]]",
    re.IGNORECASE,
)

EXCLUDED_NOT_ANSWERABLE = "not_answerable"
EXCLUDED_NO_LOCUS = "no_locus"
EXCLUDED_NO_PASSAGES = "no_passages"
EXCLUDED_JUDGE_FAILED = "judge_failed"


@dataclass
class CitedLocus:
    doc_id: str
    section_text: str

    def to_dict(self) -> dict:
        return {"doc_id": self.doc_id, "section_text": self.section_text}


@dataclass
class LocusFaithfulnessResult:
    """Per-item outcome; `score` is None whenever `excluded` is set."""

    score: float | None = None
    excluded: str | None = None
    loci: list[dict] = field(default_factory=list)
    n_passages: int = 0

    def to_dict(self) -> dict:
        return {
            "score": None if self.score is None else round(self.score, 4),
            "excluded": self.excluded,
            "n_loci": len(self.loci),
            "n_passages": self.n_passages,
            "loci": self.loci,
        }


def parse_loci(answer: str) -> list[CitedLocus]:
    """Distinct loci cited in an answer, in order of first appearance."""
    loci: list[CitedLocus] = []
    seen: set[tuple[str, str]] = set()
    for m in _LOCUS_RE.finditer(answer or ""):
        doc_id = f"{m.group('ticker').upper()}_{m.group('year')}"
        section_text = m.group("section").strip()
        key = (doc_id, section_text.lower())
        if key in seen:
            continue
        seen.add(key)
        loci.append(CitedLocus(doc_id=doc_id, section_text=section_text))
    return loci


def gather_passages(
    answer: str,
    loci: Sequence[CitedLocus],
    evidence_store: EvidenceStore,
    max_passages_per_locus: int = MAX_PASSAGES_PER_LOCUS,
) -> tuple[list[str], list[dict]]:
    """Union of the passages at every cited locus, ranked against the answer.

    Returns the deduplicated passages and one record per locus saying what
    the lookup found, so a low score can be traced to a fabricated filing
    (`doc_exists=False`) rather than to an unsupported claim.
    """
    passages: list[str] = []
    seen: set[str] = set()
    records: list[dict] = []
    for locus in loci:
        result = evidence_store.lookup(
            locus.doc_id, locus.section_text, answer,
            max_passages=max_passages_per_locus,
        )
        added = 0
        for passage in result.passages:
            if passage in seen:
                continue
            seen.add(passage)
            passages.append(passage)
            added += 1
        records.append({
            **locus.to_dict(),
            "doc_exists": result.doc_exists,
            "section_resolved": result.section_resolved,
            "n_passages": added,
        })
    return passages, records


def _judge_faithfulness(
    questions: Sequence[str], answers: Sequence[str], contexts: Sequence[list[str]],
) -> list[float | None]:
    """One RAGAS Faithfulness pass; None where the judge returned NaN."""
    samples: list[SingleTurnSample | MultiTurnSample] = [
        SingleTurnSample(user_input=q, response=a, retrieved_contexts=c)
        for q, a, c in zip(questions, answers, contexts)
    ]
    dataset = EvaluationDataset(samples=samples)
    # Fresh metric instance per call -- see ABLATION_METRIC_CLASSES in
    # ragas_evaluator.py for why instances are never shared.
    result = cast(EvaluationResult, evaluate(
        dataset=dataset,
        metrics=[Faithfulness()],
        llm=get_judge_llm(max_tokens=8192),
        run_config=RunConfig(timeout=300, max_workers=4),
    ))
    return [
        float(v) if v is not None and v == v else None
        for v in result["faithfulness"]
    ]


def evaluate_locus_faithfulness_batch(
    questions: Sequence[str],
    answers: Sequence[str],
    answerable: Sequence[bool],
    evidence_store: EvidenceStore,
) -> list[LocusFaithfulnessResult]:
    """Score every item; the returned list is aligned with the inputs.

    `answers` may still carry the reasoning block -- it is split off here,
    since the construct is the final answer's statements, not the chain's.
    """
    if not (len(questions) == len(answers) == len(answerable)):
        raise ValueError("questions, answers and answerable must be aligned")

    results = [LocusFaithfulnessResult() for _ in answers]
    to_judge: list[int] = []
    judge_answers: list[str] = []
    judge_contexts: list[list[str]] = []

    for i, (answer, is_answerable) in enumerate(zip(answers, answerable)):
        if not is_answerable:
            results[i].excluded = EXCLUDED_NOT_ANSWERABLE
            continue
        clean, _chain = split_answer(answer or "")
        loci = parse_loci(clean)
        if not loci:
            results[i].excluded = EXCLUDED_NO_LOCUS
            continue
        passages, records = gather_passages(clean, loci, evidence_store)
        results[i].loci = records
        results[i].n_passages = len(passages)
        if not passages:
            # Every cited filing is outside the corpus: nothing to check
            # against. citation_accuracy already scores the fabricated id.
            results[i].excluded = EXCLUDED_NO_PASSAGES
            continue
        to_judge.append(i)
        judge_answers.append(clean)
        judge_contexts.append(passages)

    if to_judge:
        logger.info(
            "locus_faithfulness: judging %d of %d items (%d answerable)",
            len(to_judge), len(answers), sum(1 for a in answerable if a),
        )
        scores = _judge_faithfulness(
            [questions[i] for i in to_judge], judge_answers, judge_contexts,
        )
        for i, score in zip(to_judge, scores):
            if score is None:
                results[i].excluded = EXCLUDED_JUDGE_FAILED
            else:
                results[i].score = score

    return results


def summarise(results: Sequence[dict]) -> dict:
    """Run-level summary over the per-item dicts (`to_dict()` output)."""
    scored = [r["score"] for r in results if r.get("score") is not None]
    answerable = [r for r in results if r.get("excluded") != EXCLUDED_NOT_ANSWERABLE]
    reasons: dict[str, int] = {}
    for r in answerable:
        if r.get("excluded"):
            reasons[r["excluded"]] = reasons.get(r["excluded"], 0) + 1
    return {
        "mean": round(sum(scored) / len(scored), 4) if scored else None,
        "n_scored": len(scored),
        "n_answerable": len(answerable),
        "coverage": (
            round(len(scored) / len(answerable), 4) if answerable else None
        ),
        "excluded": reasons,
    }
