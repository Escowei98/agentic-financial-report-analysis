"""
RAGAS evaluation wrapper.

Evaluates RAG pipeline outputs using five RAGAS metrics:
  - Context Precision: Are the retrieved contexts relevant?
  - Context Recall: Do retrieved contexts cover the ground truth?
  - Faithfulness: Is the answer grounded in the contexts?
  - Answer Relevancy: Does the answer address the user's question?
  - Answer Correctness: How well does the answer match the ground truth
    (semantic + factual)?

Uses OpenAI (via get_judge_llm) as the LLM judge, deliberately decoupled
from the Gemini models used by the systems under test.
"""

import logging
from dataclasses import dataclass

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.metrics._answer_correctness import answer_correctness
from ragas.metrics._answer_relevance import answer_relevancy
from ragas.metrics._context_precision import context_precision
from ragas.metrics._context_recall import context_recall
from ragas.metrics._faithfulness import faithfulness
from ragas.run_config import RunConfig

from src.common.llm_client import get_embeddings, get_judge_llm
from src.evaluation.gold_standard_loader import GoldStandardItem

logger = logging.getLogger(__name__)

# The five RAGAS metrics (pre-instantiated singletons). Name kept as
# ABLATION_METRICS for backward compatibility with existing consumers.
ABLATION_METRICS = [
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
    answer_correctness,
]

# Metrics that compare the final answer against the question/ground truth
# directly and don't depend on `retrieved_contexts` at all — meaningful for
# every system regardless of architecture. The complementary three
# (context_precision, context_recall, faithfulness) score the
# `retrieved_contexts` field and are therefore only computed for systems
# with an observable retrieval step; see eval_runner.py's
# include_context_metrics decision and
# docs/decisions/EVAL_DECISION_LOG.md [2026-08-16].
ANSWER_METRICS = [answer_relevancy, answer_correctness]


@dataclass
class EvalScores:
    """
    Evaluation scores from a single evaluation run.

    context_precision/context_recall/faithfulness are None when the
    system under test has no observable retrieval step whose output can
    stand in for "the evidence the answer was grounded in" — see
    eval_runner.py. answer_relevancy/answer_correctness are always
    computed since they don't depend on `retrieved_contexts`.
    """

    answer_relevancy: float
    answer_correctness: float
    context_precision: float | None = None
    context_recall: float | None = None
    faithfulness: float | None = None

    @property
    def composite_score(self) -> float:
        """
        Equal-weighted mean across whichever RAGAS metrics were actually
        computed (3 or 5, depending on include_context_metrics).

        Used as a coarse single-number summary; not a thesis-grade
        result on its own, and NOT comparable between a 5-metric and a
        3-metric composite — see the report's RAGAS caveat note.
        Hypotheses are evaluated against the individual metrics plus the
        custom metrics from base.yaml.
        """
        values = [
            v for v in (
                self.context_precision, self.context_recall, self.faithfulness,
                self.answer_relevancy, self.answer_correctness,
            ) if v is not None
        ]
        return sum(values) / len(values) if values else 0.0

    def to_dict(self) -> dict:
        """Return scores as dict. Uncomputed context metrics serialize as null."""
        return {
            "context_precision": round(self.context_precision, 4) if self.context_precision is not None else None,
            "context_recall": round(self.context_recall, 4) if self.context_recall is not None else None,
            "faithfulness": round(self.faithfulness, 4) if self.faithfulness is not None else None,
            "answer_relevancy": round(self.answer_relevancy, 4),
            "answer_correctness": round(self.answer_correctness, 4),
            "composite_score": round(self.composite_score, 4),
        }


def _to_str(value) -> str:
    """Coerce a value to a plain string (handles Gemini Content parts)."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        # Gemini returns [{'text': '...', 'type': 'text'}]
        parts = [p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in value]
        return "\n".join(parts)
    return str(value)


def evaluate_run(
    gold_standard: list[GoldStandardItem],
    answers: list[str],
    contexts: list[list[str]],
    include_context_metrics: bool = True,
) -> EvalScores:
    """
    Evaluate a RAG run using RAGAS metrics.

    Args:
        gold_standard: Gold-standard Q&A items.
        answers: Generated answers (one per gold_standard item).
        contexts: Retrieved contexts per query (list of string lists).
        include_context_metrics: When False, skips context_precision/
            context_recall/faithfulness entirely (not computed, not just
            hidden) and only requests answer_relevancy/answer_correctness
            from RAGAS. Use for systems without an observable retrieval
            step — see eval_runner.py.

    Returns:
        EvalScores with answer_relevancy, answer_correctness always set;
        context_precision, context_recall, faithfulness set only when
        include_context_metrics=True.
    """
    if len(gold_standard) != len(answers) or len(gold_standard) != len(contexts):
        raise ValueError(
            f"Length mismatch: {len(gold_standard)} items, "
            f"{len(answers)} answers, {len(contexts)} contexts"
        )

    # Build RAGAS v2 EvaluationDataset directly (bypasses buggy HF Dataset conversion)
    samples = [
        SingleTurnSample(
            user_input=item.question,
            response=_to_str(answer),
            retrieved_contexts=[_to_str(c) for c in ctx],
            reference=item.ground_truth,
        )
        for item, answer, ctx in zip(gold_standard, answers, contexts)
    ]
    dataset = EvaluationDataset(samples=samples)

    logger.info("Running RAGAS evaluation on %d samples...", len(dataset))

    # Judge LLM is deliberately a different provider than the systems
    # under test (see src.common.llm_client.get_judge_llm)
    llm = get_judge_llm(max_tokens=8192)
    embeddings = get_embeddings()

    # Increase timeout and lower concurrency to prevent Vertex AI Timeouts
    run_config = RunConfig(timeout=300, max_workers=4)

    # Run RAGAS evaluation — only request the context-dependent metrics
    # when they're actually meaningful for this system (see docstring).
    metrics = ABLATION_METRICS if include_context_metrics else ANSWER_METRICS
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )

    def _mean(values: list) -> float:
        """Average a list of per-sample metric scores, ignoring None/NaN."""
        valid = [v for v in values if v is not None and v == v]  # v != v filters NaN
        return sum(valid) / len(valid) if valid else 0.0

    scores = EvalScores(
        answer_relevancy=_mean(result["answer_relevancy"]),
        answer_correctness=_mean(result["answer_correctness"]),
        context_precision=_mean(result["context_precision"]) if include_context_metrics else None,
        context_recall=_mean(result["context_recall"]) if include_context_metrics else None,
        faithfulness=_mean(result["faithfulness"]) if include_context_metrics else None,
    )

    if include_context_metrics:
        logger.info(
            "RAGAS scores: precision=%.3f, recall=%.3f, faithfulness=%.3f, "
            "ans_relevancy=%.3f, ans_correctness=%.3f → composite=%.3f",
            scores.context_precision, scores.context_recall, scores.faithfulness,
            scores.answer_relevancy, scores.answer_correctness, scores.composite_score,
        )
    else:
        logger.info(
            "RAGAS scores (context metrics skipped — no observable retrieval "
            "step): ans_relevancy=%.3f, ans_correctness=%.3f → composite=%.3f",
            scores.answer_relevancy, scores.answer_correctness, scores.composite_score,
        )

    return scores

