"""
RAGAS evaluation wrapper.

Evaluates RAG pipeline outputs using five RAGAS metrics:
  - Context Precision: Are the retrieved contexts relevant?
  - Context Recall: Do retrieved contexts cover the ground truth?
  - Faithfulness: Is the answer grounded in the contexts?
  - Answer Relevancy: Does the answer address the user's question?
  - Answer Correctness: How well does the answer match the ground truth
    (semantic + factual)?

Uses Gemini as the LLM judge via langchain-google-genai.
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

from src.common.llm_client import get_embeddings, get_llm
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


@dataclass
class EvalScores:
    """Evaluation scores from a single evaluation run."""

    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float
    answer_correctness: float

    @property
    def composite_score(self) -> float:
        """
        Equal-weighted mean across all five RAGAS metrics.

        Used as a coarse single-number summary; not a thesis-grade
        result on its own. Hypotheses are evaluated against the
        individual metrics plus the custom metrics from base.yaml.
        """
        return (
            self.context_precision
            + self.context_recall
            + self.faithfulness
            + self.answer_relevancy
            + self.answer_correctness
        ) / 5.0

    def to_dict(self) -> dict:
        """Return scores as dict."""
        return {
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "faithfulness": round(self.faithfulness, 4),
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
) -> EvalScores:
    """
    Evaluate a RAG run using RAGAS metrics.

    Args:
        gold_standard: Gold-standard Q&A items.
        answers: Generated answers (one per gold_standard item).
        contexts: Retrieved contexts per query (list of string lists).

    Returns:
        EvalScores with context_precision, context_recall, faithfulness,
        answer_relevancy, answer_correctness.
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

    # Use cheaper Gemini 2.0 Flash as LLM judge (saves ~80% API costs vs 3.0 Preview)
    llm = get_llm(
        model_name="gemini-2.0-flash",
        max_output_tokens=8192
    )
    embeddings = get_embeddings()

    # Increase timeout and lower concurrency to prevent Vertex AI Timeouts
    run_config = RunConfig(timeout=300, max_workers=4)

    # Run RAGAS evaluation
    result = evaluate(
        dataset=dataset,
        metrics=ABLATION_METRICS,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    )

    def _mean(values: list) -> float:
        """Average a list of per-sample metric scores, ignoring None/NaN."""
        valid = [v for v in values if v is not None and v == v]  # v != v filters NaN
        return sum(valid) / len(valid) if valid else 0.0

    scores = EvalScores(
        context_precision=_mean(result["context_precision"]),
        context_recall=_mean(result["context_recall"]),
        faithfulness=_mean(result["faithfulness"]),
        answer_relevancy=_mean(result["answer_relevancy"]),
        answer_correctness=_mean(result["answer_correctness"]),
    )

    logger.info(
        "RAGAS scores: precision=%.3f, recall=%.3f, faithfulness=%.3f, "
        "ans_relevancy=%.3f, ans_correctness=%.3f → composite=%.3f",
        scores.context_precision, scores.context_recall, scores.faithfulness,
        scores.answer_relevancy, scores.answer_correctness, scores.composite_score,
    )

    return scores

