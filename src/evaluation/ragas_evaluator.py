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
from dataclasses import dataclass, field
from typing import cast

from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, EvaluationResult, MultiTurnSample, SingleTurnSample
from ragas.metrics._answer_correctness import AnswerCorrectness
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import ContextRecall
from ragas.metrics._faithfulness import Faithfulness
from ragas.metrics.base import Metric
from ragas.run_config import RunConfig

from src.common.llm_client import get_embeddings, get_judge_llm
from src.evaluation.gold_standard_loader import GoldStandardItem

logger = logging.getLogger(__name__)

# The five RAGAS metrics, as CLASSES. Instances are created per evaluate_run
# call by `build_metrics` below -- never shared.
#
# Until 2026-09-09 this module used ragas' pre-instantiated module-level
# singletons (`ragas.metrics._faithfulness.faithfulness` etc.). ragas'
# `evaluate()` binds llm/embeddings onto whatever metric objects it is handed,
# calls `metric.init()` (which, for AnswerCorrectness, creates the nested
# AnswerSimilarity), and RESETS those attributes when it returns, so the
# caller's objects are left as it found them. With the four systems evaluated
# concurrently, the first `evaluate()` to finish reset the shared
# AnswerCorrectness while the others were still scoring with it:
# `AssertionError: AnswerSimilarity must be set`, the item's
# answer_correctness becomes NaN, and the item drops out of a metric mapped
# to H1-H3 -- for whichever system happened to lose the race. Seen twice in
# the 2026-09-07 smoke test and twice in the 2026-09-09 one, never in the
# sequential n=150 run. See EVAL_DECISION_LOG.md [2026-09-09].
ABLATION_METRIC_CLASSES: tuple[type[Metric], ...] = (
    ContextPrecision,
    ContextRecall,
    Faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
)

# Metrics that compare the final answer against the question/ground truth
# directly and don't depend on `retrieved_contexts` at all — meaningful for
# every system regardless of architecture. The complementary three
# (context_precision, context_recall, faithfulness) score the
# `retrieved_contexts` field and are therefore only computed for systems
# with an observable retrieval step; see eval_runner.py's
# include_context_metrics decision and
# docs/decisions/EVAL_DECISION_LOG.md [2026-08-16].
ANSWER_METRIC_CLASSES: tuple[type[Metric], ...] = (AnswerRelevancy, AnswerCorrectness)


def build_metrics(include_context_metrics: bool = True) -> list[Metric]:
    """Fresh metric instances for one `evaluate()` call.

    One list per call, never cached: sharing instances across concurrent
    calls is the race described above.
    """
    classes = ABLATION_METRIC_CLASSES if include_context_metrics else ANSWER_METRIC_CLASSES
    return [cls() for cls in classes]


@dataclass
class EvalScores:
    """
    Evaluation scores from a single evaluation run.

    context_precision/context_recall/faithfulness are None when the
    system under test has no observable retrieval step whose output can
    stand in for "the evidence the answer was grounded in" — see
    eval_runner.py. answer_relevancy/answer_correctness are always
    computed since they don't depend on `retrieved_contexts`.

    The aggregate answer_relevancy/answer_correctness are averaged over
    the *answerable* items only (see evaluate_run); `per_sample` keeps
    every individual score, including those of the refusal items, so
    downstream code can stratify and run significance tests.
    """

    answer_relevancy: float
    answer_correctness: float
    context_precision: float | None = None
    context_recall: float | None = None
    faithfulness: float | None = None

    per_sample: dict[str, list[float | None]] = field(default_factory=dict)
    """Per-sample metric scores, keyed by metric name.

    Each list is aligned with the `gold_standard`/`answers`/`contexts`
    inputs of evaluate_run, one entry per sample; NaN and missing scores
    are normalised to None. Only the metrics actually requested are
    present (two keys when include_context_metrics=False, five otherwise).
    Empty when EvalScores is constructed by hand.
    """

    answer_metrics_n: int | None = None
    """How many samples the aggregate answer metrics were averaged over.

    None when EvalScores is constructed by hand rather than by
    evaluate_run.
    """

    @property
    def composite_score(self) -> float:
        """
        Equal-weighted mean across whichever RAGAS metrics were actually
        computed (2 or 5, depending on include_context_metrics).

        Used as a coarse single-number summary; not a thesis-grade
        result on its own, and NOT comparable between a 5-metric and a
        2-metric composite — the latter is just the mean of the two
        answer metrics, since the three context metrics are the ones
        that drop out. See the report's RAGAS caveat note. Hypotheses
        are evaluated against the individual metrics plus the custom
        metrics from base.yaml.
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
            "answer_metrics_n": self.answer_metrics_n,
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
        include_context_metrics=True. `per_sample` carries the individual
        score of every sample for every requested metric.

    Note on the answer-metric aggregate: answer_relevancy and
    answer_correctness are averaged over the items with
    expected_answerable=True only. RAGAS scores a (correct) refusal as
    "noncommittal", i.e. answer_relevancy 0, and the refusal items carry
    an empty ground truth, so answer_correctness has nothing to match
    against either — including them would penalise exactly the behaviour
    the refusal stratum is there to reward. This mirrors the convention
    eval_runner.py already uses for citation_accuracy and the custom
    metrics. The per-sample scores of the refusal items are kept in
    `per_sample` unchanged.
    """
    if len(gold_standard) != len(answers) or len(gold_standard) != len(contexts):
        raise ValueError(
            f"Length mismatch: {len(gold_standard)} items, "
            f"{len(answers)} answers, {len(contexts)} contexts"
        )

    # Build RAGAS v2 EvaluationDataset directly (bypasses buggy HF Dataset conversion)
    samples: list[SingleTurnSample | MultiTurnSample] = [
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
    # Fresh instances: see the note on ABLATION_METRIC_CLASSES.
    metrics = build_metrics(include_context_metrics)
    # `evaluate()` is typed to return `EvaluationResult | Executor`, but it
    # only returns an `Executor` when called with `return_executor=True`
    # (not done here) — this call always gets an `EvaluationResult` back.
    result = cast(EvaluationResult, evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
    ))

    def _mean(values: list) -> float:
        """Average a list of per-sample metric scores, ignoring None/NaN."""
        valid = [v for v in values if v is not None and v == v]  # v != v filters NaN
        return sum(valid) / len(valid) if valid else 0.0

    def _clean(values: list) -> list[float | None]:
        """Per-sample scores with NaN and missing values normalised to None."""
        return [float(v) if v is not None and v == v else None for v in values]

    requested = ["answer_relevancy", "answer_correctness"]
    if include_context_metrics:
        requested = ["context_precision", "context_recall", "faithfulness"] + requested
    per_sample: dict[str, list[float | None]] = {m: _clean(result[m]) for m in requested}

    # Answer metrics are averaged over the answerable items only — see the
    # docstring. The context metrics keep averaging over everything: they
    # are only computed for S1/S2 at all, so narrowing them would change
    # what the two systems' numbers mean relative to S3/S4's.
    answerable = [i for i, item in enumerate(gold_standard) if item.expected_answerable]
    if not answerable:
        logger.warning(
            "No answerable items in this run — averaging the answer metrics "
            "over all %d samples instead.", len(gold_standard),
        )
        answerable = list(range(len(gold_standard)))

    def _answerable_mean(metric: str) -> float:
        return _mean([per_sample[metric][i] for i in answerable])

    scores = EvalScores(
        answer_relevancy=_answerable_mean("answer_relevancy"),
        answer_correctness=_answerable_mean("answer_correctness"),
        context_precision=_mean(per_sample["context_precision"]) if include_context_metrics else None,
        context_recall=_mean(per_sample["context_recall"]) if include_context_metrics else None,
        faithfulness=_mean(per_sample["faithfulness"]) if include_context_metrics else None,
        per_sample=per_sample,
        answer_metrics_n=len(answerable),
    )

    if include_context_metrics:
        logger.info(
            "RAGAS scores: precision=%.3f, recall=%.3f, faithfulness=%.3f, "
            "ans_relevancy=%.3f, ans_correctness=%.3f (n=%d answerable of %d) "
            "→ composite=%.3f",
            scores.context_precision, scores.context_recall, scores.faithfulness,
            scores.answer_relevancy, scores.answer_correctness,
            scores.answer_metrics_n, len(gold_standard), scores.composite_score,
        )
    else:
        logger.info(
            "RAGAS scores (context metrics skipped — no observable retrieval "
            "step): ans_relevancy=%.3f, ans_correctness=%.3f "
            "(n=%d answerable of %d) → composite=%.3f",
            scores.answer_relevancy, scores.answer_correctness,
            scores.answer_metrics_n, len(gold_standard), scores.composite_score,
        )

    return scores

