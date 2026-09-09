"""
Process Evaluator for the Financial RAG/Agent Pipeline.

Implements:
1. Generic set-overlap scoring (Precision, Recall, F1) — used by
   citation_evaluator.py for document-id citation scoring.
2. Cost per Correct Answer (Aggregated across a run)
"""

import logging
from dataclasses import dataclass
from typing import Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SetOverlapResult:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    def to_dict(self):
        return {
            "precision": round(self.precision, 2),
            "recall": round(self.recall, 2),
            "f1": round(self.f1, 2),
        }

# ---------------------------------------------------------------------------
#  Evaluation Logic
# ---------------------------------------------------------------------------

def evaluate_set_overlap(expected: list[str], actual: list[str]) -> SetOverlapResult:
    """
    Compare two lists of string identifiers using set-based Precision, Recall, and F1.

    Args:
        expected: List of expected identifiers (e.g., document ids from the gold standard).
        actual: List of actually observed identifiers (can include duplicates).
    """
    if not expected:
        # If nothing was expected, and nothing was observed, perfect score.
        # If something was observed when nothing was expected, 0 score.
        if not actual:
            return SetOverlapResult(precision=1.0, recall=1.0, f1=1.0)
        else:
            return SetOverlapResult(precision=0.0, recall=0.0, f1=0.0)

    # Use sets to ignore duplicates
    expected_set = set(expected)
    actual_set = set(actual)

    true_positives = len(expected_set.intersection(actual_set))
    false_positives = len(actual_set - expected_set)
    false_negatives = len(expected_set - actual_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return SetOverlapResult(
        precision=precision,
        recall=recall,
        f1=f1
    )

def calculate_cost_per_correct_answer(
    total_cost_usd: float,
    correctness_scores: Sequence[float],
    threshold: float = 0.8
) -> float | None:
    """
    Calculate the average cost to produce a 'correct' answer.

    Args:
        total_cost_usd: Total API cost for the run.
        correctness_scores: List of correctness scores (e.g., exact_match or answer_correctness) for each query.
        threshold: Score threshold above which an answer is considered 'correct' (default 0.8).

    Returns:
        Cost per correct answer in USD, or ``None`` when no answer cleared the
        threshold. ``None`` rather than ``0.0``: a system that got nothing
        right has an undefined cost per correct answer, and 0.0 sorts as the
        best value in the comparison table — the one place the number is
        actually read. Callers must render ``None`` as "not defined", never as
        a zero. See EVAL_DECISION_LOG.md [2026-09-06].
    """
    num_correct = sum(1 for score in correctness_scores if score >= threshold)
    if num_correct == 0:
        return None
    return total_cost_usd / num_correct
