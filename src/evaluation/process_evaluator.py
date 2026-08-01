"""
Process Evaluator for the Financial RAG/Agent Pipeline.

Implements agentic and process metrics:
1. Tool Selection Accuracy (Precision, Recall, F1)
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
class ProcessEvalResult:
    tool_precision: float = 0.0
    tool_recall: float = 0.0
    tool_f1: float = 0.0

    def to_dict(self):
        return {
            "tool_precision": round(self.tool_precision, 2),
            "tool_recall": round(self.tool_recall, 2),
            "tool_f1": round(self.tool_f1, 2),
        }

# ---------------------------------------------------------------------------
#  Evaluation Logic
# ---------------------------------------------------------------------------

def evaluate_tool_selection(expected_tools: list[str], actual_tools: list[str]) -> ProcessEvalResult:
    """
    Evaluate tool selection accuracy using set-based Precision, Recall, and F1.
    
    Args:
        expected_tools: List of tool names expected by the gold standard (e.g., ["search_section", "calculate"]).
        actual_tools: List of tool names actually called by the agent (can include duplicates).
    """
    if not expected_tools:
        # If no tools were expected, and none were called, perfect score.
        # If tools were called when none were expected, 0 score.
        if not actual_tools:
            return ProcessEvalResult(tool_precision=1.0, tool_recall=1.0, tool_f1=1.0)
        else:
            return ProcessEvalResult(tool_precision=0.0, tool_recall=0.0, tool_f1=0.0)

    # Use sets to ignore duplicate calls to the same tool
    expected_set = set(expected_tools)
    actual_set = set(actual_tools)

    true_positives = len(expected_set.intersection(actual_set))
    false_positives = len(actual_set - expected_set)
    false_negatives = len(expected_set - actual_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return ProcessEvalResult(
        tool_precision=precision,
        tool_recall=recall,
        tool_f1=f1
    )

def calculate_cost_per_correct_answer(
    total_cost_usd: float,
    correctness_scores: Sequence[float],
    threshold: float = 0.8
) -> float:
    """
    Calculate the average cost to produce a 'correct' answer.
    
    Args:
        total_cost_usd: Total API cost for the run.
        correctness_scores: List of correctness scores (e.g., exact_match or answer_correctness) for each query.
        threshold: Score threshold above which an answer is considered 'correct' (default 0.8).
        
    Returns:
        Cost per correct answer in USD. Returns 0.0 if no answers were correct.
    """
    num_correct = sum(1 for score in correctness_scores if score >= threshold)
    if num_correct == 0:
        return 0.0
    return total_cost_usd / num_correct
