"""
Evaluation Runner Orchestrator.

Orchestrates the complete evaluation pipeline for a given system:
1. Runs the system on a batch of gold standard queries.
2. Computes RAGAS metrics.
3. Computes Custom metrics (Exact Match, Answer Recall, Refusal).
4. Computes Reasoning metrics (Core & Agentic).
5. Computes efficiency/process outputs (Cost per Correct Answer, Correction Rate).
6. Compiles a consolidated result JSON.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.evaluation.citation_evaluator import evaluate_citation_accuracy
from src.evaluation.custom_evaluator import evaluate_custom_metrics
from src.evaluation.gold_standard_loader import GoldStandardItem
from src.evaluation.process_evaluator import calculate_cost_per_correct_answer
from src.evaluation.ragas_evaluator import evaluate_run
from src.evaluation.reasoning_evaluator import evaluate_reasoning_batch, format_trajectory

logger = logging.getLogger(__name__)

def run_full_evaluation(
    pipeline: Any,
    system_name: str,
    gold_items: list[GoldStandardItem],
    output_dir: str | Path,
) -> dict:
    """
    Run the full evaluation suite on a given system.
    
    Args:
        pipeline: An instantiated system pipeline (e.g., RAGMonolithPipeline, AgentRAGPipeline).
                  Must have a `.query(question: str)` method.
        system_name: Identifier string (e.g., 'rag_monolith', 'rag_agent').
        gold_items: List of GoldStandardItem objects to evaluate on.
        output_dir: Directory to save the consolidated JSON results.
        
    Returns:
        A dictionary containing the full evaluation results and summary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"eval_{system_name}_{timestamp}.json"

    results = []
    answers = []
    contexts = []
    total_cost_usd = 0.0

    logger.info("Starting evaluation for %s on %d items...", system_name, len(gold_items))

    # 1. Run the system
    for i, item in enumerate(gold_items):
        logger.info("[%s] Processing %d/%d (id=%d)", system_name, i + 1, len(gold_items), item.id)
        try:
            res = pipeline.query(item.question)
            results.append(res)

            # Extract answer and context for RAGAS
            ans = res.answer if hasattr(res, "answer") else str(res)
            answers.append(ans)

            ctx = res.contexts if hasattr(res, "contexts") else []
            contexts.append(ctx)

            if hasattr(res, "metrics") and hasattr(res.metrics, "estimated_cost_usd"):
                total_cost_usd += res.metrics.estimated_cost_usd

        except Exception as e:
            logger.error("[%s] Query failed for id=%d: %s", system_name, item.id, e)
            results.append(None)
            answers.append("Error")
            contexts.append([])

    # Filter out failures for the metric pipelines
    valid_indices = [i for i, r in enumerate(results) if r is not None]
    valid_items = [gold_items[i] for i in valid_indices]
    valid_answers = [answers[i] for i in valid_indices]
    valid_contexts = [contexts[i] for i in valid_indices]
    valid_results = [results[i] for i in valid_indices]

    if not valid_indices:
        logger.error("All queries failed. Aborting evaluation.")
        return {"error": "All queries failed"}

    logger.info("Running RAGAS evaluation...")
    ragas_scores = evaluate_run(valid_items, valid_answers, valid_contexts)

    logger.info("Running Reasoning evaluation...")
    reasoning_scores = evaluate_reasoning_batch(valid_items, valid_results, system_name)

    logger.info("Running Custom, Citation and Efficiency evaluation...")
    detailed_results = []
    correctness_scores = []

    for i, item in enumerate(valid_items):
        res = valid_results[i]
        ans = valid_answers[i]

        custom_res = evaluate_custom_metrics(item, ans)
        citation_res = evaluate_citation_accuracy(item, ans)

        # Decide which correctness score to use for cost_per_correct_answer
        # We can use RAGAS Answer Correctness if exact_match is 0.0 or not applicable.
        # But RAGAS is computed in batch, so we don't easily have per-item RAGAS scores here.
        # For simplicity, we use exact_match for math questions, and answer_recall for text.
        score_to_track = custom_res.exact_match if custom_res.exact_match > 0 else custom_res.answer_recall
        correctness_scores.append(score_to_track)

        query_data = {
            "query_id": item.id,
            "fa_type": item.fa_type,
            "expected_answerable": item.expected_answerable,
            "question": item.question,
            "ground_truth": item.ground_truth,
            "answer": ans,
            "trajectory": format_trajectory(res, system_name),
            "custom_metrics": custom_res.to_dict(),
            "citation_metrics": citation_res.to_dict(),
            "reasoning_metrics": reasoning_scores[i].to_dict(),
        }

        if hasattr(res, "metrics"):
            query_data["run_metrics"] = {
                "latency_seconds": res.metrics.latency_seconds,
                "total_tokens": res.metrics.token_usage.total_tokens,
                "estimated_cost_usd": res.metrics.estimated_cost_usd,
                "num_steps": res.metrics.num_steps,
                "corrections": res.metrics.corrections,
            }

        detailed_results.append(query_data)

    # Process FA-4 specifically for cross-document success rate
    fa4_items = [d for d in detailed_results if d["fa_type"] == "FA-4"]
    cross_doc_success = 0.0
    if fa4_items:
        # Assuming success if exact_match == 1.0 or answer_recall >= 0.5
        successes = sum(
            1 for d in fa4_items
            if d["custom_metrics"]["exact_match"] == 1.0 or d["custom_metrics"]["answer_recall"] >= 0.5
        )
        cross_doc_success = successes / len(fa4_items)

    cost_per_correct = calculate_cost_per_correct_answer(
        total_cost_usd=total_cost_usd,
        correctness_scores=correctness_scores,
        threshold=0.8
    )

    # Correction rate: share of queries where the reflection/reviewer stage
    # triggered a revision (constant 0 for System 1, which has no reflection
    # stage; RunMetrics.corrections defaults to 0 and is never set there).
    items_with_run_metrics = [d for d in detailed_results if "run_metrics" in d]
    correction_rate = 0.0
    if items_with_run_metrics:
        correction_rate = sum(
            d["run_metrics"]["corrections"] for d in items_with_run_metrics
        ) / len(items_with_run_metrics)

    # Citation accuracy is only meaningful for items that actually have a
    # source to cite (expected_answerable=True) — FA-Refusal items are
    # skipped by evaluate_citation_accuracy and excluded here too.
    answerable_citation_scores = [
        d["citation_metrics"]["citation_accuracy"]
        for d in detailed_results
        if d["expected_answerable"]
    ]
    avg_citation_accuracy = (
        sum(answerable_citation_scores) / len(answerable_citation_scores)
        if answerable_citation_scores else 0.0
    )

    summary = {
        "system_name": system_name,
        "total_queries": len(gold_items),
        "successful_queries": len(valid_indices),
        "ragas_summary": ragas_scores.to_dict(),
        "cross_document_success_rate": round(cross_doc_success, 4),
        "citation_accuracy": round(avg_citation_accuracy, 4),
        "correction_rate": round(correction_rate, 4),
        "cost_per_correct_answer_usd": round(cost_per_correct, 4),
        "total_cost_usd": round(total_cost_usd, 4),
    }

    final_output = {
        "summary": summary,
        "detailed_results": detailed_results,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    logger.info("Evaluation complete. Results saved to %s", out_file)
    return final_output
