"""
Re-validation of top Optuna configs with the fixed RAGAS evaluator.

Purpose:
  The original ablation run had faithfulness truncation errors (~60-80%
  of NLI evaluations failed due to max_output_tokens=8192 being too low).
  This script re-evaluates the top-N configs with the fix applied
  (max_output_tokens=16384 in ragas_evaluator.py).

Usage (from notebook):
    from src.systems.rag_monolith.ablation.validate_top_configs import validate_top_configs
    results = validate_top_configs(filings, gold_standard, study, top_n=5)
"""

import logging
from typing import Sequence

import optuna
import pandas as pd

from src.common.ingestion import ProcessedFiling
from src.common.retrieval import build_hybrid_retriever
from src.evaluation.gold_standard_loader import GoldStandardItem
from src.evaluation.ragas_evaluator import evaluate_run
from src.systems.rag_monolith.pipeline import MonolithRAGPipeline

logger = logging.getLogger(__name__)


def validate_top_configs(
    filings: Sequence[ProcessedFiling],
    gold_standard: list[GoldStandardItem],
    study: optuna.Study,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Re-evaluate the top-N Optuna configs with the fixed RAGAS evaluator.

    This reuses the same pipeline build logic (including vectorstore caching)
    but runs a fresh RAGAS evaluation with max_output_tokens=16384, which
    prevents the NLI JSON truncation errors.

    Args:
        filings: Processed SEC filings (same as used in ablation).
        gold_standard: Gold-standard Q&A pairs.
        study: Completed Optuna study.
        top_n: Number of top configs to re-evaluate.

    Returns:
        DataFrame with clean RAGAS scores for each config.
    """
    # Get top-N trials sorted by composite score
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    completed.sort(key=lambda t: t.value, reverse=True)
    top_trials = completed[:top_n]

    logger.info(
        "Re-validating top %d configs (of %d total trials)",
        len(top_trials), len(completed),
    )

    # Cache pipelines by chunk config to avoid redundant rebuilds
    pipeline_cache: dict = {}
    results = []

    for rank, trial in enumerate(top_trials):
        params = trial.params
        cache_key = f"cs{params['chunk_size']}_ov{int(params['overlap_pct']*100)}"

        logger.info(
            "[%d/%d] Trial %d: chunk=%d, overlap=%.0f%%, bm25=%.1f, "
            "pre_k=%d, post_k=%d",
            rank + 1, len(top_trials), trial.number,
            params["chunk_size"], params["overlap_pct"] * 100,
            params["bm25_weight"], params["pre_rerank_top_k"],
            params["post_rerank_top_k"],
        )

        # Build pipeline (or reuse cached vectorstore)
        config_override = {
            "chunk_size": params["chunk_size"],
            "chunk_overlap_pct": params["overlap_pct"],
            "bm25_weight": params["bm25_weight"],
            "pre_rerank_top_k": params["pre_rerank_top_k"],
            "post_rerank_top_k": params["post_rerank_top_k"],
        }

        pipeline = MonolithRAGPipeline(config_override=config_override)

        if cache_key in pipeline_cache:
            cached = pipeline_cache[cache_key]
            pipeline._documents = cached["documents"]
            pipeline._vectorstore = cached["vectorstore"]
            pipeline._llm = cached["llm"]

            pipeline._retriever = build_hybrid_retriever(
                vectorstore=pipeline._vectorstore,
                documents=pipeline._documents,
                bm25_weight=params["bm25_weight"],
                pre_rerank_top_k=params["pre_rerank_top_k"],
            )
            logger.info("  Reused cached vectorstore for %s", cache_key)
        else:
            pipeline.build(filings)
            pipeline_cache[cache_key] = {
                "documents": pipeline._documents,
                "vectorstore": pipeline._vectorstore,
                "llm": pipeline._llm,
            }
            logger.info("  Built and cached vectorstore for %s", cache_key)

        # Run all queries
        answers = []
        all_contexts = []
        total_latency = 0.0
        total_tokens = 0

        for item in gold_standard:
            result = pipeline.query(item.question)
            answers.append(result.answer)
            all_contexts.append(result.contexts)
            total_latency += result.metrics.latency_seconds
            total_tokens += result.metrics.token_usage.total_tokens

        # Evaluate with RAGAS (now with max_output_tokens=16384)
        scores = evaluate_run(
            gold_standard=gold_standard,
            answers=answers,
            contexts=all_contexts,
        )

        row = {
            "rank": rank + 1,
            "trial": trial.number,
            "chunk_size": params["chunk_size"],
            "overlap_pct": params["overlap_pct"],
            "bm25_weight": params["bm25_weight"],
            "pre_rerank_top_k": params["pre_rerank_top_k"],
            "post_rerank_top_k": params["post_rerank_top_k"],
            "composite_score": scores.composite_score,
            "context_precision": scores.context_precision,
            "context_recall": scores.context_recall,
            "faithfulness": scores.faithfulness,
            "avg_latency": total_latency / len(gold_standard),
            "total_tokens": total_tokens,
            # Original scores for comparison
            "orig_composite": trial.user_attrs.get("composite_score", trial.value),
            "orig_faithfulness": trial.user_attrs.get("faithfulness", 0.0),
        }
        results.append(row)

        logger.info(
            "  Result: composite=%.3f (prec=%.3f, rec=%.3f, faith=%.3f) "
            "[was: composite=%.3f, faith=%.3f]",
            scores.composite_score, scores.context_precision,
            scores.context_recall, scores.faithfulness,
            row["orig_composite"], row["orig_faithfulness"],
        )

    df = pd.DataFrame(results)
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 90)
    print("  Validated RAGAS Scores (with fixed max_output_tokens=16384)")
    print("=" * 90)
    print(df[[
        "rank", "trial", "chunk_size", "overlap_pct", "bm25_weight",
        "pre_rerank_top_k", "post_rerank_top_k",
        "composite_score", "context_precision", "context_recall", "faithfulness",
    ]].to_string(index=False))
    print()

    return df
