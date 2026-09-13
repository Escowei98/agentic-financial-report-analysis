"""
Optuna TPE Bayesian Optimization for the RAG Monolith ablation study.

Searches for optimal parameters across:
  - chunk_size: [1000, 1500, 2000]
  - overlap_pct: [0.10, 0.20]
  - bm25_weight: [0.3, 0.5, 0.7]
  - pre_rerank_top_k: [15, 20, 25]
  - post_rerank_top_k: [3..8]

Uses Optuna TPE sampler with vectorstore caching to avoid
redundant index rebuilds when only retrieval params change.
"""

import logging
from pathlib import Path
from typing import Sequence

import optuna

from src.common.ingestion import ProcessedFiling
from src.evaluation.gold_standard_loader import GoldStandardItem
from src.evaluation.ragas_evaluator import evaluate_run
from src.systems.rag_monolith.pipeline import MonolithRAGPipeline

logger = logging.getLogger(__name__)

# Default study storage (SQLite for resume capability)
DEFAULT_STORAGE = "sqlite:///data/ablation_study.db"
DEFAULT_STUDY_NAME = "rag_monolith_ablation"


def create_objective(
    filings: Sequence[ProcessedFiling],
    gold_standard: list[GoldStandardItem],
    pipeline_cache: dict | None = None,
):
    """
    Create an Optuna objective function that evaluates a RAG config.

    Uses a pipeline cache to avoid rebuilding the vectorstore when only
    retrieval parameters (bm25_weight, top_k) change — the chunking +
    embedding step (most expensive) is reused.

    Args:
        filings: Processed SEC filings.
        gold_standard: Gold-standard Q&A pairs.
        pipeline_cache: Shared dict for caching built pipelines by chunk config.

    Returns:
        Optuna objective function.
    """
    if pipeline_cache is None:
        pipeline_cache = {}

    def objective(trial: optuna.Trial) -> float:
        # --- Suggest parameters ---
        chunk_size = trial.suggest_categorical("chunk_size", [1000, 1500, 2000])
        overlap_pct = trial.suggest_categorical("overlap_pct", [0.10, 0.20])
        bm25_weight = trial.suggest_categorical("bm25_weight", [0.3, 0.5, 0.7])
        pre_rerank_top_k = trial.suggest_categorical("pre_rerank_top_k", [15, 20, 25])
        post_rerank_top_k = trial.suggest_int("post_rerank_top_k", 3, 8)

        config_override = {
            "chunk_size": chunk_size,
            "chunk_overlap_pct": overlap_pct,
            "bm25_weight": bm25_weight,
            "pre_rerank_top_k": pre_rerank_top_k,
            "post_rerank_top_k": post_rerank_top_k,
        }

        logger.info(
            "Trial %d: chunk=%d, overlap=%.0f%%, bm25=%.1f, pre_k=%d, post_k=%d",
            trial.number, chunk_size, overlap_pct * 100,
            bm25_weight, pre_rerank_top_k, post_rerank_top_k,
        )

        # --- Build pipeline (with caching) ---
        cache_key = f"cs{chunk_size}_ov{int(overlap_pct*100)}"

        pipeline = MonolithRAGPipeline(config_override=config_override)

        if cache_key in pipeline_cache:
            # Reuse cached chunked documents + vectorstore
            cached = pipeline_cache[cache_key]
            pipeline._documents = cached["documents"]
            pipeline._vectorstore = cached["vectorstore"]
            pipeline._llm = cached["llm"]

            # Rebuild only the hybrid retriever with new weights
            from src.common.retrieval import (
                build_hybrid_retriever,
            )
            pipeline._retriever = build_hybrid_retriever(
                vectorstore=pipeline._vectorstore,
                documents=pipeline._documents,
                bm25_weight=bm25_weight,
                pre_rerank_top_k=pre_rerank_top_k,
            )
            logger.info("Reused cached vectorstore for %s", cache_key)
        else:
            # Full build (chunk + embed + index)
            pipeline.build(filings)
            pipeline_cache[cache_key] = {
                "documents": pipeline._documents,
                "vectorstore": pipeline._vectorstore,
                "llm": pipeline._llm,
            }
            logger.info("Built and cached vectorstore for %s", cache_key)

        # --- Run queries ---
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

        # --- Evaluate with RAGAS ---
        scores = evaluate_run(
            gold_standard=gold_standard,
            answers=answers,
            contexts=all_contexts,
        )

        # --- Store user attributes for analysis ---
        trial.set_user_attr("context_precision", scores.context_precision)
        trial.set_user_attr("context_recall", scores.context_recall)
        trial.set_user_attr("faithfulness", scores.faithfulness)
        trial.set_user_attr("composite_score", scores.composite_score)
        trial.set_user_attr("avg_latency", total_latency / len(gold_standard))
        trial.set_user_attr("total_tokens", total_tokens)

        logger.info(
            "Trial %d result: composite=%.3f (prec=%.3f, rec=%.3f, faith=%.3f)",
            trial.number, scores.composite_score,
            scores.context_precision, scores.context_recall, scores.faithfulness,
        )

        return scores.composite_score

    return objective


def run_ablation(
    filings: Sequence[ProcessedFiling],
    gold_standard: list[GoldStandardItem],
    n_trials: int = 50,
    study_name: str = DEFAULT_STUDY_NAME,
    storage: str = DEFAULT_STORAGE,
) -> optuna.Study:
    """
    Run the full Optuna ablation study.

    Args:
        filings: Processed SEC filings.
        gold_standard: Gold-standard Q&A pairs.
        n_trials: Number of Optuna trials (40-60 recommended).
        study_name: Name for the Optuna study (for persistence).
        storage: SQLite storage URL for resume capability.

    Returns:
        Completed Optuna study with all trial results.
    """
    # Ensure storage directory exists
    storage_path = storage.replace("sqlite:///", "")
    Path(storage_path).parent.mkdir(parents=True, exist_ok=True)

    # Create or load study (for resume capability)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",  # Maximize composite RAGAS score
        sampler=optuna.samplers.TPESampler(seed=42),  # Reproducible
        load_if_exists=True,
    )

    # Create objective with shared pipeline cache
    pipeline_cache: dict = {}
    objective = create_objective(filings, gold_standard, pipeline_cache)

    remaining_trials = n_trials - len(study.trials)
    if remaining_trials <= 0:
        logger.info(
            "Study already has %d trials (target: %d). No new trials needed.",
            len(study.trials), n_trials,
        )
        return study

    logger.info(
        "Starting ablation: %d trials (already completed: %d)",
        remaining_trials, len(study.trials),
    )

    study.optimize(objective, n_trials=remaining_trials)

    logger.info(
        "Ablation complete: best score=%.3f, best params=%s",
        study.best_value, study.best_params,
    )

    return study
