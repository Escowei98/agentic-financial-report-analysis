"""
Evaluation Runner Orchestrator.

Orchestrates the complete evaluation pipeline for a given system:
1. Runs the system on a batch of gold standard queries.
2. Computes RAGAS metrics.
3. Computes Custom metrics (Exact Match, Answer Recall, Refusal).
4. Computes Reasoning metrics (groundedness / validity / completeness).
5. Computes efficiency/process outputs (Cost per Correct Answer, Correction Rate).
6. Compiles a consolidated result JSON.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.evaluation.citation_evaluator import evaluate_citation_accuracy
from src.evaluation.custom_evaluator import evaluate_custom_metrics
from src.evaluation.evidence_store import EvidenceStore
from src.evaluation.gold_standard_loader import GoldStandardItem
from src.evaluation.process_evaluator import calculate_cost_per_correct_answer
from src.evaluation.locus_faithfulness import (
    EXCLUDED_NOT_ANSWERABLE,
    evaluate_locus_faithfulness_batch,
    summarise as summarise_locus_faithfulness,
)
from src.evaluation.ragas_evaluator import evaluate_run
from src.evaluation.reasoning_chain_evaluator import (
    ChainReasoningScores,
    evaluate_reasoning_chain_batch,
)
from src.evaluation.reasoning_chain_parser import parse_chain
from src.evaluation.trajectory_formatter import format_trajectory

logger = logging.getLogger(__name__)

# Systems whose `contexts` field is not "the evidence the answer was
# grounded in" and therefore can't meaningfully support RAGAS's
# context_precision/context_recall/faithfulness:
#   - long_context (S3): contexts only ever contains calculate/list_filings
#     tool outputs; the actual grounding (inlined filings) never appears
#     there (see LongContextPipeline._parse_messages docstring).
#   - multi_agent (S4): contexts is the specialist agents' own generated
#     answer text (see MultiAgentPipeline.query), not primary-source text
#     — faithfulness against it is near-tautological (synthesizer vs. its
#     own upstream paraphrase, not vs. real evidence).
# Both architectures inline the full filing set into an LLM prompt ahead
# of time rather than retrieving it at query time, so there is no
# observable retrieval step whose output RAGAS could score. Comparable
# work (FinanceBench, Islam et al. 2023; Li et al. 2024 "RAG or
# Long-Context LLMs?"; Lithgow-Serrano et al. 2025 FinDoc-RAG) evaluates
# such conditions purely on final-answer metrics for the same reason.
SYSTEMS_WITHOUT_MEANINGFUL_CONTEXT_METRICS = {"long_context", "multi_agent"}

# Attempts per query before it counts as failed, and the base of the
# exponential backoff between them.
#
# A query that dies on a transient endpoint error is dropped from every
# average (see the failed-queries note below), which silently changes the
# base a system is measured on. Retrying is applied identically to all four
# systems, so it introduces no asymmetry, but it does blur the line between
# "the endpoint hiccuped" and "this system fails more often". Every attempt
# is therefore counted and reported: per item in `query_attempts`, per run
# in `summary.retried_query_ids` / `summary.retried_queries`.
QUERY_MAX_ATTEMPTS = 3
RETRY_BASE_SECONDS = 5.0


# gt_units that denote a figure exact_match can actually rule on. "text" and
# "n/a" are qualitative or empty, and "count" carries no unit signal — for
# those, completeness (answer_recall) is the meaningful correctness signal.
_NUMERIC_GT_UNITS = {
    "usd_billion", "usd_million", "usd_per_share", "percent", "pp_change",
}


def _has_numeric_ground_truth(item: GoldStandardItem) -> bool:
    """Whether this item's ground truth is a figure exact_match can score."""
    return (item.gt_unit or "").strip().lower() in _NUMERIC_GT_UNITS


def _round_or_none(value: float | None) -> float | None:
    """Round a per-sample score, passing a missing one (RAGAS NaN) through."""
    return round(value, 4) if value is not None else None


def _query_with_retry(
    pipeline: Any,
    system_name: str,
    item: GoldStandardItem,
    max_attempts: int,
) -> tuple[Any, int]:
    """Run one query, retrying transient failures with exponential backoff.

    Returns the result (None if every attempt failed) and the number of
    attempts made. KeyboardInterrupt and SystemExit are BaseExceptions and
    deliberately not caught: a run of this size has to stay interruptible.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return pipeline.query(item.question), attempt
        except Exception as exc:  # noqa: BLE001 — retried, then reported
            last_exc = exc
            if attempt < max_attempts:
                delay = RETRY_BASE_SECONDS * 2 ** (attempt - 1)
                logger.warning(
                    "[%s] Query failed for id=%d (attempt %d/%d): %s — retrying in %.0fs",
                    system_name, item.id, attempt, max_attempts, exc, delay,
                )
                time.sleep(delay)
    logger.error(
        "[%s] Query failed for id=%d after %d attempts: %s",
        system_name, item.id, max_attempts, last_exc,
    )
    return None, max_attempts


def run_full_evaluation(
    pipeline: Any,
    system_name: str,
    gold_items: list[GoldStandardItem],
    output_dir: str | Path,
    evidence_store: EvidenceStore | None = None,
    keep_evidence: bool = False,
    max_attempts: int = QUERY_MAX_ATTEMPTS,
) -> dict:
    """
    Run the full evaluation suite on a given system.
    
    Args:
        pipeline: An instantiated system pipeline (e.g., RAGMonolithPipeline, AgentRAGPipeline).
                  Must have a `.query(question: str)` method.
        system_name: Identifier string (e.g., 'rag_monolith', 'rag_agent').
        gold_items: List of GoldStandardItem objects to evaluate on.
        output_dir: Directory to save the consolidated JSON results.
        evidence_store: Corpus access for the groundedness dimension. Without
                  it the reasoning chain is still parsed and its shape
                  reported, but the three dimensions stay None -- there is
                  nothing to verify a cited locus against.
        keep_evidence: Store the passages the judge saw alongside its
                  verdicts. Needed to build a human-validation sample that
                  shows raters the same evidence; off for production runs,
                  where it would multiply the result JSON several times over.
        max_attempts: Attempts per query before it counts as failed, applied
                  identically to every system. 1 disables retrying. Every
                  attempt is counted and reported (see QUERY_MAX_ATTEMPTS).
        
    Returns:
        A dictionary containing the full evaluation results and summary.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = output_dir / f"eval_{system_name}_{timestamp}.json"

    results: list[Any] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    total_cost_usd = 0.0

    logger.info("Starting evaluation for %s on %d items...", system_name, len(gold_items))

    # 1. Run the system
    #
    # Attempts per item are kept for every item, failed ones included: only
    # `total_cost_usd` sees the successful attempts, because a system's own
    # metrics object is the only cost signal there is and a crashed attempt
    # never produces one. The provider was still billed for it, so a run with
    # many retries costs more than its reported total says.
    attempts_by_id: dict[int, int] = {}
    for i, item in enumerate(gold_items):
        logger.info("[%s] Processing %d/%d (id=%d)", system_name, i + 1, len(gold_items), item.id)
        res, attempts = _query_with_retry(pipeline, system_name, item, max_attempts)
        attempts_by_id[item.id] = attempts
        results.append(res)

        if res is None:
            answers.append("Error")
            contexts.append([])
            continue

        # Extract answer and context for RAGAS
        ans = res.answer if hasattr(res, "answer") else str(res)
        answers.append(ans)

        ctx = res.contexts if hasattr(res, "contexts") else []
        contexts.append(ctx)

        if hasattr(res, "metrics") and hasattr(res.metrics, "estimated_cost_usd"):
            total_cost_usd += res.metrics.estimated_cost_usd

    # Split the reasoning chain off every answer BEFORE any other evaluator
    # sees it.
    #
    # The chain restates the answer's figures and repeats its citations, so an
    # answer with the block still attached would hand citation_accuracy a
    # second copy of every citation and the recall judges a second copy of
    # every number. `raw_answers` is kept only for the reasoning evaluator,
    # which is the one consumer that needs the block.
    raw_answers = list(answers)
    parsed_chains = [parse_chain(a) for a in answers]
    answers = [p.answer_without_chain for p in parsed_chains]

    # Filter out failures for the metric pipelines
    valid_indices = [i for i, r in enumerate(results) if r is not None]
    valid_items = [gold_items[i] for i in valid_indices]
    valid_answers = [answers[i] for i in valid_indices]
    valid_raw_answers = [raw_answers[i] for i in valid_indices]
    valid_chains = [parsed_chains[i] for i in valid_indices]
    valid_contexts = [contexts[i] for i in valid_indices]
    valid_results = [results[i] for i in valid_indices]

    if not valid_indices:
        logger.error("All queries failed. Aborting evaluation.")
        return {"error": "All queries failed"}

    logger.info("Running RAGAS evaluation...")
    include_context_metrics = system_name not in SYSTEMS_WITHOUT_MEANINGFUL_CONTEXT_METRICS
    ragas_scores = evaluate_run(
        valid_items, valid_answers, valid_contexts,
        include_context_metrics=include_context_metrics,
    )

    if evidence_store is None:
        logger.warning(
            "[%s] No evidence store passed — reasoning dimensions will be None. "
            "Pass one to score groundedness/validity/completeness.",
            system_name,
        )
        reasoning_scores = [
            ChainReasoningScores(
                query_id=item.id,
                system_name=system_name,
                chain_emitted=chain.chain_emitted,
                num_steps=len(chain.steps),
                num_evidential=len(chain.evidential_steps),
                num_inferential=len(chain.steps) - len(chain.evidential_steps),
                num_untagged=chain.untagged_count,
                steps=[step.to_dict() for step in chain.steps],
            )
            for item, chain in zip(valid_items, valid_chains)
        ]
    else:
        logger.info("Running Reasoning evaluation...")
        reasoning_scores = evaluate_reasoning_chain_batch(
            valid_items, valid_raw_answers, evidence_store, system_name,
            parsed_chains=valid_chains,
            keep_evidence=keep_evidence,
        )

    # Faithfulness of the final answer against the corpus at the loci it
    # cites -- the one faithfulness measure defined for all four systems.
    # Uses the same evidence store as groundedness; without one it stays
    # excluded rather than silently 0. See locus_faithfulness.py.
    locus_faithfulness: list[dict]
    if evidence_store is None:
        locus_faithfulness = [
            {"score": None,
             "excluded": (
                 EXCLUDED_NOT_ANSWERABLE if not item.expected_answerable
                 else "no_evidence_store"
             ),
             "n_loci": 0, "n_passages": 0, "loci": []}
            for item in valid_items
        ]
    else:
        logger.info("Running locus faithfulness evaluation...")
        locus_faithfulness = [
            r.to_dict() for r in evaluate_locus_faithfulness_batch(
                [item.question for item in valid_items],
                valid_answers,
                [item.expected_answerable for item in valid_items],
                evidence_store,
            )
        ]

    logger.info("Running Custom, Citation and Efficiency evaluation...")
    detailed_results = []
    correctness_scores = []

    for i, item in enumerate(valid_items):
        res = valid_results[i]
        ans = valid_answers[i]

        custom_res = evaluate_custom_metrics(item, ans)
        citation_res = evaluate_citation_accuracy(item, ans)

        # Which correctness score feeds cost_per_correct_answer.
        # evaluate_custom_metrics computes exact_match/answer_recall only for
        # answerable items and refusal_accuracy only for the unanswerable
        # ones (the others stay at their 0.0 default), so the score has to be
        # picked per stratum: for a refusal item a correct refusal IS the
        # correct answer, and reading exact_match/answer_recall there would
        # make 30 of 150 items unable to count as correct by construction —
        # inflating cost per correct answer by the same factor for every
        # system. Within the answerable stratum: exact_match for the items
        # that carry a numeric ground truth, answer_recall otherwise.
        if not item.expected_answerable:
            score_to_track = custom_res.refusal_accuracy
        elif _has_numeric_ground_truth(item):
            score_to_track = custom_res.exact_match
        else:
            score_to_track = custom_res.answer_recall
        correctness_scores.append(score_to_track)

        query_data = {
            "query_id": item.id,
            "fa_type": item.fa_type,
            "subtype": item.subtype,
            # Carried through so the per-query table can break the FA-Refusal
            # stratum down by evidence class without re-joining the gold
            # standard. Empty on answerable items.
            "refusal_evidence": item.refusal_evidence,
            # Reference text for the refusal_quality judge, never an expected
            # answer string (gt_value stays empty on FA-Refusal). Carried
            # through so the judge-validation sample can show the rater the
            # same reference the judge graded against.
            "gt_correction": item.gt_correction,
            "gt_unit": item.gt_unit,
            "window_class": item.window_class,
            "expected_answerable": item.expected_answerable,
            "question": item.question,
            "ground_truth": item.ground_truth,
            "answer": ans,
            # How many attempts this answer took. 1 for everything that
            # worked first time; anything above it belongs in the report.
            "query_attempts": attempts_by_id[item.id],
            "trajectory": format_trajectory(res, system_name),
            "custom_metrics": custom_res.to_dict(),
            "citation_metrics": citation_res.to_dict(),
            "reasoning_metrics": reasoning_scores[i].to_dict(),
            # Per-item RAGAS scores. evaluate_run() returns them aligned
            # with valid_items, so index i addresses this item. Without
            # them RAGAS would only exist as a run-level average in
            # summary.ragas_summary, which rules out both the per-stratum
            # breakdown and any significance test on answer_correctness.
            # Keys mirror whichever metrics were requested for this system
            # (two or five, see include_context_metrics); a None value
            # means RAGAS returned NaN for that sample.
            "ragas_metrics": {
                metric: _round_or_none(values[i])
                for metric, values in ragas_scores.per_sample.items()
            },
            "locus_faithfulness": locus_faithfulness[i],
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

    # Cross-document success rate.
    #
    # Population: every item that genuinely requires more than one filing:
    # all of FA-4 plus the FA-3 items marked `cross_window`, i.e. multi-year
    # items whose span does not fit inside one report's comparative columns.
    #
    # Threshold: exact_match only. Accepting a partially present answer
    # (answer_recall >= 0.5) saturates the metric; a cross-document synthesis
    # that is only half right is not a success.
    cross_doc_items = [
        d for d in detailed_results
        if d["fa_type"] == "FA-4" or d.get("window_class") == "cross_window"
    ]
    cross_doc_success = 0.0
    if cross_doc_items:
        successes = sum(
            1 for d in cross_doc_items
            if d["custom_metrics"]["exact_match"] == 1.0
        )
        cross_doc_success = successes / len(cross_doc_items)

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

    # Failed queries are excluded from the judge and RAGAS pipelines (there is
    # no result object to score), so every average below is over the
    # successful ones. That silently favours a system that crashes more
    # often, hence the explicit count and id list: any downstream table has
    # to state them, and a run with failures is not comparable to a clean one
    # without saying so.
    failed_ids = [gold_items[i].id for i, r in enumerate(results) if r is None]

    retried = {
        str(query_id): n for query_id, n in attempts_by_id.items() if n > 1
    }

    summary = {
        "system_name": system_name,
        "total_queries": len(gold_items),
        "successful_queries": len(valid_indices),
        "failed_queries": len(failed_ids),
        "failed_query_ids": failed_ids,
        # A system that needed many retries is not the same as one that
        # needed none, even when both end at 150/150 -- reported so the
        # difference cannot vanish into an identical success count.
        "query_max_attempts": max_attempts,
        "retried_queries": len(retried),
        "retried_query_ids": retried,
        "ragas_summary": ragas_scores.to_dict(),
        "locus_faithfulness_summary": summarise_locus_faithfulness(locus_faithfulness),
        "cross_document_success_rate": round(cross_doc_success, 4),
        "citation_accuracy": round(avg_citation_accuracy, 4),
        "correction_rate": round(correction_rate, 4),
        "cost_per_correct_answer_usd": _round_or_none(cost_per_correct),
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
