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
from datetime import datetime
from pathlib import Path
from typing import Any

from src.evaluation.citation_evaluator import evaluate_citation_accuracy
from src.evaluation.custom_evaluator import evaluate_custom_metrics
from src.evaluation.evidence_store import EvidenceStore
from src.evaluation.gold_standard_loader import GoldStandardItem
from src.evaluation.process_evaluator import calculate_cost_per_correct_answer
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
# such conditions purely on final-answer metrics for the same reason —
# see docs/decisions/EVAL_DECISION_LOG.md [2026-08-16].
SYSTEMS_WITHOUT_MEANINGFUL_CONTEXT_METRICS = {"long_context", "multi_agent"}


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


def run_full_evaluation(
    pipeline: Any,
    system_name: str,
    gold_items: list[GoldStandardItem],
    output_dir: str | Path,
    evidence_store: EvidenceStore | None = None,
    keep_evidence: bool = False,
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
            # standard. Empty on answerable items and on pre-v5 files.
            "refusal_evidence": item.refusal_evidence,
            # Reference text for the refusal_quality judge, never an expected
            # answer string (gt_value stays empty on FA-Refusal). Carried
            # through so the judge-validation sample can show the rater the
            # same reference the judge graded against.
            "gt_correction": item.gt_correction,
            "gt_unit": item.gt_unit,
            "difficulty": item.difficulty,
            "window_class": item.window_class,
            "expected_answerable": item.expected_answerable,
            "question": item.question,
            "ground_truth": item.ground_truth,
            "answer": ans,
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
    # Population: every item that genuinely requires more than one filing —
    # all of FA-4 plus the FA-3 items marked `cross_window`. FA-4 alone left
    # out 15 multi-year items whose span does not fit inside one report's
    # comparative columns, which is precisely what `window_class` was added
    # to identify.
    #
    # Threshold: exact_match only. The old rule also accepted
    # `answer_recall >= 0.5`, i.e. a partially present answer, and the metric
    # saturated — S2, S3 and S4 all landed on exactly 28/30. A cross-document
    # synthesis that is only half right is not a success.
    # See EVAL_DECISION_LOG.md [2026-09-06].
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
    # without saying so. See EVAL_DECISION_LOG.md [2026-09-06].
    failed_ids = [gold_items[i].id for i, r in enumerate(results) if r is None]

    summary = {
        "system_name": system_name,
        "total_queries": len(gold_items),
        "successful_queries": len(valid_indices),
        "failed_queries": len(failed_ids),
        "failed_query_ids": failed_ids,
        "ragas_summary": ragas_scores.to_dict(),
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
