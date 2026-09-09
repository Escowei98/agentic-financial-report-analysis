"""
Runs S1-S4 on the judge-validation primary sample (10 stratified gold-standard
IDs, 2x per fa_type) and evaluates them with the current (OpenAI-backed)
judge. Output is separate from data/results/mini_eval_complete/ (which holds
the pre-judge-swap Gemini-judged runs) to avoid mixing judge versions.

See docs/decisions/EVAL_DECISION_LOG.md [2026-08-01] for why this sample
exists and how it will be used (human-validation blind rating).
"""
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.common.ingestion import download_all_filings
from src.evaluation.eval_runner import run_full_evaluation
from src.evaluation.evidence_store import EvidenceStore
from src.evaluation.gold_standard_loader import GOLD_STANDARD_EN, load_gold_standard
from src.systems.long_context.pipeline import LongContextPipeline
from src.systems.multi_agent.pipeline import MultiAgentPipeline
from src.systems.rag_agent.pipeline import AgentRAGPipeline
from src.systems.rag_monolith.pipeline import MonolithRAGPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# The two id lists live in src/evaluation/judge_validation_sample.py, next
# to the reasoning for their composition, because run_full_eval.py needs
# the same lists to mark these items in the n=150 output.
from src.evaluation.judge_validation_sample import (  # noqa: E402
    PRIMARY_SAMPLE_IDS,
    RESERVE_SAMPLE_IDS,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reserve", action="store_true",
        help="Use the disjoint reserve sample instead of the primary sample "
             "(for post-fix re-validation, see EVAL_DECISION_LOG.md).",
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help="Evaluate the four systems concurrently. Wall time then tracks "
             "the slowest system (S3) instead of the sum of all four. The "
             "pipelines are still BUILT sequentially: S1 and S2 share a "
             "vectorstore directory and would race on it. Off by default "
             "because it quadruples the request rate against both the Gemini "
             "and the judge endpoint, and neither client is configured with "
             "a backoff policy.",
    )
    args = parser.parse_args()

    sample_ids = RESERVE_SAMPLE_IDS if args.reserve else PRIMARY_SAMPLE_IDS
    out_subdir = "raw_eval_reserve" if args.reserve else "raw_eval"

    project_root = Path(__file__).resolve().parent.parent
    gs_path = GOLD_STANDARD_EN
    filings = download_all_filings()

    all_gs = load_gold_standard(gs_path)
    id_to_gs = {g.id: g for g in all_gs}
    sample_gs = [id_to_gs[i] for i in sample_ids if i in id_to_gs]
    if len(sample_gs) != len(sample_ids):
        missing = set(sample_ids) - {g.id for g in sample_gs}
        raise ValueError(f"Gold standard IDs not found: {missing}")

    logger.info("Initializing pipelines...")
    s1 = MonolithRAGPipeline()
    s2 = AgentRAGPipeline()
    s3 = LongContextPipeline()
    s4 = MultiAgentPipeline()

    # Corpus access for the groundedness dimension. Built once and shared:
    # it is read-only, and rebuilding it per system would be four identical
    # chunkings of the same 12 filings.
    evidence_store = EvidenceStore(filings)

    s1.build(filings)
    s2.build(filings)
    s3.build(filings)
    s4.build(filings)

    out_dir = project_root / "data" / "results" / "judge_validation" / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    systems = [
        (s1, "rag_monolith"),
        (s2, "rag_agent"),
        (s3, "long_context"),
        (s4, "multi_agent"),
    ]

    if args.parallel:
        _run_parallel(systems, sample_gs, out_dir, evidence_store)
    else:
        for pipeline, sys_name in systems:
            logger.info("=" * 60)
            logger.info("Running full eval for %s", sys_name)
            logger.info("=" * 60)
            run_full_evaluation(
                pipeline, sys_name, sample_gs, out_dir,
                evidence_store=evidence_store,
                # The human raters must see the same passages the judge saw,
                # or they are validating a different task. Only this script
                # sets it; a production run would multiply its JSON several
                # times over for material nobody reads.
                keep_evidence=True,
            )

    logger.info("Done. Raw eval JSONs written to %s", out_dir)


def _run_parallel(systems, sample_gs, out_dir, evidence_store) -> None:
    """Evaluate the four systems concurrently.

    Safe only after every pipeline has been built: S1 and S2 resolve to the
    same vectorstore directory (both default to
    `data/vectorstores/rag_monolith` and, on equal chunk settings, to the same
    config and filings hash), so building them at the same time would have two
    writers on one ChromaDB. Querying and judging are pure I/O against remote
    endpoints, so threads are the right tool and the GIL is not in the way.

    The judge clients are constructed here, before the pool starts, so the
    lazy singletons in custom_evaluator and citation_evaluator are not raced
    into existence four times over. `evidence_store` is passed in already
    built for the same reason, and shared rather than copied: it is read-only,
    and one store for all four systems IS the fairness property the
    groundedness dimension rests on.

    A system that raises does not take the others down: its exception is
    logged and re-raised at the end, so a partial run is still on disk and the
    failure is not silent.
    """
    from src.evaluation.citation_evaluator import get_eval_llm as _warm_citation
    from src.evaluation.custom_evaluator import get_eval_llm as _warm_custom

    _warm_custom()
    _warm_citation()

    logger.info("=" * 60)
    logger.info("Running all %d systems concurrently", len(systems))
    logger.info("=" * 60)

    failures: dict[str, BaseException] = {}
    with ThreadPoolExecutor(max_workers=len(systems)) as pool:
        futures = {
            pool.submit(
                run_full_evaluation, pipeline, sys_name, sample_gs, out_dir,
                evidence_store=evidence_store, keep_evidence=True,
            ): sys_name
            for pipeline, sys_name in systems
        }
        for future in as_completed(futures):
            sys_name = futures[future]
            try:
                future.result()
                logger.info("✓ %s finished", sys_name)
            except BaseException as exc:  # noqa: BLE001 — reported below
                failures[sys_name] = exc
                logger.error("✗ %s failed: %s", sys_name, exc, exc_info=True)

    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(systems)} systems failed: "
            f"{', '.join(sorted(failures))}"
        )


if __name__ == "__main__":
    main()
