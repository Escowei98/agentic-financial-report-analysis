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
from pathlib import Path

from src.common.ingestion import download_all_filings
from src.evaluation.eval_runner import run_full_evaluation
from src.evaluation.gold_standard_loader import load_gold_standard
from src.systems.long_context.pipeline import LongContextPipeline
from src.systems.multi_agent.pipeline import MultiAgentPipeline
from src.systems.rag_agent.pipeline import AgentRAGPipeline
from src.systems.rag_monolith.pipeline import MonolithRAGPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Same 10 IDs as scripts/run_mini_eval_complete.py (stratified 2x per fa_type:
# FA-1, FA-2, FA-3, FA-4, FA-Refusal), reused for cost/comparability continuity.
PRIMARY_SAMPLE_IDS = [85, 16, 21, 26, 45, 106, 59, 147, 124, 76]

# Disjoint reserve sample (same stratification, seed=42), reserved in
# EVAL_DECISION_LOG.md [2026-08-01] specifically for re-validating a judge
# prompt/logic fix WITHOUT reusing the items that motivated the fix.
RESERVE_SAMPLE_IDS = [18, 20, 24, 44, 52, 74, 91, 119, 137, 144]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reserve", action="store_true",
        help="Use the disjoint reserve sample instead of the primary sample "
             "(for post-fix re-validation, see EVAL_DECISION_LOG.md).",
    )
    args = parser.parse_args()

    sample_ids = RESERVE_SAMPLE_IDS if args.reserve else PRIMARY_SAMPLE_IDS
    out_subdir = "raw_eval_reserve" if args.reserve else "raw_eval"

    project_root = Path(__file__).resolve().parent.parent
    gs_path = project_root / "data" / "gold_standard" / "gold_standard_v3_en.csv"
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

    for pipeline, sys_name in systems:
        logger.info("=" * 60)
        logger.info("Running full eval for %s", sys_name)
        logger.info("=" * 60)
        run_full_evaluation(pipeline, sys_name, sample_gs, out_dir)

    logger.info("Done. Raw eval JSONs written to %s", out_dir)


if __name__ == "__main__":
    main()
