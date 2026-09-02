"""
Re-runs S3 (long_context) on ONLY the gold-standard ids that failed in the
original full_eval_n150 run due to the reflection-resend bug (see
docs/decisions/EVAL_DECISION_LOG.md), now fixed in
src/systems/long_context/pipeline.py via a fallback-to-draft-answer guard.

The other 140 items already completed cleanly in the first run and are
NOT re-run here (their code path never touched the fix, so re-running
them would just spend money for identical results). Use
scripts/merge_s3_results.py afterward to fold this batch back into the
original 140-item result and recompute the summary.
"""
import logging
from pathlib import Path

from src.common.config import load_config
from src.common.ingestion import download_all_filings
from src.evaluation.eval_runner import run_full_evaluation
from src.evaluation.gold_standard_loader import load_gold_standard
from src.systems.long_context.pipeline import LongContextPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path(__file__).resolve().parent.parent / "data" / "results" / "full_eval_n150.log"),
    ],
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FAILED_IDS = [21, 32, 43, 64, 67, 69, 77, 96, 106, 117]


def main():
    gs_path = PROJECT_ROOT / "data" / "gold_standard" / "gold_standard_v3_en.csv"
    filings = download_all_filings()

    all_gs = load_gold_standard(gs_path)
    id_to_gs = {g.id: g for g in all_gs}
    retry_items = [id_to_gs[i] for i in FAILED_IDS if i in id_to_gs]
    logger.info("Re-running %d previously-failed S3 items: %s", len(retry_items), FAILED_IDS)

    lc_cfg = load_config(PROJECT_ROOT / "configs" / "long_context.yaml")
    s3 = LongContextPipeline(lc_cfg)
    s3.build(filings)

    out_dir = PROJECT_ROOT / "data" / "results" / "full_eval_n150"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("RETRY: S3 (long_context) with fallback fix — %d items", len(retry_items))
    logger.info("=" * 70)
    res = run_full_evaluation(s3, "long_context", retry_items, out_dir)
    logger.info("S3 retry summary: %s", res.get("summary", {}))
    logger.info("S3 retry complete.")


if __name__ == "__main__":
    main()
