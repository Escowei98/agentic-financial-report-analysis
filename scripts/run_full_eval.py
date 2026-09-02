"""
Runs the full evaluation pipeline for S1, S2, S3, S4 on the complete
n=150 gold standard (data/gold_standard/gold_standard_v3_en.csv).

Saves per-system JSON via eval_runner (full detail, incl. rationales and
trajectories) plus two aggregate CSVs for cross-system weakness analysis:
- full_eval_summary.csv: one row per system (RAGAS, custom, citation,
  process, cost/latency aggregates)
- full_eval_per_query.csv: one row per (system, query) with the key
  scoring dimensions, for slicing by fa_type/difficulty/system to spot
  systematic failure patterns.
"""
import csv
import logging
from pathlib import Path

from src.common.config import load_config
from src.common.ingestion import download_all_filings
from src.evaluation.eval_runner import run_full_evaluation
from src.evaluation.gold_standard_loader import load_gold_standard
from src.systems.long_context.pipeline import LongContextPipeline
from src.systems.multi_agent.pipeline import MultiAgentPipeline
from src.systems.rag_agent.pipeline import AgentRAGPipeline
from src.systems.rag_monolith.pipeline import MonolithRAGPipeline

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


def main():
    gs_path = PROJECT_ROOT / "data" / "gold_standard" / "gold_standard_v3_en.csv"
    filings = download_all_filings()

    gold_items = load_gold_standard(gs_path)
    logger.info("Loaded %d gold standard items", len(gold_items))

    logger.info("Initializing pipelines...")
    base_cfg = load_config(PROJECT_ROOT / "configs" / "base.yaml")
    agent_cfg = load_config(PROJECT_ROOT / "configs" / "agent.yaml")
    lc_cfg = load_config(PROJECT_ROOT / "configs" / "long_context.yaml")
    ma_cfg = load_config(PROJECT_ROOT / "configs" / "multi_agent.yaml")

    s1 = MonolithRAGPipeline(base_cfg)
    s2 = AgentRAGPipeline(agent_cfg)
    s3 = LongContextPipeline(lc_cfg)
    s4 = MultiAgentPipeline(ma_cfg)

    s1.build(filings)
    s2.build(filings)
    s3.build(filings)
    s4.build(filings)

    out_dir = PROJECT_ROOT / "data" / "results" / "full_eval_n150"
    out_dir.mkdir(parents=True, exist_ok=True)

    systems = [
        ("S1", s1, "rag_monolith"),
        ("S2", s2, "rag_agent"),
        ("S3", s3, "long_context"),
        ("S4", s4, "multi_agent"),
    ]

    results = {}
    for sys_label, pipeline, sys_name in systems:
        logger.info("=" * 70)
        logger.info("Running full eval for %s (%s) — %d items", sys_label, sys_name, len(gold_items))
        logger.info("=" * 70)
        res = run_full_evaluation(pipeline, sys_name, gold_items, out_dir)
        results[sys_label] = res
        logger.info("%s summary: %s", sys_label, res.get("summary", {}))

    write_summary_csv(results, out_dir / "full_eval_summary.csv")
    write_per_query_csv(results, gold_items, out_dir / "full_eval_per_query.csv")
    write_markdown_report(results, out_dir / "full_eval_report.md")
    logger.info("Full evaluation complete. Outputs in %s", out_dir)


def _avg(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def compute_extra_aggregates(detailed_results: list) -> dict:
    """
    Aggregate metrics not already present in eval_runner's summary dict.

    Custom metrics (exact_match/answer_recall/refusal_accuracy) are only
    ever computed by evaluate_custom_metrics() for a subset of items —
    exact_match/answer_recall only when expected_answerable=True,
    refusal_accuracy only when expected_answerable=False (see
    src/evaluation/custom_evaluator.py::evaluate_custom_metrics). Averaging
    over all 150 items would dilute each score with 0.0-default values from
    items where the metric was never computed, so we filter to the subset
    it actually applies to — the same convention eval_runner.py already
    uses for citation_accuracy/correction_rate.

    Agentic reasoning dimensions (tool_selection/error_recovery) are only
    scored for agentic systems (S2-S4, see AGENTIC_SYSTEMS in
    reasoning_evaluator.py); returned as None for a non-agentic system
    (S1) so callers can render "-" instead of a misleading 0.
    """
    answerable = [d for d in detailed_results if d.get("expected_answerable")]
    refusal = [d for d in detailed_results if not d.get("expected_answerable")]
    agentic_items = [d for d in detailed_results if d.get("reasoning_metrics", {}).get("agentic")]

    return {
        "exact_match": _avg([d["custom_metrics"]["exact_match"] for d in answerable]),
        "answer_recall": _avg([d["custom_metrics"]["answer_recall"] for d in answerable]),
        "refusal_accuracy": _avg([d["custom_metrics"]["refusal_accuracy"] for d in refusal]),
        "logical_soundness": _avg([d["reasoning_metrics"]["core"]["logical_soundness"] for d in detailed_results]),
        "synthesis_quality": _avg([d["reasoning_metrics"]["core"]["synthesis_quality"] for d in detailed_results]),
        "evidence_faithfulness": _avg([d["reasoning_metrics"]["core"]["evidence_faithfulness"] for d in detailed_results]),
        "tool_selection": (
            _avg([d["reasoning_metrics"]["agentic"]["tool_selection"] for d in agentic_items])
            if agentic_items else None
        ),
        "error_recovery": (
            _avg([d["reasoning_metrics"]["agentic"]["error_recovery"] for d in agentic_items])
            if agentic_items else None
        ),
        "total_tokens": _avg([d.get("run_metrics", {}).get("total_tokens") for d in detailed_results]),
        "latency_seconds": _avg([d.get("run_metrics", {}).get("latency_seconds") for d in detailed_results]),
    }


def write_summary_csv(results: dict, out_path: Path) -> None:
    fieldnames = [
        "system", "total_queries", "successful_queries",
        "ragas_faithfulness", "ragas_answer_relevancy",
        "ragas_context_precision", "ragas_context_recall",
        "ragas_answer_correctness", "ragas_composite_score",
        "exact_match", "answer_recall", "refusal_accuracy",
        "logical_soundness", "synthesis_quality", "evidence_faithfulness",
        "tool_selection", "error_recovery",
        "cross_document_success_rate", "citation_accuracy",
        "correction_rate", "avg_total_tokens", "avg_latency_seconds",
        "cost_per_correct_answer_usd", "total_cost_usd",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sys_label, res in results.items():
            s = res.get("summary", {})
            ragas = s.get("ragas_summary", {})
            extra = compute_extra_aggregates(res.get("detailed_results", []))
            writer.writerow({
                "system": sys_label,
                "total_queries": s.get("total_queries"),
                "successful_queries": s.get("successful_queries"),
                "ragas_faithfulness": ragas.get("faithfulness"),
                "ragas_answer_relevancy": ragas.get("answer_relevancy"),
                "ragas_context_precision": ragas.get("context_precision"),
                "ragas_context_recall": ragas.get("context_recall"),
                "ragas_answer_correctness": ragas.get("answer_correctness"),
                "ragas_composite_score": ragas.get("composite_score"),
                "exact_match": extra["exact_match"],
                "answer_recall": extra["answer_recall"],
                "refusal_accuracy": extra["refusal_accuracy"],
                "logical_soundness": extra["logical_soundness"],
                "synthesis_quality": extra["synthesis_quality"],
                "evidence_faithfulness": extra["evidence_faithfulness"],
                "tool_selection": extra["tool_selection"],
                "error_recovery": extra["error_recovery"],
                "cross_document_success_rate": s.get("cross_document_success_rate"),
                "citation_accuracy": s.get("citation_accuracy"),
                "correction_rate": s.get("correction_rate"),
                "avg_total_tokens": extra["total_tokens"],
                "avg_latency_seconds": extra["latency_seconds"],
                "cost_per_correct_answer_usd": s.get("cost_per_correct_answer_usd"),
                "total_cost_usd": s.get("total_cost_usd"),
            })
    logger.info("Summary CSV saved to %s", out_path)


def write_per_query_csv(results: dict, gold_items: list, out_path: Path) -> None:
    difficulty_by_id = {g.id: g.difficulty for g in gold_items}

    fieldnames = [
        "query_id", "fa_type", "difficulty", "expected_answerable", "system",
        "exact_match", "answer_recall", "refusal_accuracy",
        "citation_accuracy", "logical_soundness", "synthesis_quality",
        "evidence_faithfulness", "tool_selection", "error_recovery",
        "latency_seconds", "total_tokens", "estimated_cost_usd",
        "num_steps", "corrections",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sys_label, res in results.items():
            for d in res.get("detailed_results", []):
                custom = d.get("custom_metrics", {})
                citation = d.get("citation_metrics", {})
                reasoning = d.get("reasoning_metrics", {})
                core = reasoning.get("core", {})
                agentic = reasoning.get("agentic", {}) or {}
                run_metrics = d.get("run_metrics", {})
                writer.writerow({
                    "query_id": d.get("query_id"),
                    "fa_type": d.get("fa_type"),
                    "difficulty": difficulty_by_id.get(d.get("query_id"), ""),
                    "expected_answerable": d.get("expected_answerable"),
                    "system": sys_label,
                    "exact_match": custom.get("exact_match"),
                    "answer_recall": custom.get("answer_recall"),
                    "refusal_accuracy": custom.get("refusal_accuracy"),
                    "citation_accuracy": citation.get("citation_accuracy"),
                    "logical_soundness": core.get("logical_soundness"),
                    "synthesis_quality": core.get("synthesis_quality"),
                    "evidence_faithfulness": core.get("evidence_faithfulness"),
                    "tool_selection": agentic.get("tool_selection"),
                    "error_recovery": agentic.get("error_recovery"),
                    "latency_seconds": run_metrics.get("latency_seconds"),
                    "total_tokens": run_metrics.get("total_tokens"),
                    "estimated_cost_usd": run_metrics.get("estimated_cost_usd"),
                    "num_steps": run_metrics.get("num_steps"),
                    "corrections": run_metrics.get("corrections"),
                })
    logger.info("Per-query CSV saved to %s", out_path)


def write_markdown_report(results: dict, out_path: Path) -> None:
    labels = list(results.keys())
    extras = {lbl: compute_extra_aggregates(results[lbl].get("detailed_results", [])) for lbl in labels}

    def fmt(x, nd=3):
        return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "-"

    def row(name: str, getter) -> str:
        vals = [getter(results[lbl].get("summary", {})) for lbl in labels]
        return f"| {name} | " + " | ".join(vals) + " |"

    def extra_row(name: str, key: str, nd: int = 3) -> str:
        vals = [fmt(extras[lbl][key], nd) for lbl in labels]
        return f"| {name} | " + " | ".join(vals) + " |"

    lines = ["# Full Evaluation Report (n=150)", ""]
    lines.append("| Metric | " + " | ".join(labels) + " |")
    lines.append("|---|" + "---|" * len(labels))
    lines.append(row("Successful Queries", lambda s: f"{s.get('successful_queries')}/{s.get('total_queries')}"))

    lines.append("| **RAGAS** | | | | |")
    lines.append(row("RAGAS Composite", lambda s: fmt(s.get("ragas_summary", {}).get("composite_score"))))
    lines.append(row("RAGAS Context Precision", lambda s: fmt(s.get("ragas_summary", {}).get("context_precision"))))
    lines.append(row("RAGAS Context Recall", lambda s: fmt(s.get("ragas_summary", {}).get("context_recall"))))
    lines.append(row("RAGAS Faithfulness", lambda s: fmt(s.get("ragas_summary", {}).get("faithfulness"))))
    lines.append(row("RAGAS Answer Relevancy", lambda s: fmt(s.get("ragas_summary", {}).get("answer_relevancy"))))
    lines.append(row("RAGAS Answer Correctness", lambda s: fmt(s.get("ragas_summary", {}).get("answer_correctness"))))

    lines.append("| **Custom Metrics** | | | | |")
    lines.append(extra_row("Exact Match (answerable items)", "exact_match"))
    lines.append(extra_row("Answer Recall (answerable items)", "answer_recall"))
    lines.append(extra_row("Refusal Accuracy (refusal items)", "refusal_accuracy"))

    lines.append("| **Reasoning (LLM-Judge, 1-5)** | | | | |")
    lines.append(extra_row("Logical Soundness (core)", "logical_soundness", 2))
    lines.append(extra_row("Synthesis Quality (core)", "synthesis_quality", 2))
    lines.append(extra_row("Evidence Faithfulness (core)", "evidence_faithfulness", 2))
    lines.append(extra_row("Tool Selection (agentic, S2-S4)", "tool_selection", 2))
    lines.append(extra_row("Error Recovery (agentic, S2-S4)", "error_recovery", 2))

    lines.append("| **Process / Efficiency** | | | | |")
    lines.append(row("Cross-Doc Success", lambda s: fmt(s.get("cross_document_success_rate"))))
    lines.append(row("Citation Accuracy", lambda s: fmt(s.get("citation_accuracy"))))
    lines.append(row("Correction Rate", lambda s: fmt(s.get("correction_rate"))))
    lines.append(extra_row("Avg. Total Tokens / Query", "total_tokens", 0))
    lines.append(extra_row("Avg. Latency (s) / Query", "latency_seconds", 2))
    lines.append(row("Cost / Correct Answer (USD)", lambda s: fmt(s.get("cost_per_correct_answer_usd"), 4)))
    lines.append(row("Total Cost (USD)", lambda s: fmt(s.get("total_cost_usd"), 4)))
    lines.append("")

    from src.evaluation.eval_runner import SYSTEMS_WITHOUT_MEANINGFUL_CONTEXT_METRICS
    context_metrics_skipped = any(
        res.get("summary", {}).get("system_name") in SYSTEMS_WITHOUT_MEANINGFUL_CONTEXT_METRICS
        for res in results.values()
    )
    if context_metrics_skipped:
        lines.append(
            "> **Note (Long-Context-family systems):** RAGAS Context Precision, "
            "Context Recall and Faithfulness are shown as `-` for systems that "
            "inline the full document set into the prompt ahead of time instead "
            "of retrieving it at query time (no observable retrieval step for "
            "RAGAS to score against — see `SYSTEMS_WITHOUT_MEANINGFUL_CONTEXT_METRICS` "
            "in `src/evaluation/eval_runner.py`). They are not computed at all "
            "for these systems, not merely hidden: an earlier run showed this "
            "produces a structurally deflated score for a single-agent long-context "
            "system (near-empty `contexts`) and a structurally inflated score "
            "for a multi-agent one (`contexts` = an upstream agent's own generated "
            "answer, checked for self-consistency rather than grounding in the "
            "primary source). Comparable work handles this the same way — "
            "FinanceBench (Islam et al., 2023), Li et al. (2024, \"RAG or "
            "Long-Context LLMs?\"), and Lithgow-Serrano et al. (2025, FinDoc-RAG) "
            "all compare retrieval- and non-retrieval-based conditions purely on "
            "final-answer metrics, not on retrieval-specific context metrics. "
            "RAGAS Answer Relevancy and Answer Correctness are unaffected (they "
            "score the answer against the question/ground truth, not against "
            "`contexts`) and remain comparable across all systems. RAGAS "
            "Composite is therefore a 5-metric average for retrieval-based "
            "systems and a 3-metric average for these systems — the two are "
            "not on the same scale."
        )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Markdown report saved to %s", out_path)


if __name__ == "__main__":
    main()
