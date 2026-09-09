"""
Runs the full evaluation pipeline for S1, S2, S3, S4 on the complete
n=150 gold standard (see GOLD_STANDARD_EN in
src/evaluation/gold_standard_loader.py).

Saves per-system JSON via eval_runner (full detail, incl. rationales and
trajectories) plus two aggregate CSVs for cross-system weakness analysis:
- full_eval_summary.csv: one row per system (RAGAS, custom, citation,
  process, cost/latency aggregates)
- full_eval_per_query.csv: one row per (system, query) with the key
  scoring dimensions, for slicing by fa_type/difficulty/system to spot
  systematic failure patterns.
"""
import argparse
import csv
import logging
from pathlib import Path

from src.common.config import load_config
from src.common.ingestion import download_all_filings, fiscal_year_from_metadata
from src.evaluation.eval_runner import run_full_evaluation
from src.evaluation.evidence_store import EvidenceStore
from src.evaluation.gold_standard_loader import GOLD_STANDARD_EN, load_gold_standard
from src.evaluation.judge_validation_sample import JUDGE_VALIDATION_IDS
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", default="data/results/full_eval_n150",
        help="Where to write the run. Was hard-wired until 2026-09-08; a "
             "rerun that lands on top of a previous run's directory silently "
             "mixes two runs' JSONs, and _latest_eval_json downstream then "
             "picks whichever sorted last.",
    )
    args = parser.parse_args()

    gs_path = GOLD_STANDARD_EN
    filings = download_all_filings()
    logger.info(
        "Corpus: %d filings — %s",
        len(filings),
        ", ".join(sorted(f"{f.metadata.ticker} FY{fiscal_year_from_metadata(f)}" for f in filings)),
    )

    gold_items = load_gold_standard(gs_path)
    logger.info("Loaded %d gold standard items", len(gold_items))

    logger.info("Initializing pipelines...")
    # load_config() takes a SYSTEM NAME, not a path — it resolves
    # configs/<name>.yaml itself and merges it over base.yaml. Passing a Path
    # made it look for a non-existent "<abs path>.yaml" and silently fall back
    # to base.yaml alone, which left S1 and S2 on diverging built-in defaults
    # (chunk_size 1000 vs. 2000) and would have broken the shared retrieval
    # stack that FP-3 requires.
    s1 = MonolithRAGPipeline(load_config("rag_monolith"))
    s2 = AgentRAGPipeline(load_config("rag_agent"))
    s3 = LongContextPipeline(load_config("long_context"))
    s4 = MultiAgentPipeline(load_config("multi_agent"))

    s1.build(filings)
    s2.build(filings)
    s3.build(filings)
    s4.build(filings)

    # Corpus access for the groundedness dimension. Built once, shared by all
    # four systems -- that sharing IS the fairness property: every step is
    # verified against passages fetched by one procedure from one store,
    # whatever the system's own retrieval did.
    evidence_store = EvidenceStore(filings)

    out_dir = PROJECT_ROOT / args.out_dir
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
        res = run_full_evaluation(
            pipeline, sys_name, gold_items, out_dir,
            evidence_store=evidence_store,
        )
        results[sys_label] = res
        logger.info("%s summary: %s", sys_label, res.get("summary", {}))

    write_summary_csv(results, out_dir / "full_eval_summary.csv")
    write_per_query_csv(results, gold_items, out_dir / "full_eval_per_query.csv")
    write_markdown_report(results, out_dir / "full_eval_report.md")
    logger.info("Full evaluation complete. Outputs in %s", out_dir)


def _avg(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _as_float(flag: bool | None) -> float | None:
    """Booleans into a form _avg can average, keeping None as missing."""
    return None if flag is None else float(flag)


def compute_extra_aggregates(detailed_results: list) -> dict:
    """
    Aggregate metrics not already present in eval_runner's summary dict.

    Custom metrics (exact_match/answer_recall/refusal_accuracy/
    refusal_quality/over_refusal) are only
    ever computed by evaluate_custom_metrics() for a subset of items —
    exact_match/answer_recall only when expected_answerable=True,
    refusal_accuracy only when expected_answerable=False (see
    src/evaluation/custom_evaluator.py::evaluate_custom_metrics). Averaging
    over all 150 items would dilute each score with 0.0-default values from
    items where the metric was never computed, so we filter to the subset
    it actually applies to — the same convention eval_runner.py already
    uses for citation_accuracy/correction_rate.

    The three reasoning dimensions follow the same rule for the same reason:
    they are defined on the answerable stratum only (REASONING_QUALITY_SPEC.md
    section 8.4), and an item whose chain was never emitted carries None
    rather than 0.0 -- a format failure is not a reasoning defect and must not
    be averaged in as one. `_avg` drops Nones, so both cases fall out of the
    denominator instead of deflating it.
    """
    answerable = [d for d in detailed_results if d.get("expected_answerable")]
    refusal = [d for d in detailed_results if not d.get("expected_answerable")]
    reasoning = [d.get("reasoning_metrics", {}) for d in answerable]
    scored = [r for r in reasoning if r.get("chain_emitted")]

    return {
        "exact_match": _avg([d["custom_metrics"]["exact_match"] for d in answerable]),
        "answer_recall": _avg([d["custom_metrics"]["answer_recall"] for d in answerable]),
        "refusal_accuracy": _avg([d["custom_metrics"]["refusal_accuracy"] for d in refusal]),
        "refusal_quality": _avg(
            [d["custom_metrics"].get("refusal_quality") for d in refusal]
        ),
        # Share of ANSWERABLE items the system declined instead of attempting.
        # The counterweight to refusal_accuracy, which a system that declines
        # everything would otherwise score a perfect 1.0 on. Items scored fully
        # correct are recorded as 0.0 without a judge call, so the denominator
        # is the whole answerable stratum, not just the failures.
        "over_refusal_rate": _avg(
            [d["custom_metrics"].get("over_refusal") for d in answerable]
        ),
        # Share of answerable items where a parseable chain came back at all.
        # Reported per system because an uneven rate is itself a finding: it
        # would mean the three dimensions below are averaged over differently
        # sized, and possibly differently selected, sets of items.
        "chain_emission_rate": (
            round(len(scored) / len(answerable), 4) if answerable else None
        ),
        "groundedness": _avg([r.get("groundedness") for r in scored]),
        "validity": _avg([r.get("validity") for r in scored]),
        "completeness": _avg([r.get("completeness") for r in scored]),
        # Weakest-link aggregation (Jacovi et al. 2024): the share of chains
        # with NO violation at all. Reported next to the means because the two
        # respond differently to chain length, which is exactly what differs
        # between a monolith and a multi-agent system.
        "fully_grounded_rate": _avg(
            [_as_float(r.get("fully_grounded")) for r in scored]
        ),
        "fully_valid_rate": _avg([_as_float(r.get("fully_valid")) for r in scored]),
        "avg_chain_steps": _avg([r.get("num_steps") for r in scored]),
        "avg_evidential_steps": _avg([r.get("num_evidential") for r in scored]),
        "avg_inferential_steps": _avg([r.get("num_inferential") for r in scored]),
        "untagged_steps": sum(r.get("num_untagged") or 0 for r in scored),
        "total_tokens": _avg([d.get("run_metrics", {}).get("total_tokens") for d in detailed_results]),
        "latency_seconds": _avg([d.get("run_metrics", {}).get("latency_seconds") for d in detailed_results]),
    }


def write_summary_csv(results: dict, out_path: Path) -> None:
    fieldnames = [
        "system", "total_queries", "successful_queries", "failed_queries",
        "ragas_faithfulness", "ragas_answer_relevancy",
        "ragas_context_precision", "ragas_context_recall",
        "ragas_answer_correctness", "ragas_composite_score",
        "exact_match", "answer_recall", "refusal_accuracy",
        "refusal_quality", "over_refusal_rate",
        "chain_emission_rate", "groundedness", "validity", "completeness",
        "fully_grounded_rate", "fully_valid_rate",
        "avg_chain_steps", "avg_evidential_steps", "avg_inferential_steps",
        "untagged_steps",
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
                "failed_queries": s.get("failed_queries", 0),
                "ragas_faithfulness": ragas.get("faithfulness"),
                "ragas_answer_relevancy": ragas.get("answer_relevancy"),
                "ragas_context_precision": ragas.get("context_precision"),
                "ragas_context_recall": ragas.get("context_recall"),
                "ragas_answer_correctness": ragas.get("answer_correctness"),
                "ragas_composite_score": ragas.get("composite_score"),
                "exact_match": extra["exact_match"],
                "answer_recall": extra["answer_recall"],
                "refusal_accuracy": extra["refusal_accuracy"],
                "refusal_quality": extra["refusal_quality"],
                "over_refusal_rate": extra["over_refusal_rate"],
                "chain_emission_rate": extra["chain_emission_rate"],
                "groundedness": extra["groundedness"],
                "validity": extra["validity"],
                "completeness": extra["completeness"],
                "fully_grounded_rate": extra["fully_grounded_rate"],
                "fully_valid_rate": extra["fully_valid_rate"],
                "avg_chain_steps": extra["avg_chain_steps"],
                "avg_evidential_steps": extra["avg_evidential_steps"],
                "avg_inferential_steps": extra["avg_inferential_steps"],
                "untagged_steps": extra["untagged_steps"],
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
        "query_id", "fa_type", "subtype", "refusal_evidence", "difficulty",
        "expected_answerable",
        # True for the 34 ids the judge was validated and tuned on. Robustness
        # checks in chapter 5 filter on this column: a verdict that holds on
        # the remaining 116 items does not rest on items the judge was fitted
        # to. See src/evaluation/judge_validation_sample.py.
        "in_judge_validation_sample",
        "system",
        "exact_match", "answer_recall", "refusal_accuracy",
        "refusal_quality", "over_refusal",
        "citation_accuracy",
        "chain_emitted", "groundedness", "validity", "completeness",
        "fully_grounded", "fully_valid",
        "chain_steps", "evidential_steps", "inferential_steps",
        # Per-item RAGAS scores (see eval_runner.py). The three context
        # metrics stay empty for the systems without an observable
        # retrieval step; a NaN from RAGAS is written as an empty cell too.
        "ragas_answer_relevancy", "ragas_answer_correctness",
        "ragas_context_precision", "ragas_context_recall",
        "ragas_faithfulness",
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
                run_metrics = d.get("run_metrics", {})
                ragas = d.get("ragas_metrics", {})
                writer.writerow({
                    "query_id": d.get("query_id"),
                    "fa_type": d.get("fa_type"),
                    "subtype": d.get("subtype"),
                    "refusal_evidence": d.get("refusal_evidence"),
                    "difficulty": difficulty_by_id.get(d.get("query_id"), ""),
                    "expected_answerable": d.get("expected_answerable"),
                    "in_judge_validation_sample": d.get("query_id") in JUDGE_VALIDATION_IDS,
                    "system": sys_label,
                    "exact_match": custom.get("exact_match"),
                    "answer_recall": custom.get("answer_recall"),
                    "refusal_accuracy": custom.get("refusal_accuracy"),
                    "refusal_quality": custom.get("refusal_quality"),
                    "over_refusal": custom.get("over_refusal"),
                    "citation_accuracy": citation.get("citation_accuracy"),
                    "chain_emitted": reasoning.get("chain_emitted"),
                    "groundedness": reasoning.get("groundedness"),
                    "validity": reasoning.get("validity"),
                    "completeness": reasoning.get("completeness"),
                    "fully_grounded": reasoning.get("fully_grounded"),
                    "fully_valid": reasoning.get("fully_valid"),
                    "chain_steps": reasoning.get("num_steps"),
                    "evidential_steps": reasoning.get("num_evidential"),
                    "inferential_steps": reasoning.get("num_inferential"),
                    "ragas_answer_relevancy": ragas.get("answer_relevancy"),
                    "ragas_answer_correctness": ragas.get("answer_correctness"),
                    "ragas_context_precision": ragas.get("context_precision"),
                    "ragas_context_recall": ragas.get("context_recall"),
                    "ragas_faithfulness": ragas.get("faithfulness"),
                    "latency_seconds": run_metrics.get("latency_seconds"),
                    "total_tokens": run_metrics.get("total_tokens"),
                    "estimated_cost_usd": run_metrics.get("estimated_cost_usd"),
                    "num_steps": run_metrics.get("num_steps"),
                    "corrections": run_metrics.get("corrections"),
                })
    logger.info("Per-query CSV saved to %s", out_path)



_REFUSAL_SUBTYPES = ("not_in_corpus", "false_premise", "ambiguous_entity")


def _refusal_subtype_section(results: dict, labels: list) -> list[str]:
    """Break the FA-Refusal stratum down by subtype.

    The aggregate refusal_accuracy hides the finding this stratum was built
    for: the three subtypes are three different defect classes, and a system
    can be perfect at declining out-of-corpus questions while never noticing a
    false premise. Reported per subtype so that difference is visible.

    Ten items per subtype: descriptive only. The power analysis behind the
    design (thesis 3.6.1) justifies n=30 per stratum for the McNemar
    comparisons, not n=10 per subtype -- no significance test belongs on these
    cells.
    """
    lines = ["", "## FA-Refusal by subtype", ""]
    lines.append(
        "Refusal Accuracy = did the system avoid fabricating an answer "
        "(binary). Refusal Quality = did it diagnose WHY the question fails "
        "(0 = answered as though it were sound, 0.5 = declined without a "
        "diagnosis, 1.0 = named the defect). n=10 per subtype: descriptive, "
        "no significance tests."
    )
    lines.append("")
    lines.append("| Subtype | Metric | " + " | ".join(labels) + " |")
    lines.append("|---|---|" + "---|" * len(labels))
    for subtype in _REFUSAL_SUBTYPES:
        for metric, nice in (("refusal_accuracy", "Accuracy"), ("refusal_quality", "Quality")):
            cells = []
            for lbl in labels:
                items = [
                    d for d in results[lbl].get("detailed_results", [])
                    if d.get("subtype") == subtype
                ]
                value = _avg([d.get("custom_metrics", {}).get(metric) for d in items])
                cells.append(f"{value:.2f}" if isinstance(value, (int, float)) else "-")
            lines.append(f"| {subtype} | {nice} | " + " | ".join(cells) + " |")
    lines.append("")
    return lines


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
    lines.append(row("Failed Queries", lambda s: str(s.get("failed_queries", 0))))

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
    lines.append(extra_row("Refusal Quality (refusal items, 0/0.5/1)", "refusal_quality"))
    lines.append(extra_row("Over-Refusal Rate (answerable items)", "over_refusal_rate"))

    # Three values, reported separately and never combined into one. The
    # systems are expected to differ in their PROFILE across the three, which
    # a composite would average away — and there is no defensible weighting
    # between grounding a claim, drawing a valid inference, and covering the
    # question. See REASONING_QUALITY_SPEC.md section 10.2.
    lines.append("| **Reasoning (answerable items, per-unit)** | | | | |")
    lines.append(extra_row("Chain Emission Rate", "chain_emission_rate"))
    lines.append(extra_row("Groundedness (mean over steps)", "groundedness"))
    lines.append(extra_row("Validity (mean over transitions)", "validity"))
    lines.append(extra_row("Completeness (sub-question recall)", "completeness"))
    # Weakest-link aggregation beside the means (Jacovi et al. 2024): a chain
    # is only as strong as its worst step. The two respond differently to
    # chain length, which is precisely what separates a monolith from a
    # multi-agent system, so reporting one without the other would hide the
    # effect rather than measure it.
    lines.append(extra_row("…chains with no groundedness defect", "fully_grounded_rate"))
    lines.append(extra_row("…chains with no validity defect", "fully_valid_rate"))
    lines.append("| **Reasoning covariates** | | | | |")
    lines.append(extra_row("Avg. Chain Steps", "avg_chain_steps", 2))
    lines.append(extra_row("Avg. Evidential Steps", "avg_evidential_steps", 2))
    lines.append(extra_row("Avg. Inferential Steps", "avg_inferential_steps", 2))
    lines.append(extra_row("Untagged Steps (total)", "untagged_steps", 0))

    lines.append("| **Process / Efficiency** | | | | |")
    lines.append(row("Cross-Doc Success (FA-4 + FA-3 cross_window)",
                     lambda s: fmt(s.get("cross_document_success_rate"))))
    lines.append(row("Citation Accuracy", lambda s: fmt(s.get("citation_accuracy"))))
    lines.append(row("Correction Rate", lambda s: fmt(s.get("correction_rate"))))
    lines.append(extra_row("Avg. Total Tokens / Query", "total_tokens", 0))
    lines.append(extra_row("Avg. Latency (s) / Query", "latency_seconds", 2))
    lines.append(row("Cost / Correct Answer (USD)", lambda s: fmt(s.get("cost_per_correct_answer_usd"), 4)))
    lines.append(row("Total Cost (USD)", lambda s: fmt(s.get("total_cost_usd"), 4)))
    lines.append("")

    lines.extend(_refusal_subtype_section(results, labels))

    if any(res.get("summary", {}).get("failed_queries", 0) for res in results.values()):
        lines.append(
            "> **Note (failed queries):** averages above are taken over the "
            "successful queries only — a failed query has no result object for "
            "the judge or RAGAS to score. Where the counts differ between "
            "systems, the averages are not on the same base; the failed ids "
            "are listed per system in the JSON under `summary.failed_query_ids`."
        )
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
            "systems and a 2-metric average for these systems (`ANSWER_METRICS` "
            "in `src/evaluation/ragas_evaluator.py`) — the two are "
            "not on the same scale."
        )
        lines.append("")

    lines.append(
        "> **Note (RAGAS answer metrics):** Answer Relevancy and Answer "
        "Correctness are averaged over the answerable items only "
        "(`expected_answerable=True`, 120 of 150 in the current gold "
        "standard). RAGAS scores a correct refusal as \"noncommittal\" "
        "(Answer Relevancy 0) and the refusal items carry an empty ground "
        "truth, so including them would penalise exactly the behaviour the "
        "refusal stratum rewards — `Refusal Accuracy` measures it instead. "
        "The per-item scores in `full_eval_per_query.csv` are unfiltered; "
        "`answer_metrics_n` in each system's JSON summary records how many "
        "items the average covers."
    )
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Markdown report saved to %s", out_path)


if __name__ == "__main__":
    main()
