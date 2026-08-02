"""
Runs the full evaluation pipeline for S1, S2, S3, S4 on the 10 mini-eval queries.
Generates individual JSON reports via eval_runner and a combined Markdown report.
"""
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def main():
    # 1. Load data
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    gs_path = PROJECT_ROOT / "data" / "gold_standard" / "gold_standard_v3_en.csv"
    filings = download_all_filings()

    # 2. Filter Gold Standard to the 10 mini-eval queries
    MINI_EVAL_IDS = [85, 16, 21, 26, 45, 106, 59, 147, 124, 76]
    all_gs = load_gold_standard(gs_path)
    mini_gs = [g for g in all_gs if g.id in MINI_EVAL_IDS]

    # Sort them to maintain consistent order
    id_to_gs = {g.id: g for g in mini_gs}
    mini_gs = [id_to_gs[i] for i in MINI_EVAL_IDS if i in id_to_gs]

    # 3. Load Configurations & Pipelines
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

    out_dir = PROJECT_ROOT / "data" / "results" / "mini_eval_complete"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4. Run evaluations
    systems = [
        ("S1", s1, "rag_monolith"),
        ("S2", s2, "rag_agent"),
        ("S3", s3, "long_context"),
        ("S4", s4, "multi_agent")
    ]

    results = {}
    for sys_label, pipeline, sys_name in systems:
        logger.info("=========================================")
        logger.info(f"Running full eval for {sys_label} ({sys_name})")
        logger.info("=========================================")
        # Note: eval_runner saves JSON with a timestamp. We capture the returned dict.
        res = run_full_evaluation(pipeline, sys_name, mini_gs, out_dir)
        results[sys_label] = res

    # 5. Generate Markdown Report
    generate_markdown_report(results, mini_gs, out_dir / "mini_eval_complete_report.md")

def generate_markdown_report(results: dict, mini_gs: list, out_path: Path):
    SYSTEMS = ["S1", "S2", "S3", "S4"]

    md = []
    md.append("# Mini-Eval: Complete Evaluation (Including RAGAS & Reasoning)\n")
    md.append("## Summary\n")
    md.append("| Metric | S1 (Monolith) | S2 (Agent) | S3 (Long Context) | S4 (Multi-Agent) |")
    md.append("|---|---|---|---|---|")

    row_acc, row_em, row_cost = [], [], []
    row_ragas_comp, row_ragas_prec, row_ragas_rec = [], [], []

    for sys in SYSTEMS:
        s = results.get(sys, {}).get("summary", {})
        detailed = results.get(sys, {}).get("detailed_results", [])

        # Calculate accuracy from detailed results (Exact Match >= 0.8 OR Answer Recall >= 0.8 OR Refusal Accuracy >= 0.8)
        if detailed:
            successes = sum(
                1 for d in detailed
                if d.get("custom_metrics", {}).get("exact_match", 0) >= 0.8
                or d.get("custom_metrics", {}).get("answer_recall", 0) >= 0.8
                or d.get("custom_metrics", {}).get("refusal_accuracy", 0) >= 0.8
            )
            acc = successes / len(detailed)
        else:
            acc = 0.0

        row_acc.append(f"{acc:.0%}")
        row_cost.append(f"${s.get('total_cost_usd', 0):.4f}")

        rs = s.get("ragas_summary", {})
        row_ragas_comp.append(f"{rs.get('composite_score', 0):.2f}")
        row_ragas_prec.append(f"{rs.get('context_precision', 0):.2f}")
        row_ragas_rec.append(f"{rs.get('context_recall', 0):.2f}")

    md.append(f"| **Accuracy** | {' | '.join(row_acc)} |")
    md.append(f"| **RAGAS Composite** | {' | '.join(row_ragas_comp)} |")
    md.append(f"| **Context Precision** | {' | '.join(row_ragas_prec)} |")
    md.append(f"| **Context Recall** | {' | '.join(row_ragas_rec)} |")
    md.append(f"| **Total Cost** | {' | '.join(row_cost)} |")
    md.append("\n## Detailed Results\n")

    # We map results by query id
    for item in mini_gs:
        q_id = item.id
        md.append(f"### Query {q_id} — {item.fa_type} ({item.difficulty})")
        md.append(f"**Question:** {item.question}")
        md.append(f"**Ground Truth:** {item.ground_truth}")
        md.append("")

        md.append("| | S1 (Monolith) | S2 (Agent) | S3 (Long Context) | S4 (Multi-Agent) |")
        md.append("|---|---|---|---|---|")

        row_ans, row_tok, row_rea, row_cor, row_em, row_ar = [], [], [], [], [], []

        for sys in SYSTEMS:
            # find item in system results
            sys_res = results.get(sys, {})
            det = sys_res.get("detailed_results", [])
            q_res = next((d for d in det if d["query_id"] == q_id), None)

            if q_res:
                ans = q_res.get("answer", "Error").replace("\n", " ")
                tok = q_res.get("run_metrics", {}).get("total_tokens", 0)

                custom = q_res.get("custom_metrics", {})
                em = custom.get("exact_match", 0)
                ar = custom.get("answer_recall", 0)

                reasoning = q_res.get("reasoning_metrics", {}).get("core", {})
                ls = reasoning.get("logical_soundness", 0)

                cor = "✅" if (em >= 0.8 or ar >= 0.8 or custom.get("refusal_accuracy", 0) >= 0.8) else "❌"

                row_ans.append(ans)
                row_tok.append(str(tok))
                row_em.append(f"{em:.2f}")
                row_ar.append(f"{ar:.2f}")
                row_rea.append(f"{ls}/5")
                row_cor.append(cor)
            else:
                row_ans.append("-")
                row_tok.append("-")
                row_em.append("-")
                row_ar.append("-")
                row_rea.append("-")
                row_cor.append("-")

        md.append(f"| **Answer** | {' | '.join(row_ans)} |")
        md.append(f"| **Tokens** | {' | '.join(row_tok)} |")
        md.append(f"| **Exact Match** | {' | '.join(row_em)} |")
        md.append(f"| **Answer Recall** | {' | '.join(row_ar)} |")
        md.append(f"| **Logical Soundness** | {' | '.join(row_rea)} |")
        md.append(f"| **Correct?** | {' | '.join(row_cor)} |")
        md.append("")

        md.append("<details>")
        md.append("<summary>Evaluator Begründungen</summary>")
        md.append("")

        for sys in SYSTEMS:
            sys_res = results.get(sys, {})
            det = sys_res.get("detailed_results", [])
            q_res = next((d for d in det if d["query_id"] == q_id), None)

            if q_res:
                md.append(f"**{sys}:**")

                custom = q_res.get("custom_metrics", {})
                rationales = custom.get("rationales", {})

                # Combine custom metrics rationales
                custom_rat = []
                for k, v in rationales.items():
                    if v:
                        custom_rat.append(f"**{k}**: {v.replace('`', '')}")
                if custom_rat:
                    md.append(f"- *Custom Metrics*: {' | '.join(custom_rat)}")

                reasoning = q_res.get("reasoning_metrics", {}).get("core", {})
                reasoning_rats = reasoning.get("rationales", {})

                ls_rat = reasoning_rats.get("logical_soundness", "")
                if ls_rat:
                    md.append(f"- *Logical Soundness*: {ls_rat.replace('`', '')}")

                # Add a blank line if we added any rationales for this system
                if custom_rat or ls_rat:
                    md.append("")

        md.append("</details>")
        md.append("")
        md.append("---")
        md.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    logger.info(f"Report saved to {out_path}")

if __name__ == '__main__':
    main()
