import json
from pathlib import Path


def _to_str(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [p.get("text", str(p)) if isinstance(p, dict) else str(p) for p in value]
        return "\n".join(parts)
    return str(value)

def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    out_dir = PROJECT_ROOT / "data" / "results" / "mini_eval_complete"
    out_path = out_dir / "mini_eval_complete_report.md"

    SYSTEMS = ["S1", "S2", "S3", "S4"]

    results = {}
    for sys in SYSTEMS:
        # find the json for this system
        json_files = list(out_dir.glob(f"eval_{sys}_*.json"))
        if not json_files:
            print(f"No json found for {sys}")
            continue
        # get the latest
        latest_json = sorted(json_files)[-1]
        with open(latest_json, "r", encoding="utf-8") as f:
            results[sys] = json.load(f)

    # Assuming mini_gs has the order of IDs we want
    MINI_EVAL_IDS = [85, 16, 21, 26, 45, 106, 59, 147, 124, 76]

    md = []
    md.append("# Mini-Eval: Complete Evaluation (Including RAGAS & Reasoning)\n")
    md.append("## Summary\n")
    md.append("| Metric | S1 (Monolith) | S2 (Agent) | S3 (Long Context) | S4 (Multi-Agent) |")
    md.append("|---|---|---|---|---|")

    row_acc, row_em, row_cost = [], [], []
    row_ragas_comp, row_ragas_prec, row_ragas_rec = [], [], []

    for sys in SYSTEMS:
        s = results.get(sys, {}).get("summary", {})
        det = results.get(sys, {}).get("detailed_results", [])

        # Calculate accuracy dynamically based on our correctness definition
        correct_count = 0
        for q_res in det:
            em = q_res.get("custom_metrics", {}).get("exact_match", 0)
            ar = q_res.get("custom_metrics", {}).get("answer_recall", 0)
            ra = q_res.get("custom_metrics", {}).get("refusal_accuracy", 0)
            if em >= 0.8 or ar >= 0.8 or ra >= 0.8:
                correct_count += 1

        acc = correct_count / max(len(det), 1)
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

    for q_id in MINI_EVAL_IDS:
        # Get item details from one of the systems (since they are identical)
        item = None
        for sys in SYSTEMS:
            det = results.get(sys, {}).get("detailed_results", [])
            q_res = next((d for d in det if d["query_id"] == q_id), None)
            if q_res:
                item = q_res
                break

        if not item:
            continue

        md.append(f"### Query {q_id} — {item.get('fa_type')} ")
        md.append(f"**Question:** {item.get('question')}")
        gt = item.get('ground_truth')
        if not gt:
            gt = "*N/A (Refusal expected)*"
        md.append(f"**Ground Truth:** {gt}")
        md.append("")

        md.append("| | S1 (Monolith) | S2 (Agent) | S3 (Long Context) | S4 (Multi-Agent) |")
        md.append("|---|---|---|---|---|")

        row_ans, row_tok, row_rea, row_cor, row_em, row_ar = [], [], [], [], [], []

        for sys in SYSTEMS:
            sys_res = results.get(sys, {})
            det = sys_res.get("detailed_results", [])
            q_res = next((d for d in det if d["query_id"] == q_id), None)

            if q_res:
                ans_raw = q_res.get("answer", "Error")
                ans_str = _to_str(ans_raw)
                ans = ans_str.replace("\n", " ").replace("|", "\\|")

                tok = q_res.get("run_metrics", {}).get("total_tokens", 0)
                reasoning = q_res.get("reasoning_metrics", {}).get("groundedness")

                # Try to find exact match
                em = q_res.get("custom_metrics", {}).get("exact_match", 0)
                ar = q_res.get("custom_metrics", {}).get("answer_recall", 0)
                ra = q_res.get("custom_metrics", {}).get("refusal_accuracy", 0)
                cor = "✅" if (em >= 0.8 or ar >= 0.8 or ra >= 0.8) else "❌"

                row_ans.append(ans)
                row_tok.append(str(tok))
                row_rea.append("—" if reasoning is None else f"{reasoning:.2f}")
                row_cor.append(cor)
                row_em.append(f"{em:.2f}")
                row_ar.append(f"{ar:.2f}")
            else:
                row_ans.append("-")
                row_tok.append("-")
                row_rea.append("-")
                row_cor.append("-")
                row_em.append("-")
                row_ar.append("-")

        md.append(f"| **Answer** | {' | '.join(row_ans)} |")
        md.append(f"| **Tokens** | {' | '.join(row_tok)} |")
        md.append(f"| **Exact Match** | {' | '.join(row_em)} |")
        md.append(f"| **Answer Recall** | {' | '.join(row_ar)} |")
        md.append(f"| **Logical Soundness** | {' | '.join(row_rea)} |")
        md.append(f"| **Correct?** | {' | '.join(row_cor)} |")

        # Add reasoning rationales
        md.append("")
        md.append("<details>")
        md.append("<summary>Evaluator Begründungen</summary>\n")
        for sys in SYSTEMS:
            det = results.get(sys, {}).get("detailed_results", [])
            q_res = next((d for d in det if d["query_id"] == q_id), None)
            if q_res:
                reasoning = q_res.get("reasoning_metrics", {})
                # Only the failing units. A clean chain has every step
                # grounded and every transition valid, so listing the passing
                # ones would bury the diagnosis under the common case.
                defects = [
                    f"Schritt {v['step']} nicht belegt ({v.get('code') or '?'}):"
                    f" {v.get('rationale', '')}"
                    for v in reasoning.get("step_verdicts", []) if not v.get("grounded")
                ] + [
                    f"Übergang auf {v['step']} nicht valide ({v.get('code') or '?'}):"
                    f" {v.get('rationale', '')}"
                    for v in reasoning.get("transition_verdicts", []) if not v.get("valid")
                ]
                if defects:
                    md.append(f"**{sys}:**")
                    md.extend(f"- {d}" for d in defects)
                    md.append("")
        md.append("</details>\n")
        md.append("---\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Report saved to {out_path}")

    # Save the full results dictionary as a single JSON file
    json_out_path = out_dir / "mini_eval_complete_results.json"
    with open(json_out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Full JSON results saved to {json_out_path}")

if __name__ == '__main__':
    main()
