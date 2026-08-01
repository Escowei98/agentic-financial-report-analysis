import json
from pathlib import Path


def main():
    json_path = Path("data/results/mini_eval_full_evaluation.json")
    out_file = Path("data/results/mini_eval_full_report.md")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = data["summary"]
    all_results = data["detailed_results"]
    SYSTEMS = ["S1", "S2", "S3", "S4"]

    md_lines = []
    md_lines.append("# Mini-Eval: Vollständige Evaluierung (8 Queries × 4 Systeme)\n")
    md_lines.append("## Zusammenfassung\n")
    md_lines.append("| Metrik | S1 (Monolith) | S2 (Agent) | S3 (Long Context) | S4 (Multi-Agent) |")
    md_lines.append("|--------|---------------|------------|--------------------|--------------------|")
    md_lines.append(f"| **Korrekt / Gesamt** | {summary['S1']['correct']}/{summary['S1']['total']} | {summary['S2']['correct']}/{summary['S2']['total']} | {summary['S3']['correct']}/{summary['S3']['total']} | {summary['S4']['correct']}/{summary['S4']['total']} |")
    md_lines.append(f"| **Accuracy** | {summary['S1']['accuracy']:.0%} | {summary['S2']['accuracy']:.0%} | {summary['S3']['accuracy']:.0%} | {summary['S4']['accuracy']:.0%} |")
    md_lines.append(f"| **Avg. Exact Match** | {summary['S1']['avg_exact_match']} | {summary['S2']['avg_exact_match']} | {summary['S3']['avg_exact_match']} | {summary['S4']['avg_exact_match']} |")
    md_lines.append(f"| **Avg. Refusal Acc.** | {summary['S1']['avg_refusal_accuracy']} | {summary['S2']['avg_refusal_accuracy']} | {summary['S3']['avg_refusal_accuracy']} | {summary['S4']['avg_refusal_accuracy']} |")
    md_lines.append(f"| **Total Tokens** | {summary['S1']['total_tokens']:,} | {summary['S2']['total_tokens']:,} | {summary['S3']['total_tokens']:,} | {summary['S4']['total_tokens']:,} |")
    md_lines.append("")

    md_lines.append("## Detailergebnisse pro Query\n")

    for r in all_results:
        md_lines.append(f"### Query {r['id']} — {r['fa_type']} ({r['difficulty']})")
        md_lines.append(f"**Frage:** {r['question']}")
        md_lines.append(f"**Ground Truth:** {r['ground_truth']} {r['gt_unit']}")
        md_lines.append(f"**Expected Answerable:** {'Ja' if r['expected_answerable'] else 'Nein (Refusal erwartet)'}")
        md_lines.append(f"**Expected Tools:** {', '.join(r['expected_tools']) if r['expected_tools'] else 'Keine'}")
        md_lines.append("")

        md_lines.append("| | S1 (Monolith) | S2 (Agent) | S3 (Long Context) | S4 (Multi-Agent) |")
        md_lines.append("|---|---|---|---|---|")

        # Answer row
        answers = []
        for s in SYSTEMS:
            a = r["systems"][s]["answer"]
            a = a.replace("|", "\\|").replace("\n", " ")
            answers.append(a)
        md_lines.append(f"| **Antwort** | {answers[0]} | {answers[1]} | {answers[2]} | {answers[3]} |")

        # Tokens
        toks = [str(r["systems"][s]["total_tokens"]) for s in SYSTEMS]
        md_lines.append(f"| **Tokens** | {toks[0]} | {toks[1]} | {toks[2]} | {toks[3]} |")

        # Exact Match
        em = [str(r["systems"][s]["custom_metrics"].get("exact_match", "N/A")) for s in SYSTEMS]
        md_lines.append(f"| **Exact Match** | {em[0]} | {em[1]} | {em[2]} | {em[3]} |")

        # Answer Recall
        ar = [str(r["systems"][s]["custom_metrics"].get("answer_recall", "N/A")) for s in SYSTEMS]
        md_lines.append(f"| **Answer Recall** | {ar[0]} | {ar[1]} | {ar[2]} | {ar[3]} |")

        # Refusal Accuracy
        ra = [str(r["systems"][s]["custom_metrics"].get("refusal_accuracy", "N/A")) for s in SYSTEMS]
        md_lines.append(f"| **Refusal Accuracy** | {ra[0]} | {ra[1]} | {ra[2]} | {ra[3]} |")

        # Correct?
        corr = ["✅" if r["systems"][s]["is_correct"] else "❌" for s in SYSTEMS]
        md_lines.append(f"| **Korrekt?** | {corr[0]} | {corr[1]} | {corr[2]} | {corr[3]} |")

        # Rationales
        md_lines.append("")
        md_lines.append("<details>")
        md_lines.append("<summary>LLM-Judge Begründungen</summary>\n")
        for s in SYSTEMS:
            rats = r["systems"][s]["custom_metrics"].get("rationales", {})
            if rats:
                md_lines.append(f"**{s}:**")
                for metric_name, rationale in rats.items():
                    md_lines.append(f"- *{metric_name}*: {rationale}")
                md_lines.append("")
        md_lines.append("</details>\n")
        md_lines.append("---\n")

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Markdown generated at {out_file}")

if __name__ == '__main__':
    main()
