# Full Evaluation Report (n=150)

| Metric | S1 | S2 | S3 | S4 |
|---|---|---|---|---|
| Successful Queries | 150/150 | 150/150 | 150/150 | 150/150 |
| Failed Queries | 0 | 0 | 0 | 0 |
| Retried Queries | 0 | 0 | 0 | 0 |
| **RAGAS** | | | | |
| RAGAS Composite | 0.315 | 0.575 | 0.744 | 0.729 |
| RAGAS Context Precision | 0.155 | 0.503 | - | - |
| RAGAS Context Recall | 0.158 | 0.320 | - | - |
| RAGAS Faithfulness | 0.710 | 0.641 | - | - |
| RAGAS Answer Relevancy | 0.214 | 0.805 | 0.841 | 0.801 |
| RAGAS Answer Correctness | 0.335 | 0.605 | 0.647 | 0.656 |
| **Custom Metrics** | | | | |
| Exact Match (answerable items) | 0.217 | 0.821 | 0.867 | 0.933 |
| Answer Recall (answerable items) | 0.250 | 0.879 | 0.908 | 0.942 |
| Refusal Accuracy (refusal items) | 0.800 | 0.833 | 0.667 | 0.867 |
| Refusal Quality (refusal items, 0/0.5/1) | 0.650 | 0.900 | 0.783 | 0.650 |
| Over-Refusal Rate (answerable items) | 0.625 | 0.033 | 0.017 | 0.025 |
| **Reasoning (answerable items, per-unit)** | | | | |
| Chain Emission Rate | 0.992 | 0.967 | 0.883 | 0.992 |
| Groundedness (mean over steps) | 0.730 | 0.890 | 0.870 | 0.643 |
| Validity (mean over transitions) | 0.630 | 0.973 | 0.985 | 0.959 |
| Completeness (sub-question recall) | 0.306 | 0.972 | 0.972 | 0.854 |
| …chains with no groundedness defect | 0.645 | 0.793 | 0.783 | 0.573 |
| …chains with no validity defect | 0.547 | 0.957 | 0.954 | 0.918 |
| **Reasoning covariates** | | | | |
| Avg. Chain Steps | 4.43 | 4.71 | 4.54 | 3.62 |
| Avg. Evidential Steps | 2.78 | 2.72 | 2.49 | 2.07 |
| Avg. Inferential Steps | 1.65 | 1.99 | 2.05 | 1.55 |
| Untagged Steps (total) | 0 | 0 | 0 | 0 |
| **Process / Efficiency** | | | | |
| Cross-Doc Success (FA-4 + FA-3 cross_window) | 0.044 | 0.756 | 0.867 | 0.956 |
| Citation Accuracy | 0.554 | 0.794 | 0.793 | 0.831 |
| Correction Rate | 0.000 | 0.067 | 0.113 | 0.013 |
| Avg. Total Tokens / Query | 3121 | 17960 | 1381930 | 148762 |
| Avg. Latency (s) / Query | 3.95 | 10.88 | 41.64 | 13.19 |
| Cost / Correct Answer (USD) | 0.0043 | 0.0075 | 0.5023 | 0.0500 |
| Total Cost (USD) | 0.2147 | 0.9247 | 62.2795 | 6.8521 |


## FA-Refusal by subtype

Refusal Accuracy = did the system avoid fabricating an answer (binary). Refusal Quality = did it diagnose WHY the question fails (0 = answered as though it were sound, 0.5 = declined without a diagnosis, 1.0 = named the defect). n=10 per subtype: descriptive, no significance tests.

| Subtype | Metric | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| not_in_corpus | Accuracy | 0.90 | 1.00 | 0.60 | 1.00 |
| not_in_corpus | Quality | 0.85 | 1.00 | 0.65 | 0.65 |
| false_premise | Accuracy | 0.80 | 0.60 | 0.40 | 0.60 |
| false_premise | Quality | 0.60 | 0.85 | 0.80 | 0.75 |
| ambiguous_entity | Accuracy | 0.70 | 0.90 | 1.00 | 1.00 |
| ambiguous_entity | Quality | 0.50 | 0.85 | 0.90 | 0.55 |

> **Note (Long-Context-family systems):** RAGAS Context Precision, Context Recall and Faithfulness are shown as `-` for systems that inline the full document set into the prompt ahead of time instead of retrieving it at query time (no observable retrieval step for RAGAS to score against — see `SYSTEMS_WITHOUT_MEANINGFUL_CONTEXT_METRICS` in `src/evaluation/eval_runner.py`). They are not computed at all for these systems, not merely hidden: an earlier run showed this produces a structurally deflated score for a single-agent long-context system (near-empty `contexts`) and a structurally inflated score for a multi-agent one (`contexts` = an upstream agent's own generated answer, checked for self-consistency rather than grounding in the primary source). Comparable work handles this the same way — FinanceBench (Islam et al., 2023), Li et al. (2024, "RAG or Long-Context LLMs?"), and Lithgow-Serrano et al. (2025, FinDoc-RAG) all compare retrieval- and non-retrieval-based conditions purely on final-answer metrics, not on retrieval-specific context metrics. RAGAS Answer Relevancy and Answer Correctness are unaffected (they score the answer against the question/ground truth, not against `contexts`) and remain comparable across all systems. RAGAS Composite is therefore a 5-metric average for retrieval-based systems and a 2-metric average for these systems (`ANSWER_METRICS` in `src/evaluation/ragas_evaluator.py`) — the two are not on the same scale.

> **Note (RAGAS answer metrics):** Answer Relevancy and Answer Correctness are averaged over the answerable items only (`expected_answerable=True`, 120 of 150 in the current gold standard). RAGAS scores a correct refusal as "noncommittal" (Answer Relevancy 0) and the refusal items carry an empty ground truth, so including them would penalise exactly the behaviour the refusal stratum rewards — `Refusal Accuracy` measures it instead. The per-item scores in `full_eval_per_query.csv` are unfiltered; `answer_metrics_n` in each system's JSON summary records how many items the average covers.
