# Full Evaluation Report (n=150)

| Metric | S1 | S2 | S3 | S4 |
|---|---|---|---|---|
| Successful Queries | 150/150 | 150/150 | 150/150 | 150/150 |
| Failed Queries | 0 | 0 | 0 | 0 |
| Retried Queries | 1 | 0 | 19 | 0 |
| **RAGAS** | | | | |
| RAGAS Composite | 0.312 | 0.575 | 0.724 | 0.720 |
| RAGAS Context Precision | 0.163 | 0.485 | - | - |
| RAGAS Context Recall | 0.158 | 0.320 | - | - |
| RAGAS Faithfulness | 0.689 | 0.655 | - | - |
| RAGAS Answer Relevancy | 0.214 | 0.813 | 0.810 | 0.787 |
| RAGAS Answer Correctness | 0.336 | 0.604 | 0.637 | 0.653 |
| **Custom Metrics** | | | | |
| Exact Match (answerable items) | 0.217 | 0.817 | 0.842 | 0.925 |
| Answer Recall (answerable items) | 0.254 | 0.875 | 0.850 | 0.929 |
| Refusal Accuracy (refusal items) | 0.833 | 0.833 | 0.633 | 0.833 |
| Refusal Quality (refusal items, 0/0.5/1) | 0.650 | 0.817 | 0.683 | 0.600 |
| Over-Refusal Rate (answerable items) | 0.642 | 0.033 | 0.042 | 0.025 |
| **Reasoning (answerable items, per-unit)** | | | | |
| Chain Emission Rate | 0.992 | 0.975 | 0.875 | 1.000 |
| Groundedness (mean over steps) | 0.728 | 0.891 | 0.872 | 0.655 |
| Validity (mean over transitions) | 0.676 | 0.991 | 0.994 | 0.959 |
| Completeness (sub-question recall) | 0.306 | 0.974 | 0.971 | 0.863 |
| …chains with no groundedness defect | 0.638 | 0.786 | 0.808 | 0.573 |
| …chains with no validity defect | 0.607 | 0.979 | 0.988 | 0.933 |
| **Reasoning covariates** | | | | |
| Avg. Chain Steps | 4.37 | 4.70 | 4.26 | 3.76 |
| Avg. Evidential Steps | 2.72 | 2.74 | 2.45 | 2.12 |
| Avg. Inferential Steps | 1.65 | 1.97 | 1.81 | 1.63 |
| Untagged Steps (total) | 0 | 0 | 0 | 0 |
| **Process / Efficiency** | | | | |
| Cross-Doc Success (FA-4 + FA-3 cross_window) | 0.044 | 0.800 | 0.822 | 0.933 |
| Citation Accuracy | 0.558 | 0.807 | 0.767 | 0.827 |
| Correction Rate | 0.000 | 0.087 | 0.260 | 0.007 |
| Avg. Total Tokens / Query | 3123 | 18411 | 1382159 | 145606 |
| Avg. Latency (s) / Query | 4.57 | 29.57 | 61.45 | 14.78 |
| Cost / Correct Answer (USD) | 0.0042 | 0.0077 | 0.5242 | 0.0497 |
| Total Cost (USD) | 0.2139 | 0.9448 | 62.3748 | 6.7074 |


## FA-Refusal by subtype

Refusal Accuracy = did the system avoid fabricating an answer (binary). Refusal Quality = did it diagnose WHY the question fails (0 = answered as though it were sound, 0.5 = declined without a diagnosis, 1.0 = named the defect). n=10 per subtype: descriptive, no significance tests.

| Subtype | Metric | S1 | S2 | S3 | S4 |
|---|---|---|---|---|---|
| not_in_corpus | Accuracy | 0.90 | 1.00 | 0.70 | 0.90 |
| not_in_corpus | Quality | 0.80 | 1.00 | 0.65 | 0.60 |
| false_premise | Accuracy | 0.90 | 0.60 | 0.40 | 0.60 |
| false_premise | Quality | 0.65 | 0.65 | 0.75 | 0.65 |
| ambiguous_entity | Accuracy | 0.70 | 0.90 | 0.80 | 1.00 |
| ambiguous_entity | Quality | 0.50 | 0.80 | 0.65 | 0.55 |

> **Note (Long-Context-family systems):** RAGAS Context Precision, Context Recall and Faithfulness are shown as `-` for systems that inline the full document set into the prompt ahead of time instead of retrieving it at query time (no observable retrieval step for RAGAS to score against — see `SYSTEMS_WITHOUT_MEANINGFUL_CONTEXT_METRICS` in `src/evaluation/eval_runner.py`). They are not computed at all for these systems, not merely hidden: an earlier run showed this produces a structurally deflated score for a single-agent long-context system (near-empty `contexts`) and a structurally inflated score for a multi-agent one (`contexts` = an upstream agent's own generated answer, checked for self-consistency rather than grounding in the primary source). Comparable work handles this the same way — FinanceBench (Islam et al., 2023), Li et al. (2024, "RAG or Long-Context LLMs?"), and Lithgow-Serrano et al. (2025, FinDoc-RAG) all compare retrieval- and non-retrieval-based conditions purely on final-answer metrics, not on retrieval-specific context metrics. RAGAS Answer Relevancy and Answer Correctness are unaffected (they score the answer against the question/ground truth, not against `contexts`) and remain comparable across all systems. RAGAS Composite is therefore a 5-metric average for retrieval-based systems and a 2-metric average for these systems (`ANSWER_METRICS` in `src/evaluation/ragas_evaluator.py`) — the two are not on the same scale.

> **Note (RAGAS answer metrics):** Answer Relevancy and Answer Correctness are averaged over the answerable items only (`expected_answerable=True`, 120 of 150 in the current gold standard). RAGAS scores a correct refusal as "noncommittal" (Answer Relevancy 0) and the refusal items carry an empty ground truth, so including them would penalise exactly the behaviour the refusal stratum rewards — `Refusal Accuracy` measures it instead. The per-item scores in `full_eval_per_query.csv` are unfiltered; `answer_metrics_n` in each system's JSON summary records how many items the average covers.
