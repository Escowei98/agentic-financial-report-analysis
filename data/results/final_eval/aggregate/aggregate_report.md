# Aggregated evaluation across runs

Runs: run1, run2, run3 (n=3). Cells are mean ± sample SD across runs; a single run has no SD.

RAGAS metrics are averaged over the **120 answerable items**. The `_all150` columns in the CSVs carry the values as eval_runner reported them, over all 150 items including the 30 FA-Refusal items, where a 0 is the correct outcome rather than a defect. Refusal behaviour is reported separately by refusal_accuracy, refusal_quality and over_refusal_rate.

**avg_latency_seconds excludes run1.** Those runs executed against the `global` Vertex endpoint and spent much of their wall time in quota backoff, which `perf_counter` counts as latency; run1's S2 was additionally inflated by the machine suspending. Every other metric in this report uses all 3 run(s) — see the `latency_runs` column in aggregate_summary.csv for which runs each latency figure actually covers.

> **The `composite` row is not comparable across systems.** S3 and S4 perform no chunk retrieval, so context_precision, context_recall and faithfulness do not exist for them and their composite averages 2 metrics where S1's and S2's averages 5. Fewer metrics is not a better system. For a cross-system statement use `composite (answer-only)`, which is the mean of answer_relevancy and answer_correctness for all four, or compare the individual metrics.

## RAGAS (answerable stratum, n=120)

| metric | S1 | S2 | S3 | S4 |
|---|---|---|---|---|
| context_precision | 0.1951 ± 0.0056 | 0.6149 ± 0.0123 | — | — |
| context_recall | 0.1639 ± 0.0096 | 0.3361 ± 0.0127 | — | — |
| faithfulness | 0.7162 ± 0.0181 | 0.6942 ± 0.0084 | — | — |
| answer_relevancy | 0.2140 ± 0.0001 | 0.8127 ± 0.0073 | 0.8276 ± 0.0159 | 0.7988 ± 0.0108 |
| answer_correctness | 0.3355 ± 0.0005 | 0.6100 ± 0.0097 | 0.6442 ± 0.0061 | 0.6567 ± 0.0043 |
| composite (answer-only, comparable) | 0.2747 ± 0.0002 | 0.7114 ± 0.0082 | 0.7359 ± 0.0108 | 0.7278 ± 0.0075 |
| composite (all available metrics) | 0.3249 ± 0.0033 | 0.6136 ± 0.0037 | 0.7359 ± 0.0108 | 0.7278 ± 0.0075 |
| └ metrics averaged | 5.0 ± 0.0 | 5.0 ± 0.0 | 2.0 ± 0.0 | 2.0 ± 0.0 |

## Faithfulness (answer vs. corpus at cited loci, all four systems; post hoc, exploratory)

| metric | S1 | S2 | S3 | S4 |
|---|---|---|---|---|
| locus_faithfulness | 0.6868 ± 0.0071 | 0.5586 ± 0.0035 | 0.5334 ± 0.0379 | 0.4910 ± 0.0107 |
| └ coverage (share of answerable items scored) | 0.6111 ± 0.0048 | 0.9472 ± 0.0127 | 0.9056 ± 0.0096 | 0.9667 ± 0.0084 |

## Answer and refusal quality

| metric | S1 | S2 | S3 | S4 |
|---|---|---|---|---|
| exact_match | 0.2167 ± 0.0000 | 0.8250 ± 0.0110 | 0.8528 ± 0.0127 | 0.9305 ± 0.0048 |
| answer_recall | 0.2500 ± 0.0042 | 0.8833 ± 0.0110 | 0.8861 ± 0.0315 | 0.9403 ± 0.0105 |
| refusal_accuracy | 0.8111 ± 0.0192 | 0.8222 ± 0.0192 | 0.6444 ± 0.0193 | 0.8444 ± 0.0193 |
| refusal_quality | 0.6500 ± 0.0000 | 0.8667 ± 0.0441 | 0.7389 ± 0.0509 | 0.6222 ± 0.0255 |
| over_refusal_rate | 0.6361 ± 0.0096 | 0.0305 ± 0.0048 | 0.0278 ± 0.0127 | 0.0250 ± 0.0000 |

## Reasoning chains

| metric | S1 | S2 | S3 | S4 |
|---|---|---|---|---|
| chain_emission_rate | 0.9917 ± 0.0000 | 0.9750 ± 0.0083 | 0.8750 ± 0.0083 | 0.9945 ± 0.0048 |
| groundedness | 0.7282 ± 0.0015 | 0.8898 ± 0.0014 | 0.8701 ± 0.0021 | 0.6516 ± 0.0073 |
| validity | 0.6472 ± 0.0252 | 0.9848 ± 0.0099 | 0.9837 ± 0.0107 | 0.9553 ± 0.0062 |
| completeness | 0.3041 ± 0.0035 | 0.9727 ± 0.0013 | 0.9746 ± 0.0053 | 0.8562 ± 0.0064 |
| fully_grounded_rate | 0.6415 ± 0.0034 | 0.7864 ± 0.0067 | 0.7963 ± 0.0125 | 0.5755 ± 0.0050 |
| fully_valid_rate | 0.5704 ± 0.0324 | 0.9719 ± 0.0126 | 0.9649 ± 0.0198 | 0.9275 ± 0.0085 |
| avg_chain_steps | 4.4090 ± 0.0340 | 4.7292 ± 0.0439 | 4.4060 ± 0.1411 | 3.6729 ± 0.0744 |

## Task and cost

| metric | S1 | S2 | S3 | S4 |
|---|---|---|---|---|
| cross_document_success_rate | 0.0444 ± 0.0000 | 0.7778 ± 0.0222 | 0.8296 ± 0.0340 | 0.9556 ± 0.0223 |
| citation_accuracy | 0.5558 ± 0.0024 | 0.8035 ± 0.0082 | 0.7827 ± 0.0135 | 0.8316 ± 0.0047 |
| correction_rate | 0.0000 ± 0.0000 | 0.0689 ± 0.0168 | 0.1844 ± 0.0735 | 0.0067 ± 0.0067 |
| failed_queries | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| avg_total_tokens | 3121.3511 ± 1.4049 | 18099.9911 ± 269.6163 | 1389673.6533 ± 13214.4788 | 147751.6645 ± 1859.3078 |
| avg_latency_seconds (ohne run1) | 3.9402 ± 0.0155 | 10.4993 ± 0.5312 | 42.7473 ± 1.5691 | 13.0753 ± 0.1684 |
| total_cost_usd | 0.2144 ± 0.0005 | 0.9306 ± 0.0123 | 62.6641 ± 0.5856 | 6.8056 ± 0.0851 |

## Per-run detail

| run | system | source file | items | answerable | ragas composite (120) | ragas composite (150) | failed |
|---|---|---|---|---|---|---|---|
| run1 | S1 | eval_rag_monolith_20260909_200731.json | 150 | 120 | 0.3212 | 0.3122 | 0 |
| run1 | S2 | eval_rag_agent_20260909_205041.json | 150 | 120 | 0.6104 | 0.5754 | 0 |
| run1 | S3 | eval_long_context_MERGED.json | 150 | 120 | 0.7237 | 0.7237 | 0 |
| run1 | S4 | eval_multi_agent_20260910_140814.json | 150 | 120 | 0.7199 | 0.7199 | 0 |
| run2 | S1 | eval_rag_monolith_20260910_164157.json | 150 | 120 | 0.3276 | 0.3168 | 0 |
| run2 | S2 | eval_rag_agent_20260910_172051.json | 150 | 120 | 0.6177 | 0.5817 | 0 |
| run2 | S3 | eval_long_context_20260910_181000.json | 150 | 120 | 0.7399 | 0.7399 | 0 |
| run2 | S4 | eval_multi_agent_20260910_201549.json | 150 | 120 | 0.7348 | 0.7348 | 0 |
| run3 | S1 | eval_rag_monolith_20260910_212102.json | 150 | 120 | 0.326 | 0.3145 | 0 |
| run3 | S2 | eval_rag_agent_20260910_221325.json | 150 | 120 | 0.6127 | 0.5748 | 0 |
| run3 | S3 | eval_long_context_20260910_230308.json | 150 | 120 | 0.7441 | 0.7441 | 0 |
| run3 | S4 | eval_multi_agent_20260911_010153.json | 150 | 120 | 0.7286 | 0.7286 | 0 |
