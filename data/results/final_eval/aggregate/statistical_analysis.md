# Statistische Auswertung der drei Vollläufe

Läufe: run1, run2, run3. Item-Ebene; binäre Endpunkte über die Mehrheitsregel (exact_match = 1,0 in ≥ 2 von 3 Läufen), stetige über das Item-Mittel. Zweiseitig, α = 0.05, Holm je Hypothese. 95%-CIs: gepaarter Item-Bootstrap, 10000 Wiederholungen, Seed 20260911. Wilcoxon-Nullen nach Pratt. Reasoning-Endpunkte auf der Emissions-Schnittmenge (Kette bei beiden Systemen in ≥ 2 von 3 Läufen).

## 1 Konfirmatorische Hypothesenprüfung

| H | Paar | Endpunkt | n | A | B | Δ (B−A) | 95%-CI | Effekt | p | p_Holm | Richtung | Verdikt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H1 | S2 vs S3 | exact_match | 45 | 0.800 | 0.844 | 0.044 | [-0.089, 0.178] | b=6, c=4, OR=1.50 | 0.754 | 0.867 | ✓ | — (nicht bestätigt) |
| H1 | S2 vs S3 | answer_recall | 45 | 0.852 | 0.833 | -0.019 | [-0.126, 0.089] | r_rb=-0.17, nonzero=23, p_exact(wilcox)=0.777 | 0.433 | 0.867 | ✗ | — (nicht bestätigt) |
| H2 | S1 vs S2 | exact_match | 60 | 0.200 | 0.833 | 0.633 | [0.483, 0.767] | b=40, c=2, OR=20.00 | <0.001 | <0.001 | ✓ | ✓ (bestätigt) |
| H2 | S1 vs S2 | answer_recall | 60 | 0.233 | 0.853 | 0.619 | [0.481, 0.744] | r_rb=0.90, nonzero=44, p_exact(wilcox)=<0.001 | <0.001 | <0.001 | ✓ | ✓ (bestätigt) |
| H3 | S3 vs S4 | exact_match | 45 | 0.844 | 0.956 | 0.111 | [0.000, 0.222] | b=6, c=1, OR=6.00 | 0.125 | 0.125 | ✓ | — (nicht bestätigt) |
| H3 | S3 vs S4 | groundedness | 45 | 0.901 | 0.666 | -0.235 | [-0.352, -0.130] | r_rb=-0.66, nonzero=23, p_exact(wilcox)=<0.001 | 0.002 | 0.005 | ✗ | — (nicht bestätigt) |
| H3 | S3 vs S4 | completeness | 45 | 1.000 | 0.977 | -0.023 | [-0.049, -0.002] | r_rb=-1.00, nonzero=4, p_exact(wilcox)=0.125 ⚠ <10 nonzero | 0.046 | 0.091 | ✗ | — (nicht bestätigt) |

**H1** (S2 vs S3, FA-4 ∪ FA-3 cross_window, n = 45): **nicht bestätigt**.
- exact_match: Kontingenz beide richtig 32, beide falsch 3, nur S3 6, nur S2 4 (diskordant 10).
- answer_recall: n = 45, Differenzen +9 / −14 / 0: 22.

**H2** (S1 vs S2, FA-2 ∪ FA-3, n = 60): **bestätigt**.
- exact_match: Kontingenz beide richtig 10, beide falsch 8, nur S2 40, nur S1 2 (diskordant 42).
- answer_recall: n = 60, Differenzen +42 / −2 / 0: 16.

**H3** (S3 vs S4, FA-4 ∪ FA-3 cross_window, n = 45): **nicht bestätigt**.
- exact_match: Kontingenz beide richtig 37, beide falsch 1, nur S4 6, nur S3 1 (diskordant 7).
- groundedness: n = 45, Differenzen +5 / −18 / 0: 22; emission intersection (>=2/3 runs): 45/45.
- completeness: n = 45, Differenzen +0 / −4 / 0: 41; emission intersection (>=2/3 runs): 45/45.

## 2 Sensitivität der Reasoning-Endpunkte (H3) gegenüber der Emissionsregel

| Endpunkt | Variante | n | S3 | S4 | Δ | 95%-CI | p (unadj.) |
|---|---|---|---|---|---|---|---|
| groundedness | majority | 45 | 0.901 | 0.666 | -0.235 | [-0.352, -0.130] | 0.002 |
| groundedness | strict | 38 | 0.897 | 0.667 | -0.230 | [-0.361, -0.110] | 0.008 |
| groundedness | no_rule | 45 | 0.901 | 0.666 | -0.235 | [-0.352, -0.130] | 0.002 |
| completeness | majority | 45 | 1.000 | 0.977 | -0.023 | [-0.049, -0.002] | 0.046 |
| completeness | strict | 38 | 1.000 | 0.973 | -0.027 | [-0.057, -0.003] | 0.046 |
| completeness | no_rule | 45 | 1.000 | 0.977 | -0.023 | [-0.049, -0.002] | 0.046 |
| validity | majority | 45 | 0.986 | 0.939 | -0.047 | [-0.101, -0.002] | 0.161 |
| validity | strict | 38 | 0.990 | 0.946 | -0.044 | [-0.095, -0.004] | 0.125 |
| validity | no_rule | 45 | 0.986 | 0.939 | -0.047 | [-0.101, -0.002] | 0.161 |

## 3 Emissionsregel: Selektionsprüfung

| System | Lauf | Emissionsrate | ohne Kette | davon Leerantwort | nach FA | exact_match ohne / mit Kette |
|---|---|---|---|---|---|---|
| S1 | run1 | 0.992 | 1 | 0 | {'FA-2': 1} | 0.000 / 0.218 |
| S1 | run2 | 0.992 | 1 | 0 | {'FA-2': 1} | 0.000 / 0.218 |
| S1 | run3 | 0.992 | 1 | 0 | {'FA-2': 1} | 0.000 / 0.218 |
| S2 | run1 | 0.975 | 3 | 0 | {'FA-2': 2, 'FA-3': 1} | 0.000 / 0.838 |
| S2 | run2 | 0.983 | 2 | 0 | {'FA-2': 1, 'FA-3': 1} | 0.000 / 0.852 |
| S2 | run3 | 0.967 | 4 | 0 | {'FA-2': 1, 'FA-4': 1, 'FA-3': 2} | 0.000 / 0.849 |
| S3 | run1 | 0.875 | 15 | 3 | {'FA-1': 5, 'FA-2': 5, 'FA-3': 1, 'FA-4': 4} | 0.533 / 0.886 |
| S3 | run2 | 0.867 | 16 | 1 | {'FA-1': 9, 'FA-2': 4, 'FA-3': 3} | 0.875 / 0.846 |
| S3 | run3 | 0.883 | 14 | 0 | {'FA-1': 11, 'FA-2': 2, 'FA-4': 1} | 0.857 / 0.868 |
| S4 | run1 | 1.000 | 0 | 0 | {} | — / 0.925 |
| S4 | run2 | 0.992 | 1 | 0 | {'FA-3': 1} | 1.000 / 0.933 |
| S4 | run3 | 0.992 | 1 | 0 | {'FA-3': 1} | 0.000 / 0.941 |

## 4 Explorative Paarvergleiche (unadjustiert)

Alle sechs Paare auf allen Populationen; die drei geplanten Paare auf ihrer Hypothesenpopulation sind bereits oben konfirmatorisch berichtet.

| Paar | Population | Endpunkt | n | A | B | Δ | 95%-CI | p (unadj.) | geplant |
|---|---|---|---|---|---|---|---|---|---|
| S1 vs S2 | answerable | exact_match | 120 | 0.217 | 0.833 | 0.617 | [0.517, 0.717] | <0.001 | ja |
| S1 vs S2 | answerable | answer_recall | 120 | 0.250 | 0.883 | 0.633 | [0.543, 0.718] | <0.001 | ja |
| S1 vs S2 | answerable | groundedness | 108 | 0.704 | 0.887 | 0.183 | [0.101, 0.265] | <0.001 | ja |
| S1 vs S2 | answerable | completeness | 117 | 0.301 | 0.973 | 0.672 | [0.601, 0.741] | <0.001 | ja |
| S1 vs S2 | answerable | citation_accuracy | 120 | 0.556 | 0.803 | 0.248 | [0.193, 0.301] | <0.001 | ja |
| S1 vs S2 | answerable | answer_correctness | 120 | 0.335 | 0.610 | 0.275 | [0.224, 0.322] | <0.001 | ja |
| S1 vs S2 | answerable | locus_faithfulness | 72 | 0.709 | 0.618 | -0.091 | [-0.180, -0.001] | 0.045 | ja |
| S1 vs S2 | FA-1 | exact_match | 30 | 0.433 | 0.933 | 0.500 | [0.267, 0.700] | <0.001 | ja |
| S1 vs S2 | FA-1 | answer_recall | 30 | 0.433 | 0.989 | 0.556 | [0.367, 0.733] | <0.001 | ja |
| S1 vs S2 | FA-1 | groundedness | 27 | 0.679 | 0.864 | 0.185 | [0.012, 0.358] | 0.034 | ja |
| S1 vs S2 | FA-1 | completeness | 30 | 0.433 | 1.000 | 0.567 | [0.400, 0.733] | <0.001 | ja |
| S1 vs S2 | FA-1 | citation_accuracy | 30 | 0.670 | 0.836 | 0.166 | [0.069, 0.272] | 0.002 | ja |
| S1 vs S2 | FA-1 | answer_correctness | 30 | 0.386 | 0.659 | 0.273 | [0.167, 0.371] | <0.001 | ja |
| S1 vs S2 | FA-1 | locus_faithfulness | 24 | 0.659 | 0.646 | -0.013 | [-0.174, 0.145] | 0.752 | ja |
| S1 vs S2 | FA-2 | exact_match | 30 | 0.267 | 0.833 | 0.567 | [0.333, 0.767] | <0.001 | ja |
| S1 vs S2 | FA-2 | answer_recall | 30 | 0.333 | 0.856 | 0.522 | [0.300, 0.722] | <0.001 | ja |
| S1 vs S2 | FA-2 | groundedness | 27 | 0.753 | 0.926 | 0.173 | [0.037, 0.327] | 0.031 | ja |
| S1 vs S2 | FA-2 | completeness | 28 | 0.278 | 0.940 | 0.663 | [0.516, 0.806] | <0.001 | ja |
| S1 vs S2 | FA-2 | citation_accuracy | 30 | 0.528 | 0.790 | 0.262 | [0.129, 0.387] | <0.001 | ja |
| S1 vs S2 | FA-2 | answer_correctness | 30 | 0.325 | 0.547 | 0.222 | [0.099, 0.333] | 0.002 | ja |
| S1 vs S2 | FA-2 | locus_faithfulness | 17 | 0.607 | 0.521 | -0.085 | [-0.268, 0.106] | 0.286 | ja |
| S1 vs S2 | FA-3 | exact_match | 30 | 0.133 | 0.833 | 0.700 | [0.533, 0.867] | <0.001 | ja |
| S1 vs S2 | FA-3 | answer_recall | 30 | 0.133 | 0.850 | 0.717 | [0.561, 0.850] | <0.001 | ja |
| S1 vs S2 | FA-3 | groundedness | 25 | 0.631 | 0.803 | 0.172 | [-0.049, 0.385] | 0.125 | ja |
| S1 vs S2 | FA-3 | completeness | 29 | 0.269 | 1.000 | 0.731 | [0.604, 0.846] | <0.001 | ja |
| S1 vs S2 | FA-3 | citation_accuracy | 30 | 0.480 | 0.754 | 0.274 | [0.158, 0.389] | <0.001 | ja |
| S1 vs S2 | FA-3 | answer_correctness | 30 | 0.253 | 0.517 | 0.264 | [0.169, 0.352] | <0.001 | ja |
| S1 vs S2 | FA-3 | locus_faithfulness | 16 | 0.793 | 0.685 | -0.108 | [-0.293, 0.079] | 0.277 | ja |
| S1 vs S2 | FA-4 | exact_match | 30 | 0.033 | 0.733 | 0.700 | [0.533, 0.867] | <0.001 | ja |
| S1 vs S2 | FA-4 | answer_recall | 30 | 0.100 | 0.839 | 0.739 | [0.594, 0.867] | <0.001 | ja |
| S1 vs S2 | FA-4 | groundedness | 29 | 0.744 | 0.944 | 0.200 | [0.089, 0.323] | 0.004 | ja |
| S1 vs S2 | FA-4 | completeness | 30 | 0.220 | 0.949 | 0.729 | [0.621, 0.826] | <0.001 | ja |
| S1 vs S2 | FA-4 | citation_accuracy | 30 | 0.545 | 0.834 | 0.289 | [0.213, 0.363] | <0.001 | ja |
| S1 vs S2 | FA-4 | answer_correctness | 30 | 0.379 | 0.717 | 0.339 | [0.267, 0.406] | <0.001 | ja |
| S1 vs S2 | FA-4 | locus_faithfulness | 15 | 0.815 | 0.610 | -0.205 | [-0.384, -0.040] | 0.077 | ja |
| S1 vs S2 | FA-2 ∪ FA-3 | exact_match | 60 | 0.200 | 0.833 | 0.633 | [0.483, 0.767] | <0.001 | ja |
| S1 vs S2 | FA-2 ∪ FA-3 | answer_recall | 60 | 0.233 | 0.853 | 0.619 | [0.481, 0.744] | <0.001 | ja |
| S1 vs S2 | FA-2 ∪ FA-3 | groundedness | 52 | 0.694 | 0.867 | 0.172 | [0.044, 0.299] | 0.012 | ja |
| S1 vs S2 | FA-2 ∪ FA-3 | completeness | 57 | 0.273 | 0.971 | 0.697 | [0.600, 0.789] | <0.001 | ja |
| S1 vs S2 | FA-2 ∪ FA-3 | citation_accuracy | 60 | 0.504 | 0.772 | 0.268 | [0.180, 0.355] | <0.001 | ja |
| S1 vs S2 | FA-2 ∪ FA-3 | answer_correctness | 60 | 0.289 | 0.532 | 0.243 | [0.167, 0.316] | <0.001 | ja |
| S1 vs S2 | FA-2 ∪ FA-3 | locus_faithfulness | 33 | 0.697 | 0.601 | -0.096 | [-0.229, 0.037] | 0.122 | ja |
| S1 vs S2 | FA-4 ∪ FA-3 cross_window | exact_match | 45 | 0.044 | 0.800 | 0.756 | [0.622, 0.867] | <0.001 | ja |
| S1 vs S2 | FA-4 ∪ FA-3 cross_window | answer_recall | 45 | 0.089 | 0.852 | 0.763 | [0.648, 0.863] | <0.001 | ja |
| S1 vs S2 | FA-4 ∪ FA-3 cross_window | groundedness | 41 | 0.704 | 0.931 | 0.227 | [0.125, 0.336] | <0.001 | ja |
| S1 vs S2 | FA-4 ∪ FA-3 cross_window | completeness | 45 | 0.228 | 0.966 | 0.738 | [0.651, 0.817] | <0.001 | ja |
| S1 vs S2 | FA-4 ∪ FA-3 cross_window | citation_accuracy | 45 | 0.507 | 0.818 | 0.311 | [0.244, 0.378] | <0.001 | ja |
| S1 vs S2 | FA-4 ∪ FA-3 cross_window | answer_correctness | 45 | 0.328 | 0.654 | 0.326 | [0.265, 0.383] | <0.001 | ja |
| S1 vs S2 | FA-4 ∪ FA-3 cross_window | locus_faithfulness | 21 | 0.784 | 0.632 | -0.152 | [-0.317, 0.006] | 0.121 | ja |
| S2 vs S3 | answerable | exact_match | 120 | 0.833 | 0.858 | 0.025 | [-0.058, 0.108] | 0.690 | ja |
| S2 vs S3 | answerable | answer_recall | 120 | 0.883 | 0.886 | 0.003 | [-0.061, 0.068] | 0.579 | ja |
| S2 vs S3 | answerable | groundedness | 110 | 0.890 | 0.881 | -0.009 | [-0.056, 0.037] | 0.437 | ja |
| S2 vs S3 | answerable | completeness | 110 | 0.971 | 0.973 | 0.002 | [-0.029, 0.030] | 0.490 | ja |
| S2 vs S3 | answerable | citation_accuracy | 120 | 0.803 | 0.783 | -0.021 | [-0.056, 0.016] | 0.041 | ja |
| S2 vs S3 | answerable | answer_correctness | 120 | 0.610 | 0.644 | 0.034 | [-0.005, 0.074] | 0.165 | ja |
| S2 vs S3 | answerable | locus_faithfulness | 113 | 0.548 | 0.521 | -0.027 | [-0.081, 0.025] | 0.473 | ja |
| S2 vs S3 | FA-1 | exact_match | 30 | 0.933 | 0.900 | -0.033 | [-0.167, 0.100] | 1.000 | ja |
| S2 vs S3 | FA-1 | answer_recall | 30 | 0.989 | 0.939 | -0.050 | [-0.133, 0.011] | 0.301 | ja |
| S2 vs S3 | FA-1 | groundedness | 24 | 0.861 | 0.743 | -0.118 | [-0.257, -0.007] | 0.096 | ja |
| S2 vs S3 | FA-1 | completeness | 24 | 1.000 | 0.979 | -0.021 | [-0.062, 0.000] | 0.317 | ja |
| S2 vs S3 | FA-1 | citation_accuracy | 30 | 0.836 | 0.819 | -0.017 | [-0.061, 0.032] | 0.120 | ja |
| S2 vs S3 | FA-1 | answer_correctness | 30 | 0.659 | 0.627 | -0.032 | [-0.097, 0.030] | 0.360 | ja |
| S2 vs S3 | FA-1 | locus_faithfulness | 30 | 0.646 | 0.671 | 0.025 | [-0.056, 0.108] | 0.624 | ja |
| S2 vs S3 | FA-2 | exact_match | 30 | 0.833 | 0.867 | 0.033 | [-0.133, 0.200] | 1.000 | ja |
| S2 vs S3 | FA-2 | answer_recall | 30 | 0.856 | 0.883 | 0.028 | [-0.122, 0.183] | 0.746 | ja |
| S2 vs S3 | FA-2 | groundedness | 27 | 0.926 | 0.974 | 0.048 | [-0.003, 0.113] | 0.343 | ja |
| S2 vs S3 | FA-2 | completeness | 27 | 0.938 | 0.907 | -0.031 | [-0.130, 0.056] | 0.564 | ja |
| S2 vs S3 | FA-2 | citation_accuracy | 30 | 0.790 | 0.779 | -0.011 | [-0.084, 0.074] | 0.692 | ja |
| S2 vs S3 | FA-2 | answer_correctness | 30 | 0.547 | 0.590 | 0.044 | [-0.035, 0.129] | 0.453 | ja |
| S2 vs S3 | FA-2 | locus_faithfulness | 28 | 0.438 | 0.366 | -0.073 | [-0.183, 0.029] | 0.331 | ja |
| S2 vs S3 | FA-3 | exact_match | 30 | 0.833 | 0.833 | 0.000 | [-0.167, 0.167] | 1.000 | ja |
| S2 vs S3 | FA-3 | answer_recall | 30 | 0.850 | 0.917 | 0.067 | [-0.061, 0.211] | 0.490 | ja |
| S2 vs S3 | FA-3 | groundedness | 29 | 0.821 | 0.928 | 0.107 | [0.023, 0.203] | 0.053 | ja |
| S2 vs S3 | FA-3 | completeness | 29 | 1.000 | 1.000 | 0.000 | [0.000, 0.000] | 1.000 | ja |
| S2 vs S3 | FA-3 | citation_accuracy | 30 | 0.754 | 0.734 | -0.020 | [-0.119, 0.084] | 0.363 | ja |
| S2 vs S3 | FA-3 | answer_correctness | 30 | 0.517 | 0.651 | 0.134 | [0.045, 0.222] | 0.012 | ja |
| S2 vs S3 | FA-3 | locus_faithfulness | 26 | 0.559 | 0.464 | -0.094 | [-0.237, 0.038] | 0.182 | ja |
| S2 vs S3 | FA-4 | exact_match | 30 | 0.733 | 0.833 | 0.100 | [-0.067, 0.267] | 0.453 | ja |
| S2 vs S3 | FA-4 | answer_recall | 30 | 0.839 | 0.806 | -0.033 | [-0.161, 0.094] | 0.321 | ja |
| S2 vs S3 | FA-4 | groundedness | 30 | 0.946 | 0.860 | -0.085 | [-0.155, -0.024] | 0.007 | ja |
| S2 vs S3 | FA-4 | completeness | 30 | 0.949 | 1.000 | 0.051 | [0.006, 0.112] | 0.046 | ja |
| S2 vs S3 | FA-4 | citation_accuracy | 30 | 0.834 | 0.798 | -0.035 | [-0.079, 0.007] | 0.188 | ja |
| S2 vs S3 | FA-4 | answer_correctness | 30 | 0.717 | 0.708 | -0.009 | [-0.075, 0.051] | 0.992 | ja |
| S2 vs S3 | FA-4 | locus_faithfulness | 29 | 0.541 | 0.567 | 0.025 | [-0.063, 0.114] | 0.664 | ja |
| S2 vs S3 | FA-2 ∪ FA-3 | exact_match | 60 | 0.833 | 0.850 | 0.017 | [-0.100, 0.133] | 1.000 | ja |
| S2 vs S3 | FA-2 ∪ FA-3 | answer_recall | 60 | 0.853 | 0.900 | 0.047 | [-0.056, 0.150] | 0.474 | ja |
| S2 vs S3 | FA-2 ∪ FA-3 | groundedness | 56 | 0.872 | 0.950 | 0.079 | [0.027, 0.136] | 0.033 | ja |
| S2 vs S3 | FA-2 ∪ FA-3 | completeness | 56 | 0.970 | 0.955 | -0.015 | [-0.062, 0.027] | 0.564 | ja |
| S2 vs S3 | FA-2 ∪ FA-3 | citation_accuracy | 60 | 0.772 | 0.757 | -0.016 | [-0.077, 0.049] | 0.282 | ja |
| S2 vs S3 | FA-2 ∪ FA-3 | answer_correctness | 60 | 0.532 | 0.621 | 0.089 | [0.029, 0.151] | 0.016 | ja |
| S2 vs S3 | FA-2 ∪ FA-3 | locus_faithfulness | 54 | 0.496 | 0.413 | -0.083 | [-0.170, 0.001] | 0.114 | ja |
| S2 vs S3 | FA-4 ∪ FA-3 cross_window | exact_match | 45 | 0.800 | 0.844 | 0.044 | [-0.089, 0.178] | 0.754 | ja |
| S2 vs S3 | FA-4 ∪ FA-3 cross_window | answer_recall | 45 | 0.852 | 0.833 | -0.019 | [-0.126, 0.089] | 0.433 | ja |
| S2 vs S3 | FA-4 ∪ FA-3 cross_window | groundedness | 45 | 0.937 | 0.901 | -0.036 | [-0.093, 0.021] | 0.085 | ja |
| S2 vs S3 | FA-4 ∪ FA-3 cross_window | completeness | 45 | 0.966 | 1.000 | 0.034 | [0.004, 0.076] | 0.046 | ja |
| S2 vs S3 | FA-4 ∪ FA-3 cross_window | citation_accuracy | 45 | 0.818 | 0.784 | -0.033 | [-0.077, 0.011] | 0.098 | ja |
| S2 vs S3 | FA-4 ∪ FA-3 cross_window | answer_correctness | 45 | 0.654 | 0.702 | 0.048 | [-0.011, 0.108] | 0.113 | ja |
| S2 vs S3 | FA-4 ∪ FA-3 cross_window | locus_faithfulness | 44 | 0.539 | 0.533 | -0.006 | [-0.080, 0.067] | 0.921 | ja |
| S3 vs S4 | answerable | exact_match | 120 | 0.858 | 0.933 | 0.075 | [0.017, 0.142] | 0.035 | ja |
| S3 vs S4 | answerable | answer_recall | 120 | 0.886 | 0.940 | 0.054 | [0.006, 0.104] | 0.001 | ja |
| S3 vs S4 | answerable | groundedness | 110 | 0.876 | 0.640 | -0.236 | [-0.319, -0.157] | <0.001 | ja |
| S3 vs S4 | answerable | completeness | 112 | 0.973 | 0.847 | -0.126 | [-0.189, -0.068] | <0.001 | ja |
| S3 vs S4 | answerable | citation_accuracy | 120 | 0.783 | 0.832 | 0.049 | [0.028, 0.071] | <0.001 | ja |
| S3 vs S4 | answerable | answer_correctness | 120 | 0.644 | 0.657 | 0.012 | [-0.018, 0.043] | 0.102 | ja |
| S3 vs S4 | answerable | locus_faithfulness | 115 | 0.532 | 0.491 | -0.042 | [-0.091, 0.011] | 0.027 | ja |
| S3 vs S4 | FA-1 | exact_match | 30 | 0.900 | 1.000 | 0.100 | [0.000, 0.200] | 0.250 | ja |
| S3 vs S4 | FA-1 | answer_recall | 30 | 0.939 | 1.000 | 0.061 | [0.000, 0.144] | 0.083 | ja |
| S3 vs S4 | FA-1 | groundedness | 24 | 0.743 | 0.833 | 0.090 | [0.000, 0.208] | 0.156 | ja |
| S3 vs S4 | FA-1 | completeness | 24 | 0.979 | 1.000 | 0.021 | [0.000, 0.062] | 0.317 | ja |
| S3 vs S4 | FA-1 | citation_accuracy | 30 | 0.819 | 0.853 | 0.033 | [0.007, 0.066] | 0.025 | ja |
| S3 vs S4 | FA-1 | answer_correctness | 30 | 0.627 | 0.680 | 0.053 | [0.024, 0.087] | <0.001 | ja |
| S3 vs S4 | FA-1 | locus_faithfulness | 30 | 0.671 | 0.615 | -0.057 | [-0.132, 0.023] | 0.141 | ja |
| S3 vs S4 | FA-2 | exact_match | 30 | 0.867 | 0.867 | 0.000 | [-0.133, 0.133] | 1.000 | ja |
| S3 vs S4 | FA-2 | answer_recall | 30 | 0.883 | 0.856 | -0.028 | [-0.161, 0.100] | 0.936 | ja |
| S3 vs S4 | FA-2 | groundedness | 26 | 0.954 | 0.453 | -0.501 | [-0.672, -0.327] | <0.001 | ja |
| S3 vs S4 | FA-2 | completeness | 28 | 0.911 | 0.425 | -0.486 | [-0.665, -0.309] | <0.001 | ja |
| S3 vs S4 | FA-2 | citation_accuracy | 30 | 0.779 | 0.797 | 0.017 | [-0.022, 0.058] | 0.473 | ja |
| S3 vs S4 | FA-2 | answer_correctness | 30 | 0.590 | 0.579 | -0.011 | [-0.070, 0.046] | 0.734 | ja |
| S3 vs S4 | FA-2 | locus_faithfulness | 27 | 0.380 | 0.274 | -0.106 | [-0.210, 0.009] | 0.008 | ja |
| S3 vs S4 | FA-3 | exact_match | 30 | 0.833 | 0.933 | 0.100 | [0.000, 0.233] | 0.250 | ja |
| S3 vs S4 | FA-3 | answer_recall | 30 | 0.917 | 0.967 | 0.050 | [-0.011, 0.133] | 0.180 | ja |
| S3 vs S4 | FA-3 | groundedness | 30 | 0.931 | 0.610 | -0.321 | [-0.480, -0.165] | 0.002 | ja |
| S3 vs S4 | FA-3 | completeness | 30 | 1.000 | 1.000 | 0.000 | [0.000, 0.000] | 1.000 | ja |
| S3 vs S4 | FA-3 | citation_accuracy | 30 | 0.734 | 0.811 | 0.077 | [0.027, 0.132] | 0.005 | ja |
| S3 vs S4 | FA-3 | answer_correctness | 30 | 0.651 | 0.610 | -0.041 | [-0.114, 0.031] | 0.262 | ja |
| S3 vs S4 | FA-3 | locus_faithfulness | 28 | 0.496 | 0.517 | 0.022 | [-0.110, 0.155] | 0.714 | ja |
| S3 vs S4 | FA-4 | exact_match | 30 | 0.833 | 0.933 | 0.100 | [-0.033, 0.233] | 0.375 | ja |
| S3 vs S4 | FA-4 | answer_recall | 30 | 0.806 | 0.939 | 0.133 | [0.039, 0.233] | 0.001 | ja |
| S3 vs S4 | FA-4 | groundedness | 30 | 0.860 | 0.678 | -0.183 | [-0.304, -0.071] | 0.027 | ja |
| S3 vs S4 | FA-4 | completeness | 30 | 1.000 | 0.965 | -0.035 | [-0.072, -0.004] | 0.046 | ja |
| S3 vs S4 | FA-4 | citation_accuracy | 30 | 0.798 | 0.866 | 0.068 | [0.030, 0.111] | 0.005 | ja |
| S3 vs S4 | FA-4 | answer_correctness | 30 | 0.708 | 0.757 | 0.048 | [-0.014, 0.112] | 0.079 | ja |
| S3 vs S4 | FA-4 | locus_faithfulness | 30 | 0.564 | 0.537 | -0.027 | [-0.103, 0.046] | 0.529 | ja |
| S3 vs S4 | FA-2 ∪ FA-3 | exact_match | 60 | 0.850 | 0.900 | 0.050 | [-0.033, 0.133] | 0.453 | ja |
| S3 vs S4 | FA-2 ∪ FA-3 | answer_recall | 60 | 0.900 | 0.911 | 0.011 | [-0.067, 0.089] | 0.436 | ja |
| S3 vs S4 | FA-2 ∪ FA-3 | groundedness | 56 | 0.941 | 0.537 | -0.404 | [-0.521, -0.287] | <0.001 | ja |
| S3 vs S4 | FA-2 ∪ FA-3 | completeness | 58 | 0.957 | 0.722 | -0.235 | [-0.341, -0.130] | <0.001 | ja |
| S3 vs S4 | FA-2 ∪ FA-3 | citation_accuracy | 60 | 0.757 | 0.804 | 0.047 | [0.014, 0.081] | 0.005 | ja |
| S3 vs S4 | FA-2 ∪ FA-3 | answer_correctness | 60 | 0.621 | 0.595 | -0.026 | [-0.073, 0.021] | 0.494 | ja |
| S3 vs S4 | FA-2 ∪ FA-3 | locus_faithfulness | 55 | 0.439 | 0.398 | -0.041 | [-0.129, 0.049] | 0.125 | ja |
| S3 vs S4 | FA-4 ∪ FA-3 cross_window | exact_match | 45 | 0.844 | 0.956 | 0.111 | [0.000, 0.222] | 0.125 | ja |
| S3 vs S4 | FA-4 ∪ FA-3 cross_window | answer_recall | 45 | 0.833 | 0.959 | 0.126 | [0.048, 0.207] | <0.001 | ja |
| S3 vs S4 | FA-4 ∪ FA-3 cross_window | groundedness | 45 | 0.901 | 0.666 | -0.235 | [-0.352, -0.130] | 0.002 | ja |
| S3 vs S4 | FA-4 ∪ FA-3 cross_window | completeness | 45 | 1.000 | 0.977 | -0.023 | [-0.049, -0.002] | 0.046 | ja |
| S3 vs S4 | FA-4 ∪ FA-3 cross_window | citation_accuracy | 45 | 0.784 | 0.840 | 0.056 | [0.023, 0.090] | 0.004 | ja |
| S3 vs S4 | FA-4 ∪ FA-3 cross_window | answer_correctness | 45 | 0.702 | 0.712 | 0.010 | [-0.050, 0.067] | 0.748 | ja |
| S3 vs S4 | FA-4 ∪ FA-3 cross_window | locus_faithfulness | 45 | 0.532 | 0.533 | 0.000 | [-0.074, 0.075] | 0.799 | ja |
| S1 vs S3 | answerable | exact_match | 120 | 0.217 | 0.858 | 0.642 | [0.550, 0.733] | <0.001 | nein |
| S1 vs S3 | answerable | answer_recall | 120 | 0.250 | 0.886 | 0.636 | [0.549, 0.718] | <0.001 | nein |
| S1 vs S3 | answerable | groundedness | 102 | 0.726 | 0.877 | 0.151 | [0.070, 0.232] | 0.002 | nein |
| S1 vs S3 | answerable | completeness | 111 | 0.308 | 0.973 | 0.665 | [0.591, 0.736] | <0.001 | nein |
| S1 vs S3 | answerable | citation_accuracy | 120 | 0.556 | 0.783 | 0.227 | [0.172, 0.281] | <0.001 | nein |
| S1 vs S3 | answerable | answer_correctness | 120 | 0.335 | 0.644 | 0.309 | [0.263, 0.353] | <0.001 | nein |
| S1 vs S3 | answerable | locus_faithfulness | 72 | 0.690 | 0.563 | -0.127 | [-0.231, -0.024] | 0.019 | nein |
| S1 vs S3 | FA-1 | exact_match | 30 | 0.433 | 0.900 | 0.467 | [0.267, 0.667] | <0.001 | nein |
| S1 vs S3 | FA-1 | answer_recall | 30 | 0.433 | 0.939 | 0.506 | [0.317, 0.689] | <0.001 | nein |
| S1 vs S3 | FA-1 | groundedness | 21 | 0.706 | 0.722 | 0.016 | [-0.190, 0.214] | 0.574 | nein |
| S1 vs S3 | FA-1 | completeness | 24 | 0.458 | 0.979 | 0.521 | [0.333, 0.708] | <0.001 | nein |
| S1 vs S3 | FA-1 | citation_accuracy | 30 | 0.670 | 0.819 | 0.149 | [0.062, 0.248] | 0.028 | nein |
| S1 vs S3 | FA-1 | answer_correctness | 30 | 0.386 | 0.627 | 0.241 | [0.146, 0.334] | <0.001 | nein |
| S1 vs S3 | FA-1 | locus_faithfulness | 24 | 0.659 | 0.687 | 0.027 | [-0.117, 0.160] | 0.718 | nein |
| S1 vs S3 | FA-2 | exact_match | 30 | 0.267 | 0.867 | 0.600 | [0.400, 0.800] | <0.001 | nein |
| S1 vs S3 | FA-2 | answer_recall | 30 | 0.333 | 0.883 | 0.550 | [0.344, 0.739] | <0.001 | nein |
| S1 vs S3 | FA-2 | groundedness | 26 | 0.801 | 0.964 | 0.162 | [0.036, 0.314] | 0.114 | nein |
| S1 vs S3 | FA-2 | completeness | 27 | 0.325 | 0.907 | 0.582 | [0.424, 0.733] | <0.001 | nein |
| S1 vs S3 | FA-2 | citation_accuracy | 30 | 0.528 | 0.779 | 0.251 | [0.135, 0.364] | <0.001 | nein |
| S1 vs S3 | FA-2 | answer_correctness | 30 | 0.325 | 0.590 | 0.266 | [0.171, 0.356] | <0.001 | nein |
| S1 vs S3 | FA-2 | locus_faithfulness | 18 | 0.573 | 0.427 | -0.146 | [-0.343, 0.060] | 0.156 | nein |
| S1 vs S3 | FA-3 | exact_match | 30 | 0.133 | 0.833 | 0.700 | [0.533, 0.867] | <0.001 | nein |
| S1 vs S3 | FA-3 | answer_recall | 30 | 0.133 | 0.917 | 0.783 | [0.644, 0.906] | <0.001 | nein |
| S1 vs S3 | FA-3 | groundedness | 26 | 0.645 | 0.933 | 0.287 | [0.113, 0.470] | 0.013 | nein |
| S1 vs S3 | FA-3 | completeness | 30 | 0.260 | 1.000 | 0.740 | [0.617, 0.851] | <0.001 | nein |
| S1 vs S3 | FA-3 | citation_accuracy | 30 | 0.480 | 0.734 | 0.254 | [0.119, 0.382] | 0.002 | nein |
| S1 vs S3 | FA-3 | answer_correctness | 30 | 0.253 | 0.651 | 0.398 | [0.303, 0.482] | <0.001 | nein |
| S1 vs S3 | FA-3 | locus_faithfulness | 15 | 0.754 | 0.522 | -0.233 | [-0.499, 0.056] | 0.147 | nein |
| S1 vs S3 | FA-4 | exact_match | 30 | 0.033 | 0.833 | 0.800 | [0.633, 0.933] | <0.001 | nein |
| S1 vs S3 | FA-4 | answer_recall | 30 | 0.100 | 0.806 | 0.706 | [0.556, 0.833] | <0.001 | nein |
| S1 vs S3 | FA-4 | groundedness | 29 | 0.744 | 0.861 | 0.117 | [-0.001, 0.250] | 0.173 | nein |
| S1 vs S3 | FA-4 | completeness | 30 | 0.220 | 1.000 | 0.780 | [0.692, 0.860] | <0.001 | nein |
| S1 vs S3 | FA-4 | citation_accuracy | 30 | 0.545 | 0.798 | 0.253 | [0.171, 0.337] | <0.001 | nein |
| S1 vs S3 | FA-4 | answer_correctness | 30 | 0.379 | 0.708 | 0.330 | [0.244, 0.409] | <0.001 | nein |
| S1 vs S3 | FA-4 | locus_faithfulness | 15 | 0.815 | 0.570 | -0.245 | [-0.440, -0.045] | 0.073 | nein |
| S1 vs S3 | FA-2 ∪ FA-3 | exact_match | 60 | 0.200 | 0.850 | 0.650 | [0.517, 0.783] | <0.001 | nein |
| S1 vs S3 | FA-2 ∪ FA-3 | answer_recall | 60 | 0.233 | 0.900 | 0.667 | [0.542, 0.786] | <0.001 | nein |
| S1 vs S3 | FA-2 ∪ FA-3 | groundedness | 52 | 0.723 | 0.948 | 0.225 | [0.114, 0.342] | 0.003 | nein |
| S1 vs S3 | FA-2 ∪ FA-3 | completeness | 57 | 0.291 | 0.956 | 0.665 | [0.565, 0.759] | <0.001 | nein |
| S1 vs S3 | FA-2 ∪ FA-3 | citation_accuracy | 60 | 0.504 | 0.757 | 0.252 | [0.166, 0.338] | <0.001 | nein |
| S1 vs S3 | FA-2 ∪ FA-3 | answer_correctness | 60 | 0.289 | 0.621 | 0.332 | [0.264, 0.397] | <0.001 | nein |
| S1 vs S3 | FA-2 ∪ FA-3 | locus_faithfulness | 33 | 0.655 | 0.470 | -0.185 | [-0.349, -0.014] | 0.044 | nein |
| S1 vs S3 | FA-4 ∪ FA-3 cross_window | exact_match | 45 | 0.044 | 0.844 | 0.800 | [0.689, 0.911] | <0.001 | nein |
| S1 vs S3 | FA-4 ∪ FA-3 cross_window | answer_recall | 45 | 0.089 | 0.833 | 0.744 | [0.633, 0.848] | <0.001 | nein |
| S1 vs S3 | FA-4 ∪ FA-3 cross_window | groundedness | 41 | 0.704 | 0.900 | 0.196 | [0.085, 0.316] | 0.006 | nein |
| S1 vs S3 | FA-4 ∪ FA-3 cross_window | completeness | 45 | 0.228 | 1.000 | 0.772 | [0.694, 0.841] | <0.001 | nein |
| S1 vs S3 | FA-4 ∪ FA-3 cross_window | citation_accuracy | 45 | 0.507 | 0.784 | 0.277 | [0.201, 0.355] | <0.001 | nein |
| S1 vs S3 | FA-4 ∪ FA-3 cross_window | answer_correctness | 45 | 0.328 | 0.702 | 0.374 | [0.301, 0.443] | <0.001 | nein |
| S1 vs S3 | FA-4 ∪ FA-3 cross_window | locus_faithfulness | 21 | 0.784 | 0.581 | -0.203 | [-0.391, -0.009] | 0.092 | nein |
| S2 vs S4 | answerable | exact_match | 120 | 0.833 | 0.933 | 0.100 | [0.033, 0.175] | 0.012 | nein |
| S2 vs S4 | answerable | answer_recall | 120 | 0.883 | 0.940 | 0.057 | [0.003, 0.113] | 0.024 | nein |
| S2 vs S4 | answerable | groundedness | 116 | 0.888 | 0.659 | -0.229 | [-0.303, -0.156] | <0.001 | nein |
| S2 vs S4 | answerable | completeness | 118 | 0.973 | 0.855 | -0.118 | [-0.179, -0.063] | <0.001 | nein |
| S2 vs S4 | answerable | citation_accuracy | 120 | 0.803 | 0.832 | 0.028 | [0.000, 0.058] | 0.081 | nein |
| S2 vs S4 | answerable | answer_correctness | 120 | 0.610 | 0.657 | 0.047 | [0.014, 0.080] | <0.001 | nein |
| S2 vs S4 | answerable | locus_faithfulness | 114 | 0.553 | 0.502 | -0.051 | [-0.105, 0.000] | 0.155 | nein |
| S2 vs S4 | FA-1 | exact_match | 30 | 0.933 | 1.000 | 0.067 | [0.000, 0.167] | 0.500 | nein |
| S2 vs S4 | FA-1 | answer_recall | 30 | 0.989 | 1.000 | 0.011 | [0.000, 0.033] | 0.317 | nein |
| S2 vs S4 | FA-1 | groundedness | 30 | 0.878 | 0.867 | -0.011 | [-0.072, 0.039] | 0.944 | nein |
| S2 vs S4 | FA-1 | completeness | 30 | 1.000 | 1.000 | 0.000 | [0.000, 0.000] | 1.000 | nein |
| S2 vs S4 | FA-1 | citation_accuracy | 30 | 0.836 | 0.853 | 0.017 | [0.000, 0.050] | 0.317 | nein |
| S2 vs S4 | FA-1 | answer_correctness | 30 | 0.659 | 0.680 | 0.022 | [-0.022, 0.069] | 0.067 | nein |
| S2 vs S4 | FA-1 | locus_faithfulness | 30 | 0.646 | 0.615 | -0.031 | [-0.135, 0.071] | 0.899 | nein |
| S2 vs S4 | FA-2 | exact_match | 30 | 0.833 | 0.867 | 0.033 | [-0.100, 0.167] | 1.000 | nein |
| S2 vs S4 | FA-2 | answer_recall | 30 | 0.856 | 0.856 | 0.000 | [-0.144, 0.156] | 0.989 | nein |
| S2 vs S4 | FA-2 | groundedness | 27 | 0.907 | 0.436 | -0.471 | [-0.634, -0.311] | <0.001 | nein |
| S2 vs S4 | FA-2 | completeness | 29 | 0.943 | 0.444 | -0.498 | [-0.659, -0.333] | <0.001 | nein |
| S2 vs S4 | FA-2 | citation_accuracy | 30 | 0.790 | 0.797 | 0.006 | [-0.057, 0.083] | 0.702 | nein |
| S2 vs S4 | FA-2 | answer_correctness | 30 | 0.547 | 0.579 | 0.033 | [-0.040, 0.106] | 0.237 | nein |
| S2 vs S4 | FA-2 | locus_faithfulness | 27 | 0.423 | 0.267 | -0.155 | [-0.247, -0.068] | 0.005 | nein |
| S2 vs S4 | FA-3 | exact_match | 30 | 0.833 | 0.933 | 0.100 | [-0.033, 0.233] | 0.375 | nein |
| S2 vs S4 | FA-3 | answer_recall | 30 | 0.850 | 0.967 | 0.117 | [0.006, 0.244] | 0.059 | nein |
| S2 vs S4 | FA-3 | groundedness | 29 | 0.821 | 0.631 | -0.190 | [-0.361, -0.025] | 0.119 | nein |
| S2 vs S4 | FA-3 | completeness | 29 | 1.000 | 1.000 | 0.000 | [0.000, 0.000] | 1.000 | nein |
| S2 vs S4 | FA-3 | citation_accuracy | 30 | 0.754 | 0.811 | 0.057 | [-0.015, 0.142] | 0.326 | nein |
| S2 vs S4 | FA-3 | answer_correctness | 30 | 0.517 | 0.610 | 0.093 | [0.019, 0.168] | 0.026 | nein |
| S2 vs S4 | FA-3 | locus_faithfulness | 28 | 0.590 | 0.551 | -0.040 | [-0.144, 0.055] | 0.383 | nein |
| S2 vs S4 | FA-4 | exact_match | 30 | 0.733 | 0.933 | 0.200 | [0.033, 0.367] | 0.070 | nein |
| S2 vs S4 | FA-4 | answer_recall | 30 | 0.839 | 0.939 | 0.100 | [0.006, 0.206] | 0.083 | nein |
| S2 vs S4 | FA-4 | groundedness | 30 | 0.946 | 0.678 | -0.268 | [-0.398, -0.148] | <0.001 | nein |
| S2 vs S4 | FA-4 | completeness | 30 | 0.949 | 0.965 | 0.016 | [-0.046, 0.087] | 0.958 | nein |
| S2 vs S4 | FA-4 | citation_accuracy | 30 | 0.834 | 0.866 | 0.033 | [0.001, 0.070] | 0.063 | nein |
| S2 vs S4 | FA-4 | answer_correctness | 30 | 0.717 | 0.757 | 0.039 | [-0.023, 0.096] | 0.066 | nein |
| S2 vs S4 | FA-4 | locus_faithfulness | 29 | 0.541 | 0.556 | 0.015 | [-0.106, 0.123] | 0.473 | nein |
| S2 vs S4 | FA-2 ∪ FA-3 | exact_match | 60 | 0.833 | 0.900 | 0.067 | [-0.033, 0.167] | 0.344 | nein |
| S2 vs S4 | FA-2 ∪ FA-3 | answer_recall | 60 | 0.853 | 0.911 | 0.058 | [-0.036, 0.158] | 0.166 | nein |
| S2 vs S4 | FA-2 ∪ FA-3 | groundedness | 56 | 0.863 | 0.537 | -0.326 | [-0.452, -0.204] | <0.001 | nein |
| S2 vs S4 | FA-2 ∪ FA-3 | completeness | 58 | 0.971 | 0.722 | -0.249 | [-0.352, -0.149] | <0.001 | nein |
| S2 vs S4 | FA-2 ∪ FA-3 | citation_accuracy | 60 | 0.772 | 0.804 | 0.031 | [-0.017, 0.086] | 0.566 | nein |
| S2 vs S4 | FA-2 ∪ FA-3 | answer_correctness | 60 | 0.532 | 0.595 | 0.063 | [0.010, 0.116] | 0.016 | nein |
| S2 vs S4 | FA-2 ∪ FA-3 | locus_faithfulness | 55 | 0.508 | 0.412 | -0.096 | [-0.165, -0.030] | 0.009 | nein |
| S2 vs S4 | FA-4 ∪ FA-3 cross_window | exact_match | 45 | 0.800 | 0.956 | 0.156 | [0.044, 0.289] | 0.039 | nein |
| S2 vs S4 | FA-4 ∪ FA-3 cross_window | answer_recall | 45 | 0.852 | 0.959 | 0.107 | [0.033, 0.189] | 0.007 | nein |
| S2 vs S4 | FA-4 ∪ FA-3 cross_window | groundedness | 45 | 0.937 | 0.666 | -0.271 | [-0.389, -0.162] | <0.001 | nein |
| S2 vs S4 | FA-4 ∪ FA-3 cross_window | completeness | 45 | 0.966 | 0.977 | 0.011 | [-0.032, 0.060] | 0.973 | nein |
| S2 vs S4 | FA-4 ∪ FA-3 cross_window | citation_accuracy | 45 | 0.818 | 0.840 | 0.022 | [-0.012, 0.059] | 0.195 | nein |
| S2 vs S4 | FA-4 ∪ FA-3 cross_window | answer_correctness | 45 | 0.654 | 0.712 | 0.058 | [0.007, 0.105] | 0.012 | nein |
| S2 vs S4 | FA-4 ∪ FA-3 cross_window | locus_faithfulness | 44 | 0.539 | 0.545 | 0.005 | [-0.079, 0.085] | 0.542 | nein |
| S1 vs S4 | answerable | exact_match | 120 | 0.217 | 0.933 | 0.717 | [0.633, 0.800] | <0.001 | nein |
| S1 vs S4 | answerable | answer_recall | 120 | 0.250 | 0.940 | 0.690 | [0.607, 0.769] | <0.001 | nein |
| S1 vs S4 | answerable | groundedness | 108 | 0.699 | 0.645 | -0.055 | [-0.163, 0.055] | 0.416 | nein |
| S1 vs S4 | answerable | completeness | 119 | 0.304 | 0.864 | 0.560 | [0.471, 0.650] | <0.001 | nein |
| S1 vs S4 | answerable | citation_accuracy | 120 | 0.556 | 0.832 | 0.276 | [0.226, 0.327] | <0.001 | nein |
| S1 vs S4 | answerable | answer_correctness | 120 | 0.335 | 0.657 | 0.321 | [0.278, 0.362] | <0.001 | nein |
| S1 vs S4 | answerable | locus_faithfulness | 73 | 0.690 | 0.543 | -0.147 | [-0.249, -0.042] | 0.011 | nein |
| S1 vs S4 | FA-1 | exact_match | 30 | 0.433 | 1.000 | 0.567 | [0.400, 0.733] | <0.001 | nein |
| S1 vs S4 | FA-1 | answer_recall | 30 | 0.433 | 1.000 | 0.567 | [0.400, 0.733] | <0.001 | nein |
| S1 vs S4 | FA-1 | groundedness | 27 | 0.679 | 0.852 | 0.173 | [0.000, 0.352] | 0.057 | nein |
| S1 vs S4 | FA-1 | completeness | 30 | 0.433 | 1.000 | 0.567 | [0.400, 0.733] | <0.001 | nein |
| S1 vs S4 | FA-1 | citation_accuracy | 30 | 0.670 | 0.853 | 0.183 | [0.097, 0.282] | <0.001 | nein |
| S1 vs S4 | FA-1 | answer_correctness | 30 | 0.386 | 0.680 | 0.295 | [0.203, 0.385] | <0.001 | nein |
| S1 vs S4 | FA-1 | locus_faithfulness | 24 | 0.659 | 0.623 | -0.036 | [-0.189, 0.109] | 0.840 | nein |
| S1 vs S4 | FA-2 | exact_match | 30 | 0.267 | 0.867 | 0.600 | [0.400, 0.800] | <0.001 | nein |
| S1 vs S4 | FA-2 | answer_recall | 30 | 0.333 | 0.856 | 0.522 | [0.322, 0.722] | <0.001 | nein |
| S1 vs S4 | FA-2 | groundedness | 26 | 0.724 | 0.421 | -0.303 | [-0.526, -0.077] | 0.035 | nein |
| S1 vs S4 | FA-2 | completeness | 29 | 0.303 | 0.479 | 0.176 | [-0.046, 0.398] | 0.178 | nein |
| S1 vs S4 | FA-2 | citation_accuracy | 30 | 0.528 | 0.797 | 0.268 | [0.158, 0.382] | <0.001 | nein |
| S1 vs S4 | FA-2 | answer_correctness | 30 | 0.325 | 0.579 | 0.255 | [0.155, 0.353] | <0.001 | nein |
| S1 vs S4 | FA-2 | locus_faithfulness | 17 | 0.567 | 0.317 | -0.250 | [-0.426, -0.072] | 0.024 | nein |
| S1 vs S4 | FA-3 | exact_match | 30 | 0.133 | 0.933 | 0.800 | [0.633, 0.933] | <0.001 | nein |
| S1 vs S4 | FA-3 | answer_recall | 30 | 0.133 | 0.967 | 0.833 | [0.700, 0.944] | <0.001 | nein |
| S1 vs S4 | FA-3 | groundedness | 26 | 0.645 | 0.629 | -0.016 | [-0.258, 0.224] | 0.823 | nein |
| S1 vs S4 | FA-3 | completeness | 30 | 0.260 | 1.000 | 0.740 | [0.617, 0.851] | <0.001 | nein |
| S1 vs S4 | FA-3 | citation_accuracy | 30 | 0.480 | 0.811 | 0.331 | [0.224, 0.437] | <0.001 | nein |
| S1 vs S4 | FA-3 | answer_correctness | 30 | 0.253 | 0.610 | 0.357 | [0.278, 0.426] | <0.001 | nein |
| S1 vs S4 | FA-3 | locus_faithfulness | 17 | 0.746 | 0.678 | -0.068 | [-0.318, 0.176] | 0.552 | nein |
| S1 vs S4 | FA-4 | exact_match | 30 | 0.033 | 0.933 | 0.900 | [0.767, 1.000] | <0.001 | nein |
| S1 vs S4 | FA-4 | answer_recall | 30 | 0.100 | 0.939 | 0.839 | [0.739, 0.928] | <0.001 | nein |
| S1 vs S4 | FA-4 | groundedness | 29 | 0.744 | 0.667 | -0.078 | [-0.258, 0.109] | 0.451 | nein |
| S1 vs S4 | FA-4 | completeness | 30 | 0.220 | 0.965 | 0.745 | [0.656, 0.829] | <0.001 | nein |
| S1 vs S4 | FA-4 | citation_accuracy | 30 | 0.545 | 0.866 | 0.322 | [0.250, 0.396] | <0.001 | nein |
| S1 vs S4 | FA-4 | answer_correctness | 30 | 0.379 | 0.757 | 0.378 | [0.309, 0.441] | <0.001 | nein |
| S1 vs S4 | FA-4 | locus_faithfulness | 15 | 0.815 | 0.517 | -0.298 | [-0.530, -0.055] | 0.025 | nein |
| S1 vs S4 | FA-2 ∪ FA-3 | exact_match | 60 | 0.200 | 0.900 | 0.700 | [0.567, 0.817] | <0.001 | nein |
| S1 vs S4 | FA-2 ∪ FA-3 | answer_recall | 60 | 0.233 | 0.911 | 0.678 | [0.550, 0.800] | <0.001 | nein |
| S1 vs S4 | FA-2 ∪ FA-3 | groundedness | 52 | 0.685 | 0.525 | -0.160 | [-0.332, 0.008] | 0.088 | nein |
| S1 vs S4 | FA-2 ∪ FA-3 | completeness | 59 | 0.281 | 0.744 | 0.463 | [0.318, 0.601] | <0.001 | nein |
| S1 vs S4 | FA-2 ∪ FA-3 | citation_accuracy | 60 | 0.504 | 0.804 | 0.300 | [0.223, 0.378] | <0.001 | nein |
| S1 vs S4 | FA-2 ∪ FA-3 | answer_correctness | 60 | 0.289 | 0.595 | 0.306 | [0.241, 0.367] | <0.001 | nein |
| S1 vs S4 | FA-2 ∪ FA-3 | locus_faithfulness | 34 | 0.657 | 0.498 | -0.159 | [-0.313, -0.002] | 0.053 | nein |
| S1 vs S4 | FA-4 ∪ FA-3 cross_window | exact_match | 45 | 0.044 | 0.956 | 0.911 | [0.822, 0.978] | <0.001 | nein |
| S1 vs S4 | FA-4 ∪ FA-3 cross_window | answer_recall | 45 | 0.089 | 0.959 | 0.870 | [0.789, 0.941] | <0.001 | nein |
| S1 vs S4 | FA-4 ∪ FA-3 cross_window | groundedness | 41 | 0.704 | 0.659 | -0.045 | [-0.201, 0.115] | 0.515 | nein |
| S1 vs S4 | FA-4 ∪ FA-3 cross_window | completeness | 45 | 0.228 | 0.977 | 0.749 | [0.670, 0.821] | <0.001 | nein |
| S1 vs S4 | FA-4 ∪ FA-3 cross_window | citation_accuracy | 45 | 0.507 | 0.840 | 0.333 | [0.270, 0.399] | <0.001 | nein |
| S1 vs S4 | FA-4 ∪ FA-3 cross_window | answer_correctness | 45 | 0.328 | 0.712 | 0.384 | [0.330, 0.434] | <0.001 | nein |
| S1 vs S4 | FA-4 ∪ FA-3 cross_window | locus_faithfulness | 21 | 0.784 | 0.573 | -0.211 | [-0.426, 0.008] | 0.060 | nein |

## 5 Aufschlüsselung nach Stratum (Mittel ± SD über Läufe)

### nach fa_type

| Gruppe | n | System | exact_match | answer_recall | answer_correctness | citation_accuracy | groundedness | completeness | over_refusal |
|---|---|---|---|---|---|---|---|---|---|
| FA-1 | 30 | S1 | 0.433 ± 0.000 | 0.433 ± 0.000 | 0.386 ± 0.008 | 0.670 ± 0.000 | 0.701 ± 0.018 | 0.433 ± 0.000 | 0.556 ± 0.019 |
| FA-1 | 30 | S2 | 0.933 ± 0.033 | 0.989 ± 0.019 | 0.658 ± 0.006 | 0.836 ± 0.000 | 0.878 ± 0.006 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| FA-1 | 30 | S3 | 0.900 ± 0.033 | 0.939 ± 0.025 | 0.627 ± 0.019 | 0.819 ± 0.017 | 0.712 ± 0.047 | 0.987 ± 0.023 | 0.011 ± 0.019 |
| FA-1 | 30 | S4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.680 ± 0.001 | 0.853 ± 0.000 | 0.867 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| FA-2 | 30 | S1 | 0.267 ± 0.000 | 0.333 ± 0.000 | 0.325 ± 0.005 | 0.528 ± 0.000 | 0.772 ± 0.011 | 0.303 ± 0.007 | 0.633 ± 0.000 |
| FA-2 | 30 | S2 | 0.822 ± 0.019 | 0.856 ± 0.019 | 0.547 ± 0.018 | 0.790 ± 0.016 | 0.913 ± 0.002 | 0.942 ± 0.001 | 0.056 ± 0.038 |
| FA-2 | 30 | S3 | 0.844 ± 0.038 | 0.883 ± 0.044 | 0.591 ± 0.041 | 0.779 ± 0.041 | 0.944 ± 0.024 | 0.912 ± 0.017 | 0.078 ± 0.051 |
| FA-2 | 30 | S4 | 0.856 ± 0.019 | 0.856 ± 0.019 | 0.580 ± 0.011 | 0.797 ± 0.010 | 0.444 ± 0.036 | 0.463 ± 0.032 | 0.100 ± 0.000 |
| FA-3 | 30 | S1 | 0.133 ± 0.000 | 0.133 ± 0.000 | 0.253 ± 0.002 | 0.480 ± 0.003 | 0.671 ± 0.028 | 0.260 ± 0.006 | 0.656 ± 0.019 |
| FA-3 | 30 | S2 | 0.789 ± 0.019 | 0.850 ± 0.017 | 0.517 ± 0.029 | 0.754 ± 0.028 | 0.819 ± 0.012 | 1.000 ± 0.000 | 0.044 ± 0.019 |
| FA-3 | 30 | S3 | 0.844 ± 0.038 | 0.917 ± 0.033 | 0.651 ± 0.054 | 0.734 ± 0.028 | 0.930 ± 0.029 | 1.000 ± 0.000 | 0.011 ± 0.019 |
| FA-3 | 30 | S4 | 0.933 ± 0.033 | 0.967 ± 0.017 | 0.610 ± 0.018 | 0.811 ± 0.017 | 0.601 ± 0.015 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| FA-4 | 30 | S1 | 0.033 ± 0.000 | 0.100 ± 0.017 | 0.379 ± 0.005 | 0.545 ± 0.006 | 0.762 ± 0.022 | 0.220 ± 0.006 | 0.700 ± 0.000 |
| FA-4 | 30 | S2 | 0.756 ± 0.010 | 0.839 ± 0.019 | 0.717 ± 0.026 | 0.834 ± 0.017 | 0.948 ± 0.006 | 0.948 ± 0.005 | 0.022 ± 0.019 |
| FA-4 | 30 | S3 | 0.822 ± 0.019 | 0.806 ± 0.063 | 0.708 ± 0.023 | 0.798 ± 0.020 | 0.858 ± 0.009 | 1.000 ± 0.000 | 0.011 ± 0.019 |
| FA-4 | 30 | S4 | 0.933 ± 0.033 | 0.939 ± 0.019 | 0.757 ± 0.003 | 0.866 ± 0.008 | 0.678 ± 0.005 | 0.965 ± 0.012 | 0.000 ± 0.000 |

### nach subtype

| Gruppe | n | System | exact_match | answer_recall | answer_correctness | citation_accuracy | groundedness | completeness | over_refusal |
|---|---|---|---|---|---|---|---|---|---|
| basis_kpi | 24 | S1 | 0.458 ± 0.000 | 0.458 ± 0.000 | 0.408 ± 0.009 | 0.664 ± 0.000 | 0.692 ± 0.014 | 0.458 ± 0.000 | 0.528 ± 0.024 |
| basis_kpi | 24 | S2 | 0.917 ± 0.042 | 0.986 ± 0.024 | 0.662 ± 0.008 | 0.809 ± 0.000 | 0.951 ± 0.007 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| basis_kpi | 24 | S3 | 0.986 ± 0.024 | 0.986 ± 0.024 | 0.669 ± 0.015 | 0.788 ± 0.022 | 0.871 ± 0.027 | 0.983 ± 0.029 | 0.014 ± 0.024 |
| basis_kpi | 24 | S4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.698 ± 0.001 | 0.830 ± 0.000 | 0.958 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| cloud_comparison | 3 | S1 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.481 ± 0.023 | 0.454 ± 0.065 | 0.667 ± 0.000 | 0.185 ± 0.064 | 1.000 ± 0.000 |
| cloud_comparison | 3 | S2 | 0.333 ± 0.000 | 1.000 ± 0.000 | 0.766 ± 0.023 | 0.830 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| cloud_comparison | 3 | S3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.917 ± 0.049 | 0.794 ± 0.031 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| cloud_comparison | 3 | S4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.953 ± 0.036 | 0.849 ± 0.033 | 0.667 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| cross_firm_diff | 2 | S1 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.286 ± 0.000 | 0.540 ± 0.000 | 0.917 ± 0.144 | 0.333 ± 0.000 | 0.500 ± 0.000 |
| cross_firm_diff | 2 | S2 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.666 ± 0.075 | 0.830 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| cross_firm_diff | 2 | S3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.666 ± 0.037 | 0.775 ± 0.095 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| cross_firm_diff | 2 | S4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.646 ± 0.035 | 0.830 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| cross_firm_pairwise | 13 | S1 | 0.077 ± 0.000 | 0.103 ± 0.022 | 0.429 ± 0.013 | 0.523 ± 0.000 | 0.811 ± 0.021 | 0.197 ± 0.015 | 0.769 ± 0.000 |
| cross_firm_pairwise | 13 | S2 | 0.769 ± 0.000 | 0.859 ± 0.044 | 0.807 ± 0.059 | 0.819 ± 0.044 | 0.912 ± 0.014 | 0.947 ± 0.003 | 0.051 ± 0.044 |
| cross_firm_pairwise | 13 | S3 | 0.731 ± 0.115 | 0.769 ± 0.038 | 0.744 ± 0.053 | 0.772 ± 0.055 | 0.789 ± 0.051 | 1.000 ± 0.000 | 0.026 ± 0.044 |
| cross_firm_pairwise | 13 | S4 | 0.846 ± 0.077 | 0.885 ± 0.038 | 0.797 ± 0.037 | 0.878 ± 0.007 | 0.635 ± 0.000 | 0.940 ± 0.015 | 0.000 ± 0.000 |
| cross_firm_pairwise_growth | 3 | S1 | 0.000 ± 0.000 | 0.333 ± 0.000 | 0.455 ± 0.000 | 0.447 ± 0.000 | 0.667 ± 0.000 | 0.259 ± 0.064 | 0.333 ± 0.000 |
| cross_firm_pairwise_growth | 3 | S2 | 0.889 ± 0.096 | 0.833 ± 0.000 | 0.844 ± 0.031 | 0.833 ± 0.000 | 0.926 ± 0.064 | 0.833 ± 0.000 | 0.000 ± 0.000 |
| cross_firm_pairwise_growth | 3 | S3 | 0.944 ± 0.096 | 1.000 ± 0.000 | 0.857 ± 0.036 | 0.830 ± 0.000 | 0.917 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| cross_firm_pairwise_growth | 3 | S4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.883 ± 0.029 | 0.830 ± 0.000 | 0.833 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| eps | 1 | S1 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.167 ± 0.000 | 0.500 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| eps | 1 | S2 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.676 ± 0.000 | 0.830 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| eps | 1 | S3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.678 ± 0.000 | 0.830 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| eps | 1 | S4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.676 ± 0.000 | 0.830 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| growth_cagr | 4 | S1 | 0.250 ± 0.000 | 0.250 ± 0.000 | 0.364 ± 0.000 | 0.425 ± 0.000 | 0.556 ± 0.096 | 0.542 ± 0.036 | 0.750 ± 0.000 |
| growth_cagr | 4 | S2 | 0.750 ± 0.000 | 0.750 ± 0.000 | 0.471 ± 0.018 | 0.780 ± 0.000 | 0.750 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| growth_cagr | 4 | S3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.620 ± 0.031 | 0.813 ± 0.029 | 0.917 ± 0.144 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| growth_cagr | 4 | S4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.536 ± 0.000 | 0.830 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| growth_yoy | 14 | S1 | 0.143 ± 0.000 | 0.143 ± 0.000 | 0.261 ± 0.005 | 0.601 ± 0.000 | 0.667 ± 0.000 | 0.262 ± 0.000 | 0.405 ± 0.041 |
| growth_yoy | 14 | S2 | 0.809 ± 0.041 | 0.941 ± 0.021 | 0.566 ± 0.017 | 0.783 ± 0.000 | 0.878 ± 0.007 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| growth_yoy | 14 | S3 | 0.762 ± 0.041 | 0.917 ± 0.055 | 0.621 ± 0.075 | 0.701 ± 0.048 | 0.960 ± 0.042 | 1.000 ± 0.000 | 0.024 ± 0.041 |
| growth_yoy | 14 | S4 | 0.857 ± 0.071 | 0.976 ± 0.041 | 0.618 ± 0.028 | 0.783 ± 0.041 | 0.816 ± 0.013 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| liquidity | 1 | S1 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.174 ± 0.000 | 0.330 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| liquidity | 1 | S2 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.685 ± 0.000 | 0.830 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| liquidity | 1 | S3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.685 ± 0.000 | 0.830 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| liquidity | 1 | S4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.718 ± 0.057 | 0.830 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| max_min | 9 | S1 | 0.000 ± 0.000 | 0.074 ± 0.032 | 0.266 ± 0.028 | 0.640 ± 0.000 | 0.726 ± 0.119 | 0.228 ± 0.000 | 0.667 ± 0.000 |
| max_min | 9 | S2 | 0.778 ± 0.000 | 0.722 ± 0.000 | 0.540 ± 0.012 | 0.857 ± 0.019 | 0.977 ± 0.016 | 0.959 ± 0.016 | 0.000 ± 0.000 |
| max_min | 9 | S3 | 0.815 ± 0.170 | 0.685 ± 0.160 | 0.548 ± 0.086 | 0.832 ± 0.064 | 0.853 ± 0.094 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| max_min | 9 | S4 | 1.000 ± 0.000 | 0.963 ± 0.032 | 0.616 ± 0.061 | 0.876 ± 0.019 | 0.620 ± 0.016 | 0.970 ± 0.026 | 0.000 ± 0.000 |
| multi_step_compute | 4 | S1 | 0.250 ± 0.000 | 0.250 ± 0.000 | 0.287 ± 0.000 | 0.415 ± 0.000 | 0.583 ± 0.000 | 0.188 ± 0.000 | 0.750 ± 0.000 |
| multi_step_compute | 4 | S2 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.530 ± 0.056 | 0.848 ± 0.000 | 0.778 ± 0.048 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| multi_step_compute | 4 | S3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.730 ± 0.073 | 0.604 ± 0.072 | 0.972 ± 0.048 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| multi_step_compute | 4 | S4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.551 ± 0.021 | 0.827 ± 0.036 | 0.604 ± 0.036 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| pp_change | 5 | S1 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.166 ± 0.000 | 0.300 ± 0.000 | 0.722 ± 0.255 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| pp_change | 5 | S2 | 0.533 ± 0.116 | 0.533 ± 0.116 | 0.403 ± 0.072 | 0.609 ± 0.096 | 0.792 ± 0.036 | 1.000 ± 0.000 | 0.267 ± 0.116 |
| pp_change | 5 | S3 | 0.733 ± 0.116 | 0.800 ± 0.000 | 0.650 ± 0.082 | 0.830 ± 0.000 | 0.950 ± 0.087 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| pp_change | 5 | S4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.622 ± 0.001 | 0.830 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| ratio_compute | 7 | S1 | 0.286 ± 0.000 | 0.429 ± 0.000 | 0.319 ± 0.000 | 0.533 ± 0.000 | 0.794 ± 0.069 | 0.095 ± 0.000 | 0.571 ± 0.000 |
| ratio_compute | 7 | S2 | 0.857 ± 0.000 | 1.000 ± 0.000 | 0.622 ± 0.006 | 0.819 ± 0.000 | 0.929 ± 0.000 | 0.857 ± 0.000 | 0.000 ± 0.000 |
| ratio_compute | 7 | S3 | 0.809 ± 0.083 | 0.952 ± 0.083 | 0.634 ± 0.054 | 0.790 ± 0.069 | 0.956 ± 0.042 | 0.754 ± 0.069 | 0.048 ± 0.083 |
| ratio_compute | 7 | S4 | 0.857 ± 0.000 | 0.857 ± 0.000 | 0.592 ± 0.050 | 0.783 ± 0.000 | 0.786 ± 0.000 | 0.619 ± 0.000 | 0.095 ± 0.083 |
| segment_margin | 5 | S1 | 0.400 ± 0.000 | 0.400 ± 0.000 | 0.335 ± 0.000 | 0.764 ± 0.000 | 0.900 ± 0.000 | 0.467 ± 0.000 | 0.400 ± 0.000 |
| segment_margin | 5 | S2 | 0.733 ± 0.116 | 0.733 ± 0.116 | 0.511 ± 0.075 | 0.775 ± 0.096 | 0.892 ± 0.014 | 0.856 ± 0.019 | 0.133 ± 0.231 |
| segment_margin | 5 | S3 | 0.733 ± 0.116 | 0.733 ± 0.116 | 0.505 ± 0.074 | 0.708 ± 0.038 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.133 ± 0.231 |
| segment_margin | 5 | S4 | 0.333 ± 0.116 | 0.333 ± 0.116 | 0.342 ± 0.059 | 0.697 ± 0.058 | 0.333 ± 0.333 | 0.000 ± 0.000 | 0.467 ± 0.116 |
| segment_share | 7 | S1 | 0.429 ± 0.000 | 0.571 ± 0.000 | 0.452 ± 0.021 | 0.640 ± 0.000 | 0.622 ± 0.067 | 0.476 ± 0.000 | 0.429 ± 0.000 |
| segment_share | 7 | S2 | 0.809 ± 0.083 | 0.809 ± 0.083 | 0.493 ± 0.058 | 0.711 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.143 ± 0.000 |
| segment_share | 7 | S3 | 0.905 ± 0.083 | 0.929 ± 0.071 | 0.618 ± 0.054 | 0.775 ± 0.060 | 0.921 ± 0.007 | 0.952 ± 0.083 | 0.048 ± 0.083 |
| segment_share | 7 | S4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.610 ± 0.018 | 0.830 ± 0.000 | 0.421 ± 0.110 | 0.651 ± 0.138 | 0.000 ± 0.000 |
| solvency | 4 | S1 | 0.500 ± 0.000 | 0.500 ± 0.000 | 0.361 ± 0.000 | 0.833 ± 0.000 | 0.583 ± 0.072 | 0.500 ± 0.000 | 0.500 ± 0.000 |
| solvency | 4 | S2 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.629 ± 0.000 | 1.000 ± 0.000 | 0.625 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| solvency | 4 | S3 | 0.333 ± 0.144 | 0.625 ± 0.125 | 0.347 ± 0.066 | 1.000 ± 0.000 | 0.208 ± 0.144 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| solvency | 4 | S4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.563 ± 0.007 | 1.000 ± 0.000 | 0.500 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| synthesis | 11 | S1 | 0.091 ± 0.000 | 0.091 ± 0.000 | 0.243 ± 0.000 | 0.347 ± 0.000 | 0.783 ± 0.029 | 0.244 ± 0.019 | 0.909 ± 0.000 |
| synthesis | 11 | S2 | 0.849 ± 0.052 | 0.849 ± 0.052 | 0.549 ± 0.016 | 0.830 ± 0.000 | 0.864 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| synthesis | 11 | S3 | 0.879 ± 0.052 | 0.879 ± 0.052 | 0.584 ± 0.037 | 0.808 ± 0.053 | 0.924 ± 0.030 | 0.967 ± 0.058 | 0.091 ± 0.000 |
| synthesis | 11 | S4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.660 ± 0.020 | 0.830 ± 0.000 | 0.258 ± 0.052 | 0.455 ± 0.000 | 0.000 ± 0.000 |
| trend_qualitative | 3 | S1 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.165 ± 0.000 | 0.377 ± 0.029 | 0.926 ± 0.064 | 0.407 ± 0.064 | 1.000 ± 0.000 |
| trend_qualitative | 3 | S2 | 0.889 ± 0.193 | 0.889 ± 0.096 | 0.524 ± 0.129 | 0.702 ± 0.160 | 0.722 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| trend_qualitative | 3 | S3 | 1.000 ± 0.000 | 0.889 ± 0.096 | 0.729 ± 0.134 | 0.794 ± 0.160 | 0.722 ± 0.111 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| trend_qualitative | 3 | S4 | 1.000 ± 0.000 | 0.778 ± 0.096 | 0.735 ± 0.086 | 0.859 ± 0.048 | 0.111 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |

### nach window_class

| Gruppe | n | System | exact_match | answer_recall | answer_correctness | citation_accuracy | groundedness | completeness | over_refusal |
|---|---|---|---|---|---|---|---|---|---|
| (none) | 90 | S1 | 0.244 ± 0.000 | 0.289 ± 0.006 | 0.363 ± 0.000 | 0.581 ± 0.002 | 0.746 ± 0.007 | 0.319 ± 0.004 | 0.630 ± 0.006 |
| (none) | 90 | S2 | 0.837 ± 0.014 | 0.894 ± 0.011 | 0.641 ± 0.017 | 0.820 ± 0.007 | 0.913 ± 0.003 | 0.964 ± 0.002 | 0.026 ± 0.006 |
| (none) | 90 | S3 | 0.856 ± 0.019 | 0.876 ± 0.042 | 0.642 ± 0.023 | 0.799 ± 0.018 | 0.847 ± 0.009 | 0.965 ± 0.008 | 0.033 ± 0.019 |
| (none) | 90 | S4 | 0.930 ± 0.017 | 0.931 ± 0.013 | 0.672 ± 0.005 | 0.839 ± 0.005 | 0.669 ± 0.010 | 0.809 ± 0.008 | 0.033 ± 0.000 |
| cross_window | 15 | S1 | 0.067 ± 0.000 | 0.067 ± 0.000 | 0.227 ± 0.000 | 0.431 ± 0.006 | 0.642 ± 0.046 | 0.244 ± 0.010 | 0.711 ± 0.038 |
| cross_window | 15 | S2 | 0.844 ± 0.038 | 0.878 ± 0.038 | 0.526 ± 0.034 | 0.786 ± 0.047 | 0.917 ± 0.009 | 1.000 ± 0.000 | 0.022 ± 0.038 |
| cross_window | 15 | S3 | 0.867 ± 0.067 | 0.889 ± 0.038 | 0.689 ± 0.058 | 0.757 ± 0.050 | 0.983 ± 0.018 | 1.000 ± 0.000 | 0.022 ± 0.038 |
| cross_window | 15 | S4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.622 ± 0.062 | 0.788 ± 0.023 | 0.633 ± 0.028 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| within_window | 15 | S1 | 0.200 ± 0.000 | 0.200 ± 0.000 | 0.279 ± 0.005 | 0.529 ± 0.000 | 0.696 ± 0.030 | 0.276 ± 0.013 | 0.600 ± 0.000 |
| within_window | 15 | S2 | 0.733 ± 0.000 | 0.822 ± 0.019 | 0.508 ± 0.024 | 0.722 ± 0.010 | 0.716 ± 0.014 | 1.000 ± 0.000 | 0.067 ± 0.000 |
| within_window | 15 | S3 | 0.822 ± 0.102 | 0.944 ± 0.038 | 0.614 ± 0.059 | 0.711 ± 0.006 | 0.878 ± 0.044 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| within_window | 15 | S4 | 0.867 ± 0.067 | 0.933 ± 0.033 | 0.599 ± 0.073 | 0.834 ± 0.011 | 0.568 ± 0.017 | 1.000 ± 0.000 | 0.000 ± 0.000 |

### nach math_type

| Gruppe | n | System | exact_match | answer_recall | answer_correctness | citation_accuracy | groundedness | completeness | over_refusal |
|---|---|---|---|---|---|---|---|---|---|
| cagr | 4 | S1 | 0.250 ± 0.000 | 0.250 ± 0.000 | 0.364 ± 0.000 | 0.425 ± 0.000 | 0.556 ± 0.096 | 0.542 ± 0.036 | 0.750 ± 0.000 |
| cagr | 4 | S2 | 0.750 ± 0.000 | 0.750 ± 0.000 | 0.471 ± 0.018 | 0.780 ± 0.000 | 0.750 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| cagr | 4 | S3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.620 ± 0.031 | 0.813 ± 0.029 | 0.917 ± 0.144 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| cagr | 4 | S4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.536 ± 0.000 | 0.830 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| multi_step | 4 | S1 | 0.250 ± 0.000 | 0.250 ± 0.000 | 0.287 ± 0.000 | 0.415 ± 0.000 | 0.583 ± 0.000 | 0.188 ± 0.000 | 0.750 ± 0.000 |
| multi_step | 4 | S2 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.530 ± 0.056 | 0.848 ± 0.000 | 0.778 ± 0.048 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| multi_step | 4 | S3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.730 ± 0.073 | 0.604 ± 0.072 | 0.972 ± 0.048 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| multi_step | 4 | S4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.551 ± 0.021 | 0.827 ± 0.036 | 0.604 ± 0.036 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| n/a | 90 | S1 | 0.244 ± 0.000 | 0.289 ± 0.006 | 0.363 ± 0.000 | 0.581 ± 0.002 | 0.746 ± 0.007 | 0.319 ± 0.004 | 0.630 ± 0.006 |
| n/a | 90 | S2 | 0.837 ± 0.014 | 0.894 ± 0.011 | 0.641 ± 0.017 | 0.820 ± 0.007 | 0.913 ± 0.003 | 0.964 ± 0.002 | 0.026 ± 0.006 |
| n/a | 90 | S3 | 0.856 ± 0.019 | 0.876 ± 0.042 | 0.642 ± 0.023 | 0.799 ± 0.018 | 0.847 ± 0.009 | 0.965 ± 0.008 | 0.033 ± 0.019 |
| n/a | 90 | S4 | 0.930 ± 0.017 | 0.931 ± 0.013 | 0.672 ± 0.005 | 0.839 ± 0.005 | 0.669 ± 0.010 | 0.809 ± 0.008 | 0.033 ± 0.000 |
| none | 3 | S1 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.165 ± 0.000 | 0.377 ± 0.029 | 0.926 ± 0.064 | 0.407 ± 0.064 | 1.000 ± 0.000 |
| none | 3 | S2 | 0.889 ± 0.193 | 0.889 ± 0.096 | 0.524 ± 0.129 | 0.702 ± 0.160 | 0.722 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| none | 3 | S3 | 1.000 ± 0.000 | 0.889 ± 0.096 | 0.729 ± 0.134 | 0.794 ± 0.160 | 0.722 ± 0.111 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| none | 3 | S4 | 1.000 ± 0.000 | 0.778 ± 0.096 | 0.735 ± 0.086 | 0.859 ± 0.048 | 0.111 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| pp_diff | 5 | S1 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.166 ± 0.000 | 0.300 ± 0.000 | 0.722 ± 0.255 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| pp_diff | 5 | S2 | 0.533 ± 0.116 | 0.533 ± 0.116 | 0.403 ± 0.072 | 0.609 ± 0.096 | 0.792 ± 0.036 | 1.000 ± 0.000 | 0.267 ± 0.116 |
| pp_diff | 5 | S3 | 0.733 ± 0.116 | 0.800 ± 0.000 | 0.650 ± 0.082 | 0.830 ± 0.000 | 0.950 ± 0.087 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| pp_diff | 5 | S4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.622 ± 0.001 | 0.830 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| yoy | 14 | S1 | 0.143 ± 0.000 | 0.143 ± 0.000 | 0.261 ± 0.005 | 0.601 ± 0.000 | 0.667 ± 0.000 | 0.262 ± 0.000 | 0.405 ± 0.041 |
| yoy | 14 | S2 | 0.809 ± 0.041 | 0.941 ± 0.021 | 0.566 ± 0.017 | 0.783 ± 0.000 | 0.878 ± 0.007 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| yoy | 14 | S3 | 0.762 ± 0.041 | 0.917 ± 0.055 | 0.621 ± 0.075 | 0.701 ± 0.048 | 0.960 ± 0.042 | 1.000 ± 0.000 | 0.024 ± 0.041 |
| yoy | 14 | S4 | 0.857 ± 0.071 | 0.976 ± 0.041 | 0.618 ± 0.028 | 0.783 ± 0.041 | 0.816 ± 0.013 | 1.000 ± 0.000 | 0.000 ± 0.000 |

### nach entity_form

| Gruppe | n | System | exact_match | answer_recall | answer_correctness | citation_accuracy | groundedness | completeness | over_refusal |
|---|---|---|---|---|---|---|---|---|---|
| name | 59 | S1 | 0.237 ± 0.000 | 0.271 ± 0.009 | 0.340 ± 0.003 | 0.586 ± 0.002 | 0.761 ± 0.004 | 0.321 ± 0.006 | 0.627 ± 0.000 |
| name | 59 | S2 | 0.791 ± 0.026 | 0.878 ± 0.020 | 0.613 ± 0.018 | 0.815 ± 0.021 | 0.875 ± 0.003 | 0.957 ± 0.001 | 0.023 ± 0.010 |
| name | 59 | S3 | 0.845 ± 0.013 | 0.921 ± 0.035 | 0.646 ± 0.011 | 0.784 ± 0.015 | 0.860 ± 0.015 | 0.968 ± 0.011 | 0.023 ± 0.010 |
| name | 59 | S4 | 0.921 ± 0.010 | 0.929 ± 0.018 | 0.661 ± 0.009 | 0.837 ± 0.008 | 0.661 ± 0.016 | 0.850 ± 0.007 | 0.034 ± 0.000 |
| ticker | 61 | S1 | 0.197 ± 0.000 | 0.230 ± 0.000 | 0.331 ± 0.003 | 0.526 ± 0.003 | 0.699 ± 0.006 | 0.288 ± 0.003 | 0.645 ± 0.019 |
| ticker | 61 | S2 | 0.858 ± 0.033 | 0.888 ± 0.024 | 0.607 ± 0.020 | 0.793 ± 0.008 | 0.904 ± 0.006 | 0.989 ± 0.002 | 0.038 ± 0.009 |
| ticker | 61 | S3 | 0.861 ± 0.022 | 0.853 ± 0.033 | 0.643 ± 0.008 | 0.781 ± 0.023 | 0.881 ± 0.015 | 0.981 ± 0.001 | 0.033 ± 0.016 |
| ticker | 61 | S4 | 0.940 ± 0.009 | 0.951 ± 0.008 | 0.652 ± 0.005 | 0.827 ± 0.002 | 0.642 ± 0.015 | 0.862 ± 0.007 | 0.016 ± 0.000 |

### nach gt_unit

| Gruppe | n | System | exact_match | answer_recall | answer_correctness | citation_accuracy | groundedness | completeness | over_refusal |
|---|---|---|---|---|---|---|---|---|---|
| USD_billion | 48 | S1 | 0.271 ± 0.000 | 0.285 ± 0.006 | 0.391 ± 0.001 | 0.601 ± 0.004 | 0.686 ± 0.012 | 0.345 ± 0.004 | 0.597 ± 0.012 |
| USD_billion | 48 | S2 | 0.833 ± 0.021 | 0.927 ± 0.018 | 0.666 ± 0.021 | 0.823 ± 0.000 | 0.944 ± 0.004 | 0.992 ± 0.003 | 0.000 ± 0.000 |
| USD_billion | 48 | S3 | 0.910 ± 0.012 | 0.913 ± 0.026 | 0.680 ± 0.008 | 0.785 ± 0.017 | 0.882 ± 0.025 | 0.984 ± 0.028 | 0.035 ± 0.012 |
| USD_billion | 48 | S4 | 0.979 ± 0.021 | 0.993 ± 0.012 | 0.728 ± 0.015 | 0.834 ± 0.003 | 0.822 ± 0.002 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| USD_million | 1 | S1 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.166 ± 0.000 | 0.330 ± 0.000 | 1.000 ± 0.000 | 0.333 ± 0.000 | 1.000 ± 0.000 |
| USD_million | 1 | S2 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.514 ± 0.043 | 0.750 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| USD_million | 1 | S3 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.413 ± 0.219 | 0.830 ± 0.000 | 0.889 ± 0.192 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| USD_million | 1 | S4 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.164 ± 0.000 | 0.500 ± 0.000 | 1.000 ± 0.000 | 0.333 ± 0.000 | 0.667 ± 0.577 |
| USD_per_share | 1 | S1 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.167 ± 0.000 | 0.500 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| USD_per_share | 1 | S2 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.676 ± 0.000 | 0.830 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| USD_per_share | 1 | S3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.678 ± 0.000 | 0.830 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| USD_per_share | 1 | S4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.676 ± 0.000 | 0.830 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| count | 3 | S1 | 0.333 ± 0.000 | 0.333 ± 0.000 | 0.314 ± 0.024 | 0.787 ± 0.000 | 0.944 ± 0.096 | 0.400 ± 0.000 | 0.667 ± 0.000 |
| count | 3 | S2 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.587 ± 0.103 | 0.653 ± 0.058 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| count | 3 | S3 | 0.889 ± 0.193 | 0.889 ± 0.193 | 0.718 ± 0.048 | 0.694 ± 0.096 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| count | 3 | S4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.593 ± 0.075 | 0.910 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| percent | 55 | S1 | 0.200 ± 0.000 | 0.261 ± 0.005 | 0.323 ± 0.001 | 0.553 ± 0.000 | 0.761 ± 0.007 | 0.299 ± 0.007 | 0.594 ± 0.011 |
| percent | 55 | S2 | 0.849 ± 0.014 | 0.870 ± 0.005 | 0.594 ± 0.011 | 0.816 ± 0.009 | 0.859 ± 0.002 | 0.947 ± 0.001 | 0.036 ± 0.018 |
| percent | 55 | S3 | 0.812 ± 0.028 | 0.858 ± 0.037 | 0.608 ± 0.015 | 0.784 ± 0.024 | 0.864 ± 0.007 | 0.962 ± 0.017 | 0.030 ± 0.028 |
| percent | 55 | S4 | 0.897 ± 0.021 | 0.918 ± 0.024 | 0.617 ± 0.025 | 0.826 ± 0.006 | 0.575 ± 0.011 | 0.704 ± 0.015 | 0.042 ± 0.011 |
| pp_change | 5 | S1 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.166 ± 0.000 | 0.300 ± 0.000 | 0.722 ± 0.255 | 0.000 ± 0.000 | 1.000 ± 0.000 |
| pp_change | 5 | S2 | 0.533 ± 0.116 | 0.533 ± 0.116 | 0.403 ± 0.072 | 0.609 ± 0.096 | 0.792 ± 0.036 | 1.000 ± 0.000 | 0.267 ± 0.116 |
| pp_change | 5 | S3 | 0.733 ± 0.116 | 0.800 ± 0.000 | 0.650 ± 0.082 | 0.830 ± 0.000 | 0.950 ± 0.087 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| pp_change | 5 | S4 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.622 ± 0.001 | 0.830 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| text | 7 | S1 | 0.143 ± 0.000 | 0.143 ± 0.000 | 0.236 ± 0.000 | 0.387 ± 0.012 | 0.587 ± 0.028 | 0.282 ± 0.028 | 0.857 ± 0.000 |
| text | 7 | S2 | 0.809 ± 0.083 | 0.857 ± 0.071 | 0.515 ± 0.091 | 0.776 ± 0.131 | 0.712 ± 0.028 | 1.000 ± 0.000 | 0.048 ± 0.083 |
| text | 7 | S3 | 0.952 ± 0.083 | 0.952 ± 0.041 | 0.676 ± 0.067 | 0.745 ± 0.060 | 0.717 ± 0.024 | 1.000 ± 0.000 | 0.000 ± 0.000 |
| text | 7 | S4 | 0.905 ± 0.083 | 0.809 ± 0.041 | 0.603 ± 0.055 | 0.871 ± 0.036 | 0.286 ± 0.021 | 0.952 ± 0.000 | 0.000 ± 0.000 |

### exact_match nach Mehrheitsregel je Stratum

| group | n_items | system | exact_match_majority |
|---|---|---|---|
| FA-1 | 30 | S1 | 0.433 |
| FA-1 | 30 | S2 | 0.933 |
| FA-1 | 30 | S3 | 0.900 |
| FA-1 | 30 | S4 | 1.000 |
| FA-2 | 30 | S1 | 0.267 |
| FA-2 | 30 | S2 | 0.833 |
| FA-2 | 30 | S3 | 0.867 |
| FA-2 | 30 | S4 | 0.867 |
| FA-3 | 30 | S1 | 0.133 |
| FA-3 | 30 | S2 | 0.833 |
| FA-3 | 30 | S3 | 0.833 |
| FA-3 | 30 | S4 | 0.933 |
| FA-4 | 30 | S1 | 0.033 |
| FA-4 | 30 | S2 | 0.733 |
| FA-4 | 30 | S3 | 0.833 |
| FA-4 | 30 | S4 | 0.933 |

## 6 Refusal-Stratum (Judge-Werte, Nebenbefund)

| Gruppe | n | System | refusal_accuracy | refusal_quality |
|---|---|---|---|---|
| all | 30 | S1 | 0.811 ± 0.019 | 0.650 ± 0.000 |
| all | 30 | S2 | 0.822 ± 0.019 | 0.867 ± 0.044 |
| all | 30 | S3 | 0.644 ± 0.019 | 0.739 ± 0.051 |
| all | 30 | S4 | 0.844 ± 0.019 | 0.622 ± 0.025 |
| ambiguous_entity | 10 | S1 | 0.700 ± 0.000 | 0.500 ± 0.000 |
| ambiguous_entity | 10 | S2 | 0.900 ± 0.000 | 0.850 ± 0.050 |
| ambiguous_entity | 10 | S3 | 0.833 ± 0.153 | 0.767 ± 0.126 |
| ambiguous_entity | 10 | S4 | 1.000 ± 0.000 | 0.550 ± 0.000 |
| false_premise | 10 | S1 | 0.833 ± 0.058 | 0.617 ± 0.029 |
| false_premise | 10 | S2 | 0.567 ± 0.058 | 0.750 ± 0.100 |
| false_premise | 10 | S3 | 0.367 ± 0.058 | 0.733 ± 0.076 |
| false_premise | 10 | S4 | 0.567 ± 0.058 | 0.683 ± 0.058 |
| not_in_corpus | 10 | S1 | 0.900 ± 0.000 | 0.833 ± 0.029 |
| not_in_corpus | 10 | S2 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| not_in_corpus | 10 | S3 | 0.733 ± 0.153 | 0.717 ± 0.116 |
| not_in_corpus | 10 | S4 | 0.967 ± 0.058 | 0.633 ± 0.029 |
| counter_evidence | 6 | S1 | 0.722 ± 0.096 | 0.750 ± 0.000 |
| counter_evidence | 6 | S2 | 0.333 ± 0.000 | 0.806 ± 0.210 |
| counter_evidence | 6 | S3 | 0.167 ± 0.000 | 0.861 ± 0.048 |
| counter_evidence | 6 | S4 | 0.444 ± 0.096 | 0.722 ± 0.096 |
| out_of_scope | 4 | S1 | 1.000 ± 0.000 | 0.875 ± 0.000 |
| out_of_scope | 4 | S2 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| out_of_scope | 4 | S3 | 1.000 ± 0.000 | 0.917 ± 0.072 |
| out_of_scope | 4 | S4 | 0.917 ± 0.144 | 0.458 ± 0.072 |
| silent | 10 | S1 | 0.900 ± 0.000 | 0.650 ± 0.000 |
| silent | 10 | S2 | 0.967 ± 0.058 | 0.867 ± 0.029 |
| silent | 10 | S3 | 0.600 ± 0.100 | 0.567 ± 0.029 |
| silent | 10 | S4 | 0.900 ± 0.000 | 0.700 ± 0.000 |
| underspecified | 10 | S1 | 0.700 ± 0.000 | 0.500 ± 0.000 |
| underspecified | 10 | S2 | 0.900 ± 0.000 | 0.850 ± 0.050 |
| underspecified | 10 | S3 | 0.833 ± 0.153 | 0.767 ± 0.126 |
| underspecified | 10 | S4 | 1.000 ± 0.000 | 0.550 ± 0.000 |

## 7 Stabilität über die Läufe (beantwortbare Items)

| system | hits_0_of_3 | hits_1_of_3 | hits_2_of_3 | hits_3_of_3 | unanimous_share | majority_rate | majority_from_split | answer_recall_mean_item_sd | groundedness_mean_item_sd | completeness_mean_item_sd |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 94 | 0 | 0 | 26 | 1.000 | 0.217 | 0 | 0.005 | 0.036 | 0.009 |
| S2 | 17 | 3 | 7 | 93 | 0.917 | 0.833 | 7 | 0.043 | 0.013 | 0.001 |
| S3 | 8 | 9 | 12 | 91 | 0.825 | 0.858 | 12 | 0.079 | 0.072 | 0.013 |
| S4 | 6 | 2 | 3 | 109 | 0.958 | 0.933 | 3 | 0.022 | 0.025 | 0.011 |

## 8 Belegtheits- und Validitätscodes (Summe über Ketten, alle Läufe)

| system | n_chains | B1 | B2 | B3 | B4 | B5 | B6 | B_total | V1 | V2 | V3 | V4 | V5 | V_total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 357 | 74 | 0 | 0 | 0 | 125 | 0 | 199 | 164 | 0 | 0 | 0 | 0 | 164 |
| S2 | 351 | 101 | 0 | 0 | 0 | 4 | 0 | 105 | 7 | 0 | 0 | 0 | 1 | 8 |
| S3 | 315 | 100 | 0 | 0 | 0 | 2 | 0 | 102 | 9 | 0 | 0 | 0 | 1 | 10 |
| S4 | 358 | 259 | 0 | 0 | 0 | 1 | 0 | 260 | 25 | 0 | 0 | 0 | 0 | 25 |

## 9 Effizienz und Prozess

| system | tokens_mean | tokens_median | tokens_p90 | cost_per_run_usd | latency_runs | latency_mean_s | latency_median_s | latency_p90_s | tool_calls_mean | tool_calls_median | tool_calls_zero_share | correction_rate | retried_share | empty_answers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 3121.400 | 3030.000 | 3719.000 | 0.214 | run2+run3 | 3.940 | 3.550 | 6.250 | 1.000 | 1.000 | 0.000 | 0.000 | 0.002 | 0 |
| S2 | 18100.000 | 13687.000 | 31932.400 | 0.931 | run2+run3 | 10.500 | 8.560 | 18.350 | 2.890 | 2.000 | 0.071 | 0.069 | 0.000 | 1 |
| S3 | 1389673.700 | 1394701.500 | 2096377.700 | 62.664 | run2+run3 | 42.750 | 26.320 | 80.710 | 1.090 | 1.000 | 0.424 | 0.184 | 0.011 | 6 |
| S4 | 147751.700 | 108510.500 | 290470.600 | 6.806 | run2+run3 | 13.080 | 11.730 | 21.320 | 2.160 | 2.000 | 0.138 | 0.007 | 0.000 | 0 |

### Kosten je korrekter Antwort (Definition Kap. 3.6.2, je Lauf berechnet)

| system | cost_per_run_usd | correct_per_run | cost_per_correct_usd_mean | cost_per_correct_usd_sd |
|---|---|---|---|---|
| S1 | 0.214 | 50.330 | 0.004 | 0.000 |
| S2 | 0.931 | 123 | 0.008 | 0.000 |
| S3 | 62.664 | 121 | 0.518 | 0.014 |
| S4 | 6.806 | 135.670 | 0.050 | 0.001 |
