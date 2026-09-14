# Agentic Financial Report Analysis

Code und Ergebnisse zur Masterarbeit _„Agentic Information Extraction im Finanzkontext: Vergleichende Evaluation einer Multi-Agenten-Architektur gegenüber Single-Agent-Systemen und RAG-Verfahren zur Analyse von Geschäftsberichten“_.

Verglichen werden vier Architekturen zur Beantwortung von Fragen zu SEC-Form-10-K-Berichten. Sie liegen auf zwei Designachsen: Kontextstrategie (Retrieval oder Long-Context) und Kontrollfluss (monolithisch, Single-Agent oder Multi-Agent).

**Kernergebnisse** (Kap. 5, drei Durchläufe × 150 Anfragen × 4 Systeme):

| Hypothese                     | Paar      | Ergebnis                                                                                            |
| ----------------------------- | --------- | --------------------------------------------------------------------------------------------------- |
| H1 Long-Context > Retrieval   | S2 vs. S3 | nicht bestätigt: kein signifikanter Unterschied                                                     |
| H2 agentisch > monolithisch   | S1 vs. S2 | bestätigt: `exact_match` 0,20 → 0,83, `answer_recall` 0,23 → 0,85                                   |
| H3 Multi-Agent > Single-Agent | S3 vs. S4 | nicht bestätigt: `exact_match` 0,84 → 0,96 (n. s.), Belegtheit signifikant schlechter (0,90 → 0,67) |

## Systeme

|     | System                    | Kurzbeschreibung                                                                                 | Code                        |
| --- | ------------------------- | ------------------------------------------------------------------------------------------------ | --------------------------- |
| S1  | RAG-Monolith              | Feste Pipeline: Hybrid-Retrieval (BM25 + Embeddings, RRF), FlashRank-Reranking, eine Generierung | `src/systems/rag_monolith/` |
| S2  | Single-Agent RAG          | ReAct-Agent mit `search_section`, `retrieve_chunks`, `calculate`, `list_filings` und Reflexion   | `src/systems/rag_agent/`    |
| S3  | Single-Agent Long Context | Alle 12 Berichte im Systemprompt, Werkzeuge `calculate` und `list_filings`, Reflexion            | `src/systems/long_context/` |
| S4  | Multi-Agent Long Context  | LangGraph: Supervisor, parametrisierte Specialists, Synthesizer, Reflexion                       | `src/systems/multi_agent/`  |

## Repository-Struktur

```
configs/          base.yaml (Modell, Korpus), evaluation.yaml (Judge), best_config.yaml (Retrieval-Hyperparameter), je System eine YAML
src/common/       Ingestion, Chunking, Retrieval, LLM-Clients, Reflexion, geteilte Konventionen und Werkzeuge
src/systems/      Die vier Systeme; S1 inkl. Optuna-Ablation (ablation/)
src/evaluation/   Gold-Standard-Loader, RAGAS, eigene Metriken, Zitationsgenauigkeit, Begründungsketten-Metrik, eval_runner
scripts/          Evaluationsläufe, Aggregation, Statistik, Abbildungen, Judge-Validierung, Datenprüfung
data/             Gold Standard, Evaluationsergebnisse, Judge-Validierung, Optuna-Studie
notebooks/        Showcases (Datenpipeline, S1, S2) und die S1-Ablation
tests/            pytest-Suite
```

## Wo liegen welche Ergebnisse?

Alle Zahlen aus Kapitel 4 und 5 liegen versioniert unter `data/`. Die Aggregat-Dateien lassen sich ohne API-Aufrufe aus den Rohdaten neu erzeugen (siehe [Reproduktion, Stufe A](#stufe-a--auswertung-aus-den-rohdaten-ohne-api-kosten)).

### Aggregierte Ergebnisse

Alle Dateien liegen unter `data/results/final_eval/aggregate/`.

| Datei | Inhalt |
|---|---|
| `aggregate_summary.csv` | Mittelwert und SD je System über die drei Durchläufe für alle Metriken (RAGAS, `exact_match`, `answer_recall`, Refusal, Zitation, Reasoning, Tokens, Latenz, Kosten) |
| `aggregate_per_run.csv` | dieselben Metriken je Durchlauf und System |
| `hypothesis_tests.csv` | die sieben konfirmatorischen Tests zu H1–H3 mit Effekt, 95-%-KI, p, p_Holm und Verdikt |
| `h3_reasoning_sensitivity.csv` | H3-Reasoning-Endpunkte unter den Varianten der Emissionsregel |
| `emission_check.csv` | Emissionsrate der Begründungskette je System und Lauf |
| `exploratory_pairs.csv` | alle sechs Systempaare, explorativ und unadjustiert |
| `stratum_breakdown.csv` | Metriken je Stratum, Subtyp, Fenster, Rechenart, Bezeichnungsform und Einheit |
| `refusal_breakdown.csv` | Judge-Werte des Refusal-Stratums je Subtyp und Beleglage |
| `stability.csv` | Übereinstimmung des `exact_match`-Urteils über die drei Durchläufe |
| `violation_codes.csv` | Belegtheits- und Validitätscodes B1–B6 und V1–V5 je System |
| `efficiency.csv`, `cost_per_correct.csv` | Tokens, Latenz, Werkzeugaufrufe, Korrekturrate und Kosten je korrekte Antwort |
| `item_level_collapsed.csv` | ein Wert je Anfrage × System nach Mehrheitsregel bzw. Item-Mittel, Eingabe der Hypothesentests |

`statistical_analysis.md` und `aggregate_report.md` zeigen dieselben Zahlen als lesbare Tabellen.

### Rohdaten

- `data/results/final_eval/run1–run3/`: je Durchlauf
  - `eval_<system>_<timestamp>.json`: vollständige Rohdaten je System mit Frage, Sollwert, Antwort, Trajektorie, allen Metriken samt Judge-Begründungen und Laufzeitwerten
  - `full_eval_per_query.csv`: eine Zeile je System × Anfrage, `full_eval_summary.csv`: eine Zeile je System, `full_eval_report.md`
  - `run_provenance.json` (run2, run3): Git-Commit, Hashes von Konfiguration und Gold Standard, Korpus, Laufzeiten, Kosten, Ausfälle
  - `run.log`, `latency_contamination.csv`: Lauf-Log und markierte Latenz-Ausreißer
- `data/results/final_eval/refusal_hand_rating/`: blinde Bewertungsvorlage (`refusal_rating_run2.csv`), Schlüssel (`_KEY.json`) und Zusammenführung (`_MERGED.csv`).
- `data/results/judge_validation/`: Reserve-Stichprobe (`raw_eval_reserve/`), blinde Handbewertungen (`human_review_blind_reserve.csv`), verborgene Judge-Werte (`judge_scores_reference_reserve.json`), Validierungsreport, Kalibrierung der Belegtheit und Sensitivitätstest der Validität (`human_review_blind_sensitivity.csv`, `sensitivity_*.json`).

### Gold Standard

`data/gold_standard/` enthält den Gold Standard mit 150 Fragen in fünf Strata (FA-1 bis FA-4 sowie Refusal mit je 30 Fragen). Die Evaluation liest die englische Fassung `gold_standard_en.csv`, `gold_standard.csv` ist die deutsche Fassung. Schema und Konventionen: [gold_standard_README.md](data/gold_standard/gold_standard_README.md).

## Setup

Voraussetzungen: Python 3.12 und [uv](https://github.com/astral-sh/uv). `uv.lock` friert alle Abhängigkeitsversionen ein.

```bash
uv sync
cp .env.example .env
```

In `.env` einzutragen (nur für Stufe B und C nötig, siehe unten):

- `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`: Vertex AI für Gemini und die Embeddings, Anmeldung über `gcloud auth application-default login`. Gemessen wurde mit `europe-west1`; der `global`-Endpunkt lief in run1 in Quota-Limits.
- `OPENAI_API_KEY`: Judge-Modell
- `SEC_EDGAR_EMAIL`: Identifikation gegenüber SEC EDGAR

## Reproduktion

Alle Befehle werden aus dem Repository-Wurzelverzeichnis ausgeführt. Die Reproduktion hat drei Stufen mit steigendem Aufwand.

### Stufe A – Auswertung aus den Rohdaten (ohne API-Kosten)

Erzeugt alle Aggregate, Tests und Abbildungen aus Kapitel 5 aus den versionierten Rohdaten neu. Die Aggregat-Dateien sind dabei byte-identisch zu den eingecheckten, da der Bootstrap mit festem Seed 20260911 läuft.

```bash
uv run python scripts/aggregate_runs.py run1 run2 run3        # → aggregate/aggregate_*.csv
uv run python scripts/analyze_final_eval.py run1 run2 run3    # → aggregate/*.csv (Tests, Strata, Effizienz …)
uv run python scripts/plot_chapter5_figures.py                # → final_eval/figures/
uv run python scripts/build_refusal_rating_sheet.py --run run2 --merge   # Tabelle 17
uv run python scripts/analyze_judge_validation.py --reserve              # Tabelle 15
```

Mit `--out-dir <pfad>` schreiben `aggregate_runs.py` und `analyze_final_eval.py` in ein anderes Verzeichnis, sodass sich die Ausgaben mit den eingecheckten vergleichen lassen.

### Stufe B – Korpus aufbauen

Nötig für Stufe C. Gold Standard und Belegtheitsprüfung beziehen sich auf genau diese zwölf Berichte (AAPL, MSFT, AMZN, GOOGL × FY2020/2022/2024).

```bash
uv run python -c "from src.common import download_all_filings; download_all_filings()"
uv run python scripts/validate_ingestion.py       # Abschnitte vollständig?
uv run python scripts/validate_gold_standard.py   # Gold Standard konsistent mit dem Korpus?
```

Der Rohtext landet unter `data/raw/`, die Markdown-Berichte und Metadaten unter `data/processed/`. Beides ist nicht versioniert. Der Vektorindex für S1/S2 wird beim ersten Lauf unter `data/vectorstores/` inhaltsadressiert angelegt.

### Stufe C – Evaluationsläufe (API-Kosten)

```bash
# Smoke-Test mit wenigen Anfragen (kein gültiger Messlauf, in run_provenance.json als truncated markiert)
uv run python scripts/run_full_eval.py --run-ids smoke --limit 5

# Drei vollständige Läufe, streng nacheinander
uv run python scripts/run_full_eval.py --run-ids run1 run2 run3
```

Größenordnung je Lauf mit allen vier Systemen: rund 4,5 Stunden sequenziell und Generatorkosten von etwa 71 USD, davon S3 ≈ 63 USD, S4 ≈ 6,8 USD, S2 ≈ 0,9 USD, S1 ≈ 0,2 USD. Die Judge-Kosten kommen hinzu. `--parallel` wertet die vier Systeme gleichzeitig aus, vervierfacht aber die Anfragerate. Bestehende Laufverzeichnisse werden nur mit `--force` überschrieben. Danach folgt Stufe A.

Die Retrieval-Hyperparameter (Tabelle 12) bestimmt das Notebook `notebooks/experiments/sys1_rag_monolith/exp_ablation_optuna.ipynb` über `src/systems/rag_monolith/ablation/`. Das Ergebnis steht bereits in `configs/best_config.yaml`, die Studie liegt in `data/ablation_study.db`.

### Grenzen der Reproduzierbarkeit

- **Nicht deterministisch:** Trotz `temperature = 0` streuen die agentischen Systeme und der Judge von Lauf zu Lauf (Tabelle 22). Nur S1 war über alle drei Läufe identisch. Neue Läufe treffen die berichteten Mittelwerte daher nur innerhalb dieser Streuung.
- **Modellverfügbarkeit:** `gemini-2.5-flash`, `gemini-embedding-001` und `gpt-4o-mini` sind gehostete Modelle. Werden sie geändert oder abgekündigt, ist eine exakte Wiederholung nicht mehr möglich.
- **Korpus:** Die Berichte werden live von SEC EDGAR geladen. `run_provenance.json` hält die verwendeten Filings fest.

### Wie die Messung tatsächlich ablief

Diese Abweichungen vom Idealablauf sind in den Rohdaten dokumentiert und müssen beim Nachvollziehen berücksichtigt werden:

- **Einzeln gestartete Läufe:** Die drei Läufe wurden nacheinander einzeln gestartet (`--run-ids run2`, dann `--run-ids run3`), jeweils ohne `--parallel`. run2 und run3 liefen auf Commit `caf56e1` mit uncommitteten Änderungen; die Liste steht in `run_provenance.json` → `git.dirty_files`.
- **run1:** Für run1 existiert keine `run_provenance.json`. S3 ist in run1 ein zusammengeführtes Ergebnis (`eval_long_context_MERGED.json`): 28 Anfragen wurden nach dem Reflexions-Fix vom 10.09.2026 mit `scripts/rerun_and_merge_items.py` neu gemessen. Das steht in `summary.merged_from`.
- **Latenz:** Die Latenz aus run1 ist wegen Quota-Wartezeiten und Rechner-Standby nicht verwertbar und fließt nicht in die Latenzmittel ein. Das ist Standard in `aggregate_runs.py --latency-exclude`; betroffene Anfragen stehen in `latency_contamination.csv`. Alle Qualitätsmetriken nutzen alle drei Läufe.
- **`locus_faithfulness`** wurde nachträglich mit `scripts/rescore_locus_faithfulness.py` zu den bestehenden Rohdaten ergänzt; dabei wurde nur der Judge neu aufgerufen. Neue Läufe berechnen die Metrik direkt in `eval_runner.py`.

### Weitere Skripte

- **Läufe ergänzen und reparieren:** `run_systems_into_run.py` (einzelne Systeme in ein Laufverzeichnis), `rerun_and_merge_items.py` (einzelne Anfragen nachmessen), `rescore_locus_faithfulness.py`, `generate_report.py` (Laufberichte aus JSON neu erzeugen), `flag_latency_contamination.py`
- **Judge-Validierung (Kap. 3.6.3, 4.6.4):** `run_judge_validation_batch.py --reserve` → `build_judge_validation_sample.py --reserve` → Handbewertung → `merge_human_ratings.py --reserve` → `analyze_judge_validation.py --reserve`; zusätzlich `groundedness_calibration.py` und `validity_sensitivity_test.py`
- **Refusal-Handbewertung (Kap. 5.2.2):** `build_refusal_rating_sheet.py --run run2` erzeugt die blinde Vorlage, `--merge` wertet sie aus

Jedes Skript beschreibt Zweck und Aufruf in seinem Modul-Docstring.

## Tests

```bash
uv run pytest
```

Die Suite ersetzt alle Sprachmodell-Aufrufe durch Mocks und läuft ohne API-Zugang.
