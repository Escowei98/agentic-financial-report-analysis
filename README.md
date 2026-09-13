# Agentic Financial Report Analysis

Code und Ergebnisse zur Masterarbeit über den Vergleich von vier Architekturen zur Beantwortung von Fragen zu SEC-10-K-Berichten. Untersucht wird, ob agentenbasierte und Long-Context-Systeme klassische RAG-Pipelines bei dokument- und abschnittsübergreifenden Finanzanalysen übertreffen.

## Systeme

| | System | Kurzbeschreibung | Code |
|---|---|---|---|
| S1 | RAG-Monolith | Feste Pipeline: Hybrid-Retrieval (BM25 + Embeddings), FlashRank-Reranking, Generierung | `src/systems/rag_monolith/` |
| S2 | Single-Agent RAG | ReAct-Agent mit `search_section`, `retrieve_chunks`, `calculate`, `list_filings` und Reflexion | `src/systems/rag_agent/` |
| S3 | Single-Agent Long Context | Alle 12 Berichte im Kontextfenster, Werkzeuge `calculate` und `list_filings`, Reflexion | `src/systems/long_context/` |
| S4 | Multi-Agent Long Context | LangGraph: Supervisor, parametrisierte Specialists, Synthesizer, Reflexion | `src/systems/multi_agent/` |

Alle Systeme nutzen `gemini-2.5-flash` und dieselben geteilten Komponenten aus `src/common/` (Antwortformat-, Nicht-Beantwortbarkeits- und Begründungsketten-Konvention, Few-Shot-Szenarien, Werkzeuge). Bewertet wird mit `gpt-4o-mini` als Judge aus einer anderen Modellfamilie.

## Repository-Struktur

```
configs/          Systemkonfiguration (base.yaml), Judge (evaluation.yaml), S1-Hyperparameter (best_config.yaml)
src/common/       Ingestion, Chunking, Retrieval, LLM-Clients, geteilte Konventionen und Werkzeuge
src/systems/      Die vier Systeme; S1 inkl. Optuna-Ablation
src/evaluation/   Gold-Standard-Loader, RAGAS, eigene Metriken, Zitationsgenauigkeit, Begründungsketten-Metrik
scripts/          Evaluationsläufe, Aggregation, Statistik, Abbildungen, Judge-Validierung, Datenprüfung
data/             Gold Standard, Evaluationsergebnisse, Optuna-Studie
notebooks/        Showcases (Datenpipeline, S1, S2) und die S1-Ablation
tests/            pytest-Suite
```

### Daten

- `data/gold_standard/` — Gold Standard v6.1: 150 Fragen in fünf Strata (FA-1 bis FA-4, FA-Refusal), deutsch und englisch. Schema und Konventionen: [gold_standard_README.md](data/gold_standard/gold_standard_README.md).
- `data/results/final_eval/run1–run3/` — die drei Evaluationsläufe: Rohdaten je System (`eval_<system>_*.json`), Reports, Lauf-Log, Provenienz und Latenz-Kontamination.
- `data/results/final_eval/aggregate/` — Mittelwerte über die Läufe, Hypothesentests, explorative Vergleiche, Stratum- und Refusal-Auswertungen.
- `data/results/final_eval/refusal_hand_rating/` — blinde Handbewertung des Refusal-Stratums (run2).
- `data/results/judge_validation/` — Judge-Validierung: Reserve-Stichprobe, menschliche Bewertungen, Validierungsreport, Sensitivitätstest der Validity-Dimension.
- `data/raw/`, `data/processed/` — der Korpus (AAPL, MSFT, AMZN, GOOGL × FY2020/2022/2024); nicht versioniert, wird von der Ingestion aus SEC EDGAR erzeugt.
- `data/vectorstores/`, `data/flashrank_cache/` — lokale, regenerierbare Caches (nicht versioniert).

## Setup

Voraussetzungen: Python 3.12 und [uv](https://github.com/astral-sh/uv).

```bash
uv sync
cp .env.example .env
```

In `.env` einzutragen:

- `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` — Vertex AI für die Gemini-Modelle (Anmeldung über `gcloud auth application-default login`)
- `OPENAI_API_KEY` — Judge-Modell
- `SEC_EDGAR_EMAIL` — Identifikation gegenüber SEC EDGAR

## Reproduktion

```bash
# 1. Korpus laden und prüfen
uv run python -c "from src.common import download_all_filings; download_all_filings()"
uv run python scripts/validate_ingestion.py
uv run python scripts/validate_gold_standard.py

# 2. Drei vollständige Läufe aller vier Systeme (API-Kosten!)
uv run python scripts/run_full_eval.py --parallel --run-ids run1 run2 run3

# 3. Auswertung und Abbildungen (ohne API-Aufrufe)
uv run python scripts/aggregate_runs.py run1 run2 run3
uv run python scripts/analyze_final_eval.py run1 run2 run3
uv run python scripts/plot_chapter5_figures.py   # schreibt nach docs/thesis/figures/
```

Weitere Skripte:

- **Läufe ergänzen:** `run_systems_into_run.py`, `rerun_and_merge_items.py`, `rescore_locus_faithfulness.py`, `generate_report.py`, `flag_latency_contamination.py`
- **Judge-Validierung:** `run_judge_validation_batch.py`, `build_judge_validation_sample.py`, `merge_human_ratings.py`, `analyze_judge_validation.py`, `groundedness_calibration.py`, `validity_sensitivity_test.py`
- **Refusal-Handbewertung:** `build_refusal_rating_sheet.py`

Jedes Skript beschreibt Zweck und Aufruf in seinem Modul-Docstring.

## Tests

```bash
uv run pytest
```
