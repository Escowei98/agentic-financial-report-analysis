# Agentic Financial Report Analysis

Dieses Projekt ist Teil einer Masterarbeit mit dem Ziel, vier verschiedene Architekturansätze zur Analyse von SEC EDGAR 10-K Finanzberichten zu entwerfen und vergleichend zu evaluieren. Im Zentrum steht die Forschungsfrage, ob agentenbasierte Systeme mit großem Kontext (Long-Context) klassische RAG-Baselines bei komplexen finanziellen Analyseaufgaben und Querverweisen besser abschneiden.

## Architekturen der 4 Systeme

Die Repository-Struktur hält diese vier Systeme strikt voneinander getrennt:

1. **System 1: Simple RAG (Monolith)** - Klassischer Retrieve -> Build Context -> Answer Ansatz ohne Agenten.
2. **System 2: Single-Agent RAG** - Ein Agent, der gezielt Werkzeuge (wie Retrieval) nutzt.
3. **System 3: Single-Agent Long Context** - Nutzt extrem große Kontextfenster (wie Gemini 1.5 Pro) komplett ohne vorgeschaltetes Retrieval.
4. **System 4: Multi-Agent Long Context** - Ein komplexes System (z. B. via LangGraph) aus mehreren spezialisierten Agenten-Rollen (Analyst, Reviewer).

Zusätzlich gibt es eine zentrale Daten-Pipeline im `data` und `src/common/` Bereich zum normalisierten Parsen der EDGAR 10-K HTML-Dokumente in analysierbares Markdown.

---

## Voraussetzungen

- **OS:** Windows 10/11, macOS oder Linux.
- **Python:** >= 3.10
- **uv:** Dieses Projekt nutzt [uv](https://github.com/astral-sh/uv) für blitzschnelles Dependency-Management.
- **API-Keys:**
  - Einen Key für die **Google Gemini API**, da das Projekt Modelle der Gemini-Familie als Standard nutzt.

---

## Setup & Installation

### 1. Repository klonen

```bash
git clone <dein-repository-url>
cd agentic-financial-report-analysis
```

### 2. Projektumgebung einrichten (mit uv)

Falls du `uv` noch nicht installiert hast, installiere es in deiner Konsole:

- **Windows:** `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- **Mac/Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`

Synchronisiere danach die Abhängigkeiten aus der `uv.lock`/`pyproject.toml` (dieser Befehl lädt die nötige Python Version und erstellt automatisch eine virtuelle Umgebung unter `.venv`):

```bash
uv sync
```

_(Alternativ auf Linux/Mac: Mach das Skript über `chmod +x` ausführbar falls noch Shell-Files existieren sollten)._

### 3. Umgebungsvariablen und Google API Key konfigurieren

Damit die Modelle laufen, musst du deinen Google API Schlüssel hinterlegen.

1. Gehe dazu in das [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Erstelle dir dort einen kostenlosen API-Key.
3. Erstelle im Hauptverzeichnis des Projekts eine `.env` Datei.
4. Trage deinen Key in diese Datei ein:

```env
GOOGLE_API_KEY="AIzaSy...Dein-Google-API-Key-Hier"
```

_(Optional: Hier können auch andere Umgebungsvariablen wie Tokens für LangSmith zum Tracing eingetragen werden)._

---

## Projekt ausführen

### Notebooks & Showcases

Um interaktiv mit den Systemen zu agieren, eignen sich die bereitgestellten Notebooks.

- Um Jupyter im `uv` Environment zu starten, tippe ins Terminal:
  ```bash
  uv run jupyter lab
  ```
  Gehe dann in den Ordner `notebooks/showcases/` und öffne z.B.:
  - `demo_data_pipeline.ipynb` - Zum Verständnis der Datenverarbeitung.
  - `demo_system1_rag_monolith.ipynb` - Laufendes System 1.
  - `demo_system2_agent_rag.ipynb` - Laufendes System 2.

### Evaluierungen & Experimente

Skripte für strukturierte Evaluierungen oder Experimentationen liegen überwiegend im `configs/`-Ordner verknüpft mit `src/evaluation/` oder im Bereich `notebooks/experiments/`.

Skripte können direkt im Context des Virtual Environments aufgerufen werden, zum Beispiel ein Ablation-Experiment in System 1:

```bash
uv run python -m src.systems.rag_monolith.ablation.optuna_search
```

## Tests ausführen

Um sicherzustellen, dass das rudimentäre Basis-Setup (Daten-Ingestion, Retrival etc) richtig funktioniert, stehen Pytest-Suites bereit:

```bash
# Tests für Linux/Mac/Windows starten
uv run pytest
```
