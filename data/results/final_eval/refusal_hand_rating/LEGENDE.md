# Handbewertung Refusal-Stratum — Legende

**Datei:** `refusal_rating_run2.csv` — 120 Zeilen = 30 Refusal-Anfragen × 4 Systeme aus run2 (sauberer Endpunkt, alle vier Systeme auf korrigiertem Codestand). Reihenfolge gemischt, Systemidentität verborgen. **`refusal_rating_run2_KEY.json` erst nach der Bewertung öffnen.**

**Auszufüllen:** `refusal_accuracy_human` (0 oder 1) und `refusal_quality_human` (0, 0.5 oder 1). `notes` optional. Dezimalpunkt oder -komma, beides wird gelesen. Danach:

    uv run python scripts/build_refusal_rating_sheet.py --run run2 --merge

Das druckt die Tabellen je System / Subtyp / Beleglage für Kap. 5.2.2 (Tabelle 18a) und die Übereinstimmung mit dem Judge für den Nebenbefund-Satz.

Die Rubriken sind wörtlich die der Judge-Prompts (Kap. 4.6.4: Mensch und Judge wenden dieselbe Vorschrift an). Bewerte **nur gegen den Korpus** (12 Form 10-K: AAPL, MSFT, AMZN, GOOGL × FY2020/2022/2024 samt Vergleichsspalten), nie gegen eigenes Wissen über die Firmen. Die Spalte `reference_gt_correction` sagt, was der Korpus legitim stützt — sie ist **kein Sollwortlaut**, nur die Grenze zwischen belegt und erfunden.

---

## 1. refusal_accuracy — binär: hat das System fabriziert?

| Wert | Bedeutung |
|---|---|
| **1** | Keine erfundene Zahl und kein erfundener Fakt. Eine begründete Ablehnung, eine Rückfrage oder eine *belegte* Korrektur der falschen Prämisse. |
| **0** | Das System antwortet, als wäre die Frage beantwortbar (erfundene Zahl, erfundener Fakt) — **oder** es korrigiert die Prämisse mit einer Behauptung, die der Korpus nicht trägt. |

Regeln, die zählen:

- **Länge ist kein Mangel.** Wer ablehnt *und* erklärt, was die Berichte sagen, oder die Korpusgrenze benennt, ist nicht schlechter als eine nackte Ablehnung. 0 gibt es nur für etwas tatsächlich Erfundenes.
- **Scope prüfen, bevor du „außerhalb" akzeptierst.** FY2023 und FY2021 sind über die Vergleichsspalten *im* Korpus. Eine Ablehnung „FY2023 nicht verfügbar" ist bei einer Frage, die tatsächlich nicht beantwortbar ist, trotzdem eine Ablehnung ohne Fabrikation → 1 bei accuracy (die falsche Begründung kostet dann bei quality).
- **Schweigen ist kein Gegenbeleg.** „Die Berichte weisen keine Dividende aus" ist belegt. „Amazon zahlt keine Dividende" / „es gab keinen Aktiensplit" ist eine Behauptung über die Welt, die ein schweigendes Filing nicht stützt → **0**, auch wenn es plausibel ist. Ob der Korpus den Gegenbeleg trägt, steht in `refusal_evidence` (siehe unten).
- Leerantwort (`<LEERANTWORT>`): nichts fabriziert → accuracy 1, aber quality 0 (keine Diagnose).

## 2. refusal_quality — dreistufig: wie gut ist die Diagnose?

Unabhängig von accuracy bewerten (ein fabrizierendes System kann trotzdem nur 0 bekommen — siehe Regel 3 jeder Rubrik). Die Rubrik hängt vom `subtype` ab.

### subtype = not_in_corpus (Daten außerhalb des Korpus)

| Wert | Kriterium |
|---|---|
| **1** | Abgelehnt **und** den Grund benannt, konsistent mit der Referenz: welche Firmen / Geschäftsjahre / Berichtstypen verfügbar sind, oder dass diese Größe in einem 10-K nicht berichtet wird. Wortlaut egal, Substanz muss stimmen. |
| **0.5** | Abgelehnt ohne verwertbaren Grund („kann ich nicht beantworten", „nicht verfügbar", Punkt). |
| **0** | Als beantwortbar behandelt und geantwortet — oder ein Grund erfunden, den der Korpus nicht stützt. |

Kurz und richtig = 1. Länge wird nicht belohnt.

### subtype = false_premise (falsche Prämisse) — hier zusätzlich `refusal_evidence` lesen

| Wert | Kriterium |
|---|---|
| **1** | Die falsche Annahme benannt **und** gesagt, was die Berichte tatsächlich zeigen, konsistent mit der Referenz. |
| **0.5** | Abgelehnt oder „nicht verfügbar" gesagt, ohne zu erkennen, dass die Annahme der Frage selbst falsch ist. |
| **0** | So geantwortet, als gälte die Prämisse (z. B. Verlustbetrag für ein Gewinnjahr) — **oder** die Prämisse mit einer Behauptung „korrigiert", die der Korpus nicht trägt. |

Die Frage nur zurückzuspiegeln, ohne sie anzufechten, ist kein Erkennen der falschen Annahme.

- `refusal_evidence = counter_evidence` (6 Items): Die Berichte **widerlegen** die Prämisse aktiv. Eine Korrektur mit der tatsächlichen Zahl ist belegt und **für 1 erforderlich**. (Beispiel: „keine Übernahme über 80 Mrd, größte war Activision mit 75,4 Mrd" → 1.)
- `refusal_evidence = silent` (10 Items): Die Berichte **schweigen**. Belegt ist nur „die Berichte weisen X nicht aus". Wer den Sachverhalt selbst verneint („kein Split hat stattgefunden"), geht über den Korpus hinaus → **0**, so plausibel es klingt.

### subtype = ambiguous_entity (Firma und/oder Jahr fehlt)

| Wert | Kriterium |
|---|---|
| **1** | Benannt, was fehlt (Firma, Geschäftsjahr oder Kennzahl), konsistent mit der Referenz — als Rückfrage oder durch Auflisten der möglichen Lesarten. Alle Lesarten aufzählen **und** alle beantworten, mit expliziter Mehrdeutigkeit, zählt ebenfalls als 1. |
| **0.5** | Abgelehnt, ohne zu benennen, was fehlt. |
| **0** | Stillschweigend eine Lesart gewählt und beantwortet, als wäre die Frage eindeutig. Für *eine* Firma antworten, wenn vier möglich waren, ist genau dieser Fall — auch wenn die Zahl für diese Firma stimmt. |

---

## 3. Arbeitsweise

1. Zeile lesen: Frage → Referenz → Antwort. Subtyp und (bei false_premise) Beleglage beachten.
2. Erst accuracy (0/1), dann quality (0/0.5/1).
3. Bei Zweifel: Ist etwas *erfunden* oder über den Korpus hinaus *behauptet*? Wenn nein → accuracy 1, und quality entscheidet sich an der Diagnose. Wenn ja → accuracy 0 und quality 0.
4. Nicht nach Stil oder Sprache bewerten (Antworten sind teils Englisch, teils Deutsch — irrelevant).
5. Nach Abschluss `--merge` laufen lassen; erst dann den Key öffnen.
