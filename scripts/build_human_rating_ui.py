"""
Builds a self-contained HTML rating instrument for the blind human review.

The blind CSV holds each item's trajectory as a multi-line quoted field,
which is close to unreadable in a spreadsheet — the very column the five
reasoning dimensions have to be rated on. This script renders the sample
into a single local HTML file that shows one item at a time (question,
ground truth, answer, trajectory) next to the judge's own rubric, and
collects 1-5 ratings.

Blindness is preserved: only the blind CSV is read. The judge scores in
judge_scores_reference*.json are never loaded, never embedded, and the
page has no way to reach them.

The rubric text is sliced out of CORE_JUDGE_PROMPT / AGENTIC_JUDGE_PROMPT
at build time rather than copied, so the human rater and the LLM judge can
never end up rating against different wording.

Output: data/results/judge_validation/rating_ui{,_reserve}.html
Open it in a browser, rate, then export the ratings JSON and merge it with
scripts/merge_human_ratings.py — the CSV itself is only ever written by
Python's csv module, never by the browser.
"""
import argparse
import csv
import json
import re
from pathlib import Path

from src.evaluation.custom_evaluator import (
    ANSWER_RECALL_PROMPT,
    COMPARISON_PROMPT,
    EXACT_MATCH_PROMPT,
    OVER_REFUSAL_PROMPT,
    QUALITATIVE_PROMPT,
    get_evidence_clause,
    get_refusal_accuracy_prompt,
    get_refusal_quality_prompt,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "judge_validation"
TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "rating_ui.html"
TEMPLATE_PASS2_PATH = Path(__file__).resolve().parent / "templates" / "rating_ui_pass2.html"

# Pass 1 rates reasoning PER UNIT, not per dimension: one step, one
# transition, one required sub-question, each answered yes or no.
#
# The retired instrument asked for five 1-5 scores per record and reached
# weighted kappa 0.03-0.17. The diagnosis was not that raters disagreed about
# the chains -- it was that an unanchored five-point scale gives them nothing
# to agree on, so "a logical gap" landed on a 2 for one rater and a 4 for
# another. A binary question about one concrete step removes that degree of
# freedom, which is the whole point of the rewrite.
REASONING_DIMENSIONS = ["groundedness", "validity", "completeness"]

DIMENSION_LABELS = {
    "groundedness": "Belegtheit (pro Schritt)",
    "validity": "Validität (pro Übergang)",
    "completeness": "Vollständigkeit (pro Teilfrage)",
}

# The codebook, in the rater's language, next to the cell it governs.
#
# These are the rules the judge prompt states verbatim; a rater working from a
# different rulebook than the judge is not validating the judge. The three
# marked (!) are the ones that keep the dimensions independent -- without them
# the same defect gets recorded under two dimensions by one rater and one by
# the next, and the agreement figure measures that instead of the chains.
DIMENSION_RULES = {
    "groundedness": [
        "Frage: Trägt die angezeigte Passage die Tatsachenbehauptung dieses Schritts?",
        "(!) Belegtheit ist NICHT Wahrheit. Gibt der Schritt korrekt wieder, was "
        "in der Passage steht, ist er belegt — auch wenn die Passage sachlich "
        "falsch ist. Korpusfehler sind Datenqualität, kein Reasoning-Fehler.",
        "(!) Belegtheit ist NICHT Zitationsgenauigkeit. Trägt die Passage die "
        "Aussage, ist der Schritt belegt — auch wenn eine andere Sektion die "
        "bessere Quelle gewesen wäre. Ob richtig zitiert wurde, misst "
        "citation_accuracy.",
        "Rechnet ein Schritt mit Zahlen aus früheren Schritten, ist das keine "
        "neue Tatsachenbehauptung über die Berichte. Bewerte die Behauptung, "
        "nicht die Rechnung.",
    ],
    "validity": [
        "Frage: Folgt dieser Schritt aus den vorangegangenen?",
        "(!) Validität ist UNABHÄNGIG von der Wahrheit der Prämissen. Eine "
        "korrekte Ableitung aus einer falschen Prämisse ist valide. Genau das "
        "hält die beiden Dimensionen auseinander.",
        "(!) Ein fehlender INHALT ist kein Validitätsfehler. Erwähnt die Kette "
        "etwas nicht, was sie hätte abdecken sollen, können die vorhandenen "
        "Schritte trotzdem sauber auseinander folgen — das ist eine "
        "Vollständigkeitslücke. Nur ein fehlendes ABLEITUNGSGLIED ist hier ein "
        "Fehler.",
        "Wichtigster Fall: Schritt 1 nennt eine Zahl für FY2024, Schritt 2 "
        "schließt „der Umsatz ist also gestiegen“. Die Schlussfolgerung mag "
        "zutreffen, aber nichts in der Kette sagt, wovon er gestiegen ist.",
        "Zusätzliche Schritte, Wiederholungen und Ausführlichkeit sind keine "
        "Validitätsfehler.",
        "Bewertet werden nur Übergänge auf einen Schlussfolgerungsschritt [I]. "
        "Ein [E]-Schritt bringt eine neue Tatsache aus den Berichten ein — das "
        "ist eine Prämisse, und eine Prämisse folgt aus nichts. Ob sie gedeckt "
        "ist, misst die Belegtheit.",
    ],
    "completeness": [
        "Frage: Behandelt die Kette diese Teilfrage erkennbar?",
        "(!) Schritte jenseits der Referenzzerlegung werden WEDER BELOHNT NOCH "
        "BESTRAFT. Zusätzliche Schritte, Wiederholungen, Zwischenzusammen"
        "fassungen und Länge haben hier keinerlei Gewicht. Eine lange und eine "
        "kurze Kette, die dieselben Teilfragen abdecken, sind gleich "
        "vollständig.",
        "Abgedeckt heißt erkennbar behandelt, in wie vielen Schritten auch "
        "immer — und unabhängig davon, ob der richtige Wert herauskommt. "
        "Abdeckung ist nicht Korrektheit.",
        "Erfinde keine weiteren Anforderungen. Die angezeigte Liste ist "
        "vollständig.",
    ],
}

# --- Pass 2: custom metrics, with the ground truth / gt_correction shown ---
#
# Unlike the five reasoning dimensions, each of these has its OWN rubric --
# exact_match dispatches on the item's answer type (numeric_atomic /
# comparison / qualitative, see _classify_answer_type in custom_evaluator.py)
# and refusal_quality dispatches on subtype plus, for false_premise, on the
# per-item evidence class. The rubric text is therefore computed per ITEM,
# not looked up once per dimension like build_rubrics() does for pass 1.

CUSTOM_DIMENSIONS = [
    "exact_match", "answer_recall", "refusal_accuracy", "refusal_quality", "over_refusal",
]
CUSTOM_SCALES = {
    "exact_match": ["0", "1"],
    "answer_recall": ["0", "0.5", "1"],
    "refusal_accuracy": ["0", "1"],
    "refusal_quality": ["0", "0.5", "1"],
    "over_refusal": ["0", "1"],
}
CUSTOM_LABELS = {
    "exact_match": "Exact Match",
    "answer_recall": "Answer Recall",
    "refusal_accuracy": "Refusal Accuracy",
    "refusal_quality": "Refusal Quality",
    "over_refusal": "Over-Refusal",
}

_RULES_RE = re.compile(r"Rules:\n(.*?)\n\nOutput ONLY", re.DOTALL)


def _read_rows(csv_path: Path) -> tuple[list[dict], str]:
    """Read the blind CSV, returning its rows and the delimiter used.

    The two samples were written with different delimiters (the primary
    with ',', the reserve with ';'), so it is sniffed rather than assumed —
    the same approach analyze_judge_validation.py takes, and for the same
    reason only the delimiter is taken from the sniffer.
    """
    with open(csv_path, encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;").delimiter
        except csv.Error:
            delimiter = ","
        rows = list(csv.DictReader(f, delimiter=delimiter, quotechar='"', doublequote=True))
    return rows, delimiter


def _cell_state(raw: str) -> str:
    """Classify a rating cell: 'na' (not applicable), 'rated', or 'open'."""
    value = (raw or "").strip()
    if not value:
        return "open"
    if value.upper() == "N/A":
        return "na"
    return "rated"


def _units_for(row: dict) -> dict[str, list[dict]]:
    """The concrete things this record asks the rater to decide.

    One entry per evidential step, per transition and per required
    sub-question. A record therefore carries as many cells as its chain has
    units, which is why the CSV holds JSON maps rather than scalars.
    """
    chain = json.loads(row.get("chain_json") or "[]")
    evidence = json.loads(row.get("evidence_json") or "{}")
    subquestions = json.loads(row.get("subquestions_json") or "[]")
    by_index = {int(step["step_index"]): step for step in chain}

    groundedness = [
        {
            "unit": str(step["step_index"]),
            "step": step["text"],
            # The passages THE JUDGE SAW, carried over from the run rather
            # than re-fetched. A rater shown different evidence is not
            # validating the judge, they are doing a different task.
            "evidence": evidence.get(str(step["step_index"]), []),
        }
        for step in chain if step.get("type") == "evidential"
    ]
    # Transitions INTO INFERENTIAL STEPS only -- an [E] step introduces a new
    # premise, and a premise does not follow from anything. Offering those as
    # cells is what produced 144 validity units in the pilot of which 63 could
    # not be decided; both judge and rater silently declined them.
    # See EVAL_DECISION_LOG.md [2026-09-09].
    validity = [
        {
            "unit": str(step["step_index"]),
            "step": step["text"],
            "preceding": [
                by_index[i]["text"] for i in sorted(by_index)
                if i < int(step["step_index"])
            ],
        }
        for step in chain
        if int(step["step_index"]) > 1
        and not (step.get("type") == "evidential" and step.get("loci"))
    ]
    completeness = [
        {"unit": str(i), "subquestion": text}
        for i, text in enumerate(subquestions, start=1)
    ]
    return {
        "groundedness": groundedness,
        "validity": validity,
        "completeness": completeness,
    }


def build_items(rows: list[dict]) -> list[dict]:
    """Project the blind CSV rows onto what the pass-1 page needs.

    Only the fields a rater is meant to see are carried over. Which
    dimensions an item is open for is decided from the CSV itself: a
    pre-filled "N/A" marks a dimension that does not apply (no chain, no
    transition, a refusal item), and an existing value marks one already
    rated in an earlier pass.
    """
    items = []
    for row in rows:
        units = _units_for(row)
        dimensions = {}
        for dim in REASONING_DIMENSIONS:
            state = _cell_state(row.get(f"{dim}_human", ""))
            if state == "na" or not units[dim]:
                continue
            dimensions[dim] = {
                "state": state,
                "existing": (row.get(f"{dim}_human") or "").strip(),
                "units": units[dim],
            }
        items.append({
            "review_id": row["review_id"],
            "fa_type": row.get("fa_type", ""),
            "system_label": row.get("system_label", ""),
            "question": row.get("question", ""),
            # Deliberately omitted: the reference answer is not shown in pass
            # 1. These dimensions judge the derivation, and the LLM judge does
            # not see it either -- handing it to one side only would make
            # kappa compare two different conditions.
            # See EVAL_DECISION_LOG.md [2026-09-06].
            "answer": row.get("answer", ""),
            "trajectory": row.get("trajectory", ""),
            "chain": json.loads(row.get("chain_json") or "[]"),
            "notes": row.get("notes", ""),
            "dimensions": dimensions,
        })
    return items


def _extract_rules(template: str) -> str:
    """Return the 'Rules:' block of a custom-metric prompt, verbatim.

    All six templates in custom_evaluator.py (exact/comparison/qualitative/
    recall/refusal-accuracy/over-refusal/the three refusal-quality variants)
    share this shape and the Rules text itself never contains a template
    placeholder, so this can run on the raw (or corpus-scope-filled)
    template string without a full .format() pass.
    """
    m = _RULES_RE.search(template)
    if not m:
        raise RuntimeError(
            "'Rules:' section not found — a custom_evaluator.py prompt "
            "changed shape, adjust this script."
        )
    return m.group(1).strip()


def _answer_type(fa_type: str, gt_unit: str) -> str:
    """Mirrors _classify_answer_type in custom_evaluator.py, from CSV fields."""
    if fa_type == "FA-4":
        return "comparison"
    if (gt_unit or "").strip().lower() == "text":
        return "qualitative"
    return "numeric_atomic"


def build_custom_rubric(dim: str, row: dict) -> str:
    """The Rules text for one custom dimension, dispatched exactly as the
    judge dispatches it for THIS row -- see custom_evaluator.py.
    """
    if dim == "exact_match":
        template = {
            "comparison": COMPARISON_PROMPT,
            "qualitative": QUALITATIVE_PROMPT,
            "numeric_atomic": EXACT_MATCH_PROMPT,
        }[_answer_type(row.get("fa_type", ""), row.get("gt_unit", ""))]
        return _extract_rules(template)
    if dim == "answer_recall":
        return _extract_rules(ANSWER_RECALL_PROMPT)
    if dim == "refusal_accuracy":
        return _extract_rules(get_refusal_accuracy_prompt())
    if dim == "refusal_quality":
        rules = _extract_rules(get_refusal_quality_prompt(row.get("subtype", "")))
        if row.get("subtype", "") == "false_premise":
            clause = get_evidence_clause(row.get("refusal_evidence", ""))
            return f"{clause}\n\n{rules}"
        return rules
    if dim == "over_refusal":
        return _extract_rules(OVER_REFUSAL_PROMPT)
    raise ValueError(f"Unknown custom dimension: {dim}")


def build_items_pass2(rows: list[dict]) -> list[dict]:
    """Project the blind CSV rows onto what the pass-2 page needs.

    Unlike build_items(), the ground truth / gt_correction ARE shown here --
    that is the point of the second pass, see EVAL_DECISION_LOG.md
    [2026-09-07] section 07 of the Umbauplan. Each open dimension carries
    its own rubric text and its own answer scale (0/1 or 0/0.5/1), computed
    per item rather than looked up once per dimension.
    """
    items = []
    for row in rows:
        dimensions = {}
        for dim in CUSTOM_DIMENSIONS:
            state = _cell_state(row.get(f"{dim}_human", ""))
            if state != "na":
                dimensions[dim] = {
                    "state": state,
                    "existing": (row.get(f"{dim}_human") or "").strip(),
                    "scale": CUSTOM_SCALES[dim],
                    "rubric": build_custom_rubric(dim, row),
                }
        items.append({
            "review_id": row["review_id"],
            "fa_type": row.get("fa_type", ""),
            "system_label": row.get("system_label", ""),
            "question": row.get("question", ""),
            "answer": row.get("answer", ""),
            "ground_truth": row.get("ground_truth", ""),
            "gt_correction": row.get("gt_correction", ""),
            "notes": row.get("notes", ""),
            "dimensions": dimensions,
        })
    return items


def build_rubrics() -> dict:
    """The codebook the page shows beside each cell.

    Assembled here rather than in the template so the rules a rater reads and
    the rules the judge prompt states stay in one place. There is no scale to
    describe any more -- every cell is yes or no -- so what is left is the
    codebook itself.

    The tool inventory the retired instrument shipped is gone with the agentic
    dimensions. It was already a documented compromise: the judge was told
    which tools THIS system had, and handing the rater the same per-system
    list would have identified the architecture behind the anonymous A-D label.
    """
    return {
        "labels": DIMENSION_LABELS,
        "rules": DIMENSION_RULES,
        "preamble": (
            "Bewertet wird die Herleitung, nicht die Antwort. Ob die Zahl am "
            "Ende stimmt, wird getrennt gemessen und ist hier ohne Belang: "
            "eine saubere Kette zu einem falschen Ergebnis ist hier gut, eine "
            "kaputte Kette zum richtigen Ergebnis ist es nicht. Entscheide "
            "jede Zeile einzeln — es gibt bewusst keine Gesamtnote."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample", default=None,
        help="Render an arbitrary named sample (e.g. 'sensitivity') instead of the primary/reserve pair. Reads human_review_blind_<name>.csv and writes rating_ui_<name>.html, so a one-off instrument check uses the same proven page rather than a bespoke one.",
    )
    parser.add_argument(
        "--reserve", action="store_true",
        help="Build the instrument for the reserve sample instead of the primary one.",
    )
    parser.add_argument(
        "--pass2", action="store_true",
        help="Build the second-pass instrument (exact_match / answer_recall / "
             "refusal_accuracy / refusal_quality / over_refusal, with the "
             "ground truth shown) instead of the first-pass reasoning-"
             "dimensions instrument. Rate pass 1 to completion first -- "
             "seeing the ground truth before rating the reasoning dimensions "
             "would contaminate the process judgment with outcome knowledge.",
    )
    args = parser.parse_args()

    if args.sample and args.reserve:
        raise SystemExit("--sample und --reserve schliessen sich aus.")
    sample_name = args.sample or ("reserve" if args.reserve else "primary")
    suffix = f"_{args.sample}" if args.sample else ("_reserve" if args.reserve else "")
    blind_csv = RESULTS_DIR / f"human_review_blind{suffix}.csv"

    if not blind_csv.exists():
        raise FileNotFoundError(f"Blind review CSV not found: {blind_csv}")

    rows, delimiter = _read_rows(blind_csv)

    if args.pass2:
        out_html = RESULTS_DIR / f"rating_ui_pass2{suffix}.html"
        template_path = TEMPLATE_PASS2_PATH
        items = build_items_pass2(rows)
        payload = {
            "sample": sample_name,
            "source_csv": blind_csv.name,
            "delimiter": delimiter,
            "items": items,
            "labels": CUSTOM_LABELS,
        }
    else:
        out_html = RESULTS_DIR / f"rating_ui{suffix}.html"
        template_path = TEMPLATE_PATH
        items = build_items(rows)
        payload = {
            "sample": sample_name,
            "source_csv": blind_csv.name,
            "delimiter": delimiter,
            "items": items,
            "rubrics": build_rubrics(),
        }

    def _cells(state: str) -> int:
        """Count RATING CELLS, not dimensions.

        In pass 1 one dimension of one record can hold a dozen cells (one per
        step), so counting dimensions would understate the work by an order of
        magnitude and make the progress display meaningless.
        """
        return sum(
            len(d.get("units", [1])) if d["state"] == state else 0
            for item in items for d in item["dimensions"].values()
        )

    open_cells, rated_cells = _cells("open"), _cells("rated")

    template = template_path.read_text(encoding="utf-8")
    html = template.replace(
        "__PAYLOAD__", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    )
    out_html.write_text(html, encoding="utf-8")

    print(f"Sample:        {payload['sample']} ({len(items)} items, delimiter {delimiter!r})")
    print(f"Open cells:    {open_cells}")
    print(f"Already rated: {rated_cells}")
    print(f"Written to:    {out_html}")
    print("\nOpen it in a browser, rate, then export and run:")
    print(
        f"  uv run python scripts/merge_human_ratings.py"
        f"{f' --sample {args.sample}' if args.sample else (' --reserve' if args.reserve else '')}"
        f"{' --pass2' if args.pass2 else ''} <exported.json>"
    )


if __name__ == "__main__":
    main()
