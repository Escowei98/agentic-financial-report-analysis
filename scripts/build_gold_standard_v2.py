"""
Build Gold Standard v2 from ablation_test_data.csv.

Rewrites every query in natural German sentence form with the entity
(company ticker OR name) and fiscal year explicitly present. Every query
is stratified 50/50 between ticker form (AAPL, MSFT, AMZN, GOOGL) and
name form (Apple, Microsoft, Amazon, Alphabet) within each query type,
so both surface-form variants are represented in balanced proportions.

Addresses the methodological issue surfaced in the Sys2 pre-evaluation
run: the original CSV left the primary entity in the Doc(s) metadata
column instead of the Query text, which penalized the agent for
correctly asking clarifying questions.

Output:
    notebooks/experiments/sys1_rag_monolith/ablation_test_data_v2.csv

The output CSV has one additional column 'entity_form' with values
'ticker' or 'name', which the evaluation pipeline can use for
sub-aggregation (e.g. RAGAS score by entity form).

The ground-truth values, source sections, query types, and IDs are
preserved unchanged. Only the Query column is rewritten.

Run with:
    uv run python scripts/build_gold_standard_v2.py
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_CSV = (
    PROJECT_ROOT
    / "notebooks"
    / "experiments"
    / "sys1_rag_monolith"
    / "ablation_test_data.csv"
)
TARGET_CSV = SOURCE_CSV.with_name("ablation_test_data_v2.csv")

# Deterministic stratification seed — do NOT change to preserve reproducibility.
STRATIFICATION_SEED = 42


# ---------------------------------------------------------------------------
#  Manually-authored rewrites (77 queries)
#
#  Each entry: id -> {"ticker_form": str, "name_form": str}
#
#  Patterns:
#    - Single / Cross-Sec: "Wie hoch war <metric> von <entity> in FY<year>?"
#    - Multi-Year:         "... von <entity> ... FY<year_a> ... FY<year_b> ..."
#    - Multi-Comp:         preserve compare phrasing, replace ticker vs name
# ---------------------------------------------------------------------------

REWRITES: dict[int, dict[str, str]] = {
    # ------- Single (1-20) -------
    1: {
        "ticker_form": "Wie hoch war der Gesamtumsatz (Total Revenue) von AAPL in FY2024?",
        "name_form": "Wie hoch war der Gesamtumsatz (Total Revenue) von Apple in FY2024?",
    },
    2: {
        "ticker_form": "Wie hoch war das Net Income von AAPL in FY2024?",
        "name_form": "Wie hoch war das Net Income von Apple in FY2024?",
    },
    3: {
        "ticker_form": "Wie hoch war die Debt-to-Equity Ratio von AAPL am Ende von FY2024?",
        "name_form": "Wie hoch war die Debt-to-Equity Ratio von Apple am Ende von FY2024?",
    },
    4: {
        "ticker_form": "Wie hoch war der Gesamtumsatz von AAPL in FY2023?",
        "name_form": "Wie hoch war der Gesamtumsatz von Apple in FY2023?",
    },
    5: {
        "ticker_form": "Wie hoch war das Net Income von AAPL in FY2023?",
        "name_form": "Wie hoch war das Net Income von Apple in FY2023?",
    },
    6: {
        "ticker_form": "Wie hoch war der Total Revenue von MSFT in FY2024?",
        "name_form": "Wie hoch war der Total Revenue von Microsoft in FY2024?",
    },
    7: {
        "ticker_form": "Wie hoch war das Net Income von MSFT in FY2024?",
        "name_form": "Wie hoch war das Net Income von Microsoft in FY2024?",
    },
    8: {
        "ticker_form": "Wie hoch war die Debt/Equity Ratio von MSFT in FY2024?",
        "name_form": "Wie hoch war die Debt/Equity Ratio von Microsoft in FY2024?",
    },
    9: {
        "ticker_form": "Wie hoch war der Total Revenue von MSFT in FY2023?",
        "name_form": "Wie hoch war der Total Revenue von Microsoft in FY2023?",
    },
    10: {
        "ticker_form": "Wie hoch war das Net Income von MSFT in FY2023?",
        "name_form": "Wie hoch war das Net Income von Microsoft in FY2023?",
    },
    11: {
        "ticker_form": "Wie hoch war der Total Revenue von AMZN in FY2024?",
        "name_form": "Wie hoch war der Total Revenue von Amazon in FY2024?",
    },
    12: {
        "ticker_form": "Wie hoch war das Net Income von AMZN in FY2024?",
        "name_form": "Wie hoch war das Net Income von Amazon in FY2024?",
    },
    13: {
        "ticker_form": "Wie hoch war die Debt-to-Equity Ratio von AMZN in FY2024?",
        "name_form": "Wie hoch war die Debt-to-Equity Ratio von Amazon in FY2024?",
    },
    14: {
        "ticker_form": "Wie hoch war der Total Revenue von AMZN in FY2023?",
        "name_form": "Wie hoch war der Total Revenue von Amazon in FY2023?",
    },
    15: {
        "ticker_form": "Wie hoch war das Net Income von AMZN in FY2023?",
        "name_form": "Wie hoch war das Net Income von Amazon in FY2023?",
    },
    16: {
        "ticker_form": "Wie hoch war der Total Revenue von GOOGL in FY2024?",
        "name_form": "Wie hoch war der Total Revenue von Alphabet in FY2024?",
    },
    17: {
        "ticker_form": "Wie hoch war das Net Income von GOOGL in FY2024?",
        "name_form": "Wie hoch war das Net Income von Alphabet in FY2024?",
    },
    18: {
        "ticker_form": "Wie hoch war die Debt/Equity Ratio von GOOGL in FY2024?",
        "name_form": "Wie hoch war die Debt/Equity Ratio von Alphabet in FY2024?",
    },
    19: {
        "ticker_form": "Wie hoch war der Total Revenue von GOOGL in FY2023?",
        "name_form": "Wie hoch war der Total Revenue von Alphabet in FY2023?",
    },
    20: {
        "ticker_form": "Wie hoch war das Net Income von GOOGL in FY2023?",
        "name_form": "Wie hoch war das Net Income von Alphabet in FY2023?",
    },
    # ------- Cross-Sec (21-40) -------
    21: {
        "ticker_form": "Wie hoch war die Operating Margin von AAPL in FY2024?",
        "name_form": "Wie hoch war die Operating Margin von Apple in FY2024?",
    },
    22: {
        "ticker_form": "Wie hoch war der Free Cash Flow von AAPL in FY2024?",
        "name_form": "Wie hoch war der Free Cash Flow von Apple in FY2024?",
    },
    23: {
        "ticker_form": "Welchen Anteil am Gesamtumsatz von AAPL hatten iPhones in FY2024?",
        "name_form": "Welchen Anteil am Gesamtumsatz von Apple hatten iPhones in FY2024?",
    },
    24: {
        "ticker_form": "Wie hoch war die Operating Margin von AAPL in FY2023?",
        "name_form": "Wie hoch war die Operating Margin von Apple in FY2023?",
    },
    25: {
        "ticker_form": "Wie hoch war der Anteil der R&D-Ausgaben am Umsatz von AAPL in FY2023?",
        "name_form": "Wie hoch war der Anteil der R&D-Ausgaben am Umsatz von Apple in FY2023?",
    },
    26: {
        "ticker_form": "Wie hoch war die Operating Margin von MSFT in FY2024?",
        "name_form": "Wie hoch war die Operating Margin von Microsoft in FY2024?",
    },
    27: {
        "ticker_form": "Welchen Anteil am Gesamtumsatz von MSFT hatte Azure in FY2024?",
        "name_form": "Welchen Anteil am Gesamtumsatz von Microsoft hatte Azure in FY2024?",
    },
    28: {
        "ticker_form": "Wie hoch war die Profit Margin des Cloud-Segments von MSFT in FY2024?",
        "name_form": "Wie hoch war die Profit Margin des Cloud-Segments von Microsoft in FY2024?",
    },
    29: {
        "ticker_form": "Wie hoch war die Operating Margin von MSFT in FY2023?",
        "name_form": "Wie hoch war die Operating Margin von Microsoft in FY2023?",
    },
    30: {
        "ticker_form": "Wie hoch waren die Capital Expenditures (CapEx) von MSFT in FY2023?",
        "name_form": "Wie hoch waren die Capital Expenditures (CapEx) von Microsoft in FY2023?",
    },
    31: {
        "ticker_form": "Wie hoch war die gesamte Operating Margin von AMZN in FY2024?",
        "name_form": "Wie hoch war die gesamte Operating Margin von Amazon in FY2024?",
    },
    32: {
        "ticker_form": "Wie hoch war die Operating Margin des AWS-Segments von AMZN in FY2024?",
        "name_form": "Wie hoch war die Operating Margin des AWS-Segments von Amazon in FY2024?",
    },
    33: {
        "ticker_form": "Wie hoch war der Gewinn oder Verlust des International-Segments von AMZN in FY2024?",
        "name_form": "Wie hoch war der Gewinn oder Verlust des International-Segments von Amazon in FY2024?",
    },
    34: {
        "ticker_form": "Wie hoch war die gesamte Operating Margin von AMZN in FY2023?",
        "name_form": "Wie hoch war die gesamte Operating Margin von Amazon in FY2023?",
    },
    35: {
        "ticker_form": "Welchen Anteil am Gesamtumsatz von AMZN hatte AWS in FY2023?",
        "name_form": "Welchen Anteil am Gesamtumsatz von Amazon hatte AWS in FY2023?",
    },
    36: {
        "ticker_form": "Wie hoch war die Operating Margin von GOOGL in FY2024?",
        "name_form": "Wie hoch war die Operating Margin von Alphabet in FY2024?",
    },
    37: {
        "ticker_form": "Wie hoch war die Operating Margin des Google-Cloud-Segments von GOOGL in FY2024?",
        "name_form": "Wie hoch war die Operating Margin des Google-Cloud-Segments von Alphabet in FY2024?",
    },
    38: {
        "ticker_form": "Welchen Anteil am Gesamtumsatz von GOOGL hatte der Werbeumsatz in FY2024?",
        "name_form": "Welchen Anteil am Gesamtumsatz von Alphabet hatte der Werbeumsatz in FY2024?",
    },
    39: {
        "ticker_form": "Wie hoch war die Operating Margin von GOOGL in FY2023?",
        "name_form": "Wie hoch war die Operating Margin von Alphabet in FY2023?",
    },
    40: {
        "ticker_form": "Wie hat sich die Mitarbeiterzahl (Headcount) von GOOGL in FY2023 entwickelt?",
        "name_form": "Wie hat sich die Mitarbeiterzahl (Headcount) von Alphabet in FY2023 entwickelt?",
    },
    # ------- Multi-Year (41-55, 57, 58) -------
    41: {
        "ticker_form": "Wie hoch war das Umsatzwachstum von AAPL in FY2024 im Vergleich zu FY2022?",
        "name_form": "Wie hoch war das Umsatzwachstum von Apple in FY2024 im Vergleich zu FY2022?",
    },
    42: {
        "ticker_form": "Wie hoch war das YoY-Umsatzwachstum von AAPL von FY2023 auf FY2024?",
        "name_form": "Wie hoch war das YoY-Umsatzwachstum von Apple von FY2023 auf FY2024?",
    },
    43: {
        "ticker_form": "Wie hat sich das Net Income von AAPL von FY2023 auf FY2024 verändert?",
        "name_form": "Wie hat sich das Net Income von Apple von FY2023 auf FY2024 verändert?",
    },
    44: {
        "ticker_form": "Wie entwickelte sich die Operating Margin von AAPL über die Jahre FY2022, FY2023 und FY2024 (steigend oder fallend)?",
        "name_form": "Wie entwickelte sich die Operating Margin von Apple über die Jahre FY2022, FY2023 und FY2024 (steigend oder fallend)?",
    },
    45: {
        "ticker_form": "Wie haben sich die R&D-Ausgaben von AAPL von FY2022 auf FY2024 verändert?",
        "name_form": "Wie haben sich die R&D-Ausgaben von Apple von FY2022 auf FY2024 verändert?",
    },
    46: {
        "ticker_form": "Wie hat sich das Net Income von MSFT von FY2022 auf FY2024 verändert?",
        "name_form": "Wie hat sich das Net Income von Microsoft von FY2022 auf FY2024 verändert?",
    },
    47: {
        "ticker_form": "Wie hoch war das YoY-Umsatzwachstum von MSFT von FY2023 auf FY2024?",
        "name_form": "Wie hoch war das YoY-Umsatzwachstum von Microsoft von FY2023 auf FY2024?",
    },
    48: {
        "ticker_form": "Wie hoch war das YoY-Wachstum des Azure-Umsatzes von MSFT von FY2023 auf FY2024?",
        "name_form": "Wie hoch war das YoY-Wachstum des Azure-Umsatzes von Microsoft von FY2023 auf FY2024?",
    },
    49: {
        "ticker_form": "Wie entwickelten sich die Capital Expenditures (CapEx) von MSFT über FY2022, FY2023 und FY2024 (steigend oder fallend)?",
        "name_form": "Wie entwickelten sich die Capital Expenditures (CapEx) von Microsoft über FY2022, FY2023 und FY2024 (steigend oder fallend)?",
    },
    50: {
        "ticker_form": "Wie hat sich die Mitarbeiterzahl von MSFT von FY2022 auf FY2024 verändert?",
        "name_form": "Wie hat sich die Mitarbeiterzahl von Microsoft von FY2022 auf FY2024 verändert?",
    },
    51: {
        "ticker_form": "Wie hoch war das YoY-Wachstum des AWS-Umsatzes von AMZN von FY2022 auf FY2024?",
        "name_form": "Wie hoch war das YoY-Wachstum des AWS-Umsatzes von Amazon von FY2022 auf FY2024?",
    },
    52: {
        "ticker_form": "Wie hoch war das YoY-Umsatzwachstum (gesamt) von AMZN von FY2023 auf FY2024?",
        "name_form": "Wie hoch war das YoY-Umsatzwachstum (gesamt) von Amazon von FY2023 auf FY2024?",
    },
    53: {
        "ticker_form": "Wie hat sich das Net Income von AMZN von FY2023 auf FY2024 verändert?",
        "name_form": "Wie hat sich das Net Income von Amazon von FY2023 auf FY2024 verändert?",
    },
    54: {
        "ticker_form": "Wie entwickelte sich die Operating Margin des AWS-Segments von AMZN über FY2022, FY2023 und FY2024?",
        "name_form": "Wie entwickelte sich die Operating Margin des AWS-Segments von Amazon über FY2022, FY2023 und FY2024?",
    },
    55: {
        "ticker_form": "Wie haben sich die Fulfillment Costs von AMZN von FY2022 auf FY2024 verändert?",
        "name_form": "Wie haben sich die Fulfillment Costs von Amazon von FY2022 auf FY2024 verändert?",
    },
    57: {
        "ticker_form": "Wie hoch war das YoY-Umsatzwachstum von GOOGL von FY2023 auf FY2024?",
        "name_form": "Wie hoch war das YoY-Umsatzwachstum von Alphabet von FY2023 auf FY2024?",
    },
    58: {
        "ticker_form": "Wie hoch war das YoY-Wachstum des Google-Cloud-Umsatzes von GOOGL von FY2023 auf FY2024?",
        "name_form": "Wie hoch war das YoY-Wachstum des Google-Cloud-Umsatzes von Alphabet von FY2023 auf FY2024?",
    },
    # ------- Multi-Comp (61-80) -------
    61: {
        "ticker_form": "Welches Unternehmen hatte mehr Gesamtumsatz in FY2024, AAPL oder MSFT?",
        "name_form": "Welches Unternehmen hatte mehr Gesamtumsatz in FY2024, Apple oder Microsoft?",
    },
    62: {
        "ticker_form": "Welches Unternehmen hatte das höhere Net Income in FY2024, AAPL oder MSFT?",
        "name_form": "Welches Unternehmen hatte das höhere Net Income in FY2024, Apple oder Microsoft?",
    },
    63: {
        "ticker_form": "Wie hoch war die Differenz im Gesamtumsatz zwischen AMZN und GOOGL in FY2024?",
        "name_form": "Wie hoch war die Differenz im Gesamtumsatz zwischen Amazon und Alphabet in FY2024?",
    },
    64: {
        "ticker_form": "Welches Unternehmen hatte mehr Net Income in FY2024, MSFT oder GOOGL?",
        "name_form": "Welches Unternehmen hatte mehr Net Income in FY2024, Microsoft oder Alphabet?",
    },
    65: {
        "ticker_form": "Welches Unternehmen hatte den höheren Gesamtumsatz in FY2024, AAPL oder AMZN?",
        "name_form": "Welches Unternehmen hatte den höheren Gesamtumsatz in FY2024, Apple oder Amazon?",
    },
    66: {
        "ticker_form": "Wer hatte die höhere Operating Margin in FY2024, AAPL oder MSFT?",
        "name_form": "Wer hatte die höhere Operating Margin in FY2024, Apple oder Microsoft?",
    },
    67: {
        "ticker_form": "Wer hatte die höhere Operating Margin in FY2024, AMZN oder GOOGL?",
        "name_form": "Wer hatte die höhere Operating Margin in FY2024, Amazon oder Alphabet?",
    },
    68: {
        "ticker_form": "Ist die Debt/Equity Ratio von AAPL höher als die von AMZN in FY2024?",
        "name_form": "Ist die Debt/Equity Ratio von Apple höher als die von Amazon in FY2024?",
    },
    69: {
        "ticker_form": "Wer hat die niedrigere Debt/Equity Ratio in FY2024, MSFT oder GOOGL?",
        "name_form": "Wer hat die niedrigere Debt/Equity Ratio in FY2024, Microsoft oder Alphabet?",
    },
    70: {
        "ticker_form": "Wer hatte die höhere Gross Margin in FY2024, AAPL oder GOOGL?",
        "name_form": "Wer hatte die höhere Gross Margin in FY2024, Apple oder Alphabet?",
    },
    71: {
        "ticker_form": "Cloud-Segment: Wer hatte mehr Umsatz in FY2024, AMZN mit AWS oder MSFT mit Azure?",
        "name_form": "Cloud-Segment: Wer hatte mehr Umsatz in FY2024, Amazon mit AWS oder Microsoft mit Azure?",
    },
    72: {
        "ticker_form": "Cloud-Segment: Wer hatte mehr Umsatz in FY2024, MSFT mit Azure oder GOOGL mit Google Cloud?",
        "name_form": "Cloud-Segment: Wer hatte mehr Umsatz in FY2024, Microsoft mit Azure oder Alphabet mit Google Cloud?",
    },
    73: {
        "ticker_form": "Wer hatte das höhere YoY-Umsatzwachstum von FY2023 auf FY2024, AAPL oder MSFT?",
        "name_form": "Wer hatte das höhere YoY-Umsatzwachstum von FY2023 auf FY2024, Apple oder Microsoft?",
    },
    74: {
        "ticker_form": "Wer hatte das höhere YoY-Net-Income-Wachstum von FY2023 auf FY2024, AMZN oder GOOGL?",
        "name_form": "Wer hatte das höhere YoY-Net-Income-Wachstum von FY2023 auf FY2024, Amazon oder Alphabet?",
    },
    75: {
        "ticker_form": "Wer generierte mehr Operating Cash Flow in FY2024, AAPL oder AMZN?",
        "name_form": "Wer generierte mehr Operating Cash Flow in FY2024, Apple oder Amazon?",
    },
    76: {
        "ticker_form": "Welches der 4 Unternehmen (AAPL, MSFT, AMZN, GOOGL) hatte den höchsten Gesamtumsatz in FY2024?",
        "name_form": "Welches der 4 Unternehmen (Apple, Microsoft, Amazon, Alphabet) hatte den höchsten Gesamtumsatz in FY2024?",
    },
    77: {
        "ticker_form": "Welches der 4 Unternehmen (AAPL, MSFT, AMZN, GOOGL) hatte das höchste Net Income in FY2024?",
        "name_form": "Welches der 4 Unternehmen (Apple, Microsoft, Amazon, Alphabet) hatte das höchste Net Income in FY2024?",
    },
    78: {
        "ticker_form": "Welches der 4 Unternehmen (AAPL, MSFT, AMZN, GOOGL) hatte die höchste Operating Margin in FY2024?",
        "name_form": "Welches der 4 Unternehmen (Apple, Microsoft, Amazon, Alphabet) hatte die höchste Operating Margin in FY2024?",
    },
    79: {
        "ticker_form": "Welches der 4 Unternehmen (AAPL, MSFT, AMZN, GOOGL) hatte den niedrigsten Gesamtumsatz in FY2024?",
        "name_form": "Welches der 4 Unternehmen (Apple, Microsoft, Amazon, Alphabet) hatte den niedrigsten Gesamtumsatz in FY2024?",
    },
    80: {
        "ticker_form": "Welches der 4 Unternehmen (AAPL, MSFT, AMZN, GOOGL) hat die höchste Debt/Equity Ratio in FY2024?",
        "name_form": "Welches der 4 Unternehmen (Apple, Microsoft, Amazon, Alphabet) hat die höchste Debt/Equity Ratio in FY2024?",
    },
}


def stratified_assignment(
    items_by_type: dict[str, list[int]],
    seed: int = STRATIFICATION_SEED,
) -> dict[int, str]:
    """
    Assign each ID to either 'ticker' or 'name' form, stratified by query type.

    Within each query type, roughly half the IDs get 'ticker' and the other
    half get 'name'. For odd counts the extra item goes to 'name' (arbitrary
    but deterministic choice).

    Args:
        items_by_type: Mapping from query type (e.g. 'Single') to list of IDs.
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping each ID to 'ticker' or 'name'.
    """
    assignment: dict[int, str] = {}
    rng = random.Random(seed)

    for qtype, ids in items_by_type.items():
        shuffled = ids[:]
        rng.shuffle(shuffled)
        half = len(shuffled) // 2
        ticker_ids = set(shuffled[:half])
        for i in ids:
            assignment[i] = "ticker" if i in ticker_ids else "name"

    return assignment


def build_v2() -> None:
    """Read the original CSV, apply rewrites with stratified entity form, write v2."""
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Source CSV not found: {SOURCE_CSV}")

    with open(SOURCE_CSV, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        rows = [row for row in reader if row and len(row) >= 6]

    # Parse IDs grouped by query type for stratified assignment
    items_by_type: dict[str, list[int]] = {}
    for row in rows:
        try:
            item_id = int(row[0].strip())
        except (ValueError, IndexError):
            continue
        qtype = row[1].strip()
        items_by_type.setdefault(qtype, []).append(item_id)

    assignment = stratified_assignment(items_by_type)

    # Sanity check: every row has a rewrite entry
    missing_rewrites = []
    for row in rows:
        try:
            item_id = int(row[0].strip())
        except ValueError:
            continue
        if item_id not in REWRITES:
            missing_rewrites.append(item_id)
    if missing_rewrites:
        raise ValueError(
            f"Missing rewrites for {len(missing_rewrites)} IDs: {missing_rewrites}"
        )

    # Write the v2 CSV
    out_header = header + ["entity_form"]
    with open(TARGET_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(out_header)

        for row in rows:
            try:
                item_id = int(row[0].strip())
            except ValueError:
                continue

            form = assignment[item_id]
            rewrite = REWRITES[item_id]
            new_query = rewrite["ticker_form"] if form == "ticker" else rewrite["name_form"]

            # Preserve ID, Typ, Doc(s); replace Query; preserve GT/Source/Rationale
            new_row = [
                row[0],                              # ID
                row[1],                              # Typ
                row[2],                              # Doc(s)
                new_query,                           # Query (rewritten)
                row[4] if len(row) > 4 else "",      # GT-Wert
                row[5] if len(row) > 5 else "",      # Source
                row[6] if len(row) > 6 else "",      # Rationale
                form,                                # NEW: entity_form
            ]
            writer.writerow(new_row)

    # Summary
    print(f"Wrote {TARGET_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Stratification seed: {STRATIFICATION_SEED}")
    print("\nEntity-form distribution by query type:")
    for qtype, ids in items_by_type.items():
        ticker_count = sum(1 for i in ids if assignment[i] == "ticker")
        name_count = sum(1 for i in ids if assignment[i] == "name")
        print(f"  {qtype:12s}: {ticker_count} ticker / {name_count} name (total {len(ids)})")

    total_ticker = sum(1 for v in assignment.values() if v == "ticker")
    total_name = sum(1 for v in assignment.values() if v == "name")
    print(f"  {'Total':12s}: {total_ticker} ticker / {total_name} name")


if __name__ == "__main__":
    build_v2()
