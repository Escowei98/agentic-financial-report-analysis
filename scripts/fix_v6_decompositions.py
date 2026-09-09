"""
Hand-checked corrections to the 25 reference decompositions the generator
could not extract a metric name for.

`build_gold_standard_v6.py` derives decompositions from the question text and
flags the rows where extraction was unreliable. It flagged 25 of 120, mostly
questions whose opener it does not match ("How has X changed...", "Is X higher
than Y", "Cloud segment: Who had more revenue...") and questions whose metric
sits inside an explanatory parenthesis ("R&D intensity (R&D expense as a
percentage of revenue)"). The structure it produced was right in every case;
what was missing was the name of the quantity, and in six rows the number of
periods.

This file holds the corrections literally rather than making the generator
cleverer, for the same reason the generator writes a review sheet at all: a
decomposition is the standard Dimension 3 scores against, and it should be
readable and arguable, not the output of one more regex.

THE PRINCIPLE APPLIED
---------------------
A decomposition follows the level at which the QUESTION asks. Id 18 asks for
Alphabet's Debt/Equity Ratio, so its components (Total Liabilities, Total
Stockholders' Equity) are required sub-questions. Id 66 asks which of two
companies has the lower ratio, so the required sub-questions are the two
ratios and the comparison -- not four balance-sheet lookups. Deriving the
ratio is one way to answer it, reading it off is another, and a decomposition
that demanded the derivation would mark a correct chain incomplete.

Run once, after `build_gold_standard_v6.py`, and not again afterwards:
    uv run python scripts/fix_v6_decompositions.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GS_DIR = PROJECT_ROOT / "data" / "gold_standard"


def _change(metric: str, company: str, early: str, late: str, verb: str) -> list[str]:
    return [
        f"Establish {metric} for {company} in FY{early}.",
        f"Establish {metric} for {company} in FY{late}.",
        verb,
    ]


def _compare(metric_a: str, a: str, metric_b: str, b: str, year: str, verb: str) -> list[str]:
    return [
        f"Establish {metric_a} for {a} in FY{year}.",
        f"Establish {metric_b} for {b} in FY{year}.",
        verb,
    ]


CHANGE = "Compute the change between the two figures."
NAME_COMPANY = "Compare the figures and name the company asked for."
NAME_YEAR = "Compare the values across the periods and name the fiscal year asked for."

CORRECTIONS: dict[int, list[str]] = {
    # --- metric name only; structure was already right -------------------
    30: ["Establish Capital Expenditures (CapEx) for Microsoft in FY2023."],
    43: _change("Net Income", "Apple", "2020", "2024", CHANGE),
    45: _change("R&D expenses", "Apple", "2022", "2024", CHANGE),
    46: _change("Net Income", "Microsoft", "2022", "2024", CHANGE),
    48: _change("Net Income", "Microsoft", "2020", "2024", CHANGE),
    50: _change("the number of employees", "Microsoft", "2022", "2024", CHANGE),
    53: _change("Net Income", "Amazon", "2020", "2024", CHANGE),
    55: _change("Fulfillment Costs", "Amazon", "2022", "2024", CHANGE),
    57: _change("Net Income", "Alphabet", "2020", "2024", CHANGE),
    49: _change("Capital Expenditures (CapEx)", "Microsoft", "2022", "2024",
                "State the overall direction of the development."),
    106: _change("Gross Margin", "Apple", "2023", "2024",
                 "Express the change in percentage points."),
    65: _compare("the Debt/Equity Ratio", "Apple", "the Debt/Equity Ratio", "Amazon",
                 "2024", "Compare the two ratios and answer whether Apple's is higher."),
    66: _compare("the Debt/Equity Ratio", "Microsoft", "the Debt/Equity Ratio",
                 "Alphabet", "2024", NAME_COMPANY),
    72: _compare("Operating Cash Flow", "Apple", "Operating Cash Flow", "Amazon",
                 "2024", NAME_COMPANY),
    116: _compare("Operating Income", "Alphabet", "Operating Income", "Microsoft",
                  "2024", "Compute the difference between the figures."),

    # --- cloud comparisons: each company's segment has its own name ------
    68: _compare("AWS segment revenue", "Amazon",
                 "Intelligent Cloud segment revenue", "Microsoft", "2024", NAME_COMPANY),
    69: _compare("Intelligent Cloud segment revenue", "Microsoft",
                 "Google Cloud segment revenue", "Alphabet", "2024", NAME_COMPANY),
    120: _compare("AWS segment revenue", "Amazon",
                  "Google Cloud segment revenue", "Alphabet", "2024", NAME_COMPANY),

    # --- R&D intensity: the parenthesis names the two components ---------
    88: ["Establish R&D expense for Apple in FY2024.",
         "Establish total revenue for Apple in FY2024.",
         "Express R&D expense as a percentage of revenue."],
    89: ["Establish R&D expense for Microsoft in FY2024.",
         "Establish total revenue for Microsoft in FY2024.",
         "Express R&D expense as a percentage of revenue."],
    90: ["Establish R&D expense for Alphabet in FY2024.",
         "Establish total revenue for Alphabet in FY2024.",
         "Express R&D expense as a percentage of revenue."],

    # --- structure corrected too: every period in the span is required ---
    #
    # The generator takes the fiscal years the question NAMES. Where a
    # question spans a range ("between FY2022 and FY2024", "in which fiscal
    # year"), the intermediate year is required as well -- without FY2023 you
    # cannot know which year was highest. A 10-K carries three comparative
    # years, so FY2023 is available even though no FY2023 filing is in the
    # corpus.
    107: [f"Establish Operating Margin for Microsoft in FY{y}." for y in (2020, 2022, 2024)]
         + [NAME_YEAR],
    108: [f"Establish Operating Cash Flow for Amazon in FY{y}." for y in (2022, 2023, 2024)]
         + [NAME_YEAR],
    109: [f"Establish Net Income for Apple in FY{y}." for y in (2022, 2023, 2024)]
         + [NAME_YEAR],
    110: [f"Establish total revenue for Alphabet in FY{y}." for y in (2022, 2023, 2024)]
         + ["Determine for each year whether revenue grew over the previous one.",
            "Count the consecutive years of growth."],
}


def apply(filename: str) -> int:
    path = GS_DIR / filename
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    applied = 0
    for row in rows:
        item_id = int(row["id"])
        if item_id not in CORRECTIONS:
            continue
        if "the figure the question asks for" not in row["reference_decomposition"]:
            raise SystemExit(
                f"{filename}, id {item_id}: no placeholder left. This script is "
                "a one-off; it has already run, or the row was edited by hand."
            )
        row["reference_decomposition"] = "|".join(CORRECTIONS[item_id])
        applied += 1

    if applied != len(CORRECTIONS):
        raise SystemExit(
            f"{filename}: {applied} of {len(CORRECTIONS)} corrections matched a row."
        )

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    return applied


def main() -> int:
    # HISTORICAL — both version literals are pinned on purpose. These are the
    # corrections to the v6 migration and belong to v6 alone; following
    # GOLD_STANDARD_EN would make the script rewrite whatever version is
    # canonical later, against a placeholder text that only ever existed here.
    #
    # Both files carry the identical English decomposition: the judge that
    # reads it runs in English, and validate_gold_standard.py requires the
    # column to be byte-identical across the two.
    for name in ("gold_standard_v6_en.csv", "gold_standard_v6.csv"):
        print(f"{apply(name)} Korrekturen in {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
