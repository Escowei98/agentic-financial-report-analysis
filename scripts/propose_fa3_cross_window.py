"""
Propose the 15 cross-window FA-3 items for the gold standard (Phase 1b).

Why FA-3 needs restructuring
----------------------------
A Form 10-K reports the income statement over three fiscal years in
comparative columns. With the previous corpus of consecutive filings
(FY2022/23/24) the FY2024 filing alone covered 2022-2024, so every one of the
30 FA-3 items had a year span of 1 or 2 and was answerable from a SINGLE
document. The stratum therefore measured table lookup, not the cross-document
synthesis that FA-3 is supposed to operationalise for H2/H3.

With the corpus on alternating fiscal years (FY2020/2022/2024) a span from
FY2020 to FY2024 cannot be served by any single filing: FY2020 figures live in
the FY2020 or FY2022 filing, FY2024 figures only in the FY2024 filing.

Design: matched pairs
---------------------
The stratum is split 15/15. Each cross-window item mirrors a retained
within-window item in company, metric and math_type, and differs ONLY in the
span. That makes the document boundary the single varying quantity inside one
stratum, so Kapitel 5 can report H2/H3 per window class and read the effect of
context fragmentation directly instead of inferring it across strata.

Ground truth
------------
Every base figure below was located in the processed filings and is emitted
with its file and line number so it can be checked against the SEC original.
The derived values (growth, CAGR, margin deltas) are computed here rather than
typed in, so the arithmetic is auditable and reproducible.

This script only PROPOSES. It writes a review file and does not touch the gold
standard; merging happens after the values have been signed off.

Usage:
    uv run python scripts/propose_fa3_cross_window.py
"""

from __future__ import annotations

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "data" / "gold_standard" / "fa3_cross_window_proposal.csv"

# Base figures in millions USD, each with the file and line it was read from.
# FY2020 figures are additionally present in the FY2022 filing's comparative
# columns (3-year window), which is why the FY2020 fact accepts two sources.
FIG = {
    "AAPL": {
        "rev20": (274_515, "AAPL/10K_2020-10-30.md:407"),
        "rev22": (394_328, "AAPL/10K_2024-11-01.md:471"),
        "rev24": (391_035, "AAPL/10K_2024-11-01.md:471"),
        "ni20": (57_411, "AAPL/10K_2020-10-30.md:1129"),
        "ni24": (93_736, "AAPL/10K_2024-11-01.md:731"),
        "oi20": (66_288, "AAPL/10K_2020-10-30.md:859"),
        "oi22": (119_437, "AAPL/10K_2024-11-01.md:727"),
        "oi24": (123_216, "AAPL/10K_2024-11-01.md:727"),
    },
    "MSFT": {
        "rev20": (143_015, "MSFT/10K_2020-07-30.md:957"),
        "rev22": (198_270, "MSFT/10K_2022-07-28.md:1145"),
        "rev24": (245_122, "MSFT/10K_2024-07-30.md:875"),
        "ni20": (44_281, "MSFT/10K_2020-07-30.md:997"),
        "ni24": (88_136, "MSFT/10K_2024-07-30.md:901"),
        "oi20": (52_959, "MSFT/10K_2020-07-30.md:997"),
        "oi22": (83_383, "MSFT/10K_2022-07-28.md:1163"),
        "oi24": (109_433, "MSFT/10K_2024-07-30.md:901"),
    },
    "AMZN": {
        "rev20": (386_064, "AMZN/10K_2021-02-03.md:1066"),
        "rev24": (637_959, "AMZN/10K_2025-02-07.md:1015"),
        "ni20": (21_331, "AMZN/10K_2021-02-03.md:983"),
        "ni24": (59_248, "AMZN/10K_2025-02-07.md:967"),
        "oi20": (22_899, "AMZN/10K_2021-02-03.md:1084"),
        "oi24": (68_593, "AMZN/10K_2025-02-07.md:1024"),
    },
    "GOOGL": {
        "rev20": (182_527, "GOOGL/10K_2021-02-03.md:701"),
        "rev24": (350_018, "GOOGL/10K_2025-02-05.md:21"),
        "ni20": (40_269, "GOOGL/10K_2021-02-03.md:1764"),
        "ni24": (100_118, "GOOGL/10K_2025-02-05.md:21"),
        "oi20": (41_224, "GOOGL/10K_2021-02-03.md:1756"),
        "oi24": (112_390, "GOOGL/10K_2025-02-05.md:21"),
    },
}

NAME = {"AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon", "GOOGL": "Alphabet"}

# FY2020 is reported by the FY2020 filing and by the FY2022 filing's
# comparative columns; FY2022 by the FY2022 and FY2024 filings; FY2024 only by
# the FY2024 filing. "+" = interchangeable within a group, "|" = separate facts.
GROUP_20 = "{t}_2020+{t}_2022"
GROUP_22 = "{t}_2022+{t}_2024"
GROUP_24 = "{t}_2024"


def v(t: str, k: str) -> float:
    return FIG[t][k][0]


def src(t: str, *keys: str) -> str:
    return " ; ".join(FIG[t][k][1] for k in keys)


def build() -> list[dict]:
    rows: list[dict] = []

    def add(item_id, t, subtype, math_type, q_en, q_de, gt, unit, groups, sources, calc,
            gt_de=None):
        rows.append({
            "id": item_id,
            "fa_type": "FA-3",
            "subtype": subtype,
            "window_class": "cross_window",
            "math_type": math_type,
            "query_en": q_en,
            "query_de": q_de,
            "gt_value": gt,
            # Numeric ground truths are language-neutral; a qualitative one is
            # not, and the German file must keep a German expected answer.
            "gt_value_de": gt_de or gt,
            "gt_unit": unit,
            "doc_id_groups": groups,
            "doc_ids": "|".join(sorted({d for g in groups.split("|") for d in g.split("+")})),
            "source_sections": "item_8_income_stmt",
            "sources": sources,
            "calculation": calc,
        })

    # --- revenue growth FY2020 -> FY2024 (mirrors the retained FY2022->FY2024 items)
    for item_id, t in zip((42, 47, 52, 56), ("AAPL", "MSFT", "AMZN", "GOOGL")):
        g = (v(t, "rev24") - v(t, "rev20")) / v(t, "rev20") * 100
        add(item_id, t, "growth_yoy", "yoy",
            f"What was {NAME[t]}'s total revenue growth in FY2024 compared to FY2020?",
            f"Wie hoch war das Umsatzwachstum von {NAME[t]} in FY2024 im Vergleich zu FY2020?",
            f"+{g:.1f}%", "percent",
            f"{GROUP_20.format(t=t)}|{GROUP_24.format(t=t)}",
            src(t, "rev20", "rev24"),
            f"({v(t,'rev24'):,.0f} - {v(t,'rev20'):,.0f}) / {v(t,'rev20'):,.0f} = {g:.4f}%")

    # --- net income growth FY2020 -> FY2024
    for item_id, t in zip((43, 48, 53, 57), ("AAPL", "MSFT", "AMZN", "GOOGL")):
        g = (v(t, "ni24") - v(t, "ni20")) / v(t, "ni20") * 100
        add(item_id, t, "growth_yoy", "yoy",
            f"How has {NAME[t]}'s Net Income changed from FY2020 to FY2024?",
            f"Wie hat sich das Nettoergebnis von {NAME[t]} von FY2020 bis FY2024 verändert?",
            f"+{g:.1f}%", "percent",
            f"{GROUP_20.format(t=t)}|{GROUP_24.format(t=t)}",
            src(t, "ni20", "ni24"),
            f"({v(t,'ni24'):,.0f} - {v(t,'ni20'):,.0f}) / {v(t,'ni20'):,.0f} = {g:.4f}%")

    # --- revenue CAGR over four years
    for item_id, t in zip((98, 99), ("AAPL", "MSFT")):
        c = ((v(t, "rev24") / v(t, "rev20")) ** (1 / 4) - 1) * 100
        add(item_id, t, "growth_cagr", "cagr",
            f"What was the Compound Annual Growth Rate (CAGR) of {NAME[t]}'s revenue from FY2020 to FY2024?",
            f"Wie hoch war die Compound Annual Growth Rate (CAGR) des Umsatzes von {NAME[t]} von FY2020 bis FY2024?",
            f"{c:.1f}%", "percent",
            f"{GROUP_20.format(t=t)}|{GROUP_24.format(t=t)}",
            src(t, "rev20", "rev24"),
            f"({v(t,'rev24'):,.0f} / {v(t,'rev20'):,.0f})^(1/4) - 1 = {c:.4f}%")

    # --- operating margin, percentage-point change
    for item_id, t in zip((102, 103, 104), ("AAPL", "MSFT", "AMZN")):
        m20 = v(t, "oi20") / v(t, "rev20") * 100
        m24 = v(t, "oi24") / v(t, "rev24") * 100
        add(item_id, t, "pp_change", "pp_diff",
            f"By how many percentage points has {NAME[t]}'s Operating Margin changed from FY2020 to FY2024?",
            f"Um wie viele Prozentpunkte hat sich die Operating Margin von {NAME[t]} von FY2020 auf FY2024 verändert?",
            f"{m24 - m20:+.1f}pp", "pp_change",
            f"{GROUP_20.format(t=t)}|{GROUP_24.format(t=t)}",
            src(t, "oi20", "rev20", "oi24", "rev24"),
            f"{v(t,'oi24'):,.0f}/{v(t,'rev24'):,.0f} - {v(t,'oi20'):,.0f}/{v(t,'rev20'):,.0f} "
            f"= {m24:.2f}% - {m20:.2f}% = {m24-m20:+.4f}pp")

    # --- multi-step: argmax across the three corpus years (needs all three facts)
    t = "MSFT"
    margins = {y: v(t, f"oi{y}") / v(t, f"rev{y}") * 100 for y in ("20", "22", "24")}
    best = max(margins, key=lambda k: margins[k])
    add(107, t, "multi_step_compute", "multi_step",
        "In which fiscal year (FY2020, FY2022 or FY2024) did MSFT have the highest Operating Margin?",
        "In welchem Geschäftsjahr (FY2020, FY2022 oder FY2024) hatte MSFT die höchste Operating Margin?",
        f"FY20{best}", "text",
        f"{GROUP_20.format(t=t)}|{GROUP_22.format(t=t)}|{GROUP_24.format(t=t)}",
        src(t, "oi20", "rev20", "oi22", "rev22", "oi24", "rev24"),
        " ; ".join(f"FY20{y}={margins[y]:.2f}%" for y in ("20", "22", "24")))

    # --- qualitative trend across the three corpus years
    t = "AAPL"
    tm = {y: v(t, f"oi{y}") / v(t, f"rev{y}") * 100 for y in ("20", "22", "24")}
    add(44, t, "trend_qualitative", "none",
        "What was the overall trend in Apple's Operating Margin from FY2020 to FY2024 — rising, falling, "
        "or roughly unchanged? A year-by-year breakdown may be given as supporting evidence, but the "
        "overall direction must be stated explicitly.",
        "Was war der Gesamttrend der Operating Margin von Apple von FY2020 bis FY2024 — steigend, fallend, "
        "oder in etwa gleichbleibend? Eine Jahr-für-Jahr-Aufschlüsselung kann als unterstützender Beleg "
        "angegeben werden, aber die Gesamtrichtung muss explizit genannt werden.",
        "Rising", "text",
        f"{GROUP_20.format(t=t)}|{GROUP_22.format(t=t)}|{GROUP_24.format(t=t)}",
        src(t, "oi20", "rev20", "oi22", "rev22", "oi24", "rev24"),
        " ; ".join(f"FY20{y}={tm[y]:.2f}%" for y in ("20", "22", "24")),
        gt_de="Steigend")

    return rows


def main() -> int:
    rows = build()
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} Kandidaten geschrieben nach {OUT.relative_to(PROJECT_ROOT)}\n")
    for r in rows:
        print(f"  id {r['id']:>3} {r['subtype']:18s} {r['gt_value']:>9s}  {r['query_en'][:74]}")
        print(f"          Quellen: {r['sources']}")
        print(f"          Rechnung: {r['calculation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
