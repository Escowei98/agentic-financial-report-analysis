"""
Build Gold Standard v3 from v2 (frozen) + 73 newly authored entries.

v3 grows the Gold Standard from 77 to 150 entries (5 strata x 30) per the
methodology established in Thesis Section 3.6.1 (n=150 following Islam et al.
2023 / FinanceBench size; paired-design power per Miller 2024). It also
introduces a Refusal/Adversarial stratum (NF-1) and a normalized schema with
14 columns.

Inputs:
    notebooks/experiments/sys1_rag_monolith/ablation_test_data_v2.csv
        (frozen on 2026-04-29 per EVAL_DECISION_LOG.md)

Output:
    data/gold_standard/gold_standard_v3.csv
        Semicolon-delimited, UTF-8, 14-column schema.
        See data/gold_standard/gold_standard_README.md for column spec.

Migration is deterministic: every v2 entry maps to exactly one v3 entry via
the rules in this file. The 73 new entries are authored inline (FA-1 +10,
FA-2 +10, FA-3 +13, FA-4 +10, FA-Refusal +30) with candidate ground-truth
values that the USER must verify against the SEC filings before the v3 set
is declared frozen.

Run with:
    uv run python scripts/build_gold_standard_v3.py
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_V2_CSV = (
    PROJECT_ROOT
    / "notebooks"
    / "experiments"
    / "sys1_rag_monolith"
    / "ablation_test_data_v2.csv"
)
TARGET_V3_CSV = PROJECT_ROOT / "data" / "gold_standard" / "gold_standard_v3.csv"

V3_HEADER = [
    "id",
    "fa_type",
    "subtype",
    "doc_ids",
    "query",
    "gt_value",
    "gt_unit",
    "source_sections",
    "expected_answerable",
    "difficulty",
    "math_type",
    "entity_form",
    "hypothesis_link",
    "rationale",
]


# ---------------------------------------------------------------------------
#  Mapping tables for v2 -> v3 migration
# ---------------------------------------------------------------------------

V2_TYPE_TO_FA_TYPE = {
    "Single": "FA-1",
    "Cross-Sec": "FA-2",
    "Multi-Year": "FA-3",
    "Multi-Comp": "FA-4",
}

# Map freetext "Source" values from v2 to canonical section IDs (pipe-separated).
# See data/gold_standard/gold_standard_README.md "Section ID Reference".
SOURCE_SECTION_MAP: dict[str, str] = {
    "Item 7": "item_7",
    "Item 8": "item_8",
    "Item 1": "item_1",
    "Balance Sheet": "item_8_balance_sheet",
    "Cash Flow": "item_8_cash_flow",
    "Income Stmt": "item_8_income_stmt",
    "Segment Info": "item_8_segment",
    "MD&A": "item_7",
    "MD&A + Income Stmt": "item_7|item_8_income_stmt",
    "Cash Flow + MD&A": "item_8_cash_flow|item_7",
    "Item 7 + Segment Info": "item_7|item_8_segment",
    "Segment Info + Item 7": "item_8_segment|item_7",
    "Segment Info + MD&A": "item_8_segment|item_7",
    "MD&A + Segment Info": "item_7|item_8_segment",
    "Income Stmt + MD&A": "item_8_income_stmt|item_7",
    "Item 1 + MD&A": "item_1|item_7",
    "Item 7 beide": "item_7",
    "Item 8 beide": "item_8",
    "Balance Sheet beide": "item_8_balance_sheet",
    "Cash Flow beide": "item_8_cash_flow",
    "MD&A beide": "item_7",
    "Segment beide": "item_8_segment",
    "Segment Info beide": "item_8_segment",
    "Item 1 beide": "item_1",
    "Item 7 alle": "item_7",
    "Item 8 alle": "item_8",
    "Balance Sheet alle": "item_8_balance_sheet",
    "MD&A alle": "item_7",
    "Segment Info alle": "item_8_segment",
}


# ---------------------------------------------------------------------------
#  Helper functions
# ---------------------------------------------------------------------------


def normalize_doc_ids(v2_doc: str) -> str:
    """Convert v2 'Doc(s)' freetext into pipe-separated canonical IDs.

    Examples
    --------
    >>> normalize_doc_ids("AAPL_2024")
    'AAPL_2024'
    >>> normalize_doc_ids("AAPL_22/24")
    'AAPL_2022|AAPL_2024'
    >>> normalize_doc_ids("AAPL_22/23/24")
    'AAPL_2022|AAPL_2023|AAPL_2024'
    >>> normalize_doc_ids("AAPL/MSFT_24")
    'AAPL_2024|MSFT_2024'
    >>> normalize_doc_ids("AAPL/MSFT_23/24")
    'AAPL_2023|AAPL_2024|MSFT_2023|MSFT_2024'
    >>> normalize_doc_ids("All_4_2024")
    'AAPL_2024|MSFT_2024|AMZN_2024|GOOGL_2024'
    """
    if v2_doc == "All_4_2024":
        return "AAPL_2024|MSFT_2024|AMZN_2024|GOOGL_2024"

    if "_" not in v2_doc:
        raise ValueError(f"Unexpected v2 doc format: {v2_doc!r}")

    tickers_part, years_part = v2_doc.rsplit("_", 1)
    tickers = tickers_part.split("/")
    year_tokens = years_part.split("/")
    years = [f"20{y}" if len(y) == 2 else y for y in year_tokens]

    return "|".join(f"{t}_{y}" for t in tickers for y in years)


def normalize_source(v2_source: str) -> str:
    """Map a v2 'Source' freetext value to canonical section IDs."""
    src = v2_source.strip()
    if src in SOURCE_SECTION_MAP:
        return SOURCE_SECTION_MAP[src]
    raise ValueError(
        f"Unmapped v2 Source value: {src!r}. "
        f"Add it to SOURCE_SECTION_MAP in build_gold_standard_v3.py."
    )


def infer_gt_unit(gt_value: str) -> str:
    """Infer canonical unit from a ground-truth value string.

    Returns one of: USD_billion, USD_million, percent, pp_change, ratio,
    count, boolean, text, n/a, USD_per_share.

    Convention for composite answers (FIRM/brand prefix + value in parens):
    the unit reflects the *primary numeric magnitude* embedded in the answer,
    not the wrapper type. ``"AAPL ($93.7 Mrd > MSFT $88.1 Mrd)"`` -> USD_billion;
    ``"MSFT (44.6%)"`` -> percent. ``text`` is reserved for (a) qualitative
    answers without a primary number (trends, Yes/No), (b) answers carrying a
    verification hint ("Werte verifizieren"), (c) pattern mismatches.
    """
    s = gt_value.strip()
    if s == "":
        return "n/a"
    s_lower = s.lower()

    if "verifizieren" in s_lower or "bitte prüfen" in s_lower or "bitte pruefen" in s_lower:
        return "text"
    if s.startswith("Ja") or s.startswith("Nein"):
        return "text"
    if s in ("Leicht steigend", "Stark steigend", "Fallend dann steigend", "Steigend", "Fallend"):
        return "text"

    composite_prefixes = ("AAPL", "MSFT", "AMZN", "GOOGL", "Azure", "AWS")
    if any(s.startswith(t) for t in composite_prefixes):
        inner_match = re.search(r"\(([^)]*)\)", s)
        inner = inner_match.group(1).lower() if inner_match else ""
        if "mrd" in inner or "billion" in inner:
            return "USD_billion"
        if "mio" in inner or "million" in inner:
            return "USD_million"
        if "%" in inner:
            return "percent"
        return "text"

    if "mrd" in s_lower or "billion" in s_lower:
        return "USD_billion"
    if s.endswith("%"):
        return "percent"
    if s.startswith("+") and s.endswith("%"):
        return "percent"
    if "$" in s:
        return "USD_billion"
    if s.startswith("+") or s.startswith("-"):
        if "%" in s:
            return "percent"
        if "$" in s:
            return "USD_billion"
        return "count"
    if "%" in s:
        return "percent"
    return "text"


def infer_subtype(fa_type: str, rationale: str, query: str, gt_value: str) -> str:
    """Infer subtype from rationale tags and query content."""
    r = rationale.lower()
    q = query.lower()

    if fa_type == "FA-1":
        if "solvency" in r or "debt" in q or "equity" in q:
            return "solvency"
        if "profit" in r or "net income" in q:
            return "basis_kpi"
        if "basis" in r:
            return "basis_kpi"
        return "basis_kpi"

    if fa_type == "FA-2":
        if "synthese h3" in r:
            if "anteil" in q or "segment" in q:
                return "segment_share"
            return "segment_margin"
        if "h2" in r:
            return "synthesis"
        if "synthese" in r:
            if "anteil" in q:
                return "segment_share"
            if "segment" in q and "margin" in q:
                return "segment_margin"
            return "synthesis"
        return "synthesis"

    if fa_type == "FA-3":
        if "trend" in r:
            return "trend_qualitative"
        if "entwickelte" in q or "entwicklung" in q:
            return "trend_qualitative"
        if "yoy" in q or "yoy" in r:
            return "growth_yoy"
        if "wachstum" in q or "wachstum" in r:
            return "growth_yoy"
        if "verändert" in q:
            return "growth_yoy"
        return "growth_yoy"

    if fa_type == "FA-4":
        if "max/min" in r:
            return "max_min"
        if "cloud" in r:
            return "cloud_comparison"
        if "differenz" in q:
            return "cross_firm_diff"
        if "growth" in r:
            return "cross_firm_pairwise_growth"
        if "cross-firm" in r or "h4" in r:
            return "cross_firm_pairwise"
        return "cross_firm_pairwise"

    if fa_type == "FA-Refusal":
        return ""

    return ""


def infer_difficulty(fa_type: str, subtype: str, gt_value: str) -> str:
    """Assign a difficulty rating based on type and subtype heuristics."""
    if fa_type == "FA-1":
        if subtype == "solvency":
            return "medium"
        return "easy"
    if fa_type == "FA-2":
        if subtype in ("segment_margin", "ratio_compute"):
            return "hard"
        return "medium"
    if fa_type == "FA-3":
        if subtype == "trend_qualitative":
            return "hard"
        if subtype in ("growth_cagr", "multi_step_compute"):
            return "hard"
        return "medium"
    if fa_type == "FA-4":
        if subtype in ("max_min", "cross_firm_pairwise"):
            return "easy"
        if subtype in ("cross_firm_diff", "cloud_comparison"):
            return "medium"
        return "medium"
    if fa_type == "FA-Refusal":
        return "medium"
    return "medium"


def infer_math_type(fa_type: str, subtype: str, query: str) -> str:
    """Math-type only filled for FA-3.

    Returns ``"n/a"`` for non-FA-3 strata (math not applicable). FA-3 trend
    questions return ``"none"`` (math stratum, but qualitative).
    """
    if fa_type != "FA-3":
        return "n/a"
    if subtype == "trend_qualitative":
        return "none"
    if subtype == "growth_cagr":
        return "cagr"
    if subtype == "multi_step_compute":
        return "multi_step"
    if "yoy" in query.lower() or "wachstum" in query.lower():
        return "yoy"
    return "yoy"


def infer_hypothesis_link(fa_type: str, rationale: str) -> str:
    """Map fa_type + legacy 'H' tags in rationale to canonical hypothesis IDs.

    The v2 CSV uses 'H4' as a tag for cross-firm questions; the thesis defines
    only H1, H2, H3. FA-4 questions canonically test H1 (cross-doc) and H3
    (reasoning), so 'H4' tags are translated to 'H1|H3'.
    """
    r = rationale.upper()
    hypotheses: list[str] = []

    if "H1" in r:
        hypotheses.append("H1")
    if "H2" in r:
        hypotheses.append("H2")
    if "H3" in r:
        hypotheses.append("H3")
    if "H4" in r:
        hypotheses.extend(["H1", "H3"])

    if not hypotheses:
        if fa_type == "FA-1":
            hypotheses = ["H1"]
        elif fa_type == "FA-2":
            hypotheses = ["H2"]
        elif fa_type == "FA-3":
            hypotheses = ["H2", "H3"]
        elif fa_type == "FA-4":
            hypotheses = ["H1", "H3"]

    seen: set[str] = set()
    deduped = [h for h in hypotheses if not (h in seen or seen.add(h))]
    return "|".join(deduped)


# ---------------------------------------------------------------------------
#  v2 -> v3 migration
# ---------------------------------------------------------------------------


@dataclass
class V3Entry:
    """A single Gold Standard v3 entry."""

    id: int = 0
    fa_type: str = ""
    subtype: str = ""
    doc_ids: str = ""
    query: str = ""
    gt_value: str = ""
    gt_unit: str = ""
    source_sections: str = ""
    expected_answerable: bool = True
    difficulty: str = "medium"
    math_type: str = "n/a"
    entity_form: str = "name"
    hypothesis_link: str = ""
    rationale: str = ""

    def to_row(self) -> list[str]:
        return [
            str(self.id),
            self.fa_type,
            self.subtype,
            self.doc_ids,
            self.query,
            self.gt_value,
            self.gt_unit,
            self.source_sections,
            "true" if self.expected_answerable else "false",
            self.difficulty,
            self.math_type,
            self.entity_form,
            self.hypothesis_link,
            self.rationale,
        ]


FIRM_TICKERS = ("AAPL", "MSFT", "GOOGL", "AMZN")


def normalize_pairwise_gt_value(gt_value: str, doc_ids: str) -> str:
    """Add explicit second-firm prefix to pairwise gt_values that lack one.

    Pattern targeted: ``"FIRM (X op Y)"`` (with ``op`` in ``<``, ``>``) where the
    second value ``Y`` lacks a firm prefix. The other firm is taken from
    ``doc_ids`` (pipe-separated; duplicates collapsed via set). Yes/No answers
    (``"Ja"``/``"Nein"``) and already-prefixed answers are returned unchanged.

    Additionally enforces tilde-symmetry: if the left-hand value carries a
    leading ``~`` (rounded/approximate), the inserted right-hand value gets
    the same prefix so both sides signal the same precision.
    """
    m = re.match(r"^([A-Za-z]+)\s*\((.+?)\s*([<>])\s*(.+?)\)\s*$", gt_value)
    if not m:
        return gt_value
    winner, left, op, right = m.group(1), m.group(2).strip(), m.group(3), m.group(4).strip()
    if winner not in FIRM_TICKERS:
        return gt_value
    already_prefixed = any(right.startswith(t) for t in FIRM_TICKERS)
    left_is_approx = left.startswith("~")

    if already_prefixed:
        if left_is_approx:
            firm_match = re.match(r"^([A-Z]+)\s+(.+)$", right)
            if firm_match:
                other_firm, right_value = firm_match.group(1), firm_match.group(2).strip()
                if not right_value.startswith("~"):
                    return f"{winner} ({left} {op} {other_firm} ~{right_value})"
        return gt_value

    firms = [d.split("_")[0] for d in doc_ids.split("|") if d]
    others = {f for f in firms if f != winner}
    if len(others) != 1:
        return gt_value
    other = next(iter(others))
    if left_is_approx and not right.startswith("~"):
        right = f"~{right}"
    return f"{winner} ({left} {op} {other} {right})"


def migrate_v2_row(row: dict[str, str]) -> V3Entry:
    """Convert one v2 CSV row (as a dict) into a V3Entry."""
    v2_type = row["Typ"].strip()
    fa_type = V2_TYPE_TO_FA_TYPE[v2_type]
    rationale = row["Rationale"].strip()
    query = row["Query"].strip()
    gt_value = row["GT-Wert"].strip()

    subtype = infer_subtype(fa_type, rationale, query, gt_value)
    difficulty = infer_difficulty(fa_type, subtype, gt_value)
    math_type = infer_math_type(fa_type, subtype, query)
    hypothesis_link = infer_hypothesis_link(fa_type, rationale)
    source_sections = normalize_source(row["Source"].strip())
    query_lc = query.lower().replace("-", " ")
    if "net income" in query_lc and source_sections == "item_8":
        source_sections = "item_8_income_stmt"

    doc_ids = normalize_doc_ids(row["Doc(s)"].strip())
    gt_value = normalize_pairwise_gt_value(gt_value, doc_ids)
    if "GOOGL" in doc_ids and "Cloud" in gt_value and "Google Cloud" not in gt_value:
        gt_value = gt_value.replace("Cloud", "Google Cloud")

    return V3Entry(
        id=0,
        fa_type=fa_type,
        subtype=subtype,
        doc_ids=doc_ids,
        query=query,
        gt_value=gt_value,
        gt_unit=infer_gt_unit(gt_value),
        source_sections=source_sections,
        expected_answerable=True,
        difficulty=difficulty,
        math_type=math_type,
        entity_form=row.get("entity_form", "name").strip() or "name",
        hypothesis_link=hypothesis_link,
        rationale=rationale,
    )


def read_v2_rows() -> list[dict[str, str]]:
    """Read v2 CSV into a list of dicts keyed by header column names."""
    with open(SOURCE_V2_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        return [dict(row) for row in reader if row.get("ID") and row["ID"].strip()]


# ---------------------------------------------------------------------------
#  Post-migration overrides
#
#  Corrections applied after v2 migration when a v2 gt_value has been shown to
#  disagree with the SEC-source values in the processed markdowns. v2 itself
#  is frozen (2026-04-29, see EVAL_DECISION_LOG.md) and cannot be edited;
#  overrides live here so the provenance chain (v2 -> override -> v3) stays
#  explicit and each correction is auditable via its recorded rationale.
#
#  Key: exact match of the entry's `query` field (unique across all 150).
#  Overrides are applied by `apply_post_migration_overrides()`.
# ---------------------------------------------------------------------------

INGEST_FIX_OVERRIDES: dict[str, dict[str, str]] = {
    # id 8: MSFT Debt/Equity FY2024. v2 value 76.3% does not match the
    # documented convention (Total Liabilities / Total Equity, cf. BACKLOG
    # 2026-05-08 "Häufige Fallstricke"). MSFT_2024 Item 8 Balance Sheet:
    # Total liabilities 243,686 / Total stockholders' equity 268,477 = 90.77%.
    "Wie hoch war die Debt/Equity Ratio von Microsoft in FY2024?": {
        "gt_value": "90.8%",
        "query": "Wie hoch war die Debt/Equity Ratio (Gesamtverbindlichkeiten / Gesamtes Eigenkapital) von Microsoft in FY2024?",
        "note": "corrected 2026-07-25 (was 76.3%; source MSFT 10-K FY2024 Item 8 Balance Sheet: 243,686 / 268,477 = 90.77%); query reworded 2026-08-01 to state the Debt/Equity formula explicitly (Answer Format Convention)",
    },
    # id 46: MSFT Net Income growth FY2022 -> FY2024. v2 value 23.6% does not
    # match the recomputed value from the Cash Flow Statements
    # (MSFT_2024 FY2022 col = 72,738; FY2024 col = 88,136 -> 21.17%).
    "Wie hat sich das Net Income von Microsoft von FY2022 auf FY2024 verändert?": {
        "gt_value": "21.2%",
        "note": "corrected 2026-07-25 (was 23.6%; source MSFT 10-K FY2024 Cash Flow Statement: 88,136 / 72,738 - 1 = 21.17%)",
    },
    # id 66: MSFT vs GOOGL Debt/Equity FY2024. Both operands were wrong under
    # the documented convention; comparison winner remains GOOGL.
    # MSFT_2024: 243,686 / 268,477 = 90.77%
    # GOOGL_2024 (10K_2025-02-05): Total liabilities 125,172 / Total
    # stockholders' equity 325,084 = 38.50%
    "Wer hat die niedrigere Debt/Equity Ratio in FY2024, MSFT oder GOOGL?": {
        "gt_value": "GOOGL (38.5% < MSFT 90.8%)",
        "note": "corrected 2026-07-25 (was 'GOOGL (44.5% < MSFT 76.3%)'; MSFT D/E = 243,686/268,477 = 90.8%, GOOGL D/E = 125,172/325,084 = 38.5%; source FY2024 Item 8 Balance Sheets; winner unchanged)",
    },
    # id 18: GOOGL Debt/Equity FY2024, standalone version of the id-66
    # comparison above. Never synced when id 66 was corrected on 2026-07-25 --
    # same underlying figure, same convention (Total Liabilities / Total
    # Equity), found stale during human judge-validation (R011/R030,
    # 2026-08-01). GOOGL_2024 (10K_2025-02-05) Total liabilities 125,172 /
    # Total stockholders' equity 325,084 = 38.50%.
    "Wie hoch war die Debt/Equity Ratio von Alphabet in FY2024?": {
        "gt_value": "38.5%",
        "query": "Wie hoch war die Debt/Equity Ratio (Gesamtverbindlichkeiten / Gesamtes Eigenkapital) von Alphabet in FY2024?",
        "note": "corrected 2026-08-01 (was 44.5%, never synced with the id-66 correction on 2026-07-25; GOOGL D/E = 125,172/325,084 = 38.5%; source GOOGL 10-K FY2024 Item 8 Balance Sheet); query reworded 2026-08-01 to state the Debt/Equity formula explicitly (Answer Format Convention)",
    },
    # ids 3/13: AAPL/AMZN Debt-to-Equity FY2024. gt_value already correct
    # (verified against the same Total Liabilities / Total Equity
    # convention as ids 8/18/66), only the query is reworded here for
    # consistency -- same Answer Format Convention rollout, 2026-08-01.
    "Wie hoch war die Debt-to-Equity Ratio von Apple am Ende von FY2024?": {
        "query": "Wie hoch war die Debt-to-Equity Ratio (Gesamtverbindlichkeiten / Gesamtes Eigenkapital) von Apple am Ende von FY2024?",
        "note": "query reworded 2026-08-01 to state the Debt/Equity formula explicitly (Answer Format Convention); gt_value unchanged (540.9%, already verified)",
    },
    "Wie hoch war die Debt-to-Equity Ratio von AMZN in FY2024?": {
        "query": "Wie hoch war die Debt-to-Equity Ratio (Gesamtverbindlichkeiten / Gesamtes Eigenkapital) von AMZN in FY2024?",
        "note": "query reworded 2026-08-01 to state the Debt/Equity formula explicitly (Answer Format Convention); gt_value unchanged (118.5%, already verified)",
    },
    # ids 44/49/54: trend_qualitative rewording. Human judge-validation
    # (R004/R023, 2026-08-01, see EVAL_DECISION_LOG.md) found that a
    # thorough, correct year-by-year answer was scored as a mismatch
    # against a terse one-line GT ("Slightly rising"). Reworded to ask for
    # the overall trend explicitly while tolerating supporting detail, per
    # the Answer Format Convention's trend-question rule. gt_value strings
    # are unchanged -- they already state the overall direction correctly.
    "Wie entwickelte sich die Operating Margin von Apple über die Jahre FY2022, FY2023 und FY2024 (steigend oder fallend)?": {
        "query": "Was war der Gesamttrend der Operating Margin von Apple von FY2022 bis FY2024 — steigend, fallend, oder in etwa gleichbleibend? Eine Jahr-für-Jahr-Aufschlüsselung kann als unterstützender Beleg angegeben werden, aber die Gesamtrichtung muss explizit genannt werden.",
        "note": "query reworded 2026-08-01 (Answer Format Convention, trend-question rule); gt_value unchanged ('Leicht steigend')",
    },
    "Wie entwickelten sich die Capital Expenditures (CapEx) von Microsoft über FY2022, FY2023 und FY2024 (steigend oder fallend)?": {
        "query": "Was war der Gesamttrend der Capital Expenditures (CapEx) von Microsoft von FY2022 bis FY2024 — steigend, fallend, oder in etwa gleichbleibend? Eine Jahr-für-Jahr-Aufschlüsselung kann als unterstützender Beleg angegeben werden, aber die Gesamtrichtung muss explizit genannt werden.",
        "note": "query reworded 2026-08-01 (Answer Format Convention, trend-question rule); gt_value unchanged ('Stark steigend')",
    },
    "Wie entwickelte sich die Operating Margin des AWS-Segments von Amazon über FY2022, FY2023 und FY2024?": {
        "query": "Was war der Gesamttrend der Operating Margin des AWS-Segments von Amazon von FY2022 bis FY2024 — steigend, fallend, oder gemischt? Eine Jahr-für-Jahr-Aufschlüsselung kann als unterstützender Beleg angegeben werden, aber die Gesamtrichtung muss explizit genannt werden.",
        "note": "query reworded 2026-08-01 (Answer Format Convention, trend-question rule; 'gemischt' statt 'gleichbleibend' da GT nicht monoton ist); gt_value unchanged ('Fallend dann steigend')",
    },
    # id 24: AAPL Operating Margin FY2023. v2 value 30.8% does not match the
    # 10-K source. AAPL_2023 (10K_2023-11-03) MD&A, FY2023 column:
    # Operating income $114,301M / Total net sales $383,285M = 29.82%.
    # Cross-confirmed independently by two different system answers to two
    # different questions both citing 29.82% (id 24 direct, and id 44's
    # trend breakdown). Found during human judge-validation (R023, reserve
    # sample re-run, 2026-08-01).
    "Wie hoch war die Operating Margin von AAPL in FY2023?": {
        "gt_value": "29.8%",
        "note": "corrected 2026-08-01 (was 30.8%; AAPL FY2023 Operating income 114,301 / Total net sales 383,285 = 29.82%; source AAPL 10-K FY2023 MD&A/Item 8 Income Statement)",
    },
    # id 74: highest Net Income among the 4 companies in FY2024. GOOGL_2024
    # (10K_2025-02-05) Consolidated Statements of Income, 3-year column:
    # Net income $59,972 (FY2022) / $73,795 (FY2023) / $100,118 (FY2024).
    # $100.1bn > AAPL's $93.7bn -- GOOGL is the actual winner, not AAPL.
    # Found during human judge-validation (R035, 2026-08-01).
    "Welches der 4 Unternehmen (Apple, Microsoft, Amazon, Alphabet) hatte das höchste Net Income in FY2024?": {
        "gt_value": "GOOGL ($100.1 Mrd)",
        "note": "corrected 2026-08-01 (was 'AAPL ($93.7 billion)'; GOOGL FY2024 Net income = $100,118M per 10-K Consolidated Statements of Income, exceeds AAPL's $93,736M; source GOOGL 10-K FY2024 Item 8 Income Statement)",
    },
}


def apply_post_migration_overrides(entries: list[V3Entry]) -> int:
    """Apply INGEST_FIX_OVERRIDES to the migrated entries in-place.

    Each override may set "gt_value" (re-infers gt_unit) and/or "query"
    (rewording, e.g. to add a disambiguating parenthetical per the Answer
    Format Convention in gold_standard_README.md). The lookup key is
    always the entry's ORIGINAL query text, even when the override also
    rewords it -- rewording only takes effect after the lookup.

    Returns the number of entries that were touched. Raises if an override
    key does not match any entry (guards against typos or v2 rewordings).
    """
    applied = 0
    for entry in entries:
        override = INGEST_FIX_OVERRIDES.get(entry.query)
        if override is None:
            continue
        if "gt_value" in override:
            entry.gt_value = override["gt_value"]
            entry.gt_unit = infer_gt_unit(entry.gt_value)
        if "query" in override:
            entry.query = override["query"]
        applied += 1

    if applied != len(INGEST_FIX_OVERRIDES):
        matched_queries = {
            e.query for e in entries if e.query in INGEST_FIX_OVERRIDES
        }
        missing = set(INGEST_FIX_OVERRIDES) - matched_queries
        raise AssertionError(
            f"INGEST_FIX_OVERRIDES contains {len(missing)} query keys that "
            f"did not match any migrated entry: {sorted(missing)}"
        )
    return applied


# ---------------------------------------------------------------------------
#  73 newly authored entries
#
#  GROUND TRUTH IS DRAFT — USER must verify each entry against the SEC filing
#  before the v3 set is declared frozen. See data/gold_standard/
#  gold_standard_README.md "Verification Workflow".
# ---------------------------------------------------------------------------


# +10 Single-Fact entries (FA-1)
NEW_FA1_ENTRIES: list[V3Entry] = [
    V3Entry(
        fa_type="FA-1", subtype="basis_kpi",
        doc_ids="AAPL_2022",
        query="Wie hoch war der Gesamtumsatz (Total Revenue) von Apple in FY2022?",
        gt_value="$394.3 Mrd", gt_unit="USD_billion",
        source_sections="item_7",
        difficulty="easy", entity_form="name",
        hypothesis_link="H1",
        rationale="FY2022 Single-Fact baseline, deckt Vor-Lücke",
    ),
    V3Entry(
        fa_type="FA-1", subtype="basis_kpi",
        doc_ids="MSFT_2022",
        query="Wie hoch war der Total Revenue von MSFT in FY2022?",
        gt_value="$198.3 Mrd", gt_unit="USD_billion",
        source_sections="item_7",
        difficulty="easy", entity_form="ticker",
        hypothesis_link="H1",
        rationale="FY2022 Single-Fact baseline, deckt Vor-Lücke",
    ),
    V3Entry(
        fa_type="FA-1", subtype="basis_kpi",
        doc_ids="AMZN_2022",
        query="Wie hoch war der Gesamtumsatz von Amazon in FY2022?",
        gt_value="$514.0 Mrd", gt_unit="USD_billion",
        source_sections="item_7",
        difficulty="easy", entity_form="name",
        hypothesis_link="H1",
        rationale="FY2022 Single-Fact baseline, deckt Vor-Lücke",
    ),
    V3Entry(
        fa_type="FA-1", subtype="basis_kpi",
        doc_ids="GOOGL_2022",
        query="Wie hoch war der Total Revenue von GOOGL in FY2022?",
        gt_value="$282.8 Mrd", gt_unit="USD_billion",
        source_sections="item_7",
        difficulty="easy", entity_form="ticker",
        hypothesis_link="H1",
        rationale="FY2022 Single-Fact baseline, deckt Vor-Lücke",
    ),
    V3Entry(
        fa_type="FA-1", subtype="basis_kpi",
        doc_ids="AAPL_2024",
        query="Wie hoch war das Operating Income von Apple in FY2024?",
        gt_value="$123.2 Mrd", gt_unit="USD_billion",
        source_sections="item_8_income_stmt",
        difficulty="easy", entity_form="name",
        hypothesis_link="H1",
        rationale="Operating Income als zusätzliche Basis-KPI über die 4 Firmen",
    ),
    V3Entry(
        fa_type="FA-1", subtype="basis_kpi",
        doc_ids="MSFT_2024",
        query="Wie hoch war das Operating Income von MSFT in FY2024?",
        gt_value="$109.4 Mrd", gt_unit="USD_billion",
        source_sections="item_8_income_stmt",
        difficulty="easy", entity_form="ticker",
        hypothesis_link="H1",
        rationale="Operating Income als zusätzliche Basis-KPI über die 4 Firmen",
    ),
    V3Entry(
        fa_type="FA-1", subtype="basis_kpi",
        doc_ids="AMZN_2024",
        query="Wie hoch war das Operating Income von Amazon in FY2024?",
        gt_value="$68.6 Mrd", gt_unit="USD_billion",
        source_sections="item_8_income_stmt",
        difficulty="easy", entity_form="name",
        hypothesis_link="H1",
        rationale="Operating Income als zusätzliche Basis-KPI über die 4 Firmen",
    ),
    V3Entry(
        fa_type="FA-1", subtype="basis_kpi",
        doc_ids="GOOGL_2024",
        query="Wie hoch war das Operating Income von GOOGL in FY2024?",
        gt_value="$112.4 Mrd", gt_unit="USD_billion",
        source_sections="item_8_income_stmt",
        difficulty="easy", entity_form="ticker",
        hypothesis_link="H1",
        rationale="Operating Income als zusätzliche Basis-KPI über die 4 Firmen",
    ),
    V3Entry(
        fa_type="FA-1", subtype="liquidity",
        doc_ids="AAPL_2024",
        query="Wie hoch waren die Cash and Cash Equivalents von Apple zum Ende von FY2024?",
        gt_value="$29.9 Mrd", gt_unit="USD_billion",
        source_sections="item_8_balance_sheet",
        difficulty="easy", entity_form="name",
        hypothesis_link="H1",
        rationale="Liquidity-KPI als zusätzliche Single-Fact-Variation",
    ),
    V3Entry(
        fa_type="FA-1", subtype="eps",
        doc_ids="MSFT_2024",
        query="Wie hoch war das Diluted EPS von MSFT in FY2024?",
        gt_value="$11.80", gt_unit="USD_per_share",
        source_sections="item_8_income_stmt",
        difficulty="easy", entity_form="ticker",
        hypothesis_link="H1",
        rationale="EPS als zusätzliche Single-Fact-Variation",
    ),
]


# +10 Cross-Section entries (FA-2)
NEW_FA2_ENTRIES: list[V3Entry] = [
    V3Entry(
        fa_type="FA-2", subtype="ratio_compute",
        doc_ids="AAPL_2024",
        query="Wie hoch war die R&D-Intensität (R&D-Aufwand als Prozent des Umsatzes) von AAPL in FY2024?",
        gt_value="8.0%", gt_unit="percent",
        source_sections="item_7|item_8_income_stmt",
        difficulty="medium", entity_form="ticker",
        hypothesis_link="H2",
        rationale="R&D-Intensität testet Cross-Section-Synthese",
    ),
    V3Entry(
        fa_type="FA-2", subtype="ratio_compute",
        doc_ids="MSFT_2024",
        query="Wie hoch war die R&D-Intensität (R&D als Prozent des Umsatzes) von MSFT in FY2024?",
        gt_value="12.0%", gt_unit="percent",
        source_sections="item_7|item_8_income_stmt",
        difficulty="medium", entity_form="ticker",
        hypothesis_link="H2",
        rationale="R&D-Intensität testet Cross-Section-Synthese",
    ),
    V3Entry(
        fa_type="FA-2", subtype="ratio_compute",
        doc_ids="GOOGL_2024",
        # Reworded 2026-08-01 to match the self-defining pattern already
        # used by ids 88/89 (same metric, AAPL/MSFT) -- Answer Format
        # Convention, found inconsistent during human judge-validation.
        query="Wie hoch war die R&D-Intensität (R&D-Aufwand als Prozent des Umsatzes) von Alphabet in FY2024?",
        gt_value="14.1%", gt_unit="percent",
        source_sections="item_7|item_8_income_stmt",
        difficulty="medium", entity_form="name",
        hypothesis_link="H2",
        rationale="R&D-Intensität testet Cross-Section-Synthese",
    ),
    V3Entry(
        fa_type="FA-2", subtype="segment_share",
        doc_ids="AMZN_2024",
        query="Welchen Anteil am Gesamtumsatz von AMZN hatte das International-Segment in FY2024?",
        gt_value="22.4%", gt_unit="percent",
        source_sections="item_8_segment|item_7",
        difficulty="medium", entity_form="ticker",
        hypothesis_link="H2",
        rationale="Segment-Anteil-Berechnung erweitert um International",
    ),
    V3Entry(
        fa_type="FA-2", subtype="segment_share",
        doc_ids="MSFT_2024",
        query="Welchen Anteil am Gesamtumsatz von Microsoft hatte das Segment Productivity and Business Processes in FY2024?",  # noqa: E501
        gt_value="31.6%", gt_unit="percent",
        source_sections="item_8_segment|item_7",
        difficulty="medium", entity_form="name",
        hypothesis_link="H2",
        rationale="Zweites MSFT-Segment für tiefere Cross-Section-Analyse",
    ),
    V3Entry(
        fa_type="FA-2", subtype="ratio_compute",
        doc_ids="AAPL_2024",
        query="Wie hoch war die Effective Tax Rate von Apple in FY2024?",
        gt_value="24.1%", gt_unit="percent",
        source_sections="item_8_income_stmt|item_7",
        difficulty="medium", entity_form="name",
        hypothesis_link="H2",
        rationale="Effective Tax Rate als zusätzlicher Cross-Section-Vergleich",
    ),
    V3Entry(
        fa_type="FA-2", subtype="ratio_compute",
        doc_ids="MSFT_2024",
        query="Wie hoch war die Effective Tax Rate von MSFT in FY2024?",
        gt_value="18.2%", gt_unit="percent",
        source_sections="item_8_income_stmt|item_7",
        difficulty="medium", entity_form="ticker",
        hypothesis_link="H2",
        rationale="Effective Tax Rate als zusätzlicher Cross-Section-Vergleich",
    ),
    V3Entry(
        fa_type="FA-2", subtype="segment_margin",
        doc_ids="MSFT_2023",
        query="Wie hoch war die Operating Margin des Cloud-Segments (Intelligent Cloud) von Microsoft in FY2023?",
        gt_value="43.1%", gt_unit="percent",
        source_sections="item_8_segment|item_7",
        difficulty="hard", entity_form="name",
        hypothesis_link="H2",
        rationale="Segment-Margin-Berechnung über Vorjahr (für H3-Tool-Use-Vergleich)",
    ),
    V3Entry(
        fa_type="FA-2", subtype="segment_margin",
        doc_ids="GOOGL_2023",
        query="Wie hoch war die Operating Margin des Google-Cloud-Segments von GOOGL in FY2022?",
        gt_value="-7.3%", gt_unit="percent",
        source_sections="item_8_segment|item_7",
        difficulty="hard", entity_form="ticker",
        hypothesis_link="H2",
        rationale="Negative Margin als Edge-Case (FY2022, retrievable aus FY2023-Filing-Vergleichsperiode)",
    ),
    V3Entry(
        fa_type="FA-2", subtype="ratio_compute",
        doc_ids="AAPL_2023",
        query="Wie hoch war der Anteil der SG&A-Aufwendungen am Umsatz von Apple in FY2023?",
        gt_value="6.5%", gt_unit="percent",
        source_sections="item_8_income_stmt|item_7",
        difficulty="medium", entity_form="name",
        hypothesis_link="H2",
        rationale="SG&A-Intensität als zusätzliche Cross-Section-Variante",
    ),
]


# +13 Multi-Year entries (FA-3) with math diversity:
#   4 CAGR (cagr), 5 PP-diff (pp_diff), 4 multi-step (multi_step)
NEW_FA3_ENTRIES: list[V3Entry] = [
    # CAGR (4)
    V3Entry(
        fa_type="FA-3", subtype="growth_cagr",
        doc_ids="AAPL_2022|AAPL_2023|AAPL_2024",
        query="Wie hoch war die Compound Annual Growth Rate (CAGR) des Umsatzes von Apple von FY2022 bis FY2024?",
        gt_value="-0.4%", gt_unit="percent",
        source_sections="item_7",
        difficulty="hard", math_type="cagr", entity_form="name",
        hypothesis_link="H2|H3",
        rationale="CAGR über 3 Jahre, mehrstufige Math (stresst Tool-Use)",
    ),
    V3Entry(
        fa_type="FA-3", subtype="growth_cagr",
        doc_ids="MSFT_2022|MSFT_2023|MSFT_2024",
        query="Wie hoch war die CAGR des Umsatzes von MSFT von FY2022 bis FY2024?",
        gt_value="11.2%", gt_unit="percent",
        source_sections="item_7",
        difficulty="hard", math_type="cagr", entity_form="ticker",
        hypothesis_link="H2|H3",
        rationale="CAGR über 3 Jahre, mehrstufige Math (stresst Tool-Use)",
    ),
    V3Entry(
        fa_type="FA-3", subtype="growth_cagr",
        doc_ids="AMZN_2022|AMZN_2023|AMZN_2024",
        query="Wie hoch war die CAGR des Umsatzes von Amazon von FY2022 bis FY2024?",
        gt_value="11.4%", gt_unit="percent",
        source_sections="item_7",
        difficulty="hard", math_type="cagr", entity_form="name",
        hypothesis_link="H2|H3",
        rationale="CAGR über 3 Jahre, mehrstufige Math (stresst Tool-Use)",
    ),
    V3Entry(
        fa_type="FA-3", subtype="growth_cagr",
        doc_ids="GOOGL_2022|GOOGL_2023|GOOGL_2024",
        query="Wie hoch war die CAGR des Umsatzes von GOOGL von FY2022 bis FY2024?",
        gt_value="11.6%", gt_unit="percent",
        source_sections="item_7",
        difficulty="hard", math_type="cagr", entity_form="ticker",
        hypothesis_link="H2|H3",
        rationale="CAGR über 3 Jahre, mehrstufige Math (stresst Tool-Use)",
    ),
    # PP-diff (5)
    V3Entry(
        fa_type="FA-3", subtype="pp_change",
        doc_ids="AAPL_2022|AAPL_2024",
        query="Um wie viele Prozentpunkte hat sich die Operating Margin von Apple von FY2022 auf FY2024 verändert?",
        gt_value="+0.7pp", gt_unit="pp_change",
        source_sections="item_7|item_8_income_stmt",
        difficulty="medium", math_type="pp_diff", entity_form="name",
        hypothesis_link="H2|H3",
        rationale="PP-Differenz statt YoY-%, prüft Math-Tool präzise",
    ),
    V3Entry(
        fa_type="FA-3", subtype="pp_change",
        doc_ids="MSFT_2022|MSFT_2024",
        query="Um wie viele Prozentpunkte hat sich die Operating Margin von MSFT von FY2022 auf FY2024 verändert?",
        gt_value="+2.5pp", gt_unit="pp_change",
        source_sections="item_7|item_8_income_stmt",
        difficulty="medium", math_type="pp_diff", entity_form="ticker",
        hypothesis_link="H2|H3",
        rationale="PP-Differenz statt YoY-%, prüft Math-Tool präzise",
    ),
    V3Entry(
        fa_type="FA-3", subtype="pp_change",
        doc_ids="AMZN_2022|AMZN_2024",
        query="Um wie viele Prozentpunkte hat sich die gesamte Operating Margin von Amazon von FY2022 auf FY2024 verändert?",  # noqa: E501
        gt_value="+8.0pp", gt_unit="pp_change",
        source_sections="item_7|item_8_income_stmt",
        difficulty="medium", math_type="pp_diff", entity_form="name",
        hypothesis_link="H2|H3",
        rationale="PP-Differenz statt YoY-%, prüft Math-Tool präzise",
    ),
    V3Entry(
        fa_type="FA-3", subtype="pp_change",
        doc_ids="GOOGL_2022|GOOGL_2024",
        query="Um wie viele Prozentpunkte hat sich die Operating Margin von GOOGL von FY2022 auf FY2024 verändert?",
        gt_value="+4.8pp", gt_unit="pp_change",
        source_sections="item_7|item_8_income_stmt",
        difficulty="medium", math_type="pp_diff", entity_form="ticker",
        hypothesis_link="H2|H3",
        rationale="PP-Differenz statt YoY-%, prüft Math-Tool präzise",
    ),
    V3Entry(
        fa_type="FA-3", subtype="pp_change",
        doc_ids="AAPL_2023|AAPL_2024",
        query="Um wie viele Prozentpunkte hat sich die Gross Margin von AAPL von FY2023 auf FY2024 verändert?",
        gt_value="+2.0pp", gt_unit="pp_change",
        source_sections="item_8_income_stmt",
        difficulty="medium", math_type="pp_diff", entity_form="ticker",
        hypothesis_link="H2|H3",
        rationale="PP-Differenz auf Gross-Margin als zusätzliche Variante",
    ),
    # Multi-step (4)
    V3Entry(
        fa_type="FA-3", subtype="multi_step_compute",
        doc_ids="MSFT_2022|MSFT_2023|MSFT_2024",
        query="In welchem Geschäftsjahr (FY2022, FY2023 oder FY2024) hatte MSFT die höchste Operating Margin?",
        gt_value="FY2024", gt_unit="text",
        source_sections="item_7|item_8_income_stmt",
        difficulty="hard", math_type="multi_step", entity_form="ticker",
        hypothesis_link="H2|H3",
        rationale="Argmax über 3 Jahre, mehrstufiger Vergleich",
    ),
    V3Entry(
        fa_type="FA-3", subtype="multi_step_compute",
        doc_ids="AMZN_2022|AMZN_2023|AMZN_2024",
        query="In welchem Geschäftsjahr hatte AMZN den höchsten Operating Cash Flow zwischen FY2022 und FY2024?",
        gt_value="FY2024", gt_unit="text",
        source_sections="item_8_cash_flow",
        difficulty="hard", math_type="multi_step", entity_form="ticker",
        hypothesis_link="H2|H3",
        rationale="Argmax-Aufgabe mit Cash-Flow-Bezug",
    ),
    V3Entry(
        fa_type="FA-3", subtype="multi_step_compute",
        doc_ids="AAPL_2022|AAPL_2023|AAPL_2024",
        query="Um welchen absoluten USD-Betrag haben sich die R&D-Aufwendungen von Apple von FY2022 auf FY2024 verändert?",  # noqa: E501
        gt_value="+$5.4 Mrd", gt_unit="USD_billion",
        source_sections="item_8_income_stmt",
        difficulty="medium", math_type="multi_step", entity_form="name",
        hypothesis_link="H2|H3",
        rationale="Absolute Differenz über 2 Jahre als multi-step",
    ),
    V3Entry(
        fa_type="FA-3", subtype="multi_step_compute",
        doc_ids="GOOGL_2022|GOOGL_2023|GOOGL_2024",
        query="Wie viele Geschäftsjahre in Folge ist der Umsatz von Alphabet zwischen FY2022 und FY2024 gewachsen?",
        gt_value="2", gt_unit="count",
        source_sections="item_7",
        difficulty="hard", math_type="multi_step", entity_form="name",
        hypothesis_link="H2|H3",
        rationale="Sequenzanalyse über 3 Jahre, multi-step",
    ),
]


# +10 Multi-Company entries (FA-4)
NEW_FA4_ENTRIES: list[V3Entry] = [
    V3Entry(
        fa_type="FA-4", subtype="max_min",
        doc_ids="AAPL_2024|MSFT_2024|AMZN_2024|GOOGL_2024",
        query="Welches der 4 Unternehmen (AAPL, MSFT, AMZN, GOOGL) hatte die höchsten R&D-Aufwendungen in FY2024?",
        gt_value="GOOGL ($49.3 Mrd)", gt_unit="USD_billion",
        source_sections="item_8_income_stmt",
        difficulty="easy", entity_form="ticker",
        hypothesis_link="H1|H3",
        rationale="Argmax R&D über alle 4 als Cross-Doc-Synthese",
    ),
    V3Entry(
        fa_type="FA-4", subtype="max_min",
        doc_ids="AAPL_2024|MSFT_2024|AMZN_2024|GOOGL_2024",
        query="Welches der 4 Unternehmen (AAPL, MSFT, AMZN, GOOGL) hatte den höchsten Free Cash Flow in FY2024?",
        gt_value="AAPL ($108.8 Mrd)", gt_unit="USD_billion",
        source_sections="item_8_cash_flow|item_7",
        difficulty="easy", entity_form="ticker",
        hypothesis_link="H1|H3",
        rationale="Argmax FCF, prüft FCF-Synthese aus Cash-Flow + MD&A",
    ),
    V3Entry(
        fa_type="FA-4", subtype="max_min",
        doc_ids="AAPL_2024|MSFT_2024|AMZN_2024|GOOGL_2024",
        query="Welches der 4 Unternehmen hatte die höchsten Capital Expenditures in FY2024?",
        gt_value="AMZN ($83.0 Mrd)", gt_unit="USD_billion",
        source_sections="item_8_cash_flow",
        difficulty="easy", entity_form="name",
        hypothesis_link="H1|H3",
        rationale="Argmax CapEx, gibt AWS-Investment-Thema",
    ),
    V3Entry(
        fa_type="FA-4", subtype="max_min",
        doc_ids="AAPL_2024|MSFT_2024|AMZN_2024|GOOGL_2024",
        query="Welches der 4 Unternehmen (Apple, Microsoft, Amazon, Alphabet) hatte die meisten Mitarbeiter zum Ende von FY2024?",  # noqa: E501
        gt_value="AMZN (~1.55 Mio)", gt_unit="count",
        source_sections="item_1",
        difficulty="easy", entity_form="name",
        hypothesis_link="H1|H3",
        rationale="Argmax Mitarbeiter aus Item 1 (Business)",
    ),
    V3Entry(
        fa_type="FA-4", subtype="cross_firm_pairwise",
        doc_ids="AAPL_2024|MSFT_2024",
        query="Wer hatte die höhere Stock-Based Compensation in FY2024, AAPL oder MSFT?",
        gt_value="AAPL (~$11.7 Mrd > MSFT ~$10.7 Mrd, Werte verifizieren)", gt_unit="text",
        source_sections="item_8_cash_flow|item_8_income_stmt",
        difficulty="medium", entity_form="ticker",
        hypothesis_link="H1|H3",
        rationale="SBC-Vergleich, Edge-Case bei sehr nahen Werten",
    ),
    V3Entry(
        fa_type="FA-4", subtype="cross_firm_diff",
        doc_ids="MSFT_2024|GOOGL_2024",
        query="Wie viel höher war das Operating Income von GOOGL gegenüber MSFT in FY2024?",
        gt_value="+$3.0 Mrd", gt_unit="USD_billion",
        source_sections="item_8_income_stmt|item_7",
        difficulty="medium", entity_form="ticker",
        hypothesis_link="H1|H3",
        rationale="Numerische Differenz, prüft Subtraktion und Vorzeichen",
    ),
    V3Entry(
        fa_type="FA-4", subtype="cross_firm_pairwise",
        doc_ids="AMZN_2024|GOOGL_2024",
        query="Wer hatte die höhere Effective Tax Rate in FY2024, Amazon oder Alphabet?",
        gt_value="GOOGL (~16.4% > AMZN ~13.9%)", gt_unit="percent",
        source_sections="item_8_income_stmt",
        difficulty="medium", entity_form="name",
        hypothesis_link="H1|H3",
        rationale="Tax-Rate-Vergleich als weniger triviale KPI",
    ),
    V3Entry(
        fa_type="FA-4", subtype="cross_firm_pairwise_growth",
        doc_ids="MSFT_2023|MSFT_2024|GOOGL_2023|GOOGL_2024",
        query="Wer hatte das höhere YoY-Wachstum bei R&D-Aufwendungen von FY2023 auf FY2024, MSFT oder GOOGL?",
        gt_value="MSFT (~10.6% > GOOGL ~7.8%)", gt_unit="percent",
        source_sections="item_8_income_stmt",
        difficulty="medium", entity_form="ticker",
        hypothesis_link="H1|H3",
        rationale="Cross-Firm-Wachstumsvergleich auf R&D",
    ),
    V3Entry(
        fa_type="FA-4", subtype="cross_firm_pairwise",
        doc_ids="AAPL_2024|GOOGL_2024",
        query="Wer hatte das höhere Net Income in FY2024, Apple oder Alphabet?",
        # Corrected 2026-08-01 (was "AAPL ($93.7 Mrd > GOOGL $89.6 Mrd)"):
        # GOOGL FY2024 Net income = $100,118M per 10-K Consolidated
        # Statements of Income (source GOOGL_2024 10K_2025-02-05), exceeds
        # AAPL's $93,736M -- winner flips to GOOGL. Same underlying figure
        # as the id-74 max_min correction; found during human
        # judge-validation (R035).
        gt_value="GOOGL ($100.1 Mrd > AAPL $93.7 Mrd)", gt_unit="USD_billion",
        source_sections="item_8_income_stmt",
        difficulty="easy", entity_form="name",
        hypothesis_link="H1|H3",
        rationale="Klassischer Pairwise-Vergleich, AAPL vs GOOGL noch nicht abgedeckt",
    ),
    V3Entry(
        fa_type="FA-4", subtype="cloud_comparison",
        doc_ids="AMZN_2024|GOOGL_2024",
        query="Cloud-Segment: Wer hatte den höheren Umsatz in FY2024, AMZN mit AWS oder GOOGL mit Google Cloud?",
        gt_value="AWS ($107.6 Mrd > Google Cloud $43.2 Mrd)", gt_unit="USD_billion",
        source_sections="item_8_segment|item_7",
        difficulty="easy", entity_form="ticker",
        hypothesis_link="H1|H3",
        rationale="Cloud-Vergleich AWS vs Google Cloud (noch nicht abgedeckt)",
    ),
]


# +30 Refusal entries (FA-Refusal):
#   10 not_in_corpus, 10 false_premise, 10 ambiguous_entity
NEW_REFUSAL_ENTRIES: list[V3Entry] = [
    # 10 not_in_corpus
    V3Entry(
        fa_type="FA-Refusal", subtype="not_in_corpus",
        doc_ids="AAPL_2024", query="Wie hoch war der Umsatz von Apple im Q1 FY2024?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="name",
        hypothesis_link="",
        rationale="Quartalswerte nicht in 10-K (nur Jahreswerte)",
    ),
    V3Entry(
        fa_type="FA-Refusal", subtype="not_in_corpus",
        doc_ids="AAPL_2024|MSFT_2024", query="Wie hoch war das Net Income von NVIDIA in FY2024?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="name",
        hypothesis_link="",
        rationale="NVIDIA außerhalb des Sektor-Scopes (4-Firmen-Korpus)",
    ),
    V3Entry(
        fa_type="FA-Refusal", subtype="not_in_corpus",
        doc_ids="GOOGL_2024", query="Wie hoch war der Werbeumsatz von META in FY2024?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="name",
        hypothesis_link="",
        rationale="META außerhalb des Korpus, obwohl thematisch ähnlich (Werbung)",
    ),
    V3Entry(
        fa_type="FA-Refusal", subtype="not_in_corpus",
        doc_ids="MSFT_2022|MSFT_2023|MSFT_2024", query="Wie hoch war das Net Income von MSFT in FY2020?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="ticker",
        hypothesis_link="",
        rationale="FY2020 vor Korpus-Beginn (FY2022 ist frühestes Jahr)",
    ),
    V3Entry(
        fa_type="FA-Refusal", subtype="not_in_corpus",
        doc_ids="AAPL_2024", query="Wie hoch war AAPLs Umsatz im ersten Halbjahr (H1) von FY2024?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="ticker",
        hypothesis_link="",
        rationale="H1-Werte nicht in 10-K (nur Voll-Jahres-Werte)",
    ),
    V3Entry(
        fa_type="FA-Refusal", subtype="not_in_corpus",
        doc_ids="AMZN_2024", query="Was berichtete AMZN in seinem 8-K-Filing vom März 2024?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="ticker",
        hypothesis_link="",
        rationale="8-K-Filings nicht im Korpus (nur 10-K)",
    ),
    V3Entry(
        fa_type="FA-Refusal", subtype="not_in_corpus",
        doc_ids="GOOGL_2024", query="Wie hoch war GOOGLs Cash Position zum 30. Juni 2024?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="ticker",
        hypothesis_link="",
        rationale="Mid-Year-Werte nicht in 10-K (nur Fiscal-Year-End)",
    ),
    V3Entry(
        fa_type="FA-Refusal", subtype="not_in_corpus",
        doc_ids="AAPL_2024", query="Wie viele iPhone-Einheiten hat Apple in FY2024 verkauft?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="name",
        hypothesis_link="",
        rationale="Apple meldet seit 2018 keine Stückzahlen mehr separat",
    ),
    V3Entry(
        fa_type="FA-Refusal", subtype="not_in_corpus",
        doc_ids="AMZN_2024", query="Wie hoch ist der Marktanteil von AWS im globalen Cloud-Markt laut Amazons 10-K FY2024?",  # noqa: E501
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="name",
        hypothesis_link="",
        rationale="10-K macht keine Marktanteilsangaben (Wettbewerbsdaten)",
    ),
    V3Entry(
        fa_type="FA-Refusal", subtype="not_in_corpus",
        doc_ids="MSFT_2024", query="Wer ist laut MSFTs 10-K FY2024 der größte einzelne Konkurrent von MSFT?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="ticker",
        hypothesis_link="",
        rationale="10-K listet Wettbewerb nur generisch, keine 'größter Konkurrent'-Aussage",
    ),

    # 10 false_premise
    V3Entry(
        fa_type="FA-Refusal", subtype="false_premise",
        doc_ids="AAPL_2024", query="Wie hoch war der Verlust von Apple in FY2024?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="name",
        hypothesis_link="",
        rationale="Apple hatte FY2024 keinen Verlust, sondern $93.7 Mrd Net Income",
    ),
    V3Entry(
        fa_type="FA-Refusal", subtype="false_premise",
        doc_ids="AMZN_2024", query="Wie viel Dividende hat AMZN in FY2024 pro Aktie ausgeschüttet?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="ticker",
        hypothesis_link="",
        rationale="Amazon zahlt keine Dividende",
    ),
    V3Entry(
        fa_type="FA-Refusal", subtype="false_premise",
        doc_ids="MSFT_2024", query="Welcher Prozentsatz der MSFT-Mitarbeiter ist in Deutschland tätig?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="ticker",
        hypothesis_link="",
        rationale="10-K bricht Mitarbeiter nicht nach Ländern auf, nur US/Non-US",
    ),
    V3Entry(
        fa_type="FA-Refusal", subtype="false_premise",
        doc_ids="GOOGL_2024", query="Wie hat sich Alphabets Hardware-Umsatz von FY2022 bis FY2024 entwickelt?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="name",
        hypothesis_link="",
        rationale="Alphabet weist keinen separaten 'Hardware-Umsatz' aus (Pixel ist in Google Services)",
    ),
    V3Entry(
        fa_type="FA-Refusal", subtype="false_premise",
        doc_ids="AAPL_2024", query="Wie hoch war AAPLs Cloud-Service-Umsatz in FY2024?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="ticker",
        hypothesis_link="",
        rationale="Apple meldet 'Services' aggregiert, kein separates 'Cloud-Service'-Segment",
    ),
    V3Entry(
        fa_type="FA-Refusal", subtype="false_premise",
        doc_ids="MSFT_2024", query="Welche Akquisitionen hat MSFT in FY2024 mit einem Volumen über $50 Mrd getätigt?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="ticker",
        hypothesis_link="",
        rationale="Activision-Closing war FY2023; FY2024 keine Akquisition >$50B",
    ),
    V3Entry(
        fa_type="FA-Refusal", subtype="false_premise",
        doc_ids="GOOGL_2024", query="Wann hat GOOGL im FY2024 einen Aktiensplit durchgeführt?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="ticker",
        hypothesis_link="",
        rationale="Alphabet-Split war Juli 2022, nicht FY2024",
    ),
    V3Entry(
        fa_type="FA-Refusal", subtype="false_premise",
        doc_ids="AAPL_2024", query="Wie viele neue Apple Stores wurden in FY2024 weltweit eröffnet?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="name",
        hypothesis_link="",
        rationale="10-K macht keine Store-Count-Angaben (operative Detail)",
    ),
    V3Entry(
        fa_type="FA-Refusal", subtype="false_premise",
        doc_ids="MSFT_2024", query="Welcher Bruttomarge-Verlust trat im Microsoft-Cloud-Segment in FY2024 auf?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="name",
        hypothesis_link="",
        rationale="Microsoft Cloud erwirtschaftete Gewinn, keinen Verlust",
    ),
    V3Entry(
        fa_type="FA-Refusal", subtype="false_premise",
        doc_ids="AMZN_2024", query="Wie hoch waren die Werbeeinnahmen von Apple im AMZN-10-K von FY2024?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="name",
        hypothesis_link="",
        rationale="Falsche Prämisse: Apple-Daten sind nicht in Amazons 10-K enthalten",
    ),

    # 10 ambiguous_entity
    V3Entry(
        fa_type="FA-Refusal", subtype="ambiguous_entity",
        doc_ids="AAPL_2024|MSFT_2024|AMZN_2024|GOOGL_2024",
        query="Wie hoch war der Umsatz im letzten Geschäftsjahr?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="none",
        hypothesis_link="",
        rationale="Weder Firma noch Jahr explizit, 4 Firmen × 3 Jahre möglich",
    ),
    V3Entry(
        fa_type="FA-Refusal", subtype="ambiguous_entity",
        doc_ids="AAPL_2024|MSFT_2024|AMZN_2024|GOOGL_2024",
        query="Wie performte das Cloud-Geschäft in FY2024?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="none",
        hypothesis_link="",
        rationale="Cloud bei MSFT, AMZN und GOOGL existent — Firma fehlt",
    ),
    V3Entry(
        fa_type="FA-Refusal", subtype="ambiguous_entity",
        doc_ids="AAPL_2024|MSFT_2024|AMZN_2024|GOOGL_2024",
        query="Wie viel CapEx wurde investiert?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="none",
        hypothesis_link="",
        rationale="Firma und Jahr fehlen, CapEx in allen 4 Berichten anders",
    ),
    V3Entry(
        fa_type="FA-Refusal", subtype="ambiguous_entity",
        doc_ids="AAPL_2024|MSFT_2024|AMZN_2024|GOOGL_2024",
        query="Wie hoch war die Operating Margin?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="none",
        hypothesis_link="",
        rationale="Firma und Jahr fehlen",
    ),
    V3Entry(
        fa_type="FA-Refusal", subtype="ambiguous_entity",
        doc_ids="AAPL_2024|MSFT_2024|AMZN_2024|GOOGL_2024",
        query="Welche Firma hatte das stärkste Wachstum?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="none",
        hypothesis_link="",
        rationale="'Wachstum' welcher Metrik? Welcher Zeitraum?",
    ),
    V3Entry(
        fa_type="FA-Refusal", subtype="ambiguous_entity",
        doc_ids="AAPL_2024|MSFT_2024|AMZN_2024|GOOGL_2024",
        query="Wie hat sich das Geschäft entwickelt?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="none",
        hypothesis_link="",
        rationale="Komplett unspezifisch",
    ),
    V3Entry(
        fa_type="FA-Refusal", subtype="ambiguous_entity",
        doc_ids="AAPL_2024|MSFT_2024|AMZN_2024|GOOGL_2024",
        query="Wie hoch war der Cashflow?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="none",
        hypothesis_link="",
        rationale="Welche Firma, welcher Cashflow-Typ (Operating/Investing/Financing)?",
    ),
    V3Entry(
        fa_type="FA-Refusal", subtype="ambiguous_entity",
        doc_ids="AAPL_2024|MSFT_2024|AMZN_2024|GOOGL_2024",
        query="Was ist das wichtigste Risiko laut dem Bericht?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="none",
        hypothesis_link="",
        rationale="Welche Firma? Welches Jahr? 4×3=12 mögliche Berichte",
    ),
    V3Entry(
        fa_type="FA-Refusal", subtype="ambiguous_entity",
        doc_ids="AAPL_2024|MSFT_2024|AMZN_2024|GOOGL_2024",
        query="Wie hoch war der Anteil der Werbeumsätze am Gesamtumsatz?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="none",
        hypothesis_link="",
        rationale="Werbung primär bei GOOGL (76%), AMZN (Ads-Segment), MSFT (Bing) — Firma fehlt",
    ),
    V3Entry(
        fa_type="FA-Refusal", subtype="ambiguous_entity",
        doc_ids="AAPL_2024|GOOGL_2024",
        query="Wer hatte mehr Umsatz, Apple oder Google?",
        gt_value="", gt_unit="n/a", source_sections="",
        expected_answerable=False, entity_form="none",
        hypothesis_link="",
        rationale="'Google' uneindeutig (Alphabet vs. Google-Segment); welches Jahr?",
    ),
]


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def build_v3() -> list[V3Entry]:
    """Migrate v2 + append 73 new entries; return the full list (no IDs yet)."""
    v2_rows = read_v2_rows()
    migrated = [migrate_v2_row(r) for r in v2_rows]
    apply_post_migration_overrides(migrated)

    new_entries: list[V3Entry] = []
    new_entries.extend(NEW_FA1_ENTRIES)
    new_entries.extend(NEW_FA2_ENTRIES)
    new_entries.extend(NEW_FA3_ENTRIES)
    new_entries.extend(NEW_FA4_ENTRIES)
    new_entries.extend(NEW_REFUSAL_ENTRIES)

    return migrated + new_entries


def assert_stratification(entries: list[V3Entry]) -> None:
    """Verify the final 5x30 stratification."""
    counts: dict[str, int] = {}
    for e in entries:
        counts[e.fa_type] = counts.get(e.fa_type, 0) + 1
    expected = {"FA-1": 30, "FA-2": 30, "FA-3": 30, "FA-4": 30, "FA-Refusal": 30}
    if counts != expected:
        raise AssertionError(
            f"Stratification mismatch.\n  Expected: {expected}\n  Got:      {counts}"
        )


def write_v3(entries: list[V3Entry]) -> None:
    TARGET_V3_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(TARGET_V3_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(V3_HEADER)
        for e in entries:
            writer.writerow(e.to_row())


def main() -> None:
    entries = build_v3()

    for i, e in enumerate(entries, start=1):
        e.id = i

    assert_stratification(entries)

    write_v3(entries)

    print(f"Wrote {len(entries)} entries to {TARGET_V3_CSV.relative_to(PROJECT_ROOT)}")
    print()
    print("Stratification:")
    counts: dict[str, int] = {}
    for e in entries:
        counts[e.fa_type] = counts.get(e.fa_type, 0) + 1
    for fa_type in ["FA-1", "FA-2", "FA-3", "FA-4", "FA-Refusal"]:
        print(f"  {fa_type}: {counts[fa_type]}")
    print()
    print(
        f"Note: {sum(1 for e in entries if not e.expected_answerable)} entries are "
        f"FA-Refusal (expected_answerable=false)."
    )
    print()
    print(
        "USER ACTION REQUIRED: Verify the 73 new entries (id 78-150) against "
        "the SEC filings before declaring v3 frozen. See "
        "data/gold_standard/gold_standard_README.md \"Verification Workflow\"."
    )


if __name__ == "__main__":
    main()
