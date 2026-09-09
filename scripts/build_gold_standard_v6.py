"""
Build Gold Standard v6 from v5 by adding the `reference_decomposition` column.

v6 is METADATA-ONLY, under the same constraint v5 carried: it adds one column
and changes no `query`, no `gt_value`, no existing cell whatsoever. That is
what keeps stored answers from earlier runs scoreable against v6. The moment a
v6 question differs from its v5 wording, every post-hoc rescoring compares an
answer to a question that was never asked (see EVAL_DECISION_LOG [2026-09-07]).
The guarantee is enforced below, not merely intended: `_assert_metadata_only`
diffs every pre-existing cell and refuses to write if one moved.

WHY THE COLUMN EXISTS
---------------------
Dimension 3 of the reasoning metric (completeness) asks whether a chain covers
the sub-questions the item requires. Without a reference decomposition the
judge invents its own standard per item -- and so do human raters, which is the
largest known source of disagreement on completeness-style dimensions. The
column fixes the standard once, in the dataset, where it can be reviewed.

THIS SCRIPT PRODUCES A DRAFT
----------------------------
The decompositions are derived per subtype from `subtype`, `doc_ids`,
`source_sections` and `math_type`, then written to `--review-out` for hand
checking. The rules are a first cut, not an oracle: `synthesis` covers both
two-component derivations (Operating Margin) and plain lookups (CapEx),
`max_min` covers both direct figures and per-company derived ratios, and the
metric phrase is extracted from the question text heuristically. Every row
needs eyes on it before v6 is used.

Generating them with an LLM instead was rejected: it would move the judge's
arbitrariness into the gold standard, where it is harder to see and impossible
to argue with.

Usage:
    uv run python scripts/build_gold_standard_v6.py
    uv run python scripts/build_gold_standard_v6.py --review-out review.md
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GS_DIR = PROJECT_ROOT / "data" / "gold_standard"

NEW_COLUMN = "reference_decomposition"
SEPARATOR = "|"

TICKER_TO_NAME = {
    "AAPL": "Apple", "MSFT": "Microsoft", "AMZN": "Amazon", "GOOGL": "Alphabet",
}

# Parenthetical company enumerations -- "(AAPL, MSFT, AMZN, GOOGL)" -- are
# scope, not metric, and they wreck any opener match if left in place.
_COMPANY_LIST_RE = re.compile(
    r"\(\s*(?:AAPL|MSFT|AMZN|GOOGL|Apple|Microsoft|Amazon|Alphabet)\b[^)]*\)",
    re.IGNORECASE,
)
# Parenthetical year enumerations -- "(FY2020, FY2022 or FY2024)" -- likewise.
_YEAR_LIST_RE = re.compile(r"\(\s*FY\s?20\d{2}[^)]*\)", re.IGNORECASE)

# Question openers to strip when isolating the metric phrase.
_OPENER_RE = re.compile(
    r"^(?:what\s+(?:was|were|is|are)|how\s+much|how\s+many|by\s+how\s+many|"
    r"which\s+of\s+the\s+\d+\s+companies|which\s+company|which\s+fiscal\s+year|"
    r"in\s+which\s+fiscal\s+year|who\s+had)\s*",
    re.IGNORECASE,
)
# Verb phrases left over after the opener: "did MSFT have the highest ...",
# "has Alphabet's revenue grown", "had more Total Revenue".
_VERB_TAIL_RE = re.compile(
    r"^(?:did|has|have|had|was|were)\s+(?:the\s+)?", re.IGNORECASE,
)
# Superlatives AND comparatives. "higher"/"lower" were missing on the first
# pass, which is how "Establish higher Net Income for Apple" reached the
# review sheet flagged as clean: a per-company lookup must ask for the LEVEL,
# the comparison is its own sub-question.
_SUPERLATIVE_RE = re.compile(
    r"^(?:have|had|has|generated)?\s*(?:the\s+)?"
    r"(?:highest|lowest|most|more|fewer|less|greater|higher|lower|larger|smaller|bigger)\s+",
    re.IGNORECASE,
)
# A trailing preposition or article is always residue of a stripped clause
# ("Operating Margin of AAPL in FY2023" -> "Operating Margin of").
_TRAILING_STOP_RE = re.compile(
    r"\s+(?:of|for|in|by|to|from|at|the|a|an|did|had|have|has)$", re.IGNORECASE,
)
# Change wrappers. A per-period sub-question must ask for the LEVEL, not the
# change: "revenue growth in FY2024 vs FY2022" decomposes into the revenue of
# each year plus the change, so the metric to look up is "revenue".
_CHANGE_WRAPPERS = (
    re.compile(r"^overall\s+trend\s+in\s+", re.IGNORECASE),
    re.compile(r"^compound\s+annual\s+growth\s+rate\s*(?:\(cagr\))?\s*(?:of|in)\s+", re.IGNORECASE),
    re.compile(r"^(?:yoy\s+)?growth\s+(?:rate\s+)?(?:of|in)\s+", re.IGNORECASE),
    re.compile(r"^change\s+in\s+", re.IGNORECASE),
    re.compile(r"^consecutive\s+fiscal\s+years\s+.*?\b(revenue|income|margin|cash\s+flow)\b.*$", re.IGNORECASE),
)
_TRAILING_CHANGE_RE = re.compile(
    r"\s+(?:growth|grown|changed|trend|development)\s*$", re.IGNORECASE,
)
# Trailing scope clauses to strip once the metric phrase is isolated.
_TAIL_RE = re.compile(
    r"\s*(?:\b(?:in|for|at\s+the\s+end\s+of|from|between|compared\s+to)\b\s*FY\s?20\d{2}.*|"
    r"\s+—.*|\?.*)$",
    re.IGNORECASE,
)
_POSSESSIVE_RE = re.compile(
    r"\b(?:AAPL|MSFT|AMZN|GOOGL|Apple|Microsoft|Amazon|Alphabet|Google)(?:'s)?\s+",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"FY\s?(20\d{2})", re.IGNORECASE)

# A phrase still carrying any of these has not been cleanly isolated.
_LEFTOVER_MARKERS = (
    re.compile(r"[)(]"),
    re.compile(r"\b(?:AAPL|MSFT|AMZN|GOOGL)\b"),
    re.compile(r"^(?:did|has|have|had|was|were|the)\b", re.IGNORECASE),
    # Any comparative or superlative left in a per-entity lookup.
    re.compile(r"\b(?:highest|lowest|more|most|higher|lower|larger|smaller)\b", re.IGNORECASE),
    # Question machinery: if these survived, no opener matched and the phrase
    # is a fragment of the question rather than the metric it asks for.
    re.compile(r"\b(?:what|which|who|whose|how)\b", re.IGNORECASE),
    re.compile(r"\b(?:did|does|do)\b", re.IGNORECASE),
    # A dangling preposition means a clause was cut mid-way.
    re.compile(r"\b(?:of|for|in|by|to|from|at)$", re.IGNORECASE),
)


PLACEHOLDER_METRIC = "the figure the question asks for"


def extract_metric(query: str, level_only: bool = False) -> tuple[str, bool]:
    """Isolate the metric phrase from a question. Returns (phrase, confident).

    `level_only` asks for the underlying quantity rather than the change in
    it -- what a per-period sub-question of a growth or trend item needs.
    """
    text = _COMPANY_LIST_RE.sub("", query.strip())
    text = _YEAR_LIST_RE.sub("", text)
    text = _OPENER_RE.sub("", text, count=1)
    text = _VERB_TAIL_RE.sub("", text, count=1)
    text = _SUPERLATIVE_RE.sub("", text, count=1)
    text = _POSSESSIVE_RE.sub("", text, count=1)
    text = _TAIL_RE.sub("", text)
    text = text.strip(" ,.?—-")
    text = re.sub(r"^(?:the|a|an)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text).strip(" ,.?—-")
    for _ in range(3):
        stripped = _TRAILING_STOP_RE.sub("", text).strip(" ,.?—-")
        if stripped == text:
            break
        text = stripped

    if level_only:
        for pattern in _CHANGE_WRAPPERS:
            stripped = pattern.sub("", text, count=1)
            if stripped != text:
                text = stripped.strip(" ,.?—-")
                break
        text = _TRAILING_CHANGE_RE.sub("", text).strip(" ,.?—-")
        text = _POSSESSIVE_RE.sub("", text, count=1).strip(" ,.?—-")

    confident = (
        2 <= len(text) <= 90
        and not any(p.search(text) for p in _LEFTOVER_MARKERS)
    )
    return (text if confident else PLACEHOLDER_METRIC), confident


def question_years(query: str, doc_ids: list[str]) -> list[str]:
    """Fiscal years the question is about -- not necessarily the filing years.

    A 10-K carries three comparative years, so an item can ask about FY2023
    while its only source document is the FY2024 filing. The question text is
    therefore authoritative here and `doc_ids` only the fallback.
    """
    years = sorted({m for m in _YEAR_RE.findall(query)})
    if years:
        return years
    return sorted({d.split("_")[1] for d in doc_ids if "_" in d})


def tickers_of(doc_ids: list[str]) -> list[str]:
    seen: list[str] = []
    for doc_id in doc_ids:
        ticker = doc_id.split("_")[0]
        if ticker not in seen:
            seen.append(ticker)
    return seen


def _lookup(metric: str, ticker: str, year: str) -> str:
    company = TICKER_TO_NAME.get(ticker, ticker)
    if metric == PLACEHOLDER_METRIC:
        return f"Establish, for {company} in FY{year}, the figure the question asks for."
    return f"Establish {metric} for {company} in FY{year}."


def decompose(row: dict) -> tuple[list[str], bool]:
    """Derive the reference decomposition for one row. Returns (tfs, confident)."""
    subtype = row["subtype"]
    query = row["query"]
    doc_ids = [d for d in row["doc_ids"].split("|") if d]
    level_only = subtype in {
        "growth_yoy", "pp_change", "growth_cagr", "trend_qualitative",
        "multi_step_compute", "cross_firm_pairwise_growth",
    }
    metric, confident = extract_metric(query, level_only=level_only)
    years = question_years(query, doc_ids)
    tickers = tickers_of(doc_ids)
    t0 = tickers[0] if tickers else "the company"
    company = TICKER_TO_NAME.get(t0, t0)

    # --- single lookup -----------------------------------------------------
    if subtype in {"basis_kpi", "liquidity", "eps"}:
        return [_lookup(metric, t0, years[-1] if years else "?")], confident

    # --- two components plus a combination ---------------------------------
    if subtype == "solvency":
        year = years[-1] if years else "?"
        return [
            _lookup("Total Liabilities", t0, year),
            _lookup("Total Stockholders' Equity", t0, year),
            "Divide Total Liabilities by Total Stockholders' Equity to obtain the ratio.",
        ], True

    # For the two segment subtypes the two components are fixed by the subtype
    # itself, so they are named outright instead of being read off the
    # question. "What share of X's total revenue did Y have" defeats every
    # opener pattern, and the generic wording then carried the whole question
    # fragment into the sub-question.
    if subtype == "segment_share":
        year = years[-1] if years else "?"
        return [
            f"Establish the revenue of the segment in question for {company} in FY{year}.",
            f"Establish total revenue for {company} in FY{year}.",
            "Express the segment's revenue as a share of the total.",
        ], True

    if subtype == "segment_margin":
        year = years[-1] if years else "?"
        return [
            f"Establish the revenue of the segment in question for {company} in FY{year}.",
            f"Establish the operating result of that segment for {company} in FY{year}.",
            "Divide the segment result by the segment revenue to obtain the margin.",
        ], True

    if subtype in {"ratio_compute", "synthesis"}:
        year = years[-1] if years else "?"
        return [
            f"Establish the first component of {metric} for {company} in FY{year}.",
            f"Establish the second component of {metric} for {company} in FY{year}.",
            f"Combine both components into {metric}.",
        ], confident

    # --- change between two points in time ---------------------------------
    if subtype in {"growth_yoy", "pp_change"}:
        early, late = (years[0], years[-1]) if len(years) >= 2 else ("?", "?")
        change = (
            "Express the change in percentage points."
            if subtype == "pp_change"
            else "Compute the change between the two figures."
        )
        return [
            _lookup(metric, t0, early),
            _lookup(metric, t0, late),
            change,
        ], confident and len(years) >= 2

    if subtype == "growth_cagr":
        early, late = (years[0], years[-1]) if len(years) >= 2 else ("?", "?")
        return [
            _lookup(metric, t0, early),
            _lookup(metric, t0, late),
            f"Establish the number of years between FY{early} and FY{late}.",
            "Compute the compound annual growth rate from those three quantities.",
        ], confident and len(years) >= 2

    # --- one figure per period plus a verdict ------------------------------
    if subtype in {"trend_qualitative", "multi_step_compute"}:
        verdict = (
            "State the overall direction of the development."
            if subtype == "trend_qualitative"
            else "Compare the values across the periods and name the answer."
        )
        return [_lookup(metric, t0, y) for y in years] + [verdict], confident and bool(years)

    # --- one figure per company plus a comparison --------------------------
    if subtype in {"cross_firm_pairwise", "cross_firm_diff", "cloud_comparison", "max_min"}:
        year = years[-1] if years else "?"
        verdict = {
            "cross_firm_diff": "Compute the difference between the figures.",
            "max_min": "Compare the figures and name the company asked for.",
        }.get(subtype, "Compare the figures and name the company asked for.")
        return [_lookup(metric, t, year) for t in tickers] + [verdict], confident and bool(tickers)

    if subtype == "cross_firm_pairwise_growth":
        early, late = (years[0], years[-1]) if len(years) >= 2 else ("?", "?")
        tfs = []
        for ticker in tickers:
            tfs.append(_lookup(metric, ticker, early))
            tfs.append(_lookup(metric, ticker, late))
        tfs.append("Compute the growth rate for each company.")
        tfs.append("Compare the growth rates and name the company asked for.")
        return tfs, confident and len(years) >= 2

    return [f"Answer the question for {company}."], False


def _assert_metadata_only(before: list[dict], after: list[dict], src_name: str) -> None:
    """Refuse to write if any pre-existing cell changed."""
    for old, new in zip(before, after):
        for key, value in old.items():
            if new[key] != value:
                raise SystemExit(
                    f"{src_name}: metadata-only guarantee violated at id "
                    f"{old['id']}, column '{key}': {value!r} -> {new[key]!r}"
                )


def build(
    src_name: str,
    dst_name: str,
    review: list[str] | None = None,
    precomputed: dict[str, str] | None = None,
) -> dict[str, str]:
    """Write one v6 file. Returns id -> decomposition, for reuse on the other.

    The decomposition is derived ONCE, from the English file, and the identical
    English text is written into the German file. Two reasons: the judge that
    consumes it runs in English, and `validate_gold_standard.py` requires every
    column outside {query, gt_value, rationale, gt_correction} to be
    byte-identical across the two files. Deriving it twice from the two
    question texts would produce two different columns and break that
    invariant on all 120 answerable rows.
    """
    src = GS_DIR / src_name
    dst = GS_DIR / dst_name
    with src.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter=";"))
    original = [dict(r) for r in rows]

    if NEW_COLUMN in rows[0]:
        raise SystemExit(f"{src_name} already carries '{NEW_COLUMN}'")
    fieldnames = list(rows[0].keys()) + [NEW_COLUMN]

    produced: dict[str, str] = {}
    unconfident = 0
    for row in rows:
        if row["expected_answerable"].strip().lower() != "true":
            # FA-Refusal has no meaningful decomposition: the correct chain is
            # "the premise cannot be checked". The reasoning dimensions are
            # computed on the answerable stratum only (spec section 8.4).
            row[NEW_COLUMN] = ""
            continue
        if precomputed is not None:
            row[NEW_COLUMN] = precomputed.get(row["id"], "")
            continue
        tfs, confident = decompose(row)
        row[NEW_COLUMN] = SEPARATOR.join(tfs)
        produced[row["id"]] = row[NEW_COLUMN]
        if not confident:
            unconfident += 1
        if review is not None:
            review.append(
                f"| {row['id']} | {row['subtype']} | {'ok' if confident else '**CHECK**'} | "
                f"{row['query'][:70]} | "
                f"{'<br>'.join(f'{i}. {t}' for i, t in enumerate(tfs, 1))} |"
            )

    _assert_metadata_only(original, rows, src_name)

    with dst.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    answerable = sum(1 for r in rows if r["expected_answerable"].strip().lower() == "true")
    suffix = f", {unconfident} flagged CHECK" if precomputed is None else " (copied)"
    print(
        f"wrote {dst.relative_to(PROJECT_ROOT)} "
        f"({len(rows)} rows, {answerable} decomposed{suffix})"
    )
    return produced


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--review-out", default="data/gold_standard/v6_decomposition_review.md",
        help="Where to write the hand-review table.",
    )
    args = parser.parse_args()

    review: list[str] = []
    # HISTORICAL — both version literals are pinned on purpose. This is the
    # one-off v5 -> v6 migration; pointing it at GOLD_STANDARD_EN/DE would make
    # it re-derive v6 from itself once those constants move on.
    #
    # English first: it is the file the decompositions are derived from, and
    # the German file then receives the identical column.
    decompositions = build(
        "gold_standard_v5_en.csv", "gold_standard_v6_en.csv", review=review
    )
    build(
        "gold_standard_v5.csv", "gold_standard_v6.csv", precomputed=decompositions
    )

    out = PROJECT_ROOT / args.review_out
    # HISTORICAL — the filename in the instruction text below names this
    # migration's own output, so it is pinned like the build() calls above and
    # must not follow GOLD_STANDARD_EN. Once the canonical constant moves past
    # v6, this sheet still documents the v6 review that actually happened.
    out.write_text(
        "# Gold Standard v6 — reference_decomposition, hand-review sheet\n\n"
        "Generated draft. Every row needs review before v6 is used for scoring;\n"
        "rows marked **CHECK** had an unreliable metric extraction and should be\n"
        "read first. Edit `gold_standard_v6_en.csv` (and the German file) directly\n"
        "— do NOT re-run this script afterwards, it would overwrite the edits.\n\n"
        "| id | subtype | extraction | question | reference decomposition |\n"
        "|---|---|---|---|---|\n" + "\n".join(review) + "\n",
        encoding="utf-8",
    )
    print(f"review sheet: {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
