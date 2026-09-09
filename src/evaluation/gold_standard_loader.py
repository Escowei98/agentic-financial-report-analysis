"""
Gold Standard data loader for RAGAS evaluation.

Supports both schemas:
  - v2 (frozen): notebooks/experiments/sys1_rag_monolith/ablation_test_data_v2.csv
                 8 columns, semicolon-delimited; freetext source field;
                 4 fragetypes (Single/Cross-Sec/Multi-Year/Multi-Comp).
  - v3 (frozen): data/gold_standard/gold_standard_v3.csv
                 14 columns, semicolon-delimited; canonical source IDs;
                 5 fragetypes (FA-1..FA-4 + FA-Refusal).
  - v4 (frozen):  data/gold_standard/gold_standard_v4.csv
                  v3 plus `doc_id_groups` (acceptable citation sources per
                  required fact) and `window_class` (FA-3 within/cross
                  document window).
  - v5 (frozen):  data/gold_standard/gold_standard_v5.csv
                  v4 plus `refusal_evidence` and `gt_correction`, the two
                  columns the subtype-conditioned refusal scoring needs.
                  Metadata-only: no question and no ground-truth value
                  differs from v4 (see scripts/build_gold_standard_v5.py),
                  so answers stored from a v4 run stay scoreable against v5.
  - v6 (current): data/gold_standard/gold_standard_v6.csv
                  v5 plus `reference_decomposition`, the required
                  sub-questions Dimension 3 of the reasoning metric scores
                  coverage against. Metadata-only under the same guarantee,
                  enforced by scripts/build_gold_standard_v6.py.

    Deliberately NOT part of any schema: an `expected_tools` column. A
    deterministic tool-selection metric was removed on 2026-08-15 in favour of
    the Agentic Score's judge dimension; that dimension then failed human
    validation at kappa 0.03 and was itself removed on 2026-09-08. Tool
    selection is now out of scope, deliberately and on the record — see
    EVAL_DECISION_LOG.md and REASONING_QUALITY_SPEC.md section 11.

Schema is auto-detected from the CSV header. v2 fields without a v3/v4
counterpart stay as None on newer entries; newer fields are filled with
sensible defaults when loading an older file (so consumers can rely on the
unified GoldStandardItem dataclass whichever version they load).

Converts the CSV into a format compatible with RAGAS evaluation:
question, ground_truth, doc_refs, query_type.
"""

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_GS_DIR = _PROJECT_ROOT / "data" / "gold_standard"

# --- The canonical gold standard -------------------------------------------
#
# Import these instead of spelling the filename out. Nine scripts each held
# their own literal path, and when the corpus and the dataset moved to v4 on
# 2026-09-06 not one of them followed: the whole run pipeline, the report
# generator and the judge-validation batch were still loading v3 — a file
# whose FA-3 questions ask about different fiscal years and whose doc ids
# point at filings that are not in the corpus. Nothing failed loudly, the
# numbers were simply answers to the previous version of the dataset.
#
# Scripts that operate on the SAVED ARTEFACTS of an earlier run are the
# deliberate exception and pin their own version: rescoring answers that were
# generated for v3 questions against v4 questions would be meaningless. Those
# scripts say so at their own path assignment.
GOLD_STANDARD_EN = _GS_DIR / "gold_standard_v6_en.csv"
"""English gold standard — what every system and evaluator runs against."""

GOLD_STANDARD_DE = _GS_DIR / "gold_standard_v6.csv"
"""German source of record. Hand-maintained; the English file is patched
cell-wise to match (see data/gold_standard/gold_standard_README.md,
"Correction convention")."""


# Mapping between v2 query_type freetext and v3 fa_type IDs.
V2_TYPE_TO_FA_TYPE = {
    "Single": "FA-1",
    "Cross-Sec": "FA-2",
    "Multi-Year": "FA-3",
    "Multi-Comp": "FA-4",
}
FA_TYPE_TO_V2_TYPE = {v: k for k, v in V2_TYPE_TO_FA_TYPE.items()}


@dataclass
class GoldStandardItem:
    """A single gold-standard Q&A pair for evaluation.

    Unified across v2 and v3 schemas. v2-only fields (`query_type`,
    `source_section`) and v3-only fields (`fa_type`, `subtype`, ...)
    are both present; the loader populates the appropriate ones based
    on the source schema. `question` and `ground_truth` are canonical
    fields that exist in both schemas.
    """

    id: int
    question: str
    ground_truth: str
    doc_refs: str

    query_type: str = ""
    source_section: str = ""
    rationale: str = ""
    entity_form: str | None = None

    fa_type: str = ""
    subtype: str = ""
    doc_ids: list[str] = field(default_factory=list)
    gt_unit: str = ""
    source_sections: list[str] = field(default_factory=list)
    expected_answerable: bool = True
    difficulty: str = "medium"
    math_type: str = ""
    hypothesis_link: list[str] = field(default_factory=list)

    # --- v4 fields ---
    doc_id_groups: list[list[str]] = field(default_factory=list)
    """Acceptable citation sources, grouped by the fact each one supports.

    A 10-K reports prior years in comparative columns, so one fiscal year is
    often carried by more than one filing. Each inner list holds the filings
    that are interchangeable for one required fact; citing ANY member of a
    group satisfies it. `doc_ids` stays the flat union for the judge prompt
    and for pre-v4 consumers. Empty on v2/v3 files, where callers fall back
    to `doc_ids`.
    """

    window_class: str = ""
    """FA-3 only: 'within_window' or 'cross_window'.

    Whether the item's year span fits inside a single filing's comparative
    columns. Splits the multi-year stratum so the effect of the document
    boundary can be read off directly instead of being inferred across
    strata.
    """

    # --- v5 fields ---
    refusal_evidence: str = ""
    """FA-Refusal only: what the corpus can actually support as a response.

    One of 'out_of_scope' (company/fiscal year/filing type outside the
    corpus), 'counter_evidence' (the filings positively contradict the
    premise), 'silent' (in scope, but the item is not reported) or
    'underspecified' (no single company and/or fiscal year identified).

    The distinction exists because a correction is only legitimate where the
    corpus carries the counter-evidence. For a 'silent' item the sole
    groundable statement is "the filings do not report X"; asserting that X
    did not happen is parametric knowledge and therefore exactly the NF-1
    violation this stratum is built to detect. Empty on answerable items and
    on pre-v5 files.
    """

    # --- v6 fields ---
    reference_decomposition: list[str] = field(default_factory=list)
    """Answerable items only: the sub-questions a complete chain must cover.

    The standard Dimension 3 of the reasoning metric scores against
    (REASONING_QUALITY_SPEC.md section 8). It lives in the dataset rather
    than in the judge prompt because a judge left to infer the required
    sub-questions per item invents its own standard each time -- and so do
    human raters, which is what makes completeness-style dimensions
    disagree.

    Empty on FA-Refusal items: the correct chain there is "the premise
    cannot be checked", which has no decomposition. Also empty on pre-v6
    files.
    """

    gt_correction: str = ""
    """FA-Refusal only: reference text for the refusal_quality judge.

    NOT a ground truth answer -- `gt_value` stays empty for every FA-Refusal
    row, which remains the canonical signature of the stratum. This field
    carries what a system could legitimately add beyond a bare refusal:
    the scope boundary, the contradicting figure, or the missing
    specification. Empty on answerable items and on pre-v5 files.
    """


def _detect_schema_version(header: list[str]) -> str:
    """Return 'v5', 'v4', 'v3' or 'v2' based on the header row."""
    normalized = [h.strip() for h in header]
    if normalized and normalized[0] == "id" and "fa_type" in normalized:
        # Each bump is identified by the one column it introduced: v6 the
        # reference decomposition, v5 the refusal-evidence classification, v4
        # the acceptable-source groups. Newest first, since each schema also
        # carries every column of the one before it.
        if "reference_decomposition" in normalized:
            return "v6"
        if "refusal_evidence" in normalized:
            return "v5"
        return "v4" if "doc_id_groups" in normalized else "v3"
    if normalized and normalized[0] in ("ID", "#") and "Typ" in normalized:
        return "v2"
    raise ValueError(
        f"Unknown CSV schema. Header: {normalized!r}. "
        f"Expected v2 ('ID;Typ;...'), or v3/v4/v5/v6 ('id;fa_type;...')."
    )


def _parse_v2_row(row: dict[str, str]) -> GoldStandardItem | None:
    """Parse one v2 row into a GoldStandardItem (with v3 fields filled by defaults)."""
    raw_id = (row.get("ID") or row.get("#") or "").strip()
    if not raw_id:
        return None
    try:
        item_id = int(raw_id)
    except ValueError:
        return None

    v2_type = row.get("Typ", "").strip()
    fa_type = V2_TYPE_TO_FA_TYPE.get(v2_type, "")
    doc_refs = row.get("Doc(s)", "").strip()
    question = row.get("Query", "").strip()
    ground_truth = (row.get("GT-Wert (manuell)") or row.get("GT-Wert", "")).strip()
    source_section = (row.get("Source (manuell)") or row.get("Source", "")).strip()
    rationale = (row.get("Rationale (H)") or row.get("Rationale", "")).strip()
    entity_form_raw = row.get("entity_form", "").strip()

    return GoldStandardItem(
        id=item_id,
        question=question,
        ground_truth=ground_truth,
        doc_refs=doc_refs,
        query_type=v2_type,
        source_section=source_section,
        rationale=rationale,
        entity_form=entity_form_raw or None,
        fa_type=fa_type,
        expected_answerable=True,
        difficulty="medium",
    )


def _parse_v3_row(row: dict[str, str]) -> GoldStandardItem | None:
    """Parse one v3, v4, v5 or v6 row into a GoldStandardItem.

    Handles all three schemas: every bump only adds columns, so the newer
    fields resolve to their defaults on an older file and no separate parser
    is needed. v2 fields are derived where possible.
    """
    raw_id = row.get("id", "").strip()
    if not raw_id:
        return None
    try:
        item_id = int(raw_id)
    except ValueError:
        return None

    fa_type = row.get("fa_type", "").strip()
    doc_ids_str = row.get("doc_ids", "").strip()
    doc_ids = [d.strip() for d in doc_ids_str.split("|") if d.strip()]
    source_sections_str = row.get("source_sections", "").strip()
    source_sections = [s.strip() for s in source_sections_str.split("|") if s.strip()]
    hypothesis_link_str = row.get("hypothesis_link", "").strip()
    hypothesis_link = [h.strip() for h in hypothesis_link_str.split("|") if h.strip()]

    expected_answerable_raw = row.get("expected_answerable", "true").strip().lower()
    expected_answerable = expected_answerable_raw == "true"

    entity_form_raw = row.get("entity_form", "").strip()
    entity_form: str | None = entity_form_raw if entity_form_raw else None

    # v4 columns; absent on v3 files, where these stay empty and callers fall
    # back to doc_ids / skip the corresponding metric.
    doc_id_groups = [
        [d.strip() for d in group.split("+") if d.strip()]
        for group in row.get("doc_id_groups", "").strip().split("|")
        if group.strip()
    ]

    return GoldStandardItem(
        id=item_id,
        question=row.get("query", "").strip(),
        ground_truth=row.get("gt_value", "").strip(),
        doc_refs=doc_ids_str,
        query_type=FA_TYPE_TO_V2_TYPE.get(fa_type, fa_type),
        source_section=source_sections_str,
        rationale=row.get("rationale", "").strip(),
        entity_form=entity_form,
        fa_type=fa_type,
        subtype=row.get("subtype", "").strip(),
        doc_ids=doc_ids,
        gt_unit=row.get("gt_unit", "").strip(),
        source_sections=source_sections,
        expected_answerable=expected_answerable,
        difficulty=row.get("difficulty", "medium").strip() or "medium",
        math_type=row.get("math_type", "").strip(),
        hypothesis_link=hypothesis_link,
        doc_id_groups=doc_id_groups,
        window_class=row.get("window_class", "").strip(),
        refusal_evidence=row.get("refusal_evidence", "").strip(),
        gt_correction=row.get("gt_correction", "").strip(),
        reference_decomposition=[
            tf.strip()
            for tf in row.get("reference_decomposition", "").split("|")
            if tf.strip()
        ],
    )


def load_gold_standard(
    csv_path: str | Path,
    filter_types: list[str] | None = None,
    filter_subtype: list[str] | None = None,
    filter_answerable: bool | None = None,
) -> list[GoldStandardItem]:
    """Load gold-standard Q&A pairs from a CSV (v2-v5 schema, auto-detected).

    Args:
        csv_path: Path to the gold standard CSV.
        filter_types: Optional list of types to include. Accepts both v2 names
            (``"Single"``, ``"Cross-Sec"``, ``"Multi-Year"``, ``"Multi-Comp"``)
            and v3 IDs (``"FA-1"``..``"FA-4"``, ``"FA-Refusal"``). The two
            naming schemes are matched against both ``query_type`` and
            ``fa_type`` so a v2-style filter still works on a v3 file.
        filter_subtype: Optional list of v3 subtypes to include (e.g.
            ``["cagr", "max_min"]``). Ignored when loading a v2 CSV (no
            subtype column available).
        filter_answerable: If set, restrict to rows whose
            ``expected_answerable`` matches. Ignored when loading a v2 CSV
            (where every row is implicitly answerable).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Gold standard file not found: {csv_path}")

    items: list[GoldStandardItem] = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        if reader.fieldnames is None:
            raise ValueError(f"Empty CSV: {csv_path}")
        schema_version = _detect_schema_version(list(reader.fieldnames))
        logger.debug("Detected schema version: %s", schema_version)

        for row in reader:
            # Membership test, not `!= "v2"`: a new schema version that
            # nobody wired in here would otherwise be routed to the v2 parser,
            # which returns None for every modern row and yields a silently
            # empty dataset instead of an error.
            if schema_version in ("v3", "v4", "v5", "v6"):
                item = _parse_v3_row(row)
            elif schema_version == "v2":
                item = _parse_v2_row(row)
            else:
                raise ValueError(
                    f"Schema version {schema_version!r} detected but no parser "
                    f"is wired for it in load_gold_standard ({csv_path})."
                )
            if item is None:
                continue

            if filter_types:
                if not (
                    item.query_type in filter_types
                    or item.fa_type in filter_types
                ):
                    continue
            if filter_subtype and item.subtype not in filter_subtype:
                continue
            if filter_answerable is not None and item.expected_answerable != filter_answerable:
                continue

            items.append(item)

    logger.info(
        "Loaded %d gold-standard items from %s (schema=%s, type_filter=%s, subtype_filter=%s, answerable=%s)",
        len(items),
        csv_path.name,
        schema_version,
        filter_types,
        filter_subtype,
        filter_answerable,
    )
    return items
