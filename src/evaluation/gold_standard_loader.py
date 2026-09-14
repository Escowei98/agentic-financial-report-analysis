"""
Gold standard loader.

Two CSV schemas are read, both semicolon-delimited:
  - the evaluation gold standard (data/gold_standard/gold_standard*.csv):
    150 items in five strata (FA-1..FA-4 + FA-Refusal) with canonical source
    ids, acceptable-source groups (`doc_id_groups`), the FA-3 document
    window class, the refusal evidence class with its reference correction,
    and the reference decomposition the completeness dimension scores
    against;
  - the ablation set (notebooks/experiments/sys1_rag_monolith/
    ablation_test_data.csv): 8 columns, free-text source field, four query
    types. Used only for the S1 hyperparameter search.

The schema is detected from the header. Fields one schema lacks, and
optional evaluation columns a file does not carry, are filled with defaults
so consumers can rely on the unified GoldStandardItem.

Deliberately not part of the schema: an `expected_tools` column. Tool
selection is out of scope of the evaluation.
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
# Import these instead of spelling the filename out, so that every script
# runs against the same dataset (guarded by
# tests/test_evaluation/test_gold_standard_path_discipline.py).
GOLD_STANDARD_EN = _GS_DIR / "gold_standard_en.csv"
"""English gold standard — what every system and evaluator runs against."""

GOLD_STANDARD_DE = _GS_DIR / "gold_standard.csv"
"""German source of record. Hand-maintained; the English file is patched
cell-wise to match (see data/gold_standard/gold_standard_README.md,
"Correction convention")."""


# Mapping between the ablation set's query_type freetext and fa_type IDs.
ABLATION_TYPE_TO_FA_TYPE = {
    "Single": "FA-1",
    "Cross-Sec": "FA-2",
    "Multi-Year": "FA-3",
    "Multi-Comp": "FA-4",
}
FA_TYPE_TO_ABLATION_TYPE = {v: k for k, v in ABLATION_TYPE_TO_FA_TYPE.items()}


@dataclass
class GoldStandardItem:
    """A single gold-standard Q&A pair for evaluation.

    Unified across the ablation and evaluation schemas. Ablation-only fields
    (`query_type`, `source_section`) and evaluation-only fields (`fa_type`,
    `subtype`, ...)
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
    math_type: str = ""
    hypothesis_link: list[str] = field(default_factory=list)

    # --- Citation sources and FA-3 window ---
    doc_id_groups: list[list[str]] = field(default_factory=list)
    """Acceptable citation sources, grouped by the fact each one supports.

    A 10-K reports prior years in comparative columns, so one fiscal year is
    often carried by more than one filing. Each inner list holds the filings
    that are interchangeable for one required fact; citing ANY member of a
    group satisfies it. `doc_ids` stays the flat union for the judge prompt.
    Empty on the ablation schema, where callers fall back to `doc_ids`.
    """

    window_class: str = ""
    """FA-3 only: 'within_window' or 'cross_window'.

    Whether the item's year span fits inside a single filing's comparative
    columns. Splits the multi-year stratum so the effect of the document
    boundary can be read off directly instead of being inferred across
    strata.
    """

    # --- Refusal stratum ---
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
    violation this stratum is built to detect. Empty on answerable items.
    """

    # --- Reasoning completeness and refusal reference ---
    reference_decomposition: list[str] = field(default_factory=list)
    """Answerable items only: the sub-questions a complete chain must cover.

    The standard Dimension 3 of the reasoning metric scores against. It
    lives in the dataset rather
    than in the judge prompt because a judge left to infer the required
    sub-questions per item invents its own standard each time -- and so do
    human raters, which is what makes completeness-style dimensions
    disagree.

    Empty on FA-Refusal items: the correct chain there is "the premise
    cannot be checked", which has no decomposition.
    """

    gt_correction: str = ""
    """FA-Refusal only: reference text for the refusal_quality judge.

    NOT a ground truth answer -- `gt_value` stays empty for every FA-Refusal
    row, which remains the canonical signature of the stratum. This field
    carries what a system could legitimately add beyond a bare refusal:
    the scope boundary, the contradicting figure, or the missing
    specification. Empty on answerable items.
    """


def _detect_schema(header: list[str]) -> str:
    """Return 'evaluation' or 'ablation' based on the header row."""
    normalized = [h.strip() for h in header]
    if normalized and normalized[0] == "id" and "fa_type" in normalized:
        return "evaluation"
    if normalized and normalized[0] in ("ID", "#") and "Typ" in normalized:
        return "ablation"
    raise ValueError(
        f"Unknown CSV schema. Header: {normalized!r}. "
        f"Expected the ablation schema ('ID;Typ;...') or the evaluation "
        f"schema ('id;fa_type;...')."
    )


def _parse_ablation_row(row: dict[str, str]) -> GoldStandardItem | None:
    """Parse one ablation-set row (evaluation fields filled by defaults)."""
    raw_id = (row.get("ID") or row.get("#") or "").strip()
    if not raw_id:
        return None
    try:
        item_id = int(raw_id)
    except ValueError:
        return None

    ablation_type = row.get("Typ", "").strip()
    fa_type = ABLATION_TYPE_TO_FA_TYPE.get(ablation_type, "")
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
        query_type=ablation_type,
        source_section=source_section,
        rationale=rationale,
        entity_form=entity_form_raw or None,
        fa_type=fa_type,
        expected_answerable=True,
    )


def _parse_evaluation_row(row: dict[str, str]) -> GoldStandardItem | None:
    """Parse one evaluation-schema row into a GoldStandardItem.

    Optional columns a file does not carry resolve to their defaults.
    Ablation fields are derived where possible.
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

    # Acceptable-source groups and window class; empty on the ablation
    # schema, where callers fall back to doc_ids / skip the metric.
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
        query_type=FA_TYPE_TO_ABLATION_TYPE.get(fa_type, fa_type),
        source_section=source_sections_str,
        rationale=row.get("rationale", "").strip(),
        entity_form=entity_form,
        fa_type=fa_type,
        subtype=row.get("subtype", "").strip(),
        doc_ids=doc_ids,
        gt_unit=row.get("gt_unit", "").strip(),
        source_sections=source_sections,
        expected_answerable=expected_answerable,
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
    """Load gold-standard Q&A pairs from a CSV (schema auto-detected).

    Args:
        csv_path: Path to the gold standard CSV.
        filter_types: Optional list of types to include. Accepts both ablation names
            (``"Single"``, ``"Cross-Sec"``, ``"Multi-Year"``, ``"Multi-Comp"``)
            and evaluation IDs (``"FA-1"``..``"FA-4"``, ``"FA-Refusal"``). The
            two naming schemes are matched against both ``query_type`` and
            ``fa_type`` so an ablation-style filter still works on the
            evaluation file.
        filter_subtype: Optional list of subtypes to include (e.g.
            ``["cagr", "max_min"]``). Ignored when loading the ablation CSV (no
            subtype column available).
        filter_answerable: If set, restrict to rows whose
            ``expected_answerable`` matches. Ignored when loading the ablation CSV
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
        schema = _detect_schema(list(reader.fieldnames))
        logger.debug("Detected schema: %s", schema)
        parse_row = _parse_evaluation_row if schema == "evaluation" else _parse_ablation_row

        for row in reader:
            item = parse_row(row)
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
        schema,
        filter_types,
        filter_subtype,
        filter_answerable,
    )
    return items
