"""
Gold Standard data loader for RAGAS evaluation.

Supports both schemas:
  - v2 (frozen): notebooks/experiments/sys1_rag_monolith/ablation_test_data_v2.csv
                 8 columns, semicolon-delimited; freetext source field;
                 4 fragetypes (Single/Cross-Sec/Multi-Year/Multi-Comp).
  - v3 (current): data/gold_standard/gold_standard_v3.csv
                  14 columns, semicolon-delimited; canonical source IDs;
                  5 fragetypes (FA-1..FA-4 + FA-Refusal).

Schema is auto-detected from the CSV header. v2 fields without a v3
counterpart stay as None on v3 entries; v3 fields without a v2 counterpart
are filled with sensible defaults when loading v2 (so consumers can rely
on the unified GoldStandardItem dataclass either way).

Converts the CSV into a format compatible with RAGAS evaluation:
question, ground_truth, doc_refs, query_type.
"""

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


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


def _detect_schema_version(header: list[str]) -> str:
    """Return 'v3' if header matches v3 schema; 'v2' otherwise."""
    normalized = [h.strip() for h in header]
    if normalized and normalized[0] == "id" and "fa_type" in normalized:
        return "v3"
    if normalized and normalized[0] in ("ID", "#") and "Typ" in normalized:
        return "v2"
    raise ValueError(
        f"Unknown CSV schema. Header: {normalized!r}. "
        f"Expected v2 ('ID;Typ;...') or v3 ('id;fa_type;...')."
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
    """Parse one v3 row into a GoldStandardItem (v2 fields are derived where possible)."""
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
    )


def load_gold_standard(
    csv_path: str | Path,
    filter_types: list[str] | None = None,
    filter_subtype: list[str] | None = None,
    filter_answerable: bool | None = None,
) -> list[GoldStandardItem]:
    """Load gold-standard Q&A pairs from a CSV (v2 or v3 schema, auto-detected).

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
            item = (
                _parse_v3_row(row)
                if schema_version == "v3"
                else _parse_v2_row(row)
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


def gold_standard_to_ragas_dataset(
    items: list[GoldStandardItem],
    answers: list[str],
    contexts: list[list[str]],
) -> dict:
    """Convert gold standard items + RAG results to RAGAS evaluation format.

    RAGAS expects a dict with keys: question, ground_truth, answer, contexts.

    Args:
        items: Gold standard Q&A pairs.
        answers: RAG-generated answers (one per item).
        contexts: Retrieved contexts per query (list of lists).

    Returns:
        Dict compatible with RAGAS evaluate().
    """
    if len(items) != len(answers) or len(items) != len(contexts):
        raise ValueError(
            f"Length mismatch: {len(items)} items, {len(answers)} answers, "
            f"{len(contexts)} contexts"
        )

    return {
        "question": [item.question for item in items],
        "ground_truth": [item.ground_truth for item in items],
        "answer": answers,
        "contexts": contexts,
    }
