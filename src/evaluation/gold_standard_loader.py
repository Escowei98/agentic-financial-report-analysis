"""
Gold Standard data loader for RAGAS evaluation.

Converts the ablation test dataset CSV into a format compatible
with RAGAS evaluation: question, ground_truth, doc_refs, query_type.
"""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GoldStandardItem:
    """A single gold-standard Q&A pair for evaluation."""

    id: int
    question: str
    ground_truth: str
    query_type: str         # "Single", "Multi-Year", "Cross-Sec"
    doc_refs: str           # e.g., "AAPL_2024", "MSFT_22/24"
    source_section: str     # e.g., "Item 7", "Balance Sheet"
    rationale: str          # e.g., "Basis KPI", "H1 Cross-Doc"
    entity_form: str | None = None  # v2 only: "ticker" or "name"


def load_gold_standard(
    csv_path: str | Path,
    filter_types: list[str] | None = None,
) -> list[GoldStandardItem]:
    """
    Load gold-standard Q&A pairs from the ablation test data CSV.

    CSV format (semicolon-delimited):
      #;Typ;Doc(s);Query;GT-Wert (manuell);Source (manuell);Rationale (H)

    Args:
        csv_path: Path to the ablation_test_data.csv file.
        filter_types: Optional list of query types to include
                      (e.g., ["Single"] for single-doc only).
                      If None, all types are included.

    Returns:
        List of GoldStandardItem objects.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Gold standard file not found: {csv_path}")

    items: list[GoldStandardItem] = []

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")

        # Skip header line (starts with #)
        header = next(reader)
        logger.debug("CSV header: %s", header)

        for row in reader:
            if not row or len(row) < 6:
                continue

            # Parse row: #;Typ;Doc(s);Query;GT-Wert;Source;Rationale
            try:
                item_id = int(row[0].strip())
            except (ValueError, IndexError):
                continue

            query_type = row[1].strip()
            doc_refs = row[2].strip()
            question = row[3].strip()
            ground_truth = row[4].strip()
            source_section = row[5].strip() if len(row) > 5 else ""
            rationale = row[6].strip() if len(row) > 6 else ""
            # v2 gold-standard adds an 'entity_form' column (ticker|name);
            # remains None when loading the original v1 CSV for backward compatibility.
            entity_form = row[7].strip() if len(row) > 7 and row[7].strip() else None

            # Apply type filter
            if filter_types and query_type not in filter_types:
                continue

            items.append(GoldStandardItem(
                id=item_id,
                question=question,
                ground_truth=ground_truth,
                query_type=query_type,
                doc_refs=doc_refs,
                source_section=source_section,
                rationale=rationale,
                entity_form=entity_form,
            ))

    logger.info(
        "Loaded %d gold-standard items from %s (filter=%s)",
        len(items), csv_path.name, filter_types,
    )
    return items


def gold_standard_to_ragas_dataset(
    items: list[GoldStandardItem],
    answers: list[str],
    contexts: list[list[str]],
) -> dict:
    """
    Convert gold standard items + RAG results to RAGAS evaluation format.

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
