"""
Unit tests for the gold standard loader.
"""

from pathlib import Path

import pytest

from src.evaluation.gold_standard_loader import (
    GoldStandardItem,
    gold_standard_to_ragas_dataset,
    load_gold_standard,
)

# Sample CSV content matching the actual ablation_test_data.csv format
SAMPLE_CSV = """\
#;Typ;Doc(s);Query;GT-Wert (manuell);Source (manuell);Rationale (H)
1;Single;AAPL_2024;Was war der Gesamtumsatz (Total Revenue) FY2024?;$391 Mrd;Item 7;Basis KPI
2;Single;MSFT_2024;Total Revenue FY2024?;$245,1 Mrd;Item 7;-
3;Multi-Year;AAPL_22/24;YoY-Wachstum Umsatz 2024 vs. 2022?;-0,80%;Item 7 beide;H1 Cross-Doc
4;Cross-Sec;AAPL_2024;Operating Margin FY2024?;31,50%;MD&A;Synthese H2
"""


class TestLoadGoldStandard:
    """Tests for load_gold_standard()."""

    @pytest.fixture
    def csv_file(self, tmp_path) -> Path:
        """Create a temporary CSV file with sample data."""
        csv_path = tmp_path / "test_gold_standard.csv"
        csv_path.write_text(SAMPLE_CSV, encoding="utf-8")
        return csv_path

    def test_loads_all_items(self, csv_file):
        """Should load all 4 items from sample CSV."""
        items = load_gold_standard(csv_file)
        assert len(items) == 4

    def test_item_fields(self, csv_file):
        """Should correctly parse all fields."""
        items = load_gold_standard(csv_file)
        first = items[0]

        assert first.id == 1
        assert first.question == "Was war der Gesamtumsatz (Total Revenue) FY2024?"
        assert first.ground_truth == "$391 Mrd"
        assert first.query_type == "Single"
        assert first.doc_refs == "AAPL_2024"
        assert first.source_section == "Item 7"
        assert first.rationale == "Basis KPI"

    def test_filter_single_only(self, csv_file):
        """Filter should return only 'Single' type queries."""
        items = load_gold_standard(csv_file, filter_types=["Single"])
        assert len(items) == 2
        assert all(item.query_type == "Single" for item in items)

    def test_filter_multi_year(self, csv_file):
        """Filter should return only 'Multi-Year' type queries."""
        items = load_gold_standard(csv_file, filter_types=["Multi-Year"])
        assert len(items) == 1
        assert items[0].query_type == "Multi-Year"

    def test_filter_multiple_types(self, csv_file):
        """Filter with multiple types should include all matching."""
        items = load_gold_standard(csv_file, filter_types=["Single", "Cross-Sec"])
        assert len(items) == 3

    def test_file_not_found(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_gold_standard("/nonexistent/path.csv")


class TestGoldStandardToRagasDataset:
    """Tests for gold_standard_to_ragas_dataset()."""

    def test_conversion(self):
        """Should convert to RAGAS format correctly."""
        items = [
            GoldStandardItem(
                id=1, question="Q1?", ground_truth="A1",
                query_type="Single", doc_refs="AAPL_2024",
                source_section="Item 7", rationale="",
            ),
        ]
        result = gold_standard_to_ragas_dataset(
            items=items,
            answers=["Generated A1"],
            contexts=[["Context chunk 1", "Context chunk 2"]],
        )

        assert result["question"] == ["Q1?"]
        assert result["ground_truth"] == ["A1"]
        assert result["answer"] == ["Generated A1"]
        assert result["contexts"] == [["Context chunk 1", "Context chunk 2"]]

    def test_length_mismatch(self):
        """Should raise ValueError on mismatched lengths."""
        items = [
            GoldStandardItem(
                id=1, question="Q?", ground_truth="A",
                query_type="Single", doc_refs="X",
                source_section="", rationale="",
            ),
        ]
        with pytest.raises(ValueError, match="Length mismatch"):
            gold_standard_to_ragas_dataset(items, answers=[], contexts=[["ctx"]])
