"""
Unit tests for the gold standard loader.
"""

from pathlib import Path

import pytest

from src.evaluation.gold_standard_loader import (
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


# Evaluation schema with doc_id_groups / window_class. Row 3 exercises
# the case these columns exist for: FY2020 is reported by two filings (its own
# and the FY2022 one's comparative columns), FY2024 by only one — so citing
# either member of the first group is correct.
SAMPLE_CSV_GROUPS = """\
id;fa_type;subtype;doc_ids;doc_id_groups;query;gt_value;gt_unit;source_sections;expected_answerable;math_type;entity_form;hypothesis_link;rationale;window_class
1;FA-1;basis_kpi;AAPL_2024;AAPL_2024;Total revenue FY2024?;$391.0 billion;usd_billion;item_8_income_stmt;true;n/a;ticker;H1;Basis KPI;
2;FA-Refusal;not_in_corpus;AAPL_2024;AAPL_2024;Revenue in FY2019?;;n/a;;false;n/a;name;;Out of scope;
3;FA-3;growth_yoy;AAPL_2020|AAPL_2022|AAPL_2024;AAPL_2020+AAPL_2022|AAPL_2024;Revenue growth FY2020 to FY2024?;+42.4%;percent;item_8_income_stmt;true;yoy;name;H2|H3;Cross window;cross_window
"""


class TestLoadGoldStandardGroups:
    """Acceptable-source groups and window class."""

    @pytest.fixture
    def csv_groups(self, tmp_path) -> Path:
        csv_path = tmp_path / "gold_standard.csv"
        csv_path.write_text(SAMPLE_CSV_GROUPS, encoding="utf-8")
        return csv_path

    def test_parses_doc_id_groups(self, csv_groups):
        item = load_gold_standard(csv_groups)[2]
        assert item.doc_id_groups == [["AAPL_2020", "AAPL_2022"], ["AAPL_2024"]]
        # The flat union stays available for consumers that ignore groups.
        assert item.doc_ids == ["AAPL_2020", "AAPL_2022", "AAPL_2024"]

    def test_parses_window_class(self, csv_groups):
        items = load_gold_standard(csv_groups)
        assert items[2].window_class == "cross_window"
        assert items[0].window_class == ""

    def test_file_without_optional_columns_loads_with_defaults(self, tmp_path):
        """Optional columns missing from a file default to empty."""
        minimal = tmp_path / "minimal.csv"
        minimal.write_text(
            "id;fa_type;subtype;doc_ids;query;gt_value;gt_unit;source_sections;"
            "expected_answerable;math_type;entity_form;hypothesis_link;rationale\n"
            "1;FA-1;basis_kpi;AAPL_2024;Total revenue FY2024?;$391.0 billion;usd_billion;"
            "item_8_income_stmt;true;n/a;ticker;H1;Basis KPI\n",
            encoding="utf-8",
        )
        item = load_gold_standard(minimal)[0]
        assert item.doc_ids == ["AAPL_2024"]
        assert item.doc_id_groups == []
        assert item.window_class == ""

    def test_filters_still_apply(self, csv_groups):
        assert len(load_gold_standard(csv_groups, filter_types=["FA-3"])) == 1
        assert len(load_gold_standard(csv_groups, filter_answerable=False)) == 1

    def test_no_expected_tools_field(self):
        """There is no tool-selection column.

        A shared expected tool set would penalise S3/S4 for tools they cannot
        have, so tool selection is out of scope. The data-side guard lives in
        scripts/validate_gold_standard.py.
        """
        from dataclasses import fields

        from src.evaluation.gold_standard_loader import GoldStandardItem

        assert "expected_tools" not in {f.name for f in fields(GoldStandardItem)}
