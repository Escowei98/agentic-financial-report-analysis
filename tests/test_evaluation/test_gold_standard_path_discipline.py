"""Guards the single source of truth for the gold-standard path.

If every script held its own literal path, a renamed or replaced dataset could
leave one of them silently loading a stale file: nothing would fail loudly, the
run would simply answer a different dataset. These tests make that failure
mode visible the moment it is introduced.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.evaluation.gold_standard_loader import GOLD_STANDARD_DE, GOLD_STANDARD_EN

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Scripts that drive a NEW evaluation run or read the current dataset. These
# must import the constant.
ACTIVE_SCRIPTS = [
    "run_full_eval.py",
    "run_judge_validation_batch.py",
    "generate_report.py",
]

_GS_LITERAL = re.compile(r"gold_standard[\w]*\.csv")


def _script_paths() -> list[Path]:
    return sorted(p for p in SCRIPTS_DIR.glob("*.py") if p.name != "__init__.py")


class TestCanonicalPathConstant:
    def test_constants_point_at_an_existing_file(self):
        assert GOLD_STANDARD_EN.is_file(), GOLD_STANDARD_EN
        assert GOLD_STANDARD_DE.is_file(), GOLD_STANDARD_DE

    def test_constants_name_the_same_dataset(self):
        en = GOLD_STANDARD_EN.name.replace("_en.csv", "")
        assert GOLD_STANDARD_DE.name.replace(".csv", "") == en

    def test_the_canonical_file_carries_the_current_schema(self):
        """Every column a live evaluator reads must be present.

        A constant pointed at a file lacking these columns would load and run,
        just silently without `reference_decomposition`, and Dimension 3
        would score every chain against an empty requirement list.
        """
        header = GOLD_STANDARD_EN.read_text(encoding="utf-8").splitlines()[0]
        for column in ("doc_id_groups", "window_class", "refusal_evidence",
                       "gt_correction", "reference_decomposition"):
            assert column in header, f"canonical file lacks {column!r}"


class TestScriptsUseTheConstant:
    @pytest.mark.parametrize("path", _script_paths(), ids=lambda p: p.name)
    def test_no_hardcoded_gold_standard_filename(self, path):
        found = _GS_LITERAL.findall(path.read_text(encoding="utf-8"))
        assert not found, (
            f"{path.name} spells out {found} instead of importing "
            f"GOLD_STANDARD_EN / GOLD_STANDARD_DE."
        )

    @pytest.mark.parametrize("name", ACTIVE_SCRIPTS)
    def test_imports_the_constant(self, name):
        text = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
        assert "GOLD_STANDARD_EN" in text
