"""Guards the single source of truth for the gold-standard path.

If every script held its own literal path, a dataset update could leave one
of them silently loading the previous version: nothing would fail loudly, the
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
# must never spell out a gold-standard filename.
ACTIVE_SCRIPTS = [
    "run_full_eval.py",
    "run_judge_validation_batch.py",
    "generate_report.py",
]

_GS_LITERAL = re.compile(r"gold_standard_v\d[\w]*\.csv")
_HISTORICAL_LOOKBACK = 10


def _script_paths() -> list[Path]:
    return sorted(p for p in SCRIPTS_DIR.glob("*.py") if p.name != "__init__.py")


class TestCanonicalPathConstant:
    def test_constants_point_at_an_existing_file(self):
        assert GOLD_STANDARD_EN.is_file(), GOLD_STANDARD_EN
        assert GOLD_STANDARD_DE.is_file(), GOLD_STANDARD_DE

    def test_constants_are_the_same_version(self):
        en = GOLD_STANDARD_EN.name.replace("_en.csv", "")
        assert GOLD_STANDARD_DE.name.replace(".csv", "") == en

    def test_the_canonical_file_carries_the_current_schema(self):
        """Every column a live evaluator reads must be present.

        The point of this assertion is not the version number but the
        columns: a constant pointed at an older file would load and run,
        just silently without `reference_decomposition`, and Dimension 3
        would score every chain against an empty requirement list.
        """
        header = GOLD_STANDARD_EN.read_text(encoding="utf-8").splitlines()[0]
        for column in ("doc_id_groups", "window_class", "refusal_evidence",
                       "gt_correction", "reference_decomposition"):
            assert column in header, f"canonical file lacks {column!r}"


class TestActiveScriptsUseTheConstant:
    @pytest.mark.parametrize("name", ACTIVE_SCRIPTS)
    def test_no_hardcoded_gold_standard_filename(self, name):
        text = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
        found = _GS_LITERAL.findall(text)
        assert not found, (
            f"{name} spells out {found} instead of importing GOLD_STANDARD_EN. "
            f"That is exactly how the v3/v4 drift happened."
        )

    @pytest.mark.parametrize("name", ACTIVE_SCRIPTS)
    def test_imports_the_constant(self, name):
        text = (SCRIPTS_DIR / name).read_text(encoding="utf-8")
        assert "GOLD_STANDARD_EN" in text


class TestHistoricalScriptsDeclareThemselves:
    """A script may pin an older version — rescoring answers generated for v3
    questions against v4 questions would be meaningless — but it has to say so
    at the line that does it, so the next reader can tell a deliberate pin
    from a forgotten one.
    """

    def test_every_version_literal_is_marked_historical(self):
        offenders = []
        for path in _script_paths():
            lines = path.read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(lines):
                if not _GS_LITERAL.search(line):
                    continue
                window = lines[max(0, i - _HISTORICAL_LOOKBACK):i]
                if not any("HISTORICAL" in w for w in window):
                    offenders.append(f"{path.name}:{i + 1}: {line.strip()}")
        assert not offenders, (
            "gold-standard version literals without a HISTORICAL marker above "
            "them:\n  " + "\n  ".join(offenders)
        )
