"""
Unit tests for centralized configuration loading (src/common/config.py).
"""

import pytest

from src.common.config import _deep_merge, load_config


class TestDeepMerge:
    """Tests for the recursive config-merging helper."""

    def test_override_replaces_scalar(self):
        base = {"a": 1, "b": 2}
        override = {"a": 99}
        assert _deep_merge(base, override) == {"a": 99, "b": 2}

    def test_override_merges_nested_dicts(self):
        base = {"chunking": {"chunk_size": 1000, "chunk_overlap_pct": 0.1}}
        override = {"chunking": {"chunk_size": 500}}
        result = _deep_merge(base, override)
        assert result == {"chunking": {"chunk_size": 500, "chunk_overlap_pct": 0.1}}

    def test_override_adds_new_keys(self):
        base = {"a": 1}
        override = {"b": 2}
        assert _deep_merge(base, override) == {"a": 1, "b": 2}

    def test_override_replaces_dict_with_scalar(self):
        """If override provides a non-dict for a dict key, it wins outright."""
        base = {"reranker": {"enabled": True}}
        override = {"reranker": "disabled"}
        assert _deep_merge(base, override) == {"reranker": "disabled"}

    def test_does_not_mutate_base(self):
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}
        _deep_merge(base, override)
        assert base == {"a": {"x": 1}}


class TestLoadConfig:
    """Tests for load_config's file-merging and env-injection behaviour."""

    @pytest.fixture
    def configs_dir(self, tmp_path, monkeypatch):
        configs_dir = tmp_path / "configs"
        configs_dir.mkdir()
        monkeypatch.setattr("src.common.config.CONFIGS_DIR", configs_dir)
        return configs_dir

    def test_missing_base_config_raises(self, configs_dir):
        with pytest.raises(FileNotFoundError, match="Base config not found"):
            load_config()

    def test_loads_base_config_only(self, configs_dir):
        (configs_dir / "base.yaml").write_text("llm:\n  model: gemini-2.5-flash\n")
        config = load_config()
        assert config["llm"]["model"] == "gemini-2.5-flash"

    def test_system_config_overrides_base(self, configs_dir):
        (configs_dir / "base.yaml").write_text(
            "llm:\n  model: gemini-2.5-flash\n  temperature: 0.0\n"
        )
        (configs_dir / "rag_monolith.yaml").write_text(
            "llm:\n  temperature: 0.5\n"
        )
        config = load_config("rag_monolith")
        assert config["llm"]["model"] == "gemini-2.5-flash"
        assert config["llm"]["temperature"] == 0.5

    def test_missing_system_config_is_silently_ignored(self, configs_dir):
        (configs_dir / "base.yaml").write_text("llm:\n  model: gemini-2.5-flash\n")
        config = load_config("nonexistent_system")
        assert config["llm"]["model"] == "gemini-2.5-flash"

    def test_empty_yaml_treated_as_empty_dict(self, configs_dir, monkeypatch):
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        (configs_dir / "base.yaml").write_text("")
        config = load_config()
        assert config["google_cloud_project"] == ""

    def test_env_vars_injected(self, configs_dir, monkeypatch):
        (configs_dir / "base.yaml").write_text("llm:\n  model: gemini-2.5-flash\n")
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
        monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        monkeypatch.setenv("SEC_EDGAR_EMAIL", "test@example.com")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

        config = load_config()

        assert config["google_cloud_project"] == "my-project"
        assert config["google_cloud_location"] == "us-central1"
        assert config["sec_edgar_email"] == "test@example.com"
        assert config["openai_api_key"] == "sk-test"

    def test_env_vars_default_to_empty_or_global(self, configs_dir, monkeypatch):
        (configs_dir / "base.yaml").write_text("llm:\n  model: gemini-2.5-flash\n")
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)

        config = load_config()

        assert config["google_cloud_project"] == ""
        assert config["google_cloud_location"] == "global"

    def test_env_vars_override_yaml_values(self, configs_dir, monkeypatch):
        """Env vars are injected after file merging, so they always win."""
        (configs_dir / "base.yaml").write_text(
            "google_cloud_project: yaml-project\n"
        )
        monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")

        config = load_config()

        assert config["google_cloud_project"] == "env-project"
