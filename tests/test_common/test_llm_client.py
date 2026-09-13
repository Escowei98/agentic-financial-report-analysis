"""
Unit tests for the unified LLM/embedding client (src/common/llm_client.py).

All systems and the evaluation layer MUST go through this module so that
model configuration stays identical for fair comparison; these tests pin
the parameter-building, validation, and batching logic without making any
real network calls.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.common.llm_client import (
    BatchedVertexAIEmbeddings,
    get_embeddings,
    get_judge_llm,
    get_llm,
)


class TestGetLLM:
    """Tests for get_llm's config resolution and validation."""

    def test_raises_without_project(self):
        with patch("src.common.llm_client.load_config", return_value={}):
            with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
                get_llm()

    def test_uses_config_defaults(self):
        config = {
            "google_cloud_project": "my-project",
            "google_cloud_location": "us-central1",
            "llm": {"model": "gemini-2.5-flash", "temperature": 0.0, "max_output_tokens": 8192},
        }
        with patch("src.common.llm_client.load_config", return_value=config), \
             patch("src.common.llm_client.ChatVertexAI") as mock_chat:
            mock_chat.return_value = MagicMock()
            get_llm("rag_monolith")

            mock_chat.assert_called_once_with(
                model_name="gemini-2.5-flash",
                temperature=0.0,
                max_output_tokens=8192,
                project="my-project",
                location="us-central1",
            )

    def test_kwargs_override_config_defaults(self):
        config = {
            "google_cloud_project": "my-project",
            "llm": {"model": "gemini-2.5-flash", "temperature": 0.0, "max_output_tokens": 8192},
        }
        with patch("src.common.llm_client.load_config", return_value=config), \
             patch("src.common.llm_client.ChatVertexAI") as mock_chat:
            get_llm(temperature=0.7)

            _, kwargs = mock_chat.call_args
            assert kwargs["temperature"] == 0.7

    def test_missing_llm_section_falls_back_to_hardcoded_defaults(self):
        config = {"google_cloud_project": "my-project"}
        with patch("src.common.llm_client.load_config", return_value=config), \
             patch("src.common.llm_client.ChatVertexAI") as mock_chat:
            get_llm()

            _, kwargs = mock_chat.call_args
            assert kwargs["model_name"] == "gemini-2.5-flash"
            assert kwargs["temperature"] == 0.0
            assert kwargs["max_output_tokens"] == 8192
            assert kwargs["location"] == "global"


class TestGetJudgeLLM:
    """Tests for get_judge_llm's provider guard and validation."""

    def test_raises_on_non_openai_provider(self):
        config = {"judge": {"provider": "gemini"}, "openai_api_key": "sk-test"}
        with patch("src.common.llm_client.load_config", return_value=config):
            with pytest.raises(ValueError, match="Unsupported judge provider"):
                get_judge_llm()

    def test_raises_without_api_key(self):
        config = {"judge": {"provider": "openai"}, "openai_api_key": ""}
        with patch("src.common.llm_client.load_config", return_value=config):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                get_judge_llm()

    def test_loads_evaluation_config_specifically(self):
        config = {"judge": {"provider": "openai", "model": "gpt-4o-mini"}, "openai_api_key": "sk-test"}
        with patch("src.common.llm_client.load_config", return_value=config) as mock_load, \
             patch("src.common.llm_client.ChatOpenAI"):
            get_judge_llm()
            mock_load.assert_called_once_with("evaluation")

    def test_builds_chat_openai_with_config_values(self):
        config = {
            "judge": {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.0},
            "openai_api_key": "sk-test",
        }
        with patch("src.common.llm_client.load_config", return_value=config), \
             patch("src.common.llm_client.ChatOpenAI") as mock_chat:
            get_judge_llm()

            mock_chat.assert_called_once_with(
                model="gpt-4o-mini", temperature=0.0, api_key="sk-test",
            )

    def test_kwargs_override_config_defaults(self):
        config = {
            "judge": {"provider": "openai", "model": "gpt-4o-mini", "temperature": 0.0},
            "openai_api_key": "sk-test",
        }
        with patch("src.common.llm_client.load_config", return_value=config), \
             patch("src.common.llm_client.ChatOpenAI") as mock_chat:
            get_judge_llm(max_tokens=2048)

            _, kwargs = mock_chat.call_args
            assert kwargs["max_tokens"] == 2048

    def test_default_provider_is_openai_when_unspecified(self):
        """provider defaults to 'openai' so an empty judge config doesn't crash."""
        config = {"judge": {}, "openai_api_key": "sk-test"}
        with patch("src.common.llm_client.load_config", return_value=config), \
             patch("src.common.llm_client.ChatOpenAI"):
            get_judge_llm()  # must not raise


class TestGetEmbeddings:
    """Tests for get_embeddings's validation and batching wrapper."""

    def test_raises_without_project(self):
        with patch("src.common.llm_client.load_config", return_value={}):
            with pytest.raises(ValueError, match="GOOGLE_CLOUD_PROJECT"):
                get_embeddings()

    def test_returns_batched_wrapper(self):
        config = {"google_cloud_project": "my-project", "embedding": {"model": "gemini-embedding-001"}}
        with patch("src.common.llm_client.load_config", return_value=config), \
             patch("src.common.llm_client.VertexAIEmbeddings") as mock_embed:
            result = get_embeddings()
            assert isinstance(result, BatchedVertexAIEmbeddings)
            mock_embed.assert_called_once_with(
                model="gemini-embedding-001",
                project="my-project",
                location="global",
            )


class TestBatchedVertexAIEmbeddings:
    """Tests for the >250-text batching wrapper around VertexAIEmbeddings."""

    def test_single_batch_under_limit(self):
        inner = MagicMock()
        inner.embed_documents.side_effect = lambda texts: [[0.0] for _ in texts]
        wrapper = BatchedVertexAIEmbeddings(inner)

        result = wrapper.embed_documents(["a", "b", "c"])

        inner.embed_documents.assert_called_once_with(["a", "b", "c"])
        assert len(result) == 3

    def test_splits_into_subbatches_over_limit(self):
        inner = MagicMock()
        inner.embed_documents.side_effect = lambda texts: [[0.0] for _ in texts]
        wrapper = BatchedVertexAIEmbeddings(inner)

        texts = [f"text-{i}" for i in range(600)]
        result = wrapper.embed_documents(texts)

        assert len(result) == 600
        assert inner.embed_documents.call_count == 3  # 250 + 250 + 100
        call_sizes = [len(call.args[0]) for call in inner.embed_documents.call_args_list]
        assert call_sizes == [250, 250, 100]

    def test_exact_multiple_of_batch_size(self):
        inner = MagicMock()
        inner.embed_documents.side_effect = lambda texts: [[0.0] for _ in texts]
        wrapper = BatchedVertexAIEmbeddings(inner)

        texts = [f"text-{i}" for i in range(500)]
        wrapper.embed_documents(texts)

        assert inner.embed_documents.call_count == 2

    def test_empty_input_makes_no_calls(self):
        inner = MagicMock()
        wrapper = BatchedVertexAIEmbeddings(inner)

        result = wrapper.embed_documents([])

        assert result == []
        inner.embed_documents.assert_not_called()

    def test_embed_query_delegates_to_inner(self):
        inner = MagicMock()
        inner.embed_query.return_value = [0.1, 0.2]
        wrapper = BatchedVertexAIEmbeddings(inner)

        result = wrapper.embed_query("hello")

        inner.embed_query.assert_called_once_with("hello")
        assert result == [0.1, 0.2]
