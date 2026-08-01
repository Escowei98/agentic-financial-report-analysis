"""
Unified LLM and Embedding client.

Ensures ALL 4 systems use identical model configurations
for fair comparison. This is the ONLY place where LLM/embedding
models are instantiated.

Authentication: Uses Google Cloud Application Default Credentials (ADC).
Run `gcloud auth application-default login` before first use.
"""

import logging

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_openai import ChatOpenAI

from src.common.config import load_config

logger = logging.getLogger(__name__)

# Vertex AI allows at most 250 texts per embed request.
_EMBED_MAX_BATCH_SIZE = 250


class BatchedVertexAIEmbeddings(Embeddings):
    """Wrapper that splits embed_documents into sub-batches of <= 250."""

    def __init__(self, inner: VertexAIEmbeddings) -> None:
        self._inner = inner

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_MAX_BATCH_SIZE):
            batch = texts[i : i + _EMBED_MAX_BATCH_SIZE]
            logger.debug(
                "Embedding batch %d-%d / %d",
                i, i + len(batch), len(texts),
            )
            all_embeddings.extend(self._inner.embed_documents(batch))
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)


def get_llm(system_name: str | None = None, **kwargs) -> ChatVertexAI:
    """
    Get a configured Gemini LLM instance via Vertex AI.

    All systems MUST use this function to ensure identical model settings.

    Args:
        system_name: Optional system config override.
        **kwargs: Additional overrides (e.g., temperature).

    Returns:
        Configured ChatVertexAI instance.
    """
    config = load_config(system_name)
    llm_config = config.get("llm", {})

    project = config.get("google_cloud_project")
    location = config.get("google_cloud_location", "global")

    if not project:
        raise ValueError(
            "GOOGLE_CLOUD_PROJECT not set. Add it to your .env file."
        )

    params = {
        "model_name": llm_config.get("model", "gemini-2.5-flash"),
        "temperature": llm_config.get("temperature", 0.0),
        "max_output_tokens": llm_config.get("max_output_tokens", 8192),
        "project": project,
        "location": location,
    }
    params.update(kwargs)

    llm = ChatVertexAI(**params)
    logger.info(
        "LLM initialized: model=%s, temperature=%s, project=%s, location=%s",
        params["model_name"],
        params["temperature"],
        project,
        location,
    )
    return llm


def get_judge_llm(**kwargs) -> BaseChatModel:
    """
    Get the shared LLM-as-a-judge instance used by ALL evaluation code
    (RAGAS, custom metrics, reasoning evaluator).

    Deliberately backed by a different provider (OpenAI) than the Gemini
    models the four systems under test use for generation, to avoid
    same-model self-preference bias in LLM-as-a-judge scoring.

    Args:
        **kwargs: Overrides for ChatOpenAI (e.g., max_output_tokens).

    Returns:
        Configured ChatOpenAI instance.
    """
    config = load_config()
    judge_config = config.get("evaluation", {}).get("judge", {})

    api_key = config.get("openai_api_key")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not set. Add it to your .env file."
        )

    params = {
        "model": judge_config.get("model", "gpt-4o-mini"),
        "temperature": judge_config.get("temperature", 0.0),
        "api_key": api_key,
    }
    params.update(kwargs)

    llm = ChatOpenAI(**params)
    logger.info("Judge LLM initialized: model=%s, temperature=%s", params["model"], params["temperature"])
    return llm


def get_embeddings(system_name: str | None = None) -> Embeddings:
    """
    Get a configured embedding model instance via Vertex AI.

    Used by System 1 (Monolith RAG) and System 2 (Agent RAG)
    for vector store indexing. System 3 & 4 do NOT use embeddings.

    Automatically batches requests to stay within the Vertex AI
    API limit of 250 texts per request.

    Args:
        system_name: Optional system config override.

    Returns:
        Configured embedding instance with automatic batching.
    """
    config = load_config(system_name)
    embedding_config = config.get("embedding", {})

    project = config.get("google_cloud_project")
    location = config.get("google_cloud_location", "global")

    if not project:
        raise ValueError(
            "GOOGLE_CLOUD_PROJECT not set. Add it to your .env file."
        )

    model = embedding_config.get("model", "gemini-embedding-001")

    inner = VertexAIEmbeddings(
        model_name=model,
        project=project,
        location=location,
    )
    logger.info(
        "Embeddings initialized: model=%s, project=%s, location=%s",
        model, project, location,
    )
    return BatchedVertexAIEmbeddings(inner)
