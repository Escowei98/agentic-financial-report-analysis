"""
Unified LLM and Embedding client.

Ensures ALL 4 systems use identical model configurations
for fair comparison. This is the ONLY place where LLM/embedding
models are instantiated.
"""

import logging

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.common.config import load_config

logger = logging.getLogger(__name__)


def get_llm(system_name: str | None = None, **kwargs) -> ChatGoogleGenerativeAI:
    """
    Get a configured Gemini LLM instance.

    All systems MUST use this function to ensure identical model settings.

    Args:
        system_name: Optional system config override.
        **kwargs: Additional overrides (e.g., temperature).

    Returns:
        Configured ChatGoogleGenerativeAI instance.
    """
    config = load_config(system_name)
    llm_config = config.get("llm", {})

    params = {
        "model": llm_config.get("model", "gemini-2.0-flash"),
        "temperature": llm_config.get("temperature", 0.0),
        "max_output_tokens": llm_config.get("max_output_tokens", 8192),
        "google_api_key": config.get("google_api_key"),
    }
    params.update(kwargs)

    if not params["google_api_key"]:
        raise ValueError(
            "GOOGLE_API_KEY not set. Add it to your .env file."
        )

    llm = ChatGoogleGenerativeAI(**params)
    logger.info(
        "LLM initialized: model=%s, temperature=%s",
        params["model"],
        params["temperature"],
    )
    return llm


def get_embeddings(system_name: str | None = None) -> GoogleGenerativeAIEmbeddings:
    """
    Get a configured embedding model instance.

    Used by System 1 (Monolith RAG) and System 2 (Agent RAG)
    for vector store indexing. System 3 & 4 do NOT use embeddings.

    Args:
        system_name: Optional system config override.

    Returns:
        Configured GoogleGenerativeAIEmbeddings instance.
    """
    config = load_config(system_name)
    embedding_config = config.get("embedding", {})

    model = embedding_config.get("model", "models/text-embedding-004")
    # Ensure model name has the correct prefix
    if not model.startswith("models/"):
        model = f"models/{model}"

    embeddings = GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=config.get("google_api_key"),
    )
    logger.info("Embeddings initialized: model=%s", model)
    return embeddings
