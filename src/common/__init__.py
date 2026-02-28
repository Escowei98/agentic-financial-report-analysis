"""Shared utilities used by ALL 4 RAG systems for fair comparison."""

from src.common.config import load_config
from src.common.ingestion import (
    ProcessedFiling,
    download_all_filings,
    download_filing,
    load_processed_filing,
)
from src.common.llm_client import get_embeddings, get_llm
from src.common.utils import CostTracker, RunMetrics, setup_logging, timer

__all__ = [
    "load_config",
    "download_filing",
    "download_all_filings",
    "load_processed_filing",
    "ProcessedFiling",
    "get_llm",
    "get_embeddings",
    "CostTracker",
    "RunMetrics",
    "setup_logging",
    "timer",
]
