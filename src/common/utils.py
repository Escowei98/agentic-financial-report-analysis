"""
Shared utilities for cost tracking, timing, logging, and caching.

Used by ALL 4 systems for consistent measurement and fair comparison.
"""

from __future__ import annotations

import hashlib
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence


@dataclass
class TokenUsage:
    """Tracks token usage for a single LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def extract_text(content: str | list) -> str:
    """Extracts text from LLM message content, handling both strings and lists of dicts."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return " ".join(parts)
    return str(content)


@dataclass
class RunMetrics:
    """
    Collects metrics for a single query run.

    Used by the evaluation pipeline to compute efficiency metrics.
    """

    query: str = ""
    system_name: str = ""
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    latency_seconds: float = 0.0
    num_steps: int = 0
    tool_calls: list[str] = field(default_factory=list)
    corrections: int = 0  # Correction rounds triggered by reflection (Sys2) or reviewer (Sys4)

    # Optional S4 (None for S1-S3)
    num_specialists_invoked: int | None = None
    token_breakdown: dict[str, int] | None = None  # role -> total_tokens

    @property
    def estimated_cost_usd(self) -> float:
        """
        Estimate cost based on Gemini 2.5 Flash pricing (standard tier).

        Pricing (as of 2026-04, Vertex AI standard tier):
        - Input: $0.30 / 1M tokens
        - Output: $2.50 / 1M tokens

        Matches the model configured in configs/base.yaml (gemini-2.5-flash).
        Update both this constant and base.yaml when changing the model.
        """
        input_cost = self.token_usage.prompt_tokens * 0.30 / 1_000_000
        output_cost = self.token_usage.completion_tokens * 2.50 / 1_000_000
        return input_cost + output_cost


class CostTracker:
    """
    Tracks cumulative costs and token usage across multiple queries.

    Usage:
        tracker = CostTracker("rag_monolith")
        tracker.record(run_metrics)
        print(tracker.summary())
    """

    def __init__(self, system_name: str):
        self.system_name = system_name
        self.runs: list[RunMetrics] = []

    def record(self, metrics: RunMetrics) -> None:
        """Record metrics from a single query run."""
        self.runs.append(metrics)

    @property
    def total_tokens(self) -> int:
        return sum(r.token_usage.total_tokens for r in self.runs)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.estimated_cost_usd for r in self.runs)

    @property
    def avg_latency(self) -> float:
        if not self.runs:
            return 0.0
        return sum(r.latency_seconds for r in self.runs) / len(self.runs)

    def summary(self) -> dict:
        """Return a summary of all tracked metrics."""
        return {
            "system": self.system_name,
            "total_queries": len(self.runs),
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_latency_seconds": round(self.avg_latency, 3),
            "avg_tokens_per_query": (
                self.total_tokens // len(self.runs) if self.runs else 0
            ),
        }


@contextmanager
def timer(label: str = ""):
    """
    Context manager that measures wall-clock execution time.

    Usage:
        with timer("retrieval") as t:
            results = retriever.get_relevant_documents(query)
        print(f"Took {t.elapsed:.2f}s")
    """
    t = _TimerResult()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.elapsed = time.perf_counter() - start
        if label:
            logging.getLogger(__name__).debug(
                "%s completed in %.3fs", label, t.elapsed
            )


class _TimerResult:
    """Stores elapsed time from the timer context manager."""

    elapsed: float = 0.0


def setup_logging(level: int = logging.INFO) -> None:
    """Configure consistent logging format across all systems."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
#  Caching utilities
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from src.common.ingestion import ProcessedFiling


def compute_filings_hash(filings: Sequence[ProcessedFiling]) -> str:
    """
    Compute a deterministic SHA-256 hash for a set of filings.

    The hash is derived from sorted ``ticker:accession_number`` pairs so
    that the same set of filings always produces the same hash regardless
    of input order.  This is used as a content-addressed cache key for
    the vectorstore, BM25 index, and chunked documents.

    Args:
        filings: Processed SEC 10-K filings.

    Returns:
        First 12 hex chars of the SHA-256 digest (short but collision-safe
        for the expected number of distinct filing sets).
    """
    keys = sorted(
        f"{f.metadata.ticker}:{f.metadata.accession_number}" for f in filings
    )
    digest = hashlib.sha256("|".join(keys).encode()).hexdigest()
    return digest[:12]
