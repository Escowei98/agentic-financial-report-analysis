"""
Unit tests for shared cost-tracking, timing, and caching utilities
(src/common/utils.py).
"""

import time

from src.common.ingestion import FilingMetadata, ProcessedFiling
from src.common.utils import (
    CostTracker,
    RunMetrics,
    TokenUsage,
    compute_filings_hash,
    extract_text,
    timer,
)


class TestExtractText:
    """Tests for extract_text (LLM content normalization)."""

    def test_string_content_returned_as_is(self):
        assert extract_text("hello world") == "hello world"

    def test_list_of_text_blocks_joined(self):
        content = [{"text": "hello"}, {"text": "world"}]
        assert extract_text(content) == "hello world"

    def test_list_ignores_blocks_without_text_key(self):
        content = [{"text": "hello"}, {"type": "image"}]
        assert extract_text(content) == "hello"

    def test_non_dict_list_items_are_skipped(self):
        content = ["not-a-dict", {"text": "hello"}]
        assert extract_text(content) == "hello"

    def test_empty_list_returns_empty_string(self):
        assert extract_text([]) == ""

    def test_other_type_falls_back_to_str(self):
        assert extract_text(42) == "42"


class TestRunMetricsCost:
    """Tests for RunMetrics.estimated_cost_usd."""

    def test_zero_tokens_costs_nothing(self):
        metrics = RunMetrics()
        assert metrics.estimated_cost_usd == 0.0

    def test_cost_formula_matches_documented_pricing(self):
        usage = TokenUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000)
        metrics = RunMetrics(token_usage=usage)
        # $0.30 / 1M input + $2.50 / 1M output
        assert metrics.estimated_cost_usd == 0.30 + 2.50

    def test_cost_scales_linearly(self):
        usage = TokenUsage(prompt_tokens=500_000, completion_tokens=200_000, total_tokens=700_000)
        metrics = RunMetrics(token_usage=usage)
        expected = 500_000 * 0.30 / 1_000_000 + 200_000 * 2.50 / 1_000_000
        assert metrics.estimated_cost_usd == expected


class TestCostTracker:
    """Tests for CostTracker aggregation across multiple query runs."""

    def _make_metrics(self, prompt: int, completion: int, latency: float) -> RunMetrics:
        return RunMetrics(
            token_usage=TokenUsage(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=prompt + completion,
            ),
            latency_seconds=latency,
        )

    def test_empty_tracker_summary(self):
        tracker = CostTracker("rag_monolith")
        summary = tracker.summary()
        assert summary["total_queries"] == 0
        assert summary["total_tokens"] == 0
        assert summary["total_cost_usd"] == 0.0
        assert summary["avg_latency_seconds"] == 0.0
        assert summary["avg_tokens_per_query"] == 0

    def test_records_are_accumulated(self):
        tracker = CostTracker("rag_agent")
        tracker.record(self._make_metrics(100, 20, 1.0))
        tracker.record(self._make_metrics(200, 40, 3.0))

        assert len(tracker.runs) == 2
        assert tracker.total_tokens == 360
        assert tracker.avg_latency == 2.0

    def test_summary_reflects_system_name(self):
        tracker = CostTracker("long_context")
        assert tracker.summary()["system"] == "long_context"

    def test_avg_tokens_per_query_uses_integer_division(self):
        tracker = CostTracker("rag_agent")
        tracker.record(self._make_metrics(100, 1, 1.0))
        tracker.record(self._make_metrics(100, 0, 1.0))
        # total=201, 2 runs -> floor(201/2) == 100
        assert tracker.summary()["avg_tokens_per_query"] == 100


class TestTimer:
    """Tests for the timer context manager."""

    def test_elapsed_time_is_recorded(self):
        with timer("test") as t:
            time.sleep(0.01)
        assert t.elapsed >= 0.01

    def test_elapsed_recorded_even_without_label(self):
        with timer() as t:
            pass
        assert t.elapsed >= 0.0

    def test_elapsed_recorded_even_on_exception(self):
        t_ref = None
        try:
            with timer("boom") as t:
                t_ref = t
                raise ValueError("boom")
        except ValueError:
            pass
        assert t_ref.elapsed >= 0.0


class TestComputeFilingsHash:
    """Tests for the content-addressed cache key used by vectorstore/BM25 caches."""

    def _make_filing(
        self, ticker: str, accession: str, sections: dict | None = None
    ) -> ProcessedFiling:
        return ProcessedFiling(
            metadata=FilingMetadata(
                ticker=ticker,
                company_name=f"{ticker} Inc.",
                cik="1234",
                filing_date="2024-01-01",
                accession_number=accession,
                fiscal_year_end="2023-12-31",
            ),
            sections={} if sections is None else sections,
            full_text="text",
        )

    def test_hash_is_order_independent(self):
        f1 = self._make_filing("AAPL", "001")
        f2 = self._make_filing("MSFT", "002")

        hash_ab = compute_filings_hash([f1, f2])
        hash_ba = compute_filings_hash([f2, f1])

        assert hash_ab == hash_ba

    def test_hash_is_deterministic(self):
        filings = [self._make_filing("AAPL", "001")]
        assert compute_filings_hash(filings) == compute_filings_hash(filings)

    def test_hash_changes_with_different_filings(self):
        filings_a = [self._make_filing("AAPL", "001")]
        filings_b = [self._make_filing("AAPL", "002")]
        assert compute_filings_hash(filings_a) != compute_filings_hash(filings_b)

    def test_hash_is_short_hex_string(self):
        filings = [self._make_filing("AAPL", "001")]
        digest = compute_filings_hash(filings)
        assert len(digest) == 12
        int(digest, 16)  # raises ValueError if not valid hex

    def test_empty_filings_list_still_hashes(self):
        digest = compute_filings_hash([])
        assert len(digest) == 12

    def test_hash_changes_when_section_text_changes(self):
        """A re-parsed filing keeps its accession number but must not keep
        its cache. GOOGL FY2024 was re-ingested on 2026-09-06 after its
        Risk Factors and MD&A had been truncated; an identity-only key
        would have kept S1/S2 on the broken chunks.
        """
        broken = self._make_filing("GOOGL", "0001652044-25-000014", {"MD&A": "stub"})
        repaired = self._make_filing(
            "GOOGL", "0001652044-25-000014", {"MD&A": "the full discussion ..."}
        )
        assert compute_filings_hash([broken]) != compute_filings_hash([repaired])

    def test_hash_changes_when_a_section_is_missing(self):
        complete = self._make_filing(
            "AAPL", "001", {"MD&A": "text", "Risk Factors": "text"}
        )
        partial = self._make_filing("AAPL", "001", {"MD&A": "text"})
        assert compute_filings_hash([complete]) != compute_filings_hash([partial])

    def test_hash_ignores_section_insertion_order(self):
        a = self._make_filing("AAPL", "001", {"MD&A": "x", "Business": "y"})
        b = self._make_filing("AAPL", "001", {"Business": "y", "MD&A": "x"})
        assert compute_filings_hash([a]) == compute_filings_hash([b])
