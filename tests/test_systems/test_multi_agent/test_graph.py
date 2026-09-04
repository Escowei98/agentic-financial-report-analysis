"""
Unit tests for System 4's shared graph-state helpers.
"""

from src.common.utils import TokenUsage
from src.systems.multi_agent.graph import _merge_token_usage


class TestMergeTokenUsage:
    """Tests for the in-place TokenUsage accumulator used across S4 nodes."""

    def test_accumulates_into_dest(self):
        dest = TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120)
        src = TokenUsage(prompt_tokens=50, completion_tokens=10, total_tokens=60)

        _merge_token_usage(dest, src)

        assert dest.prompt_tokens == 150
        assert dest.completion_tokens == 30
        assert dest.total_tokens == 180

    def test_merging_zero_usage_is_a_no_op(self):
        dest = TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120)
        src = TokenUsage()

        _merge_token_usage(dest, src)

        assert dest.prompt_tokens == 100
        assert dest.completion_tokens == 20
        assert dest.total_tokens == 120

    def test_does_not_mutate_src(self):
        dest = TokenUsage(prompt_tokens=10, completion_tokens=1, total_tokens=11)
        src = TokenUsage(prompt_tokens=5, completion_tokens=1, total_tokens=6)

        _merge_token_usage(dest, src)

        assert src.prompt_tokens == 5
        assert src.completion_tokens == 1
        assert src.total_tokens == 6
