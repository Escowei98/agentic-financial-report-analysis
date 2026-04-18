"""
Unit tests for the Reflexion-style reflection verifier.
"""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from src.systems.rag_agent.reflection import (
    ReflectionVerdict,
    build_reflection_chain,
    generate_feedback_message,
    unpack_reflection_result,
)


class TestReflectionVerdictSchema:
    """Pydantic-schema behavior for ReflectionVerdict."""

    def test_accept_verdict_format_valid(self):
        v = ReflectionVerdict(status="accept", feedback="", issues=[])
        assert v.status == "accept"
        assert v.feedback == ""
        assert v.issues == []

    def test_revise_verdict_with_feedback(self):
        v = ReflectionVerdict(
            status="revise",
            feedback="Missing source citation for the revenue figure.",
            issues=["missing_citation", "numerical_mismatch"],
        )
        assert v.status == "revise"
        assert "citation" in v.feedback
        assert "missing_citation" in v.issues
        assert len(v.issues) == 2


class TestGenerateFeedbackMessage:
    """Feedback message generation for different verdict statuses."""

    def test_generate_feedback_message_for_revise(self):
        verdict = ReflectionVerdict(
            status="revise",
            feedback="Re-retrieve MSFT FY2024 Financial Statements.",
            issues=["numerical_mismatch"],
        )
        msg = generate_feedback_message(verdict)
        assert msg is not None
        assert isinstance(msg, HumanMessage)
        assert "Re-retrieve MSFT FY2024 Financial Statements." in msg.content
        assert "numerical_mismatch" in msg.content
        assert "revise" in msg.content.lower() or "reviewer" in msg.content.lower()

    def test_generate_feedback_message_for_accept_returns_none(self):
        verdict = ReflectionVerdict(status="accept", feedback="", issues=[])
        msg = generate_feedback_message(verdict)
        assert msg is None

    def test_generate_feedback_message_handles_empty_feedback_on_revise(self):
        """Revise verdict with empty feedback should still produce a usable message."""
        verdict = ReflectionVerdict(status="revise", feedback="", issues=[])
        msg = generate_feedback_message(verdict)
        assert msg is not None
        assert isinstance(msg, HumanMessage)
        # Fallback placeholder is inserted when no feedback is provided
        assert len(msg.content) > 0


class TestUnpackReflectionResult:
    """Token extraction from the different chain output shapes."""

    def test_unpack_bare_verdict_returns_zero_tokens(self):
        verdict = ReflectionVerdict(status="accept", feedback="", issues=[])
        out, prompt_t, completion_t = unpack_reflection_result(verdict)
        assert out is verdict
        assert prompt_t == 0
        assert completion_t == 0

    def test_unpack_dict_with_usage_metadata(self):
        verdict = ReflectionVerdict(status="accept", feedback="", issues=[])
        raw = AIMessage(
            content="accepted",
            usage_metadata={"input_tokens": 42, "output_tokens": 7, "total_tokens": 49},
        )
        chain_result = {"raw": raw, "parsed": verdict, "parsing_error": None}
        out, prompt_t, completion_t = unpack_reflection_result(chain_result)
        assert out is verdict
        assert prompt_t == 42
        assert completion_t == 7

    def test_unpack_dict_without_usage_metadata(self):
        verdict = ReflectionVerdict(status="accept", feedback="", issues=[])
        raw = AIMessage(content="accepted")  # no usage_metadata
        chain_result = {"raw": raw, "parsed": verdict, "parsing_error": None}
        out, prompt_t, completion_t = unpack_reflection_result(chain_result)
        assert out is verdict
        assert prompt_t == 0
        assert completion_t == 0

    def test_unpack_parse_error_returns_accept_fallback(self):
        """On parse error we must not crash: return an accept-fallback verdict."""
        raw = AIMessage(
            content="malformed output",
            usage_metadata={"input_tokens": 50, "output_tokens": 5, "total_tokens": 55},
        )
        chain_result = {
            "raw": raw,
            "parsed": None,
            "parsing_error": ValueError("validation failed"),
        }
        out, prompt_t, completion_t = unpack_reflection_result(chain_result)

        assert isinstance(out, ReflectionVerdict)
        assert out.status == "accept"
        assert "parse_error" in out.issues
        assert out.feedback == ""
        # Tokens must still be counted for accurate cost tracking
        assert prompt_t == 50
        assert completion_t == 5

    def test_unpack_parse_error_without_raw_usage_metadata(self):
        """Parse error path must be safe even when raw lacks usage_metadata."""
        chain_result = {
            "raw": AIMessage(content="malformed"),
            "parsed": None,
            "parsing_error": ValueError("validation failed"),
        }
        out, prompt_t, completion_t = unpack_reflection_result(chain_result)

        assert out.status == "accept"
        assert "parse_error" in out.issues
        assert prompt_t == 0
        assert completion_t == 0


class TestBuildReflectionChain:
    """Factory construction of the reflection chain."""

    def test_reflection_chain_calls_with_structured_output(self):
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = MagicMock(name="structured_llm")

        chain = build_reflection_chain(mock_llm, include_raw=True)

        # Verify the LLM was wrapped with structured output targeting our schema
        mock_llm.with_structured_output.assert_called_once()
        call_args = mock_llm.with_structured_output.call_args
        assert call_args.args[0] is ReflectionVerdict
        assert call_args.kwargs.get("include_raw") is True
        assert chain is not None

    def test_reflection_chain_default_includes_raw(self):
        """Default invocation should include_raw=True for token tracking."""
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = MagicMock()

        build_reflection_chain(mock_llm)

        call_args = mock_llm.with_structured_output.call_args
        assert call_args.kwargs.get("include_raw") is True
