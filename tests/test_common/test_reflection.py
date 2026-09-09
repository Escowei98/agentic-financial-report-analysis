"""
Unit tests for the Reflexion-style reflection verifier.
"""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from src.common.reflection import (
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


class TestRunReflectionPass:
    """
    Shared single-pass reflection + correction round (S2 and S3).

    The re-invoke guard is the reason this lives in `common`: S2 and S3 both
    resend the full message history including prior tool calls, so both are
    exposed to the same Vertex AI thought-signature round-trip failure. When
    it was implemented per system, only S3 had the guard.
    """

    @staticmethod
    def _chain(status, feedback="", issues=None):
        chain = MagicMock()
        chain.invoke.return_value = {
            "parsed": ReflectionVerdict(
                status=status, feedback=feedback, issues=issues or []
            ),
            "raw": AIMessage(content=""),
        }
        return chain

    def _run(self, agent, chain, messages=None, placeholder="(none)", preamble=""):
        from src.common.reflection import run_reflection_pass
        return run_reflection_pass(
            agent=agent,
            messages=messages if messages is not None else [AIMessage(content="draft")],
            question="What was AAPL FY2024 revenue?",
            reflection_chain=chain,
            recursion_limit=12,
            empty_contexts_placeholder=placeholder,
            contexts_preamble=preamble,
        )

    def test_accept_verdict_does_not_reinvoke_the_agent(self):
        agent = MagicMock()
        messages, verdict, was_revised, _, _ = self._run(agent, self._chain("accept"))
        agent.invoke.assert_not_called()
        assert was_revised is False
        assert verdict.status == "accept"
        assert [m.content for m in messages] == ["draft"]

    def test_revise_verdict_triggers_exactly_one_reinvoke(self):
        agent = MagicMock()
        agent.invoke.return_value = {"messages": [AIMessage(content="revised")]}
        messages, verdict, was_revised, _, _ = self._run(
            agent, self._chain("revise", "Add a citation.", ["missing_citation"])
        )
        assert agent.invoke.call_count == 1
        assert was_revised is True
        assert verdict.status == "revise"
        assert [m.content for m in messages] == ["revised"]

    def test_reinvoke_failure_falls_back_to_the_draft(self):
        """The guard: a failed revision must not lose the query."""
        agent = MagicMock()
        agent.invoke.side_effect = RuntimeError(
            "400 InvalidArgument: must include at least one parts field"
        )
        draft = [AIMessage(content="draft answer")]
        messages, verdict, was_revised, _, _ = self._run(
            agent, self._chain("revise", "Fix it.", ["x"]), messages=draft
        )
        assert agent.invoke.call_count == 1
        assert was_revised is False
        assert [m.content for m in messages] == ["draft answer"]
        assert verdict.status == "revise"

    def test_empty_contexts_placeholder_reaches_the_verifier(self):
        chain = self._chain("accept")
        self._run(MagicMock(), chain, placeholder="(no retrieval contexts captured)")
        assert chain.invoke.call_args[0][0]["contexts"] == "(no retrieval contexts captured)"

    def test_tool_outputs_are_passed_as_contexts(self):
        from langchain_core.messages import ToolMessage
        chain = self._chain("accept")
        msgs = [
            AIMessage(content="", tool_calls=[{"name": "calculate", "args": {}, "id": "c1"}]),
            ToolMessage(content="42", tool_call_id="c1"),
            AIMessage(content="draft"),
        ]
        self._run(MagicMock(), chain, messages=msgs)
        payload = chain.invoke.call_args[0][0]
        assert payload["contexts"] == "42"
        assert payload["answer"] == "draft"
        assert payload["tool_calls"] == "- calculate({})"

    def test_preamble_precedes_tool_outputs(self):
        """A non-retrieval system must be able to say what its tool outputs are.

        Without it the verifier reads a bare `calculate` return under the
        heading "Retrieved source contexts" and rules every claim ungrounded —
        the defect that drove S3's correction_rate to 0.75 on 2026-09-08.
        """
        from langchain_core.messages import ToolMessage
        chain = self._chain("accept")
        msgs = [
            AIMessage(content="", tool_calls=[{"name": "calculate", "args": {}, "id": "c1"}]),
            ToolMessage(content="22.4005", tool_call_id="c1"),
            AIMessage(content="draft"),
        ]
        self._run(MagicMock(), chain, messages=msgs, preamble="NOTE: not source evidence.")
        contexts = chain.invoke.call_args[0][0]["contexts"]
        assert contexts.startswith("NOTE: not source evidence.")
        assert "22.4005" in contexts

    def test_preamble_also_precedes_the_empty_placeholder(self):
        chain = self._chain("accept")
        self._run(MagicMock(), chain, placeholder="(no tool outputs)", preamble="NOTE: inlined.")
        assert chain.invoke.call_args[0][0]["contexts"] == "NOTE: inlined.\n\n(no tool outputs)"

    def test_no_preamble_leaves_contexts_untouched(self):
        """A retrieval system's tool outputs ARE the evidence — no note needed."""
        from langchain_core.messages import ToolMessage
        chain = self._chain("accept")
        msgs = [
            AIMessage(content="", tool_calls=[{"name": "retrieve_chunks", "args": {}, "id": "c1"}]),
            ToolMessage(content="AAPL FY2024 revenue was $391,035M", tool_call_id="c1"),
            AIMessage(content="draft"),
        ]
        self._run(MagicMock(), chain, messages=msgs)
        assert chain.invoke.call_args[0][0]["contexts"] == "AAPL FY2024 revenue was $391,035M"
