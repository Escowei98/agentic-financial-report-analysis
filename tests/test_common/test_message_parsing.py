"""
Unit tests for the shared agent-message parser.

This loop previously existed five times across S2, S3 and S4. The tests
below pin the behaviour the four systems' token accounting, process metrics
and judge-facing trajectories all depend on.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.common.message_parsing import format_tool_calls_for_prompt, parse_agent_messages


def _ai(content="", tool_calls=None, usage=None):
    msg = AIMessage(content=content, tool_calls=tool_calls or [])
    if usage:
        msg.usage_metadata = usage
    return msg


class TestParseAgentMessages:
    def test_empty_history(self):
        answer, calls, contexts, usage = parse_agent_messages([])
        assert answer == ""
        assert calls == []
        assert contexts == []
        assert usage.total_tokens == 0

    def test_direct_answer_without_tools(self):
        msgs = [
            HumanMessage(content="What was AAPL FY2024 revenue?"),
            _ai("$391,035 million.", usage={"input_tokens": 120, "output_tokens": 30}),
        ]
        answer, calls, contexts, usage = parse_agent_messages(msgs)
        assert answer == "$391,035 million."
        assert calls == []
        assert usage.prompt_tokens == 120
        assert usage.completion_tokens == 30
        assert usage.total_tokens == 150

    def test_tool_result_is_linked_back_to_its_call(self):
        msgs = [
            _ai(tool_calls=[{"name": "calculate", "args": {"expression": "1+1"}, "id": "c1"}],
                usage={"input_tokens": 10, "output_tokens": 5}),
            ToolMessage(content="2", tool_call_id="c1"),
            _ai("The result is 2.", usage={"input_tokens": 20, "output_tokens": 4}),
        ]
        answer, calls, contexts, usage = parse_agent_messages(msgs)
        assert answer == "The result is 2."
        assert len(calls) == 1
        assert calls[0]["tool"] == "calculate"
        assert calls[0]["result"] == "2"
        assert contexts == ["2"]
        assert usage.total_tokens == 39

    def test_tokens_accumulate_across_all_ai_messages(self):
        msgs = [
            _ai(tool_calls=[{"name": "t", "args": {}, "id": "a"}], usage={"input_tokens": 100, "output_tokens": 10}),
            ToolMessage(content="x", tool_call_id="a"),
            _ai(tool_calls=[{"name": "t", "args": {}, "id": "b"}], usage={"input_tokens": 200, "output_tokens": 20}),
            ToolMessage(content="y", tool_call_id="b"),
            _ai("done", usage={"input_tokens": 300, "output_tokens": 30}),
        ]
        _, calls, contexts, usage = parse_agent_messages(msgs)
        assert len(calls) == 2
        assert contexts == ["x", "y"]
        assert usage.prompt_tokens == 600
        assert usage.completion_tokens == 60

    def test_intermediate_ai_content_does_not_override_final_answer(self):
        msgs = [
            _ai("thinking...", tool_calls=[{"name": "t", "args": {}, "id": "a"}]),
            ToolMessage(content="x", tool_call_id="a"),
            _ai("final answer"),
        ]
        answer, _, _, _ = parse_agent_messages(msgs)
        assert answer == "final answer"

    def test_unmatched_tool_message_is_still_a_context(self):
        msgs = [ToolMessage(content="orphan", tool_call_id="missing")]
        _, calls, contexts, _ = parse_agent_messages(msgs)
        assert calls == []
        assert contexts == ["orphan"]

    def test_missing_usage_metadata_is_tolerated(self):
        answer, _, _, usage = parse_agent_messages([_ai("no usage field")])
        assert answer == "no usage field"
        assert usage.total_tokens == 0


class TestFormatToolCallsForPrompt:
    def test_empty(self):
        assert format_tool_calls_for_prompt([]) == "(no tool calls)"

    def test_renders_one_line_per_call(self):
        out = format_tool_calls_for_prompt([
            {"tool": "calculate", "args": {"expression": "1+1"}, "result": "2"},
            {"tool": "list_filings", "args": {}, "result": "..."},
        ])
        assert out.splitlines() == [
            "- calculate({'expression': '1+1'})",
            "- list_filings({})",
        ]
