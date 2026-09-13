"""
Unit tests for System 4's delegate_to_specialist tool factory.
"""

from unittest.mock import MagicMock

from src.systems.multi_agent.tools import create_delegate_tool


class TestCreateDelegateTool:
    """Tests for create_delegate_tool's validation and delegation logic."""

    def _make_tool(self, specialist_runner=None):
        return create_delegate_tool(
            specialist_runner=specialist_runner or MagicMock(return_value="specialist answer"),
            available_tickers=["AAPL", "MSFT"],
            available_sections=["MD&A", "Financial Statements"],
        )

    def test_valid_call_invokes_specialist_runner(self):
        runner = MagicMock(return_value="Apple's revenue was $391B")
        tool = self._make_tool(runner)

        result = tool.invoke({
            "tickers": ["AAPL"],
            "sections": ["MD&A"],
            "sub_question": "What was AAPL's revenue?",
        })

        runner.assert_called_once_with(["AAPL"], ["MD&A"], "What was AAPL's revenue?")
        assert "Apple's revenue was $391B" in result

    def test_ticker_matching_is_case_insensitive(self):
        runner = MagicMock(return_value="ok")
        tool = self._make_tool(runner)

        result = tool.invoke({
            "tickers": ["aapl"],
            "sections": ["MD&A"],
            "sub_question": "q",
        })

        runner.assert_called_once()
        assert "Error" not in result

    def test_invalid_ticker_returns_error_without_invoking_runner(self):
        runner = MagicMock()
        tool = self._make_tool(runner)

        result = tool.invoke({
            "tickers": ["TSLA"],
            "sections": ["MD&A"],
            "sub_question": "q",
        })

        runner.assert_not_called()
        assert "Invalid tickers" in result
        assert "TSLA" in result

    def test_invalid_section_returns_error_without_invoking_runner(self):
        runner = MagicMock()
        tool = self._make_tool(runner)

        result = tool.invoke({
            "tickers": ["AAPL"],
            "sections": ["Nonexistent Section"],
            "sub_question": "q",
        })

        runner.assert_not_called()
        assert "Invalid sections" in result

    def test_empty_tickers_returns_error(self):
        runner = MagicMock()
        tool = self._make_tool(runner)

        result = tool.invoke({"tickers": [], "sections": ["MD&A"], "sub_question": "q"})

        runner.assert_not_called()
        assert "at least one ticker" in result

    def test_empty_sections_returns_error(self):
        runner = MagicMock()
        tool = self._make_tool(runner)

        result = tool.invoke({"tickers": ["AAPL"], "sections": [], "sub_question": "q"})

        runner.assert_not_called()
        assert "at least one ticker" in result

    def test_empty_sub_question_returns_error(self):
        runner = MagicMock()
        tool = self._make_tool(runner)

        result = tool.invoke({"tickers": ["AAPL"], "sections": ["MD&A"], "sub_question": ""})

        runner.assert_not_called()
        assert "sub_question cannot be empty" in result

    def test_specialist_runner_exception_is_caught(self):
        runner = MagicMock(side_effect=RuntimeError("recursion limit exceeded"))
        tool = self._make_tool(runner)

        result = tool.invoke({
            "tickers": ["AAPL"],
            "sections": ["MD&A"],
            "sub_question": "q",
        })

        assert "Error running specialist" in result
        assert "recursion limit exceeded" in result

    def test_multiple_valid_tickers_and_sections(self):
        runner = MagicMock(return_value="combined answer")
        tool = self._make_tool(runner)

        result = tool.invoke({
            "tickers": ["AAPL", "MSFT"],
            "sections": ["MD&A", "Financial Statements"],
            "sub_question": "Compare AAPL and MSFT",
        })

        runner.assert_called_once_with(
            ["AAPL", "MSFT"], ["MD&A", "Financial Statements"], "Compare AAPL and MSFT"
        )
        assert "combined answer" in result
