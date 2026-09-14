"""
Trajectory rendering (src/evaluation/trajectory_formatter.py).

No metric reads this output; scoring works off the uniform chain the
systems emit. Its consumers are human: `eval_runner` stores the trajectory
per query, and the blind rating sample is built from it. A rater deciding
whether a system reasoned soundly is looking at exactly this text.

That makes the formatter a fairness surface even though it is not a metric.
It holds four separate rendering functions, one per architecture, and if one
of them shows less of what its system did than another does, the rater sees a
systematically thinner account for that system and has no way to notice. The
tests below pin the properties that keep the four accounts comparable:

  - no step, delegation or specialist output is silently dropped
  - primary filing text is withheld everywhere it could appear
  - non-evidence tool output (calculate, list_filings) IS shown, for all
  - the reflection stage is always accounted for, present or not
  - each account ends with the same cost signal
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.evaluation import trajectory_formatter as tf
from src.evaluation.trajectory_formatter import format_trajectory

SYSTEM_NAMES = ["rag_monolith", "rag_agent", "long_context", "multi_agent"]

# The three systems that run a tool loop. S1 is non-agentic by construction
# and has none, which is the architecture under test, not a gap.
TOOL_LOOP_SYSTEMS = ["rag_agent", "long_context", "multi_agent"]


def _metrics(total_tokens: int = 1234) -> SimpleNamespace:
    return SimpleNamespace(token_usage=SimpleNamespace(total_tokens=total_tokens))


def _doc(ticker: str = "AAPL", year: str = "2024", section: str = "MD&A") -> SimpleNamespace:
    return SimpleNamespace(
        metadata={"ticker": ticker, "fiscal_year": year, "section_name": section},
        page_content="secret filing text",
    )


def _tool_call(tool: str, args: dict | None = None, result: str = "") -> dict:
    return {"tool": tool, "args": args or {}, "result": result}


def _result(system_name: str, **overrides) -> SimpleNamespace:
    """A plausible result object for one system, overridable per test."""
    common = {"answer": "42", "contexts": ["ctx"], "metrics": _metrics()}
    per_system = {
        "rag_monolith": {"context_documents": [_doc()]},
        "rag_agent": {
            "tool_calls_log": [_tool_call("search_section", {"ticker": "AAPL"}, "text")],
            "context_documents": [],
            "reflection_verdict": None,
            "was_revised": False,
        },
        "long_context": {
            "tool_calls_log": [_tool_call("calculate", {"expression": "1+1"}, "2")],
            "reflection_verdict": None,
            "was_revised": False,
        },
        "multi_agent": {
            "tool_calls_log": [_tool_call("calculate", {"expression": "1+1"}, "2")],
            "supervisor_plan": "Ask the AAPL specialist.",
            "delegation_requests": [
                {"ticker": "AAPL", "sections": "MD&A", "sub_question": "revenue?"}
            ],
            "specialist_outputs": {"AAPL": "Revenue was $391bn."},
            "token_breakdown": {"supervisor": SimpleNamespace(total_tokens=100)},
            "reflection_verdict": None,
            "was_revised": False,
        },
    }
    fields = {**common, **per_system[system_name], **overrides}
    return SimpleNamespace(**fields)


class TestRouting:
    @pytest.mark.parametrize("system_name", SYSTEM_NAMES)
    def test_every_system_renders_a_trajectory(self, system_name):
        out = format_trajectory(_result(system_name), system_name)
        assert out.startswith("## Trajectory")
        assert len(out.splitlines()) >= 3

    def test_an_unknown_system_is_named_rather_than_silently_empty(self):
        out = format_trajectory(_result("rag_agent"), "rag_agent_unknown")
        assert "Unknown system" in out and "rag_agent_unknown" in out


class TestNoStepIsSilentlyDropped:
    """The property that keeps the four accounts comparable.

    A rater cannot tell a system that took three steps from one whose third
    step the formatter dropped.
    """

    @pytest.mark.parametrize("system_name", TOOL_LOOP_SYSTEMS)
    def test_every_tool_call_appears(self, system_name):
        calls = [
            _tool_call("list_filings", {}, "AAPL FY2024"),
            _tool_call("calculate", {"expression": "2+2"}, "4"),
            _tool_call("search_section", {"ticker": "MSFT"}, "filing text"),
        ]
        out = format_trajectory(_result(system_name, tool_calls_log=calls), system_name)

        for name in ("list_filings", "calculate", "search_section"):
            assert name in out, f"{system_name} dropped {name}"

    @pytest.mark.parametrize("system_name", TOOL_LOOP_SYSTEMS)
    def test_the_number_of_steps_is_stated(self, system_name):
        calls = [_tool_call("calculate", {}, "1") for _ in range(7)]
        out = format_trajectory(_result(system_name, tool_calls_log=calls), system_name)
        assert "7" in out

    @pytest.mark.parametrize("system_name", TOOL_LOOP_SYSTEMS)
    def test_an_empty_tool_loop_is_stated_rather_than_omitted(self, system_name):
        """"No tools used" and "tools not rendered" must not look alike."""
        out = format_trajectory(_result(system_name, tool_calls_log=[]), system_name)
        assert "None" in out or "0" in out

    @pytest.mark.parametrize("system_name", TOOL_LOOP_SYSTEMS)
    def test_an_unnamed_tool_call_is_still_rendered(self, system_name):
        out = format_trajectory(
            _result(system_name, tool_calls_log=[{"args": {}, "result": "x"}]), system_name
        )
        assert "unknown" in out


class TestEvidenceWithholding:
    """Primary filing text stays out, everywhere it could get in.

    Only S1 and S2 can surface retrieved passages at all, so showing them
    would make the rating a measure of the instrumentation rather than of the
    system. The rule has to hold in every formatter, not just the two where
    it currently bites.
    """

    @pytest.mark.parametrize("system_name", TOOL_LOOP_SYSTEMS)
    @pytest.mark.parametrize("tool", ["retrieve_chunks", "search_section"])
    def test_retrieved_filing_text_never_reaches_the_reader(self, system_name, tool):
        secret = "Net sales were $391,035 million in fiscal 2024."
        out = format_trajectory(
            _result(system_name, tool_calls_log=[_tool_call(tool, {}, secret)]),
            system_name,
        )
        assert secret not in out
        assert "withheld" in out

    @pytest.mark.parametrize("system_name", TOOL_LOOP_SYSTEMS)
    def test_the_size_of_what_was_withheld_is_stated(self, system_name):
        """A rater must see that a step returned evidence, just not the text."""
        out = format_trajectory(
            _result(system_name, tool_calls_log=[_tool_call("search_section", {}, "x" * 640)]),
            system_name,
        )
        assert "640 chars" in out

    @pytest.mark.parametrize("system_name", TOOL_LOOP_SYSTEMS)
    @pytest.mark.parametrize("tool", ["calculate", "list_filings"])
    def test_computed_and_catalogue_output_is_shown(self, system_name, tool):
        """Withholding these would hide the reasoning instead of the evidence."""
        out = format_trajectory(
            _result(system_name, tool_calls_log=[_tool_call(tool, {}, "RESULT-MARKER")]),
            system_name,
        )
        assert "RESULT-MARKER" in out

    def test_the_tool_name_is_matched_case_insensitively(self):
        out = format_trajectory(
            _result("rag_agent", tool_calls_log=[_tool_call("Search_Section", {}, "secret text")]),
            "rag_agent",
        )
        assert "secret text" not in out

    def test_s1_shows_provenance_but_no_passage_text(self):
        out = format_trajectory(_result("rag_monolith"), "rag_monolith")
        assert "AAPL FY2024, MD&A" in out
        assert "secret filing text" not in out


class TestReflectionIsAlwaysAccountedFor:
    """Silence about reflection is indistinguishable from "did not reflect"."""

    @pytest.mark.parametrize("system_name", TOOL_LOOP_SYSTEMS)
    def test_a_verdict_is_reported_with_its_revision_flag(self, system_name):
        out = format_trajectory(
            _result(
                system_name,
                reflection_verdict=SimpleNamespace(value="revise"),
                was_revised=True,
            ),
            system_name,
        )
        assert "revise" in out
        assert "True" in out

    @pytest.mark.parametrize("system_name", TOOL_LOOP_SYSTEMS)
    def test_a_plain_string_verdict_is_reported_too(self, system_name):
        out = format_trajectory(
            _result(system_name, reflection_verdict="accept"), system_name
        )
        assert "accept" in out

    @pytest.mark.parametrize("system_name", TOOL_LOOP_SYSTEMS)
    def test_the_absence_of_reflection_is_stated_explicitly(self, system_name):
        out = format_trajectory(_result(system_name, reflection_verdict=None), system_name)
        assert "Reflection" in out

    def test_s1_reports_no_reflection_section_at_all(self):
        """S1 has no reflection stage; a line about one would be noise."""
        out = format_trajectory(_result("rag_monolith"), "rag_monolith")
        assert "Reflection" not in out


class TestMultiAgentChainIsComplete:
    """S4's account has the most parts, and therefore the most to lose."""

    def test_the_supervisor_plan_is_shown(self):
        out = format_trajectory(
            _result("multi_agent", supervisor_plan="PLAN-MARKER"), "multi_agent"
        )
        assert "PLAN-MARKER" in out

    def test_every_delegation_is_shown_with_its_sub_question(self):
        delegations = [
            {"ticker": "AAPL", "sections": "MD&A", "sub_question": "SUBQ-1"},
            {"ticker": "MSFT", "sections": "Item 8", "sub_question": "SUBQ-2"},
            {"ticker": "GOOGL", "sections": "MD&A", "sub_question": "SUBQ-3"},
        ]
        out = format_trajectory(
            _result("multi_agent", delegation_requests=delegations), "multi_agent"
        )
        assert "3 specialists invoked" in out
        for i in (1, 2, 3):
            assert f"SUBQ-{i}" in out

    def test_alternate_delegation_key_names_are_understood(self):
        """The graph has used `tickers`/`question` in places."""
        out = format_trajectory(
            _result(
                "multi_agent",
                delegation_requests=[{"tickers": "AAPL", "section": "MD&A", "question": "SUBQ-ALT"}],
            ),
            "multi_agent",
        )
        assert "SUBQ-ALT" in out
        assert "?" not in out.split("Sub-question:")[1].split("\n")[0]

    def test_every_specialist_output_is_shown(self):
        outputs = {"AAPL": "OUT-A", "MSFT": "OUT-M"}
        out = format_trajectory(
            _result("multi_agent", specialist_outputs=outputs), "multi_agent"
        )
        assert "OUT-A" in out and "OUT-M" in out

    def test_the_per_agent_token_breakdown_is_shown(self):
        breakdown = {
            "supervisor": SimpleNamespace(total_tokens=100),
            "specialist_AAPL": SimpleNamespace(total_tokens=900),
        }
        out = format_trajectory(
            _result("multi_agent", token_breakdown=breakdown), "multi_agent"
        )
        assert "supervisor: 100" in out
        assert "specialist_AAPL: 900" in out

    def test_a_bare_integer_token_breakdown_still_renders(self):
        out = format_trajectory(
            _result("multi_agent", token_breakdown={"supervisor": 100}), "multi_agent"
        )
        assert "supervisor: 100" in out


class TestCostSignalIsComparable:
    @pytest.mark.parametrize("system_name", SYSTEM_NAMES)
    def test_every_account_ends_with_the_total_token_count(self, system_name):
        out = format_trajectory(_result(system_name, metrics=_metrics(4711)), system_name)
        assert out.splitlines()[-1] == "Total tokens: 4711"


class TestTruncation:
    def test_text_at_the_limit_is_not_cut(self):
        assert tf._truncate("y" * tf._EVIDENCE_CHAR_LIMIT).endswith("y")
        assert "..." not in tf._truncate("y" * tf._EVIDENCE_CHAR_LIMIT)

    def test_text_over_the_limit_is_marked_as_cut(self):
        out = tf._truncate("y" * (tf._EVIDENCE_CHAR_LIMIT + 1))
        assert out.endswith("...")
        assert len(out) == tf._EVIDENCE_CHAR_LIMIT + 3

    def test_newlines_are_flattened_so_one_step_stays_one_line(self):
        assert "\n" not in tf._truncate("a\nb\nc")


class TestRobustness:
    """A malformed result must not take a 105-USD run down with it.

    `eval_runner` calls the formatter inside the per-item loop, outside any
    try/except: an exception here ends that system's evaluation entirely,
    after every query has already been paid for.
    """

    @pytest.mark.parametrize("system_name", SYSTEM_NAMES)
    def test_a_result_missing_every_optional_field_still_renders(self, system_name):
        bare = SimpleNamespace(answer="42", metrics=_metrics())
        out = format_trajectory(bare, system_name)
        assert out.startswith("## Trajectory")

    def test_context_documents_without_metadata_do_not_raise(self):
        doc = SimpleNamespace(metadata=None)
        out = format_trajectory(
            _result("rag_monolith", context_documents=[doc]), "rag_monolith"
        )
        assert "?" in out
