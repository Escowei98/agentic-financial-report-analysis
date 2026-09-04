"""
Structural symmetry checks across the four systems.

These tests do not exercise behaviour; they pin the *architecture* claims the
thesis makes in 3.5.1 and 4.2.3 — that the only differences between the four
systems are the ones the hypotheses actually vary. Each of them would have
caught a real drift that occurred during development: the Vertex AI
thought-signature guard existed only in S3 while S2 shared the same exposure,
and the ReAct wrapper, message parser and tool-call formatter each existed as
multiple hand-maintained copies.
"""

import inspect

import src.systems.long_context.pipeline as s3
import src.systems.multi_agent.pipeline as s4
import src.systems.rag_agent.pipeline as s2
import src.systems.rag_monolith.pipeline as s1
from src.common.agent import build_agent
from src.common.answer_format_convention import ANSWER_FORMAT_CONVENTION
from src.common.message_parsing import parse_agent_messages
from src.common.reflection import run_reflection_pass
from src.common.tools.calculate import calculate
from src.common.tools.list_filings import create_list_filings_tool

AGENTIC = (s2, s3, s4)
ALL_SYSTEMS = (s1, s2, s3, s4)


class TestSharedKnowledgeTools:
    """Tool-Symmetry Principle: knowledge tools are identical objects, not copies."""

    def test_all_agentic_systems_use_the_same_calculate_object(self):
        for mod in AGENTIC:
            assert mod.calculate is calculate

    def test_all_agentic_systems_use_the_same_list_filings_factory(self):
        for mod in AGENTIC:
            assert mod.create_list_filings_tool is create_list_filings_tool

    def test_retrieval_tools_stay_exclusive_to_system_2(self):
        assert hasattr(s2, "create_retrieve_chunks_tool")
        assert hasattr(s2, "create_search_section_tool")
        for mod in (s3, s4):
            assert not hasattr(mod, "create_retrieve_chunks_tool")
            assert not hasattr(mod, "create_search_section_tool")

    def test_delegation_primitive_stays_exclusive_to_system_4(self):
        assert hasattr(s4, "create_delegate_tool")
        for mod in (s2, s3):
            assert not hasattr(mod, "create_delegate_tool")


class TestSharedControlFlow:
    """One ReAct factory and one message parser for all agentic systems."""

    def test_all_agentic_systems_use_the_same_agent_factory(self):
        for mod in AGENTIC:
            assert mod.build_agent is build_agent

    def test_all_agentic_systems_use_the_same_message_parser(self):
        for mod in AGENTIC:
            assert mod.parse_agent_messages is parse_agent_messages

    def test_no_system_defines_its_own_parse_or_format_helper(self):
        for mod in ALL_SYSTEMS:
            for name in ("_parse_messages", "_format_tool_calls_for_prompt"):
                assert not hasattr(mod, name), f"{mod.__name__} still defines {name}"


class TestSharedReflection:
    """S2 and S3 must not differ in self-correction capability (H1/H3)."""

    def test_s2_and_s3_share_the_reflection_pass(self):
        assert s2.run_reflection_pass is run_reflection_pass
        assert s3.run_reflection_pass is run_reflection_pass

    def test_reinvoke_guard_is_shared_not_per_system(self):
        source = inspect.getsource(run_reflection_pass)
        assert "except Exception" in source
        for mod in (s2, s3):
            mod_source = inspect.getsource(mod)
            assert "thought signature" not in mod_source, (
                f"{mod.__name__} carries its own copy of the guard again"
            )

    def test_all_agentic_systems_build_the_same_reflection_chain(self):
        from src.common.reflection import build_reflection_chain
        for mod in AGENTIC:
            assert mod.build_reflection_chain is build_reflection_chain


class TestSharedAnswerFormatConvention:
    """The convention must be imported by all four prompts, never copied."""

    def test_convention_is_present_verbatim_in_every_system_prompt(self):
        from src.systems.long_context.prompt import ROLE_AND_INSTRUCTIONS  # noqa: F401
        from src.systems.multi_agent.prompts import (
            SPECIALIST_PROMPT_TEMPLATE,
            SYNTHESIZER_PROMPT,
        )
        from src.systems.rag_agent.agent import SYSTEM_PROMPT

        prompts = {
            "S1": str(s1.RAG_PROMPT.messages[0].prompt.template),
            "S2": SYSTEM_PROMPT,
            "S4.specialist": SPECIALIST_PROMPT_TEMPLATE,
            "S4.synthesizer": SYNTHESIZER_PROMPT,
        }
        for name, text in prompts.items():
            assert ANSWER_FORMAT_CONVENTION in text, f"{name} lost the convention"

    def test_s3_prompt_carries_the_convention(self):
        from src.systems.long_context.prompt import build_system_prompt
        assert ANSWER_FORMAT_CONVENTION in build_system_prompt([])
