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
from src.common.non_answerability_convention import NON_ANSWERABILITY_CONVENTION
from src.common.reasoning_chain_convention import REASONING_CHAIN_CONVENTION
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

    def test_only_the_non_retrieval_system_declares_an_evidence_note(self):
        """Shared verifier, honestly described inputs — a deliberate asymmetry.

        S2's tool outputs ARE the source evidence the groundedness criterion
        assumes, so it passes no note. S3 retrieves nothing: its filings are
        inlined in the system prompt and never reach the verifier, so without
        a note the verifier judges a bare `calculate` return as if it were the
        corpus. Equal treatment here means describing each architecture's
        evidence basis truthfully, not feeding both the same string.
        See EVAL_DECISION_LOG.md [2026-09-08].
        """
        s3_source = inspect.getsource(s3.LongContextPipeline.query)
        assert "contexts_preamble=" in s3_source, (
            "S3 no longer tells the verifier that its tool outputs are not "
            "source evidence — the 2026-09-08 spurious-revision defect is back"
        )
        note = s3._REFLECTION_EVIDENCE_NOTE
        assert "no retrieval" in note and "unsupported_claim" in note
        # The note scopes criterion 1 only; the rest must keep their teeth.
        assert "Criteria 2-4 apply unchanged" in note

        s2_source = inspect.getsource(s2.AgentRAGPipeline.query)
        assert "contexts_preamble=" not in s2_source, (
            "S2 retrieves real evidence; a note would tell the verifier to "
            "stop checking groundedness on the one system where it can"
        )


def _answer_emitting_prompts() -> dict[str, str]:
    """The prompt of every node that emits a user-facing answer."""
    from src.systems.multi_agent.prompts import (
        SPECIALIST_PROMPT_TEMPLATE,
        SYNTHESIZER_PROMPT,
    )
    from src.systems.rag_agent.agent import SYSTEM_PROMPT

    return {
        "S1": str(s1.RAG_PROMPT.messages[0].prompt.template),
        "S2": SYSTEM_PROMPT,
        "S4.specialist": SPECIALIST_PROMPT_TEMPLATE,
        "S4.synthesizer": SYNTHESIZER_PROMPT,
    }


class TestSharedAnswerFormatConvention:
    """The convention must be imported by all four prompts, never copied."""

    def test_convention_is_present_verbatim_in_every_system_prompt(self):
        for name, text in _answer_emitting_prompts().items():
            assert ANSWER_FORMAT_CONVENTION in text, f"{name} lost the convention"

    def test_s3_prompt_carries_the_convention(self):
        from src.systems.long_context.prompt import build_system_prompt
        assert ANSWER_FORMAT_CONVENTION in build_system_prompt([])


class TestSharedNonAnswerabilityConvention:
    """Second shared convention, same discipline: byte-identical everywhere,
    imported and never copied. If one system were told how to handle an
    unanswerable question and another were not, the FA-Refusal comparison
    would measure prompt placement instead of architecture."""

    def test_convention_is_present_verbatim_in_every_system_prompt(self):
        for name, text in _answer_emitting_prompts().items():
            assert NON_ANSWERABILITY_CONVENTION in text, f"{name} lost the convention"

    def test_s3_prompt_carries_the_convention(self):
        from src.systems.long_context.prompt import build_system_prompt
        assert NON_ANSWERABILITY_CONVENTION in build_system_prompt([])

    def test_s4_supervisor_carries_it_too(self):
        """Documented FP-5 exception: the supervisor holds the corpus
        metadata and is the only S4 node that can recognise an out-of-corpus
        question at all. It gets this convention although it does not get the
        answer-format one."""
        from src.systems.multi_agent.prompts import SUPERVISOR_PROMPT
        assert NON_ANSWERABILITY_CONVENTION in SUPERVISOR_PROMPT

    def test_no_system_holds_a_private_copy_of_the_rules(self):
        """A paraphrase drifting into one prompt is the failure this guards:
        the text must come from the shared constant, so the distinctive
        wording must not appear anywhere outside it."""
        import re
        from pathlib import Path

        marker = "silent on a matter is not evidence"
        src_dir = Path(__file__).resolve().parents[2] / "src"
        offenders = [
            str(p.relative_to(src_dir))
            for p in src_dir.rglob("*.py")
            if p.name != "non_answerability_convention.py"
            and re.search(marker, p.read_text(encoding="utf-8"))
        ]
        assert not offenders, f"convention text copied into: {offenders}"


class TestSharedReasoningChainConvention:
    """Third shared convention. Unlike the other two it is deliberately NOT in
    every answer-emitting node: it belongs to the node whose answer is the
    measured one. See REASONING_QUALITY_SPEC.md section 4."""

    def test_the_three_single_answer_systems_carry_it(self):
        prompts = _answer_emitting_prompts()
        for name in ("S1", "S2", "S4.synthesizer"):
            assert REASONING_CHAIN_CONVENTION in prompts[name], (
                f"{name} lost the reasoning chain convention"
            )

    def test_s3_prompt_carries_the_convention(self):
        from src.systems.long_context.prompt import build_system_prompt
        assert REASONING_CHAIN_CONVENTION in build_system_prompt([])

    def test_s4_emits_exactly_one_chain(self):
        """Specialists and supervisor must NOT carry it.

        With the convention in the specialists, S4 would be the only
        architecture producing several competing chains per question, and
        which one the parser picked would be an instrumentation choice rather
        than a property of the system.
        """
        from src.systems.multi_agent.prompts import (
            SPECIALIST_PROMPT_TEMPLATE,
            SUPERVISOR_PROMPT,
        )
        assert REASONING_CHAIN_CONVENTION not in SPECIALIST_PROMPT_TEMPLATE
        assert REASONING_CHAIN_CONVENTION not in SUPERVISOR_PROMPT

    def test_convention_carries_no_curly_braces(self):
        """S1 renders its prompt through ChatPromptTemplate, which reads
        `{...}` as a template variable. A brace in the shared text would break
        exactly one of the four systems -- the worst possible failure mode for
        a constant whose entire purpose is symmetry."""
        assert "{" not in REASONING_CHAIN_CONVENTION
        assert "}" not in REASONING_CHAIN_CONVENTION

    def test_s1_prompt_still_renders(self):
        """The behavioural half of the guard above."""
        rendered = s1.RAG_PROMPT.format(context="ctx", question="q")
        assert "## Reasoning" in rendered

    def test_no_system_holds_a_private_copy_of_the_rules(self):
        import re
        from pathlib import Path

        marker = "must still have the complete answer"
        src_dir = Path(__file__).resolve().parents[2] / "src"
        offenders = [
            str(p.relative_to(src_dir))
            for p in src_dir.rglob("*.py")
            if p.name != "reasoning_chain_convention.py"
            and re.search(marker, p.read_text(encoding="utf-8"))
        ]
        assert not offenders, f"convention text copied into: {offenders}"


class TestSharedFewShotScenarios:
    """Fourth shared component. The three worked examples used to exist in S2
    and S3 only, with different third scenarios, and in no S4 node -- a
    confounder on FF3 that the convention discipline had not covered because
    examples were never treated as a shared component. Now the scenarios
    (question + answer) are one constant and every LLM call sees exactly
    three of them; only the step lines between question and answer differ,
    because that is the architecture."""

    @staticmethod
    def _every_prompt() -> dict[str, str]:
        from src.systems.long_context.prompt import build_system_prompt
        from src.systems.multi_agent.prompts import SUPERVISOR_PROMPT

        prompts = dict(_answer_emitting_prompts())
        prompts["S3"] = build_system_prompt([])
        # The supervisor emits no answer, but it is the node that decides the
        # tool pattern for S4 -- the thing the S2/S3 examples demonstrate.
        prompts["S4.supervisor"] = SUPERVISOR_PROMPT
        return prompts

    def test_every_scenario_is_verbatim_in_every_prompt(self):
        from src.common.few_shot_examples import FEW_SHOT_SCENARIOS

        for name, text in self._every_prompt().items():
            for scenario in FEW_SHOT_SCENARIOS:
                assert scenario.question in text, f"{name} lost the question of '{scenario.key}'"
                assert scenario.answer in text, f"{name} lost the answer of '{scenario.key}'"

    def test_every_llm_call_sees_exactly_three_examples(self):
        for name, text in self._every_prompt().items():
            assert text.count("### Example ") == 3, f"{name} shows {text.count('### Example ')} examples"

    def test_scenarios_carry_no_curly_braces(self):
        """S1 renders through ChatPromptTemplate, S4's supervisor and specialist
        through str.format. A brace would break some systems and not others."""
        from src.common.few_shot_examples import FEW_SHOT_SCENARIOS

        for scenario in FEW_SHOT_SCENARIOS:
            assert "{" not in scenario.question + scenario.answer
            assert "}" not in scenario.question + scenario.answer

    def test_every_templated_prompt_still_renders(self):
        from src.systems.multi_agent.prompts import (
            SPECIALIST_PROMPT_TEMPLATE,
            SUPERVISOR_PROMPT,
            SYNTHESIZER_PROMPT,
        )

        assert "### Example 1" in s1.RAG_PROMPT.format(context="ctx", question="q")
        assert "### Example 1" in SUPERVISOR_PROMPT.format(list_filings_output="meta")
        assert "### Example 1" in SPECIALIST_PROMPT_TEMPLATE.format(
            tickers="AAPL", sections="MD&A", inlined_filing_sections="..."
        )
        assert "### Example 1" in SYNTHESIZER_PROMPT.format(
            user_query="q", formatted_specialist_outputs="x"
        )

    def test_no_system_holds_a_private_copy_of_a_scenario(self):
        import re
        from pathlib import Path

        from src.common.few_shot_examples import PRIVATE_COPY_MARKER

        src_dir = Path(__file__).resolve().parents[2] / "src"
        offenders = [
            str(p.relative_to(src_dir))
            for p in src_dir.rglob("*.py")
            if p.name != "few_shot_examples.py"
            and re.search(PRIVATE_COPY_MARKER, p.read_text(encoding="utf-8"))
        ]
        assert not offenders, f"scenario text copied into: {offenders}"


class TestSharedCorpusCoverage:
    """The corpus description -- which fiscal years a filing reports -- must
    reach every system, because the judge is given the same fact
    (custom_evaluator._corpus_scope_sentence). Before 2026-09-09 only the
    judge had it, and all four systems declined FY2023 questions as out of
    corpus while the judge scored them as answerable."""

    @staticmethod
    def _filing(ticker="AAPL", fy="2024"):
        from src.common.ingestion import FilingMetadata, ProcessedFiling

        return ProcessedFiling(
            metadata=FilingMetadata(
                ticker=ticker, company_name=f"{ticker} Inc.", cik="1",
                filing_date=f"{fy}-11-01", accession_number="x",
                fiscal_year_end=f"{fy}-09-28",
            ),
            sections={"Financial Statements": "body"},
            full_text="body",
        )

    def test_the_note_reaches_every_prompt_through_the_convention(self):
        from src.common.corpus_coverage import COMPARATIVE_COLUMNS_NOTE

        assert COMPARATIVE_COLUMNS_NOTE in NON_ANSWERABILITY_CONVENTION
        for name, text in _answer_emitting_prompts().items():
            assert COMPARATIVE_COLUMNS_NOTE in text, f"{name} lacks the coverage note"

    def test_inlined_filings_carry_the_same_coverage_line_as_list_filings(self):
        from src.common.corpus_coverage import describe_coverage
        from src.common.tools.list_filings import create_list_filings_tool
        from src.systems.long_context.prompt import _format_single_filing
        from src.systems.multi_agent.prompts import _format_single_filing_filtered

        line = f"Coverage: {describe_coverage(2024)}"
        filing = self._filing()
        assert line in create_list_filings_tool([filing]).invoke({})
        assert line in _format_single_filing(filing)
        assert line in _format_single_filing_filtered(filing, ["Financial Statements"])

    def test_the_monolith_is_told_what_its_chunk_prefix_means(self):
        """S1 has no list_filings; its equivalent is the prefix explanation."""
        text = str(s1.RAG_PROMPT.messages[0].prompt.template)
        assert "comparative columns" in text
