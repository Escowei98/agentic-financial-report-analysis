"""Parsing of the "## Reasoning" block the four systems emit.

The stakes here are higher than they look. The parser decides how many steps
a chain has, which is the denominator of two of the three reasoning metrics,
and it decides what the OTHER evaluators get to see -- an answer with the
block still attached would hand citation_accuracy a second copy of every
citation.
"""

import pytest

from src.evaluation.reasoning_chain_parser import (
    EVIDENTIAL,
    INFERENTIAL,
    parse_chain,
    split_answer,
)

ANSWER = "Apple's revenue rose to $391,035M. (AAPL, FY2024, Income Statement)"

CHAIN = """## Reasoning
1. [E] Total net sales for FY2020 were $274,515M. (AAPL, FY2020, Income Statement)
2. [E] Total net sales for FY2024 were $391,035M. (AAPL, FY2024, Income Statement)
3. [I] Sales therefore rose over the period.
"""


class TestSplitting:
    def test_the_block_is_removed_from_the_answer(self):
        clean, block = split_answer(f"{ANSWER}\n\n{CHAIN}")
        assert clean == ANSWER
        assert "Total net sales" in block

    def test_an_answer_without_a_block_is_returned_unchanged(self):
        clean, block = split_answer(ANSWER)
        assert clean == ANSWER
        assert block == ""

    def test_the_last_header_wins(self):
        """The convention's own example contains "## Reasoning".

        A system echoing its instructions back would otherwise have its answer
        truncated at the echo, losing the real answer entirely.
        """
        echoed = f"I was told to write:\n## Reasoning\n(example)\n\n{ANSWER}\n\n{CHAIN}"
        clean, _ = split_answer(echoed)
        assert ANSWER in clean

    def test_empty_answer_is_handled(self):
        assert split_answer("") == ("", "")


class TestStepTyping:
    def test_tags_are_read_and_loci_attached_to_evidential_steps_only(self):
        chain = parse_chain(f"{ANSWER}\n\n{CHAIN}")
        assert chain.chain_emitted
        assert [s.step_type for s in chain.steps] == [EVIDENTIAL, EVIDENTIAL, INFERENTIAL]
        assert chain.steps[0].loci[0].doc_id == "AAPL_2020"
        assert chain.steps[2].loci == ()

    def test_a_wrapped_step_stays_one_step(self):
        wrapped = (
            "## Reasoning\n"
            "1. [E] Total net sales for FY2024 were $391,035M.\n"
            "   (AAPL, FY2024, Income Statement)\n"
            "2. [I] That is more than in FY2020.\n"
        )
        chain = parse_chain(wrapped)
        assert len(chain.steps) == 2
        assert chain.steps[0].loci[0].doc_id == "AAPL_2024"

    @pytest.mark.parametrize("body", ["(E) a claim", "E: a claim", "[e] a claim"])
    def test_tag_punctuation_variants_are_tolerated(self, body):
        chain = parse_chain(f"## Reasoning\n1. {body} (AAPL, FY2024, Income Statement)\n")
        assert chain.steps[0].step_type == EVIDENTIAL
        assert chain.steps[0].type_was_tagged

    def test_an_untagged_step_is_typed_by_whether_it_cites(self):
        """Deterministic fallback, identical for all four systems.

        Discarding untagged steps instead would silently shrink the
        denominator for whichever architecture is sloppier about the format,
        which is the opposite of what the metric is for.
        """
        chain = parse_chain(
            "## Reasoning\n"
            "1. A claim with a source. (AAPL, FY2024, Income Statement)\n"
            "2. A conclusion with none.\n"
        )
        assert [s.step_type for s in chain.steps] == [EVIDENTIAL, INFERENTIAL]
        assert chain.untagged_count == 2

    def test_duplicate_loci_in_one_step_are_collapsed(self):
        chain = parse_chain(
            "## Reasoning\n"
            "1. [E] Twice cited. (AAPL, FY2024, Income Statement) "
            "(AAPL, FY2024, Income Statement)\n"
        )
        assert len(chain.steps[0].loci) == 1


class TestMissingChain:
    def test_no_block_means_no_chain(self):
        chain = parse_chain(ANSWER)
        assert not chain.chain_emitted
        assert chain.steps == []

    def test_a_header_with_no_numbered_step_is_not_a_chain(self):
        """Scoring an empty chain would hand the system a vacuous 1.0."""
        chain = parse_chain(f"{ANSWER}\n\n## Reasoning\nI looked it up.\n")
        assert not chain.chain_emitted
