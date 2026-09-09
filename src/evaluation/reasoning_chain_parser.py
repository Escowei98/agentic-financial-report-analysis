"""
Parser for the "## Reasoning" block the four systems emit under
`src/common/reasoning_chain_convention.py`.

This is a parser, not a normaliser. It segments nothing and rewrites nothing:
the systems are asked for an already-segmented chain precisely so that no
second LLM sits between the answer and the metric, deciding where one step
ends and the next begins. Those decisions would land straight in the
denominator of `groundedness_run` and `validity_run`.

Two jobs:

1. `split_answer` cuts the block off the answer. Every other evaluator --
   exact_match, answer_recall, citation_accuracy, RAGAS, the refusal metrics --
   must see the answer WITHOUT it. The chain repeats the answer's figures and
   citations, so leaving it in would silently inflate the citation sets
   `citation_accuracy` counts and hand the recall judges a second copy of
   every number.
2. `parse_chain` turns the block into typed steps with their cited loci.

See docs/decisions/REASONING_QUALITY_SPEC.md section 4.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.evaluation.citation_evaluator import _CITATION_RE, _map_section_text

EVIDENTIAL = "evidential"
INFERENTIAL = "inferential"

# The header the convention asks for. Tolerant about the heading level and a
# trailing colon, strict about the word: the convention says "headed exactly",
# so a system that writes something else has not followed it, and that has to
# show up as a missing chain rather than be quietly repaired here.
_HEADER_RE = re.compile(r"^[ \t]*#{1,6}[ \t]*Reasoning[ \t]*:?[ \t]*$", re.IGNORECASE | re.MULTILINE)

# "1." / "1)" / "- 1." / "* 1." , with the step body following.
_STEP_RE = re.compile(r"^[ \t]*(?:[-*][ \t]*)?(\d{1,3})[.)][ \t]+(.*)$")

# "[E]" / "[I]", also tolerated with parentheses or as a bare prefix "E:".
_TYPE_RE = re.compile(r"^[ \t]*[\[(]?\s*([EI])\s*[\])]?\s*[:\-]?\s*", re.IGNORECASE)


@dataclass(frozen=True)
class Locus:
    """One cited place in the corpus: which filing, which section."""

    ticker: str
    fiscal_year: int
    section_text: str
    section_ids: frozenset

    @property
    def doc_id(self) -> str:
        return f"{self.ticker}_{self.fiscal_year}"

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "section_text": self.section_text,
            "section_ids": sorted(self.section_ids),
        }


@dataclass(frozen=True)
class ReasoningStep:
    step_index: int
    text: str
    step_type: str
    loci: tuple[Locus, ...] = ()
    type_was_tagged: bool = True

    def to_dict(self) -> dict:
        return {
            "step_index": self.step_index,
            "text": self.text,
            "type": self.step_type,
            "loci": [locus.to_dict() for locus in self.loci],
            "type_was_tagged": self.type_was_tagged,
        }


@dataclass
class ParsedChain:
    """Result of parsing one answer."""

    answer_without_chain: str
    chain_emitted: bool = False
    steps: list[ReasoningStep] = field(default_factory=list)
    raw_block: str = ""

    @property
    def evidential_steps(self) -> list[ReasoningStep]:
        return [s for s in self.steps if s.step_type == EVIDENTIAL]

    @property
    def untagged_count(self) -> int:
        return sum(1 for s in self.steps if not s.type_was_tagged)

    def to_dict(self) -> dict:
        return {
            "chain_emitted": self.chain_emitted,
            "num_steps": len(self.steps),
            "num_evidential": len(self.evidential_steps),
            "num_inferential": len(self.steps) - len(self.evidential_steps),
            "num_untagged": self.untagged_count,
            "steps": [s.to_dict() for s in self.steps],
        }


def split_answer(answer: str) -> tuple[str, str]:
    """Split an answer into (answer without the chain, the raw chain block).

    The LAST "## Reasoning" header wins. The convention's own worked example
    contains that header, and a system echoing its instructions back would
    otherwise have its answer truncated at the echo.
    """
    if not answer:
        return "", ""
    matches = list(_HEADER_RE.finditer(answer))
    if not matches:
        return answer, ""
    last = matches[-1]
    return answer[: last.start()].rstrip(), answer[last.end():].strip()


def _parse_loci(text: str) -> tuple[Locus, ...]:
    """Extract cited loci from one step, reusing the citation evaluator's regex."""
    loci = []
    seen = set()
    for m in _CITATION_RE.finditer(text):
        ticker = m.group("ticker").upper()
        year = int(m.group("year"))
        section_text = m.group("section").strip()
        key = (ticker, year, section_text.lower())
        if key in seen:
            continue
        seen.add(key)
        loci.append(Locus(
            ticker=ticker,
            fiscal_year=year,
            section_text=section_text,
            section_ids=_map_section_text(section_text),
        ))
    return tuple(loci)


def _iter_step_bodies(block: str):
    """Yield (number, body) per numbered step, joining wrapped continuation lines."""
    current_num: int | None = None
    current_lines: list[str] = []
    for line in block.split("\n"):
        m = _STEP_RE.match(line)
        if m:
            if current_num is not None:
                yield current_num, " ".join(current_lines).strip()
            current_num = int(m.group(1))
            current_lines = [m.group(2)]
        elif current_num is not None and line.strip():
            current_lines.append(line.strip())
    if current_num is not None:
        yield current_num, " ".join(current_lines).strip()


def parse_chain(answer: str) -> ParsedChain:
    """Parse the reasoning block out of one system answer."""
    clean, block = split_answer(answer)
    if not block:
        return ParsedChain(answer_without_chain=clean, chain_emitted=False)

    steps: list[ReasoningStep] = []
    for index, (_number, body) in enumerate(_iter_step_bodies(block), start=1):
        if not body:
            continue
        tag_match = _TYPE_RE.match(body)
        if tag_match:
            step_type = EVIDENTIAL if tag_match.group(1).upper() == "E" else INFERENTIAL
            text = body[tag_match.end():].strip()
            tagged = True
        else:
            # Untagged fallback, deterministic and identical for all four
            # systems: a step that cites a locus is making a statement about
            # the filings, one that cites none is not. It is a fallback, not a
            # classifier -- `num_untagged` is reported per system so a
            # architecture that leans on it is visible rather than silently
            # scored on inferred types.
            text = body
            step_type = EVIDENTIAL if _CITATION_RE.search(body) else INFERENTIAL
            tagged = False

        steps.append(ReasoningStep(
            step_index=index,
            text=text,
            step_type=step_type,
            loci=_parse_loci(text) if step_type == EVIDENTIAL else (),
            type_was_tagged=tagged,
        ))

    # A header with no parseable numbered step is not an emitted chain. Scoring
    # it as an empty chain would hand the system a vacuous 1.0 on groundedness.
    return ParsedChain(
        answer_without_chain=clean,
        chain_emitted=bool(steps),
        steps=steps,
        raw_block=block,
    )
