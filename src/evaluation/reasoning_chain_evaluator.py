"""
Reasoning Quality Evaluator (chain-based, binary per unit).

Replaces the two-tier Core/Agentic judge retired on 2026-09-08. That judge
scored five 1-5 dimensions off each architecture's own trace format and failed
human validation on every one of them -- weighted kappa 0.03 to 0.17, and,
decisively, a per-system spread of up to 2.75 scale points. A metric that
marks one architecture down by half the scale cannot carry a comparison
between architectures.

Three dimensions, all binary at the unit of judgement:

  1. Groundedness  -- per evidential step, against the passages at the locus
                      the step itself cites (see evidence_store.py).
  2. Validity      -- per transition, does step n follow from steps 1..n-1.
  3. Completeness  -- per required sub-question of the gold standard's
                      reference decomposition, is it covered by the chain.

Nothing here is scored 1-5. The unanchored five-point scale is the single
largest identified contributor to the disagreement in the retired instrument:
raters and judge were not disagreeing about the facts of a chain so much as
about where on a scale a given flaw belonged.

Population: the 120 answerable gold-standard items. FA-Refusal items have no
meaningful reference decomposition -- the correct chain there is "the premise
cannot be checked" -- and keep refusal_accuracy / refusal_quality /
over_refusal instead.

See docs/decisions/REASONING_QUALITY_SPEC.md sections 6-10.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from src.common.llm_client import get_judge_llm
from src.evaluation.evidence_store import EvidenceStore
from src.evaluation.reasoning_chain_parser import (
    EVIDENTIAL,
    ParsedChain,
    ReasoningStep,
    parse_chain,
)

logger = logging.getLogger(__name__)

# Groundedness violation codes (spec section 6.3).
B_FABRICATION = "B1"
B_DISTORTION = "B2"
B_OVERGENERALISATION = "B3"
B_PARAMETRIC = "B4"
B_NO_LOCUS = "B5"
B_LOCUS_NOT_IN_CORPUS = "B6"

# Validity violation codes (spec section 7.2).
V_UNSTATED_PREMISE = "V1"
VALIDITY_CODES = ("V1", "V2", "V3", "V4", "V5")
GROUNDEDNESS_CODES = (
    B_FABRICATION, B_DISTORTION, B_OVERGENERALISATION,
    B_PARAMETRIC, B_NO_LOCUS, B_LOCUS_NOT_IN_CORPUS,
)


# ---------------------------------------------------------------------------
#  Result types
# ---------------------------------------------------------------------------

@dataclass
class ChainReasoningScores:
    """Reasoning scores for one query x one system."""

    query_id: int = 0
    system_name: str = ""
    chain_emitted: bool = False

    num_steps: int = 0
    num_evidential: int = 0
    num_inferential: int = 0
    num_untagged: int = 0

    groundedness: float | None = None
    validity: float | None = None
    completeness: float | None = None

    # Weakest-link aggregation (spec section 10.2). For binary per-unit
    # labels the minimum over a chain is all-or-nothing, so it is recorded as
    # a flag per chain and averaged into a share at run level.
    fully_grounded: bool | None = None
    fully_valid: bool | None = None

    steps: list[dict] = field(default_factory=list)
    """The chain itself, as parsed.

    Carried because nothing else preserves it: `eval_runner` stores the
    answer with the chain block already cut off (every other evaluator must
    see it that way), so without this the run output would hold verdicts
    about steps whose text is gone. That breaks three things at once -- the
    human-validation page has nothing to show a rater, the run is not
    reproducible from its own output, and no example chain can be quoted in
    the appendix.
    """

    step_verdicts: list[dict] = field(default_factory=list)
    transition_verdicts: list[dict] = field(default_factory=list)
    subquestion_verdicts: list[dict] = field(default_factory=list)

    evidence_shown: dict[int, list[str]] = field(default_factory=dict)
    """Passages the judge saw, per step. Populated only when `keep_evidence`.

    Needed by the human-validation sample and by nothing else. A rater who is
    shown different passages than the judge is not validating the judge, they
    are doing a different task -- so the evidence has to travel with the
    verdict rather than be re-fetched later against a store that may have
    been rebuilt. Off by default because it multiplies the size of a run's
    JSON several times over.
    """

    def violation_counts(self) -> dict[str, int]:
        counts = {code: 0 for code in GROUNDEDNESS_CODES + VALIDITY_CODES}
        for verdict in self.step_verdicts:
            code = verdict.get("code")
            if code in counts:
                counts[code] += 1
        for verdict in self.transition_verdicts:
            code = verdict.get("code")
            if code in counts:
                counts[code] += 1
        return counts

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "system_name": self.system_name,
            "chain_emitted": self.chain_emitted,
            "num_steps": self.num_steps,
            "num_evidential": self.num_evidential,
            "num_inferential": self.num_inferential,
            "num_untagged": self.num_untagged,
            "groundedness": _round_or_none(self.groundedness),
            "validity": _round_or_none(self.validity),
            "completeness": _round_or_none(self.completeness),
            "fully_grounded": self.fully_grounded,
            "fully_valid": self.fully_valid,
            "violation_counts": self.violation_counts(),
            "steps": self.steps,
            "evidence_shown": {str(k): v for k, v in self.evidence_shown.items()},
            "step_verdicts": self.step_verdicts,
            "transition_verdicts": self.transition_verdicts,
            "subquestion_verdicts": self.subquestion_verdicts,
        }


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


# ---------------------------------------------------------------------------
#  Judge prompts
# ---------------------------------------------------------------------------

_SHARED_PREAMBLE = """\
You are evaluating the reasoning chain a financial-analysis system produced to
support its answer. You are NOT evaluating whether the answer is correct --
that is measured separately, and you are deliberately not shown a reference
answer. A chain that reaches a wrong figure by sound steps scores well here; a
chain that reaches the right figure through a broken step does not.

Answer with a verdict for every unit you are asked about. Never skip one.
"""

GROUNDEDNESS_VALIDITY_PROMPT = _SHARED_PREAMBLE + """
You judge two independent things about the chain below.

## A. Groundedness (evidential steps only)

For each EVIDENTIAL step you are given the passages found at the source the
step itself cites. Decide whether those passages carry the step's factual
claim.

For every step you rule GROUNDED, list in `supporting_values` the figures YOU
FOUND IN THE PASSAGE that carry the claim, written as the passage writes them.
They are checked against the passages automatically, so copy rather than
reconstruct: a figure that does not occur there fails the step.

Watch for the derived quantity. An operating margin, an effective tax rate, a
growth rate or a share of revenue is usually NOT stated in a 10-K -- the
filing gives the inputs and the ratio is computed from them. If the passage
holds only the inputs, the step asserting the ratio is NOT grounded, however
correct the arithmetic. Do not list a figure you calculated yourself.

Rules, all binding:
- GROUNDEDNESS IS NOT TRUTH. If the passage says something and the step
  reports it accurately, the step is grounded -- even if the passage is
  factually wrong. Errors in the filings are a data-quality matter, not a
  reasoning defect.
- GROUNDEDNESS IS NOT CITATION ACCURACY. If the passage carries the claim,
  the step is grounded -- even if you think a different section would have
  been the better source. Whether the right place was cited is measured by a
  separate metric.
- Arithmetic a step performs on figures from earlier steps is not a new
  factual claim about the filings. Judge only the claim, not the sum.

Violation codes, use exactly one when a step is NOT grounded:
- B1 fabrication: the claim, or its figure, is not in the passages at all
- B2 distortion: the passages say something related but materially different
- B3 overgeneralisation: the passages support a narrower claim than the step makes
- B4 parametric knowledge: the step adds a fact or a reason not in the passages

## A2. Steps marked [E] that cite no source

These are listed separately below. For each, decide what it actually is:

- It asserts something about WHAT THE FILINGS CONTAIN -- a figure, a fact, a
  statement attributed to a report -- but names no source. That is B5, not
  grounded.
- It describes the run rather than the filings: that data was not found, what
  the user asked, what a tool returned, a value the chain itself computed.
  It makes no claim about the filings' content, so there is nothing to ground.
  Set `"not_a_claim": true`. It is then no groundedness unit at all -- neither
  passed nor failed.

Judge what the step says, not how it is tagged: the tag is the system's own
label and this section exists because it is sometimes wrong.

## B. Validity (transitions into inferential steps)

Judge the transition into each step marked [I] -- a step that draws a
conclusion. Decide whether it follows from the steps before it.

Do NOT judge a transition into a step marked [E]. Such a step introduces a new
fact from the filings; it is a PREMISE, and a premise does not follow from
anything. Whether it is warranted is measured under Groundedness, above.

ONE EXCEPTION: if an [E] step contradicts, or silently sets aside, something an
earlier step of this chain asserted, report it here as V5. That is a defect in
the chain's coherence, not in the sourcing of the step, so Groundedness cannot
catch it. Report an [E] step ONLY in that case, never to confirm one.

FOR EVERY TRANSITION, FIRST DO THIS BOOKKEEPING, THEN RULE.

List in `values_used` every figure the step RELIES ON AS AN INPUT. For each,
name the earlier step that states it. Any input you cannot trace to an earlier
step goes in `unsupported_values` -- including one you take to be obvious,
assumed, or true in the world. "Assumed" is precisely what belongs there.

INPUTS ONLY. A figure the step itself derives from those inputs is its OUTPUT,
not an input, and never belongs in `unsupported_values` -- otherwise every
calculation would be invalid, since a result is by definition not stated
beforehand. Same for a value the step computes and then carries forward.

USED, NOT MERELY NAMED. A step relies on a quantity when it asserts or
computes a value with it. Naming one is not relying on it. These steps rely on
NOTHING and are valid as they stand -- `values_used` and `unsupported_values`
are both empty for them:

    "Operating Margin is calculated as Operating Income divided by Total Net
     Sales."                                            -- a definition
    "To compute revenue growth, the revenue of both years is required."
                                                        -- a requirement
    "Operating income is not the same as revenue."      -- a distinction
    "The filings do not report Amazon's net income."    -- an absence

Each names quantities it does not use. Marking them V1 because the named
figure is not in an earlier step is a category error: the step makes no claim
that could rest on it. Compare the genuine case, where the value IS used --
"Comparing the two rates, 18% is higher than 16.4%" asserts something about
18%, so 18% must come from somewhere.

Worked example:
    1. [E] Alphabet's effective tax rate for 2024 was 16.4%.
    2. [I] Comparing the two rates, 18% is higher than 16.4%.
    -> values_used ["16.4% (step 1)", "18% (nowhere)"],
       unsupported_values ["18%"]
Do not write that such a step "follows from the previous steps": the figure is
not in them.

Rules, all binding:
- VALIDITY IS INDEPENDENT OF THE TRUTH OF THE PREMISES. A correct inference
  from a false premise is valid. This is what keeps the two dimensions apart.
  Note the difference from the check above: a premise that is stated but wrong
  is fine here; a premise that is never stated at all is V1.
- A MISSING CONTENT IS NOT A VALIDITY DEFECT. If the chain never mentions
  something it should have covered, the remaining steps can still follow from
  one another cleanly. That is a completeness gap, measured elsewhere. Only a
  missing INFERENTIAL LINK is a validity defect.
- Extra steps, restatements and verbosity are not validity defects.

Violation codes, use exactly one when a transition is NOT valid:
- V1 unstated or missing premise: the conclusion may be true, but it does not
  follow from what the chain has actually said
- V2 negation or quantifier error
- V3 correlation treated as causation
- V4 circularity: the step assumes its own conclusion
- V5 a contrary statement made in an EARLIER STEP OF THIS CHAIN is silently
  passed over

The V1 case is the most important one in this instrument. Example: step 1
states a figure for FY2024 and step 2 concludes "revenue therefore rose". The
conclusion may well be right, but nothing in the chain says what it rose from.

## Question
{question}

## Chain
{chain}

## Evidence retrieved at the cited sources
{evidence}

## Steps marked [E] that cite no source
{unsourced}

## Output
Respond with ONLY a JSON object, no markdown:
{{
  "groundedness": [
    {{"step": <int>, "grounded": true|false, "code": "B1"|"B2"|"B3"|"B4"|null,
      "supporting_values": ["<figure as written in the passage>"],
      "rationale": "<one sentence>"}}
  ],
  "unsourced": [
    {{"step": <int>, "not_a_claim": true|false, "rationale": "<one sentence>"}}
  ],
  "validity": [
    {{"step": <int>,
      "values_used": ["<figure> (step <n>)" or "<figure> (nowhere)"],
      "unsupported_values": ["<figure>"],
      "valid": true|false, "code": "V1"|..."V5"|null,
      "rationale": "<one sentence>"}}
  ]
}}
Include one groundedness entry per evidential step listed under "Evidence",
and one `unsourced` entry per step listed under A2.
Include one validity entry per [I] step other than the first step of the
chain, plus an entry for any [E] step that triggers the V5 exception.
"""

COMPLETENESS_PROMPT = _SHARED_PREAMBLE + """
You judge coverage only. For each required sub-question below, decide whether
the chain recognisably deals with it.

Rules, all binding:
- STEPS BEYOND THE REQUIRED SUB-QUESTIONS ARE NEITHER REWARDED NOR PUNISHED.
  Additional steps, repetitions, intermediate summaries and length in general
  have no bearing on this judgement. A long chain and a short chain that cover
  the same sub-questions score identically.
- A sub-question counts as covered if the chain deals with it recognisably, in
  however many steps, and whether or not it reaches the right value. Coverage
  is not correctness.
- Do not invent further requirements. The list below is exhaustive.

## Question
{question}

## Required sub-questions
{subquestions}

## Chain
{chain}

## Output
Respond with ONLY a JSON object, no markdown:
{{
  "coverage": [
    {{"subquestion": <int>, "covered": true|false, "rationale": "<one sentence>"}}
  ]
}}
Include exactly one entry per required sub-question, numbered as listed.
"""


# ---------------------------------------------------------------------------
#  JSON extraction
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Pull a JSON object out of judge output, tolerating fences and prose."""
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    try:
        parsed: dict = json.loads(cleaned)
        return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            return parsed
        except json.JSONDecodeError:
            pass
    logger.warning("Failed to parse judge JSON: %s", (text or "")[:200])
    return {}


# ---------------------------------------------------------------------------
#  Prompt rendering
# ---------------------------------------------------------------------------

def _render_chain(steps: list[ReasoningStep]) -> str:
    lines = []
    for step in steps:
        tag = "E" if step.step_type == EVIDENTIAL else "I"
        lines.append(f"{step.step_index}. [{tag}] {step.text}")
    return "\n".join(lines)


def _render_evidence(evidence_by_step: dict[int, list[str]]) -> str:
    if not evidence_by_step:
        return "(no evidential step in this chain has a checkable source)"
    blocks = []
    for step_index in sorted(evidence_by_step):
        passages = evidence_by_step[step_index]
        body = "\n".join(f"  - {p.strip()}" for p in passages) or "  (none found)"
        blocks.append(f"Step {step_index}:\n{body}")
    return "\n\n".join(blocks)


def _render_unsourced(steps: list[ReasoningStep]) -> str:
    if not steps:
        return "(none)"
    return "\n".join(f"Step {s.step_index}: {s.text}" for s in steps)


def _render_subquestions(subquestions: list[str]) -> str:
    return "\n".join(f"{i}. {q}" for i, q in enumerate(subquestions, start=1))


# ---------------------------------------------------------------------------
#  Deterministic pre-pass
# ---------------------------------------------------------------------------

def _prepare_groundedness(
    chain: ParsedChain,
    store: EvidenceStore,
) -> tuple[list[dict], dict[int, list[str]], list[ReasoningStep]]:
    """Settle what the judge does not need to decide.

    B5 (an evidential step citing nothing) and B6 (a step citing a filing the
    corpus does not contain) are mechanical facts. Deciding them here keeps
    them out of the judge's discretion and off the token bill, and leaves the
    judge only the question it is actually needed for: does this passage carry
    this claim.
    """
    settled: list[dict] = []
    evidence_by_step: dict[int, list[str]] = {}
    unsourced: list[ReasoningStep] = []

    for step in chain.evidential_steps:
        if not step.loci:
            # NOT auto-B5 any more. In the pilot this rule produced 7 of the 8
            # false alarms on groundedness: it fired on steps that make no
            # claim about the filings at all -- "No context is provided for
            # Microsoft", "The user is asking for the Total Revenue of GOOGL",
            # "The calculated growth is 65.247%. (calculate)". The rater called
            # those grounded, correctly: there is nothing in them to ground.
            # The judge now decides whether the step asserts filing content
            # (B5) or merely describes the run (no unit at all).
            unsourced.append(step)
            continue

        passages: list[str] = []
        missing_docs = []
        for locus in step.loci:
            result = store.lookup(locus.doc_id, locus.section_text, step.text)
            if not result.doc_exists:
                missing_docs.append(locus.doc_id)
                continue
            passages.extend(result.passages)

        if missing_docs and not passages:
            settled.append({
                "step": step.step_index, "grounded": False,
                "code": B_LOCUS_NOT_IN_CORPUS,
                "rationale": f"Cited filing(s) not in corpus: {', '.join(missing_docs)}.",
                "method": "deterministic",
            })
            continue

        evidence_by_step[step.step_index] = passages

    return settled, evidence_by_step, unsourced


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------

def evaluate_reasoning_chain(
    question: str,
    answer: str,
    reference_decomposition: list[str],
    store: EvidenceStore,
    system_name: str = "",
    query_id: int = 0,
    parsed: ParsedChain | None = None,
    keep_evidence: bool = False,
) -> ChainReasoningScores:
    """Score one answer's reasoning chain on the three dimensions."""
    chain = parsed if parsed is not None else parse_chain(answer)

    scores = ChainReasoningScores(
        query_id=query_id,
        system_name=system_name,
        chain_emitted=chain.chain_emitted,
        num_steps=len(chain.steps),
        num_evidential=len(chain.evidential_steps),
        num_inferential=len(chain.steps) - len(chain.evidential_steps),
        num_untagged=chain.untagged_count,
        steps=[step.to_dict() for step in chain.steps],
    )
    if not chain.chain_emitted:
        # All three stay None. A missing chain is a format failure, and
        # scoring it 0.0 would let it sink the system's mean as if it were a
        # reasoning defect. It is counted in chain_emission_rate instead.
        logger.info("[%s] id=%d: no reasoning chain emitted", system_name, query_id)
        return scores

    judge = get_judge_llm(max_tokens=4096)

    # --- Dimensions 1 + 2, one call ---
    settled, evidence_by_step, unsourced_steps = _prepare_groundedness(chain, store)
    step_verdicts = list(settled)
    transition_verdicts: list[dict] = []

    needs_judge = bool(evidence_by_step) or unsourced_steps or len(chain.steps) > 1
    if needs_judge:
        prompt = GROUNDEDNESS_VALIDITY_PROMPT.format(
            question=question,
            chain=_render_chain(chain.steps),
            evidence=_render_evidence(evidence_by_step),
            unsourced=_render_unsourced(unsourced_steps),
        )
        data = _extract_json(_to_str(judge.invoke(prompt).content))

        judged_steps = {
            int(entry["step"]): entry
            for entry in data.get("groundedness", [])
            if isinstance(entry, dict) and str(entry.get("step", "")).isdigit()
        }
        for step_index in sorted(evidence_by_step):
            entry = judged_steps.get(step_index)
            if entry is None:
                logger.warning(
                    "[%s] id=%d: judge returned no groundedness verdict for step %d",
                    system_name, query_id, step_index,
                )
                continue
            # A "grounded" verdict stands only if its quote is really in the
            # passages. Checked, not trusted: in the pilot the judge confirmed
            # seven steps the rater rejected, asserting each time that the
            # passage "contains" a derived ratio no 10-K states.
            grounded = bool(entry.get("grounded"))
            missing = _unsupported_figures(
                entry.get("supporting_values"), evidence_by_step.get(step_index, [])
            ) if grounded else []
            verdict: dict = {
                "step": step_index,
                "grounded": grounded and not missing,
                "rationale": str(entry.get("rationale", "")),
                "method": "judge",
            }
            if missing:
                verdict["code"] = B_FABRICATION
                verdict["figures_not_in_passage"] = missing
                verdict["overruled_judge"] = True
            elif not grounded:
                verdict["code"] = _clean_code(entry.get("code"), GROUNDEDNESS_CODES)
            else:
                verdict["code"] = None
            step_verdicts.append(verdict)

        # Steps marked [E] with no source: B5 only where the judge says the
        # step actually asserts filing content. One that merely describes the
        # run is dropped -- it is not a groundedness unit.
        unsourced_by_index = {
            int(e["step"]): e for e in data.get("unsourced", [])
            if isinstance(e, dict) and str(e.get("step", "")).isdigit()
        }
        for step in unsourced_steps:
            entry = unsourced_by_index.get(step.step_index)
            if entry is not None and entry.get("not_a_claim"):
                continue
            step_verdicts.append({
                "step": step.step_index,
                "grounded": False,
                "code": B_NO_LOCUS,
                "rationale": str((entry or {}).get(
                    "rationale", "Asserts filing content without naming a source.")),
                "method": "judge",
            })

        judged_transitions = {
            int(entry["step"]): entry
            for entry in data.get("validity", [])
            if isinstance(entry, dict) and str(entry.get("step", "")).isdigit()
        }
        # Validity is assessed on transitions INTO INFERENTIAL STEPS only.
        #
        # The pilot of 2026-09-08 settled this empirically: the judge returned
        # a verdict for 81 of 81 transitions into an inferential step and for
        # only 15 of 63 into an evidential one, without ever being told to
        # make that distinction. It had converged on a rule that holds -- a
        # step introducing a new fact from the filings is a PREMISE, and a
        # premise does not follow from anything. The human rater did the same.
        # Counting those transitions as trivially valid would have filled the
        # denominator with units that cannot discriminate, and their share
        # differs per architecture (S1 ran 29 evidential to 28 inferential
        # steps, S4 19 to 13), so it would have differed per architecture too.
        #
        # The V5 exception keeps the dimension honest: an evidential step that
        # contradicts an earlier one is a coherence defect that groundedness
        # cannot see, so the judge may return it and it counts.
        # Excluded is the step that genuinely introduces filing evidence: type
        # [E] AND a parseable locus. A step tagged [E] that cites nothing
        # checkable is not a premise from the corpus -- it is the system
        # narrating ("The user is asking for...") or carrying a computed
        # result ("The calculated growth is 65.247%. (calculate)"), and
        # whether that follows is a fair question.
        #
        # The sensitivity test of 2026-09-09 found this hole the hard way: in
        # V022 the deleted premise left its damage on a step tagged [E] whose
        # source was "(calculate)", so no assessable transition existed and
        # the defect was invisible to the rater. 7% of evidential steps in the
        # pilot carry no locus.
        required = [
            step.step_index for step in chain.steps[1:]
            if not (step.step_type == EVIDENTIAL and step.loci)
        ]
        for step_index in required:
            entry = judged_transitions.get(step_index)
            if entry is None:
                logger.warning(
                    "[%s] id=%d: judge returned no validity verdict for transition %d",
                    system_name, query_id, step_index,
                )
                continue
            # A figure the judge could not trace to an earlier step settles
            # the verdict mechanically, whatever the judge then concluded.
            #
            # Not defensive coding -- it is the fix for a measured defect. In
            # the sensitivity test of 2026-09-09 the judge caught 3 of 12
            # planted V1 defects, and its own rationales showed why: on one it
            # wrote that the step follows "from the figures provided in step 1
            # and the Total Net Sales, which is assumed". It SAW the gap and
            # ruled valid anyway. Asking it to do the bookkeeping and applying
            # the rule here moves the decision off its judgement and onto what
            # it extracted.
            unsupported = [
                str(v) for v in (entry.get("unsupported_values") or []) if str(v).strip()
            ]
            valid = bool(entry.get("valid")) and not unsupported
            transition: dict = {
                "step": step_index,
                "valid": valid,
                "code": None if valid else (
                    _clean_code(entry.get("code"), VALIDITY_CODES) or V_UNSTATED_PREMISE
                ),
                "rationale": str(entry.get("rationale", "")),
            }
            if unsupported:
                transition["unsupported_values"] = unsupported
                transition["overruled_judge"] = bool(entry.get("valid"))
            transition_verdicts.append(transition)

        # V5 on an evidential step: counted when raised, never when absent.
        # An unflagged premise is not evidence of coherence, it is simply not
        # a unit -- so confirming entries are dropped rather than scored 1.
        for step_index, entry in judged_transitions.items():
            if step_index in required or step_index not in {s.step_index for s in chain.steps}:
                continue
            if entry.get("valid"):
                continue
            transition_verdicts.append({
                "step": step_index,
                "valid": False,
                "code": _clean_code(entry.get("code"), VALIDITY_CODES),
                "rationale": str(entry.get("rationale", "")),
                "premise_contradiction": True,
            })
        transition_verdicts.sort(key=lambda v: v["step"])

    if keep_evidence:
        scores.evidence_shown = dict(evidence_by_step)
    scores.step_verdicts = sorted(step_verdicts, key=lambda v: v["step"])
    scores.transition_verdicts = transition_verdicts

    if scores.step_verdicts:
        grounded_n = sum(1 for v in scores.step_verdicts if v["grounded"])
        scores.groundedness = grounded_n / len(scores.step_verdicts)
        scores.fully_grounded = grounded_n == len(scores.step_verdicts)
    if transition_verdicts:
        valid_n = sum(1 for v in transition_verdicts if v["valid"])
        scores.validity = valid_n / len(transition_verdicts)
        scores.fully_valid = valid_n == len(transition_verdicts)
    # A chain with no inferential step has no assessable transition. Validity
    # stays None rather than 1.0: a chain with nothing to get wrong has not
    # demonstrated valid inference.

    # --- Dimension 3, separate call ---
    #
    # Separate so the coverage judgement is not tinted by the defects just
    # found. Asked in one call with the other two, a judge that has just
    # flagged a fabrication tends to mark the chain down here as well.
    if reference_decomposition:
        prompt = COMPLETENESS_PROMPT.format(
            question=question,
            subquestions=_render_subquestions(reference_decomposition),
            chain=_render_chain(chain.steps),
        )
        data = _extract_json(_to_str(judge.invoke(prompt).content))
        by_index = {
            int(entry["subquestion"]): entry
            for entry in data.get("coverage", [])
            if isinstance(entry, dict) and str(entry.get("subquestion", "")).isdigit()
        }
        verdicts = []
        for i, text in enumerate(reference_decomposition, start=1):
            entry = by_index.get(i, {})
            verdicts.append({
                "subquestion": i,
                "text": text,
                "covered": bool(entry.get("covered")),
                "rationale": str(entry.get("rationale", "")),
            })
        scores.subquestion_verdicts = verdicts
        if verdicts:
            scores.completeness = sum(1 for v in verdicts if v["covered"]) / len(verdicts)

    return scores


_WS_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Fold whitespace and case so a quote survives reformatting, nothing more."""
    return _WS_RE.sub(" ", str(text or "")).strip().lower()


_FIGURE_CHARS_RE = re.compile(r"[\s\xa0,$%]")


def _unsupported_figures(values: list | None, passages: list[str]) -> list[str]:
    """Which of the judge's cited figures do not occur in the passages?

    Figures, not prose. The first version of this check asked for a verbatim
    span and failed 36 steps the rater called grounded: 10-K evidence is
    tabular ("Research and development$31,370\xa05\xa0%$29,915"), so a
    sentence never appears in it word for word even when every number does.

    Only DISTINCTIVE figures are checked -- four or more digits, or a decimal
    point. A bare "18" occurs somewhere in almost any financial passage, so
    testing it would wave through exactly what the check is for while adding
    noise. That is a real limit, not a tuning choice: this catches the derived
    ratio that the filing never states (an operating margin of 24.15%), and it
    does not catch a figure that is present but attached to a different line
    item. The latter needs the passage read, not searched.
    """
    haystack = _FIGURE_CHARS_RE.sub("", " ".join(passages))
    missing = []
    for value in values or []:
        # Extract the NUMBER, do not normalise the whole string. The judge
        # writes "$450,256 million"; stripping punctuation alone leaves
        # "450256million", which matches nothing -- and 450,256 was in the
        # passage all along. That artefact accounted for most of the false
        # alarms the first figure-based version produced.
        for raw_number in re.findall(r"\d[\d.,]*", str(value)):
            token = _FIGURE_CHARS_RE.sub("", raw_number)
            digits = sum(c.isdigit() for c in token)
            if digits < 4 and "." not in token:
                continue
            if token not in haystack:
                missing.append(raw_number)
    return missing


def _clean_code(raw, allowed: tuple[str, ...]) -> str | None:
    code = str(raw or "").strip().upper()
    return code if code in allowed else None


def _to_str(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in value
        )
    return str(value)


def evaluate_reasoning_chain_batch(
    gold_items: list,
    answers: list[str],
    store: EvidenceStore,
    system_name: str,
    parsed_chains: list[ParsedChain] | None = None,
    keep_evidence: bool = False,
) -> list[ChainReasoningScores]:
    """Score a run's answers. Refusal items are skipped (spec section 8.4).

    `answers` are the RAW system answers, chain block included. Pass
    `parsed_chains` when the caller has already split them -- eval_runner does,
    because every other evaluator must see the answer without the block.
    """
    if len(gold_items) != len(answers):
        raise ValueError(
            f"Length mismatch: {len(gold_items)} gold items, {len(answers)} answers"
        )
    if parsed_chains is not None and len(parsed_chains) != len(answers):
        raise ValueError(
            f"Length mismatch: {len(parsed_chains)} parsed chains, "
            f"{len(answers)} answers"
        )

    results: list[ChainReasoningScores] = []
    for i, (item, answer) in enumerate(zip(gold_items, answers), start=1):
        if not item.expected_answerable:
            results.append(ChainReasoningScores(
                query_id=item.id, system_name=system_name, chain_emitted=False,
            ))
            continue
        logger.info(
            "[%s] Reasoning chain %d/%d (id=%d, type=%s)",
            system_name, i, len(gold_items), item.id, item.fa_type,
        )
        try:
            results.append(evaluate_reasoning_chain(
                question=item.question,
                answer=answer,
                reference_decomposition=getattr(item, "reference_decomposition", []) or [],
                store=store,
                system_name=system_name,
                query_id=item.id,
                parsed=None if parsed_chains is None else parsed_chains[i - 1],
                keep_evidence=keep_evidence,
            ))
        except Exception as exc:  # keep alignment with gold_items
            logger.error("[%s] id=%d reasoning eval failed: %s", system_name, item.id, exc)
            results.append(ChainReasoningScores(query_id=item.id, system_name=system_name))
    return results
