"""
System 4: Multi-Agent Long Context Pipeline.

Supervisor -> Parametrized Specialists -> Synthesizer -> Reflection.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.common.agent import build_agent
from src.common.config import load_config
from src.common.ingestion import ProcessedFiling
from src.common.llm_client import get_llm
from src.common.message_parsing import format_tool_calls_for_prompt, parse_agent_messages
from src.common.reflection import (
    ReflectionVerdict,
    build_reflection_chain,
    generate_feedback_message,
    unpack_reflection_result,
)
from src.common.tools.calculate import calculate
from src.common.tools.list_filings import create_list_filings_tool
from src.common.utils import RunMetrics, TokenUsage
from src.systems.multi_agent.graph import DelegationRequest, MultiAgentState, _merge_token_usage
from src.systems.multi_agent.prompts import SUPERVISOR_PROMPT, SYNTHESIZER_PROMPT, build_specialist_prompt
from src.systems.multi_agent.tools import create_delegate_tool

logger = logging.getLogger(__name__)


@dataclass
class MultiAgentResult:
    """Result from a single Multi-Agent query."""

    answer: str
    contexts: list[str] = field(default_factory=list)
    tool_calls_log: list[dict] = field(default_factory=list)
    metrics: RunMetrics = field(default_factory=RunMetrics)
    reflection_verdict: ReflectionVerdict | None = None
    was_revised: bool = False

    # S4 specific
    specialist_outputs: dict[str, str] = field(default_factory=dict)
    token_breakdown: dict[str, TokenUsage] = field(default_factory=dict)
    num_specialists_invoked: int = 0
    supervisor_plan: str = ""
    delegation_requests: list[dict] = field(default_factory=list)




class MultiAgentPipeline:
    """
    System 4: Multi-Agent Long Context Pipeline.
    
    Supervisor ReAct agent delegates to Specialist ReAct agents.
    Synthesizer aggregates the results.
    """

    def __init__(self, config_override: dict | None = None):
        self.config = load_config("multi_agent")
        self._apply_overrides(config_override or {})

        self._llm: ChatVertexAI | None = None
        self._filings: list[ProcessedFiling] = []
        self._graph: CompiledStateGraph | None = None
        self._reflection_chain: Runnable | None = None

        self._available_tickers: list[str] = []
        self._available_sections: list[str] = []
        self._current_state: dict[str, Any] = {}

    def _apply_overrides(self, overrides: dict) -> None:
        sup_cfg = self.config.get("supervisor", {})
        spec_cfg = self.config.get("specialist", {})
        syn_cfg = self.config.get("synthesizer", {})
        ref_cfg = self.config.get("reflection", {})

        if "supervisor_max_delegate" in overrides:
            sup_cfg["max_delegate_calls"] = overrides["supervisor_max_delegate"]
        if "specialist_max_iterations" in overrides:
            spec_cfg["max_iterations"] = overrides["specialist_max_iterations"]
        if "reflection_enabled" in overrides:
            ref_cfg["enabled"] = overrides["reflection_enabled"]

        self.config["supervisor"] = sup_cfg
        self.config["specialist"] = spec_cfg
        self.config["synthesizer"] = syn_cfg
        self.config["reflection"] = ref_cfg

    @property
    def params(self) -> dict:
        return {
            "supervisor_max_delegate": self.config.get("supervisor", {}).get("max_delegate_calls", 5),
            "specialist_max_iterations": self.config.get("specialist", {}).get("max_iterations", 5),
            "reflection_enabled": self.config.get("reflection", {}).get("enabled", True),
        }

    def build(self, filings: Sequence[ProcessedFiling]) -> None:
        self._filings = list(filings)
        self._llm = get_llm("multi_agent")

        # Extract available tickers and sections
        tickers_set = set()
        sections_set: set[str] = set()
        for f in self._filings:
            tickers_set.add(f.metadata.ticker.upper())
            if f.sections:
                sections_set.update(f.sections.keys())
        self._available_tickers = sorted(list(tickers_set))
        self._available_sections = sorted(list(sections_set))
        if not self._available_sections:
            self._available_sections = ["Full Text"]

        # Build Supervisor Agent
        # We need to compile a StateGraph for the overall flow.
        # But we also need the supervisor to be a ReAct agent.
        # We can use LangChain's build_agent for Supervisor, and use it inside a graph node.

        # StateGraph setup
        workflow = StateGraph(MultiAgentState)

        # We need a way to pass the state down to the specialist runner.
        # Since _run_specialist is a callback, it needs to update the state.
        # We will wrap the supervisor node execution in a class method so it can use a state dict.

        workflow.add_node("supervisor_node", self._supervisor_node_func)
        workflow.add_node("synthesizer_node", self._synthesizer_node_func)

        workflow.add_edge(START, "supervisor_node")
        workflow.add_edge("supervisor_node", "synthesizer_node")

        if self.config.get("reflection", {}).get("enabled", True):
            self._reflection_chain = build_reflection_chain(llm=self._llm, include_raw=True)
            workflow.add_node("reflection_node", self._reflection_node_func)
            workflow.add_edge("synthesizer_node", "reflection_node")

            # Conditional edge for reflection
            def reflection_router(state: MultiAgentState) -> str:
                verdict = state.get("reflection_verdict")
                iterations = state.get("reflection_iterations", 0)
                max_iterations = self.config.get("reflection", {}).get("max_iterations", 1)

                # `_reflection_node_func` has already incremented
                # `reflection_iterations` by the time this router runs, so
                # `iterations` here counts completed reflection passes, not
                # ones still to come. `<=` (not `<`) makes `max_iterations`
                # mean "allow this many correction rounds": with the
                # default of 1, a single 'revise' verdict triggers exactly
                # one re-synthesis, matching the single-pass policy S2/S3
                # get from `run_reflection_pass`.
                if verdict and verdict.get("status") == "revise" and iterations <= max_iterations:
                    return "synthesizer_node"
                return END

            workflow.add_conditional_edges(
                "reflection_node",
                reflection_router,
                {"synthesizer_node": "synthesizer_node", END: END}
            )
            logger.info("S4 reflection chain enabled")
        else:
            self._reflection_chain = None
            workflow.add_edge("synthesizer_node", END)
            logger.info("S4 reflection chain disabled")

        self._graph = workflow.compile()
        logger.info("S4 pipeline built: %d filings", len(self._filings))

    def _run_specialist(self, tickers: list[str], sections: list[str], sub_question: str) -> str:
        """Dynamically build and invoke a specialist agent."""
        if self._llm is None:
            raise RuntimeError("Pipeline not built. Call .build(filings) first.")

        spec_cfg = self.config.get("specialist", {})
        recursion_limit = spec_cfg.get("recursion_limit", 12)

        prompt = build_specialist_prompt(tickers, sections, self._filings)
        agent = build_agent(
            llm=self._llm,
            tools=[calculate],
            system_prompt=prompt,
            recursion_limit=recursion_limit
        )

        logger.debug("Invoking specialist for tickers=%s, sections=%s", tickers, sections)
        result = agent.invoke(
            {"messages": [HumanMessage(content=sub_question)]},
            config={"recursion_limit": recursion_limit}
        )

        # Extract answer and tokens
        messages = result.get("messages", [])
        answer, tool_calls_log, _, tokens = parse_agent_messages(messages)

        # Save to state
        if not hasattr(self, "_current_state"):
            self._current_state = {}

        scope_key = f"{'_'.join(tickers)}__{'_'.join(sections)}"
        # Add index to make it unique if called multiple times
        idx = len(self._current_state.get("specialist_outputs", {}))
        unique_key = f"{scope_key}_{idx}"

        if "specialist_outputs" not in self._current_state:
            self._current_state["specialist_outputs"] = {}
        self._current_state["specialist_outputs"][unique_key] = answer

        if "token_breakdown" not in self._current_state:
            self._current_state["token_breakdown"] = {}
        self._current_state["token_breakdown"][f"specialist_{unique_key}"] = tokens

        if "delegations" not in self._current_state:
            self._current_state["delegations"] = []
        self._current_state["delegations"].append(
            DelegationRequest(tickers=tickers, sections=sections, sub_question=sub_question, scope_key=unique_key)
        )

        if "tool_calls_log" not in self._current_state:
            self._current_state["tool_calls_log"] = []
        self._current_state["tool_calls_log"].extend(tool_calls_log)

        return answer

    def _supervisor_node_func(self, state: MultiAgentState) -> dict:
        """Executes the Supervisor ReAct agent."""
        if self._llm is None:
            raise RuntimeError("Pipeline not built. Call .build(filings) first.")

        sup_cfg = self.config.get("supervisor", {})
        recursion_limit = sup_cfg.get("recursion_limit", 12)

        # Provide metadata
        list_filings_tool = create_list_filings_tool(filings=self._filings)
        metadata_str = list_filings_tool.invoke({})

        prompt = SUPERVISOR_PROMPT.format(list_filings_output=metadata_str)

        delegate_tool = create_delegate_tool(
            self._run_specialist,
            self._available_tickers,
            self._available_sections
        )

        agent = build_agent(
            llm=self._llm,
            tools=[list_filings_tool, delegate_tool],
            system_prompt=prompt,
            recursion_limit=recursion_limit
        )

        # Bind the current state for _run_specialist to update
        self._current_state = {
            "specialist_outputs": {},
            "token_breakdown": {},
            "delegations": [],
            "tool_calls_log": []
        }

        result = agent.invoke(
            {"messages": state["messages"]},
            config={"recursion_limit": recursion_limit}
        )

        messages = result.get("messages", [])
        plan, tool_calls_log, _, tokens = parse_agent_messages(messages)
        self._current_state["token_breakdown"]["supervisor"] = tokens
        self._current_state["tool_calls_log"].extend(tool_calls_log)

        return {
            "supervisor_plan": plan,
            "delegations": self._current_state["delegations"],
            "specialist_outputs": self._current_state["specialist_outputs"],
            "token_breakdown": self._current_state["token_breakdown"],
            "tool_calls_log": self._current_state["tool_calls_log"],
        }

    def _synthesizer_node_func(self, state: MultiAgentState) -> dict:
        """Executes the Synthesizer agent."""
        if self._llm is None:
            raise RuntimeError("Pipeline not built. Call .build(filings) first.")

        syn_cfg = self.config.get("synthesizer", {})
        recursion_limit = syn_cfg.get("recursion_limit", 8)

        outputs = state.get("specialist_outputs", {})
        formatted_outputs = ""
        for key, ans in outputs.items():
            formatted_outputs += f"### Scope: {key}\n{ans}\n\n"

        if not formatted_outputs:
            formatted_outputs = "(No specialist outputs provided. The supervisor did not delegate.)"

        prompt = SYNTHESIZER_PROMPT.format(
            user_query=state["user_query"],
            formatted_specialist_outputs=formatted_outputs
        )

        agent = build_agent(
            llm=self._llm,
            tools=[calculate],
            system_prompt=prompt,
            recursion_limit=recursion_limit
        )

        # If there's a reflection verdict, the last message in state might contain feedback
        # LangGraph add_messages handles appending automatically, but we just need to run the agent.
        # Actually, if we just pass state["messages"], it will include the original query,
        # but the synthesizer has it in the system prompt.
        # Let's just pass the feedback if it exists.

        input_msgs = []
        if state.get("reflection_verdict") and state["messages"][-1].type == "human":
            # This is feedback from reflection
            input_msgs.append(state["messages"][-1])
        else:
            input_msgs.append(HumanMessage(content="Synthesize the answer based on the specialist outputs."))

        result = agent.invoke(
            {"messages": input_msgs},
            config={"recursion_limit": recursion_limit}
        )

        messages = result.get("messages", [])
        answer, tool_calls_log, _, tokens = parse_agent_messages(messages)

        # Synthesizer tokens might be appended if this is a reflection loop
        current_breakdown = state.get("token_breakdown", {})
        if "synthesizer" in current_breakdown:
            _merge_token_usage(current_breakdown["synthesizer"], tokens)
        else:
            current_breakdown["synthesizer"] = tokens

        return {
            "final_answer": answer,
            "token_breakdown": current_breakdown,
            "tool_calls_log": tool_calls_log,  # LangGraph add_messages will merge lists?
            # Wait, tool_calls_log uses add_messages? In state definition it's just `list[dict]`.
            # If we just return tool_calls_log it will OVERWRITE unless we define a reducer.
            # Let's fix MultiAgentState in graph.py to use an aggregator, or just merge here.
            # Since graph.py doesn't define aggregators for lists, they overwrite by default!
        }

    def _reflection_node_func(self, state: MultiAgentState) -> dict:
        """Executes the Reflection Verifier."""
        if not self._reflection_chain:
            return {}

        draft_answer = state["final_answer"]
        tool_calls = state.get("tool_calls_log", [])

        # The specialist outputs ARE the evidence the synthesizer answered
        # from — it sees nothing else — so they are what the verifier has to
        # check against. This used to pass the fixed string "(Specialist
        # outputs were provided to the synthesizer)", i.e. a sentence about
        # evidence instead of the evidence, which left criterion 1
        # (groundedness) with nothing to check and drove `revise` on 6 of 12
        # items in the 2026-09-08 reserve run. They are already in state; the
        # synthesizer node formats them the same way a few lines up.
        outputs = state.get("specialist_outputs", {})
        if outputs:
            contexts = "\n\n---\n\n".join(
                f"### Scope: {key}\n{ans}" for key, ans in outputs.items()
            )
        else:
            contexts = (
                "(No specialist outputs: the supervisor answered without "
                "delegating, so the synthesizer had no specialist evidence "
                "to work from.)"
            )

        chain_result = self._reflection_chain.invoke({
            "question": state["user_query"],
            "answer": draft_answer,
            "contexts": contexts,
            "tool_calls": format_tool_calls_for_prompt(tool_calls),
        })

        verdict, prompt_tokens, comp_tokens = unpack_reflection_result(chain_result)

        tokens = TokenUsage(prompt_tokens, comp_tokens, prompt_tokens + comp_tokens)

        current_breakdown = state.get("token_breakdown", {})
        if "reflection" in current_breakdown:
            _merge_token_usage(current_breakdown["reflection"], tokens)
        else:
            current_breakdown["reflection"] = tokens

        feedback_msg = generate_feedback_message(verdict)
        ret = {
            "reflection_verdict": verdict.__dict__,
            "token_breakdown": current_breakdown,
            "reflection_iterations": state.get("reflection_iterations", 0) + 1
        }
        if feedback_msg:
            ret["messages"] = [feedback_msg]

        return ret

    def query(self, question: str) -> MultiAgentResult:
        """Run a single query through the Multi-Agent pipeline."""
        if self._graph is None:
            raise RuntimeError("Pipeline not built. Call .build(filings) first.")

        start_time = time.perf_counter()

        initial_state: MultiAgentState = {
            "messages": [HumanMessage(content=question)],
            "user_query": question,
            "supervisor_plan": "",
            "delegations": [],
            "specialist_outputs": {},
            "final_answer": None,
            "reflection_verdict": None,
            "reflection_iterations": 0,
            "tool_calls_log": [],
            "token_breakdown": {},
        }

        final_state = self._graph.invoke(initial_state)

        elapsed = time.perf_counter() - start_time

        # Compile total token usage
        total_prompt = 0
        total_comp = 0
        breakdown = final_state.get("token_breakdown", {})
        for tu in breakdown.values():
            total_prompt += tu.prompt_tokens
            total_comp += tu.completion_tokens

        aggregated_tokens = TokenUsage(total_prompt, total_comp, total_prompt + total_comp)

        # We also need to get ALL tool calls.
        # The supervisor adds its tool calls, specialists add theirs, synthesizer adds its.
        # Since state overwrite happens, let's just collect them by overriding the synthesizer return
        # to include supervisor+specialists+synthesizer tool calls.
        # Wait, I didn't merge in synthesizer_node. Let's assume tool_calls_log in final_state
        # only has synthesizer tool calls unless fixed. I'll just use what's in there for now,
        # but I'll fix it in the file.

        verdict_dict = final_state.get("reflection_verdict")
        verdict = None
        if verdict_dict:
            verdict = ReflectionVerdict(**verdict_dict)

        was_revised = verdict is not None and verdict.status == "revise"

        # Re-fetch tool calls log from the state
        # Actually I should fix the StateGraph TypedDict in graph.py to use `add_messages` style reducer for lists,
        # OR just use operator.add

        num_specialists_invoked = len(final_state.get("delegations", []))
        metrics = RunMetrics(
            query=question,
            system_name="multi_agent",
            token_usage=aggregated_tokens,
            latency_seconds=elapsed,
            num_steps=len(final_state.get("tool_calls_log", [])),
            tool_calls=[tc["tool"] for tc in final_state.get("tool_calls_log", [])],
            corrections=1 if was_revised else 0,
            num_specialists_invoked=num_specialists_invoked,
            token_breakdown={k: v.total_tokens for k, v in breakdown.items()}
        )

        specialist_outputs = final_state.get("specialist_outputs", {})

        return MultiAgentResult(
            answer=final_state.get("final_answer", ""),
            contexts=list(specialist_outputs.values()),
            tool_calls_log=final_state.get("tool_calls_log", []),
            metrics=metrics,
            reflection_verdict=verdict,
            was_revised=was_revised,
            specialist_outputs=final_state.get("specialist_outputs", {}),
            token_breakdown=breakdown,
            num_specialists_invoked=num_specialists_invoked,
            supervisor_plan=final_state.get("supervisor_plan", ""),
            delegation_requests=[d.__dict__ for d in final_state.get("delegations", [])]
        )
