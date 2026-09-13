"""
System 4: Multi-Agent Long Context Graph Definition.

Defines the StateGraph topology:
Supervisor -> Specialist Pool -> Synthesizer -> Reflection
"""

import logging
from dataclasses import dataclass
from operator import add
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from src.common.utils import TokenUsage

logger = logging.getLogger(__name__)


@dataclass
class DelegationRequest:
    """A single specialist invocation planned by the Supervisor."""
    tickers: list[str]
    sections: list[str]
    sub_question: str
    scope_key: str


class MultiAgentState(TypedDict):
    """Central state for the S4 multi-agent graph."""
    messages: Annotated[list[BaseMessage], add_messages]
    user_query: str

    # Supervisor's output
    supervisor_plan: str
    delegations: list[DelegationRequest]

    # Specialists' outputs
    specialist_outputs: dict[str, str]  # scope_key -> answer

    # Final Synthesizer output
    final_answer: str | None

    # Reflection
    reflection_verdict: dict | None
    reflection_iterations: int

    # Tracking
    tool_calls_log: Annotated[list[dict], add]
    token_breakdown: dict[str, TokenUsage]  # role -> TokenUsage


def _merge_token_usage(dest: TokenUsage, src: TokenUsage) -> None:
    dest.prompt_tokens += src.prompt_tokens
    dest.completion_tokens += src.completion_tokens
    dest.total_tokens += src.total_tokens


# The actual node implementations (supervisor_node, specialist_pool_node, synthesizer_node, reflection_node)
# will be closures inside the pipeline's build() method so they have access to the agent runners.
