"""Shared building blocks used by ALL systems, so that fairness principles
FP-1 to FP-4 hold structurally rather than by convention.

Anything imported by more than one system lives here: the model clients, the
data pipeline, the retrieval stack, the ReAct agent factory, the knowledge
tools, the reflection verifier, the answer-format convention and the metrics
layer. Modules under `src/systems/` import from here; the reverse never
happens.
"""

from src.common.agent import build_agent
from src.common.answer_format_convention import ANSWER_FORMAT_CONVENTION
from src.common.config import load_config
from src.common.ingestion import (
    ProcessedFiling,
    download_all_filings,
    download_filing,
    fiscal_year_from_metadata,
    load_processed_filing,
)
from src.common.llm_client import get_embeddings, get_judge_llm, get_llm
from src.common.message_parsing import format_tool_calls_for_prompt, parse_agent_messages
from src.common.reflection import (
    ReflectionVerdict,
    build_reflection_chain,
    run_reflection_pass,
)
from src.common.tools import calculate, create_list_filings_tool
from src.common.utils import CostTracker, RunMetrics, TokenUsage, setup_logging, timer

__all__ = [
    # configuration
    "load_config",
    # data pipeline
    "download_filing",
    "download_all_filings",
    "load_processed_filing",
    "fiscal_year_from_metadata",
    "ProcessedFiling",
    # models
    "get_llm",
    "get_judge_llm",
    "get_embeddings",
    # agentic building blocks
    "build_agent",
    "parse_agent_messages",
    "format_tool_calls_for_prompt",
    "build_reflection_chain",
    "run_reflection_pass",
    "ReflectionVerdict",
    "calculate",
    "create_list_filings_tool",
    "ANSWER_FORMAT_CONVENTION",
    # measurement
    "CostTracker",
    "RunMetrics",
    "TokenUsage",
    "setup_logging",
    "timer",
]
