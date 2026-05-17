"""
LangGraph ReAct Agent for System 3 (Long Context).

Mirrors `src/systems/rag_agent/agent.py` but with two architectural
differences:

  1. The system prompt is dynamic — it is composed at build time from
     the loaded filings via `build_system_prompt(filings)`. There is no
     module-level SYSTEM_PROMPT constant because the prompt depends on
     the input filings.

  2. The tool set excludes retrieval (`retrieve_chunks`,
     `search_section`). Instead, the LLM reads filings directly from
     its context window. The remaining tools (`calculate`,
     `list_filings`) are imported unchanged from S2 to keep the
     architectural-comparison axis pure.
"""

import logging
from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger(__name__)


def build_agent(
    llm: BaseChatModel,
    tools: Sequence[Any],
    system_prompt: str,
    recursion_limit: int = 12,
) -> Any:
    """
    Build a LangGraph ReAct agent for the long-context system.

    Args:
        llm: Chat model instance (Gemini 2.5 Flash via Vertex AI).
        tools: Tool list (typically `[calculate, list_filings]`).
        system_prompt: Pre-composed system prompt with all filings inlined.
            See `src/systems/long_context/prompt.py`.
        recursion_limit: Max graph recursion depth.

    Returns:
        Compiled LangGraph agent (CompiledStateGraph).
    """
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )

    logger.info(
        "S3 agent built: %d tools, recursion_limit=%d, prompt_chars=%d",
        len(tools), recursion_limit, len(system_prompt),
    )
    return agent
