"""
Shared LangGraph ReAct agent factory for all agentic systems (S2, S3, S4).

Wraps `langgraph.prebuilt.create_react_agent` behind a single, prompt-
parametrized entry point. Previously this wrapper existed twice — once in
`src/systems/rag_agent/agent.py` (S2, prompt hard-wired to a module-level
constant) and once in `src/systems/long_context/agent.py` (S3/S4, prompt
passed in). Both produced the same agent; the only difference was where the
prompt came from.

Consolidating them makes the claim that S2, S3 and S4 share one ReAct
control flow (thesis 3.5.1 / 4.2.3) structural rather than a property that
holds only as long as two copies stay in sync. The system prompt remains
system-specific and is supplied by the caller:

  - S2: `src/systems/rag_agent/agent.py::SYSTEM_PROMPT` (static)
  - S3: `src/systems/long_context/prompt.py::build_system_prompt()` (composed
        from the filing corpus at build time)
  - S4: `src/systems/multi_agent/prompts.py` (one per role)
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
    Build a LangGraph ReAct agent.

    Args:
        llm: Chat model instance (see `src/common/llm_client.py::get_llm`).
        tools: Tool list for this system.
        system_prompt: Fully composed system prompt.
        recursion_limit: Max graph recursion depth. Passed by the caller at
            invoke time; recorded here for logging only.

    Returns:
        Compiled LangGraph agent (CompiledStateGraph).
    """
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )

    logger.info(
        "Agent built: %d tools, recursion_limit=%d, prompt_chars=%d",
        len(tools), recursion_limit, len(system_prompt),
    )
    return agent
