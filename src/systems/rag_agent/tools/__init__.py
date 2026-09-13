"""
Retrieval tools exclusive to System 2 (Agent RAG).

Both require the retrieval stack injected at pipeline build time. The
knowledge tools shared with S3/S4 (`calculate`, `list_filings`) live in
`src/common/tools/` — see the module docstring there.
"""

from src.systems.rag_agent.tools.retrieve_chunks import create_retrieve_chunks_tool
from src.systems.rag_agent.tools.search_section import create_search_section_tool

__all__ = [
    "create_retrieve_chunks_tool",
    "create_search_section_tool",
]
