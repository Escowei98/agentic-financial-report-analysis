"""
Knowledge tools shared by all agentic systems (S2, S3, S4).

These are the tools the Tool-Symmetry Principle (see thesis 3.5.1) requires
to be *identical* across the agentic systems: they supply information or
perform computation, so an asymmetry here would confound the architectural
comparison. Keeping them in `src/common/` rather than inside one system's
package makes that symmetry structural instead of a property that has to be
re-checked whenever one of them changes.

System-exclusive tools stay in their own package:
  - `retrieve_chunks`, `search_section`  -> src/systems/rag_agent/tools/ (S2)
  - `delegate_to_specialist`             -> src/systems/multi_agent/tools.py (S4,
    an orchestration primitive, exempt from the symmetry rule)
"""

from src.common.tools.calculate import calculate
from src.common.tools.list_filings import create_list_filings_tool

__all__ = ["calculate", "create_list_filings_tool"]
