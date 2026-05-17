"""System 4: Multi-Agent Long Context.

This package is currently a specification-only placeholder. The chosen
architecture (Supervisor + Company-Specialists + Synthesizer with shared
S2/S3 wissens-tool set and an exclusive `delegate_to_specialist`
orchestration primitive) is documented in two places:

- Decision rationale and H3-Operationalisierung:
  ``docs/decisions/SYS4_MULTI_AGENT_LOG.md`` (entries from 2026-05-17).
- Concrete topology, LangGraph state shape, tool wiring and message
  plan: ``src/systems/multi_agent/architecture.md``.

Implementation will follow in a dedicated workpaket and is intentionally
out of scope for the Capability-Triage iteration that produced these
specs.
"""
