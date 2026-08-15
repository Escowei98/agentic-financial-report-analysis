"""
Search Section Tool for System 2 (Agent RAG).

Allows the agent to retrieve chunks from a specific section of a specific
SEC 10-K filing using ChromaDB metadata filtering.

Ranking pipeline within the filtered candidate pool:
  1. ChromaDB similarity_search on a ranking_query (sub_query or fallback)
  2. Optional FlashRank cross-encoder reranking (uniform with retrieve_chunks)

This is a key architectural advantage over System 1: the agent can
make targeted retrievals instead of blindly searching all chunks.
"""

import logging
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Valid section names (must match chunker.py metadata)
VALID_SECTIONS = [
    "Business",
    "Risk Factors",
    "MD&A",
    "Financial Statements",
    "Directors and Corporate Governance",
    "Full Text",
]


def _format_section_results(
    docs: list[Document],
    ticker: str,
    fiscal_year: str,
    section: str,
    sub_query: str | None,
) -> str:
    """Format section search results for LLM consumption."""
    if not docs:
        return (
            f"No chunks found for {ticker} in section '{section}'. "
            f"Try using retrieve_chunks() for a broader search, "
            f"or check available filings with list_filings()."
        )

    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(
            f"[{ticker}, FY{fiscal_year}, {section} — Chunk {i}]\n{doc.page_content}"
        )

    ranking_hint = (
        f" ranked by sub_query={sub_query!r}"
        if sub_query
        else " (generic section ranking)"
    )
    header = f"Found {len(docs)} chunks from {ticker} — {section}{ranking_hint}:"
    return header + "\n\n---\n\n".join([""] + parts)


def create_search_section_tool(
    vectorstore: Chroma,
    retrieval_config: dict,
    reranker_config: dict,
) -> Any:
    """
    Factory that creates a search_section tool bound to a specific vectorstore.

    The tool performs a metadata-filtered ChromaDB similarity_search and
    optionally reranks the candidate pool with FlashRank — the same
    cross-encoder used by retrieve_chunks, providing uniform retrieval
    semantics within System 2.

    Args:
        vectorstore: Pre-built ChromaDB vectorstore with section metadata.
        retrieval_config: Retrieval settings. Expected keys:
            - pre_rerank_top_k (int): initial candidate pool size (default 15)
            - post_rerank_top_k (int): final chunk count after reranking
              (default 4)
        reranker_config: Reranker settings. Expected keys:
            - enabled (bool): whether to apply FlashRank (default True)
            - model (str): FlashRank model name
              (default 'ms-marco-MiniLM-L-12-v2')

    Returns:
        A LangChain @tool function.
    """
    pre_rerank_top_k = retrieval_config.get("pre_rerank_top_k", 15)
    post_rerank_top_k = retrieval_config.get("post_rerank_top_k", 4)
    reranker_enabled = reranker_config.get("enabled", True)
    reranker_model = reranker_config.get("model", "ms-marco-MiniLM-L-12-v2")

    @tool
    def search_section(
        ticker: str,
        fiscal_year: str,
        section: str,
        sub_query: str | None = None,
    ) -> str:
        """Retrieve chunks from a specific section of a specific SEC 10-K filing.

        Use this when you know which company, year, and section contains the
        answer. More precise than retrieve_chunks() — directly targets a known
        section via ChromaDB metadata filtering.

        Pass `sub_query` with the specific natural-language question to get
        chunks ranked by relevance to that question within the filtered
        section (recommended for targeted questions). Omit `sub_query` for a
        generic section overview.

        Args:
            ticker: Company ticker symbol (e.g., 'AAPL', 'MSFT', 'AMZN', 'GOOGL').
            fiscal_year: Fiscal year as string (e.g., '2024', '2023').
            section: SEC 10-K section name. Must be one of:
                     'Business', 'Risk Factors', 'MD&A', 'Financial Statements',
                     'Directors and Corporate Governance'.
            sub_query: Optional natural-language question used as ranking query
                       for similarity_search and the reranker. If omitted or
                       empty, falls back to a generic '{ticker} {section}' query.
        """
        ticker_upper = ticker.upper().strip()
        fy = fiscal_year.strip()

        if section not in VALID_SECTIONS:
            return (
                f"Invalid section '{section}'. "
                f"Valid sections: {', '.join(VALID_SECTIONS)}"
            )

        # Query selection: prefer sub_query, fall back to generic section query.
        # An empty or whitespace-only sub_query is treated as "not provided".
        effective_sub_query: str | None = None
        if sub_query and sub_query.strip():
            effective_sub_query = sub_query.strip()
            ranking_query = effective_sub_query
        else:
            ranking_query = f"{ticker_upper} {section}"

        where_filter = {
            "$and": [
                {"ticker": {"$eq": ticker_upper}},
                {"fiscal_year": {"$eq": fy}},
                {"section_name": {"$eq": section}},
            ]
        }

        try:
            candidates = vectorstore.similarity_search(
                query=ranking_query,
                k=pre_rerank_top_k,
                filter=where_filter,
            )
        except Exception as e:
            logger.error(
                "search_section failed: ticker=%s, fy=%s, section=%s: %s",
                ticker_upper, fy, section, e,
            )
            return f"Error searching section: {e}"

        # Apply FlashRank only if it can meaningfully prune the pool.
        # For small candidate sets (<= post_rerank_top_k) the reranker has
        # nothing to remove — skip the extra latency.
        if (
            reranker_enabled
            and len(candidates) > post_rerank_top_k
        ):
            from src.common.retrieval import rerank_documents
            final_docs = rerank_documents(
                query=ranking_query,
                documents=candidates,
                top_n=post_rerank_top_k,
                model=reranker_model,
            )
        else:
            final_docs = candidates[:post_rerank_top_k]

        logger.info(
            "search_section: %s FY%s %s (sub_query=%s) → %d/%d chunks",
            ticker_upper, fy, section,
            repr(effective_sub_query) if effective_sub_query else "None",
            len(final_docs), len(candidates),
        )
        return _format_section_results(
            final_docs, ticker_upper, fy, section, effective_sub_query,
        )

    return search_section
