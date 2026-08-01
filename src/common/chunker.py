"""
Parametrizable chunking for System 1 (Monolith RAG).

Splits processed SEC 10-K filings into LangChain Documents
using RecursiveCharacterTextSplitter with configurable chunk size
and overlap percentage.

Non-agentic — pure text splitting with metadata attachment.
"""

import logging
from typing import Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.common.ingestion import ProcessedFiling

logger = logging.getLogger(__name__)


def chunk_filings(
    filings: Sequence[ProcessedFiling],
    chunk_size: int = 1000,
    overlap_pct: float = 0.20,
) -> list[Document]:
    """
    Chunk a list of processed filings into LangChain Documents.

    Each chunk carries metadata for traceability:
      - ticker, company_name, filing_date, fiscal_year
      - section_name (which 10-K section it came from)
      - chunk_index (position within the section)

    Args:
        filings: Processed SEC 10-K filings.
        chunk_size: Target characters per chunk.
        overlap_pct: Overlap as fraction of chunk_size (e.g., 0.20 = 20%).

    Returns:
        Flat list of LangChain Documents with metadata.
    """
    chunk_overlap = int(chunk_size * overlap_pct)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_documents: list[Document] = []

    for filing in filings:
        meta = filing.metadata
        fiscal_year = meta.fiscal_year_end[:4] if meta.fiscal_year_end else "unknown"

        base_metadata = {
            "ticker": meta.ticker,
            "company_name": meta.company_name,
            "filing_date": meta.filing_date,
            "fiscal_year": fiscal_year,
            "accession_number": meta.accession_number,
        }

        if filing.sections:
            # Chunk each section separately for better metadata tracking
            for section_name, content in filing.sections.items():
                section_docs = splitter.create_documents(
                    texts=[content],
                    metadatas=[{**base_metadata, "section_name": section_name}],
                )
                # Add chunk_index within this section
                for i, doc in enumerate(section_docs):
                    doc.metadata["chunk_index"] = i
                all_documents.extend(section_docs)
        else:
            # Fallback: chunk full text as a single "Full Text" section
            full_docs = splitter.create_documents(
                texts=[filing.full_text],
                metadatas=[{**base_metadata, "section_name": "Full Text"}],
            )
            for i, doc in enumerate(full_docs):
                doc.metadata["chunk_index"] = i
            all_documents.extend(full_docs)

        logger.info(
            "Chunked %s FY%s: %d chunks (chunk_size=%d, overlap=%d)",
            meta.ticker, fiscal_year, len(all_documents), chunk_size, chunk_overlap,
        )

    logger.info(
        "Total: %d documents from %d filings (chunk_size=%d, overlap_pct=%.0f%%)",
        len(all_documents), len(filings), chunk_size, overlap_pct * 100,
    )
    return all_documents
