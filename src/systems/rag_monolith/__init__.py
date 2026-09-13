"""System 1: Simple RAG — Classic chunking + hybrid retrieval + reranking."""

from src.common.chunker import chunk_filings
from src.common.retrieval import (
    HybridRetriever,
    build_hybrid_retriever,
    build_vectorstore,
    rerank_documents,
    retrieve,
)
from src.systems.rag_monolith.pipeline import MonolithRAGPipeline, RAGResult

__all__ = [
    "chunk_filings",
    "MonolithRAGPipeline",
    "RAGResult",
    "HybridRetriever",
    "build_hybrid_retriever",
    "build_vectorstore",
    "rerank_documents",
    "retrieve",
]
