"""Retriever interfaces and implementations."""

from rag_claim_verification.retrieval.base import Retriever
from rag_claim_verification.retrieval.in_memory_retriever import InMemoryKeywordRetriever

__all__ = ["InMemoryKeywordRetriever", "Retriever"]
