"""
RAG Module
==========
Retrieval-Augmented Generation for forensic explanations.
"""

from .build_index import KnowledgeBaseBuilder, build_vector_index
from .retriever import DocumentRetriever
from .llm_explainer import LLMExplainer, generate_forensic_explanation

__all__ = [
    "KnowledgeBaseBuilder",
    "build_vector_index",
    "DocumentRetriever",
    "LLMExplainer",
    "generate_forensic_explanation"
]
