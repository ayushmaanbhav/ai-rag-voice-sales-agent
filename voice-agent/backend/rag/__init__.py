"""
RAG (Retrieval-Augmented Generation) module for Kotak Gold Loan Voice Agent.

Provides:
- OllamaEmbeddingService: Multilingual embeddings using Ollama qwen3-embedding
- OllamaEmbeddingFunction: ChromaDB-compatible embedding function
- ChromaVectorStore: Local vector storage with persistence
- HybridRetriever: Semantic + keyword (BM25) search
- EnhancedRetriever: Conversation-aware retrieval with reranking
"""

from rag.embeddings import OllamaEmbeddingService, OllamaEmbeddingFunction
from rag.vector_store import ChromaVectorStore
from rag.retriever import HybridRetriever
from rag.enhanced_retriever import EnhancedRetriever, ConversationStage, RetrievalResult

__all__ = [
    "OllamaEmbeddingService",
    "OllamaEmbeddingFunction",
    "ChromaVectorStore",
    "HybridRetriever",
    "EnhancedRetriever",
    "ConversationStage",
    "RetrievalResult",
]
