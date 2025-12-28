"""
ChromaDB vector store for Kotak Gold Loan knowledge base.

Provides persistent local storage for embeddings with metadata filtering.
"""

import chromadb
from chromadb.config import Settings
from typing import Optional
import logging
from pathlib import Path

from rag.embeddings import OllamaEmbeddingFunction

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    ChromaDB-based vector store for knowledge retrieval.

    Features:
    - Persistent storage
    - Metadata filtering (by type, category, language)
    - Batch operations
    - Ollama embedding integration
    """

    def __init__(
        self,
        persist_dir: str = "./data/chroma",
        collection_name: str = "kotak_knowledge",
        embedding_model: str = "qwen3-embedding:8b-q4_K_M"
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            persist_dir: Directory for persistent storage
            collection_name: Name of the collection
            embedding_model: Ollama embedding model to use
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Create embedding function
        self.embedding_fn = OllamaEmbeddingFunction(model=embedding_model)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"description": "Kotak Gold Loan knowledge base"}
        )

        logger.info(
            f"ChromaDB initialized: {collection_name} "
            f"({self.collection.count()} documents)"
        )

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: Optional[list[dict]] = None
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            ids: Unique identifiers for each document
            documents: Text content to embed and store
            metadatas: Optional metadata for filtering
        """
        if not ids or not documents:
            return

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Added {len(ids)} documents to vector store")

    def upsert_documents(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: Optional[list[dict]] = None
    ) -> None:
        """
        Add or update documents in the vector store.

        Args:
            ids: Unique identifiers for each document
            documents: Text content to embed and store
            metadatas: Optional metadata for filtering
        """
        if not ids or not documents:
            return

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"Upserted {len(ids)} documents to vector store")

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[dict] = None,
        include: Optional[list[str]] = None
    ) -> dict:
        """
        Query the vector store for similar documents.

        Args:
            query_text: Search query
            n_results: Number of results to return
            where: Metadata filter (e.g., {"type": "product"})
            include: What to include in results (documents, metadatas, distances)

        Returns:
            Dict with ids, documents, metadatas, and distances
        """
        if include is None:
            include = ["documents", "metadatas", "distances"]

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            include=include
        )

        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results.get("documents") else [],
            "metadatas": results["metadatas"][0] if results.get("metadatas") else [],
            "distances": results["distances"][0] if results.get("distances") else []
        }

    def query_by_type(
        self,
        query_text: str,
        doc_type: str,
        n_results: int = 5
    ) -> dict:
        """
        Query with type filter.

        Args:
            query_text: Search query
            doc_type: Document type (product, competitor, objection, regulation, faq)
            n_results: Number of results

        Returns:
            Query results
        """
        return self.query(
            query_text=query_text,
            n_results=n_results,
            where={"type": doc_type}
        )

    def query_by_category(
        self,
        query_text: str,
        category: str,
        n_results: int = 5
    ) -> dict:
        """
        Query with category filter.

        Args:
            query_text: Search query
            category: Category (rates, safety, auction, etc.)
            n_results: Number of results

        Returns:
            Query results
        """
        return self.query(
            query_text=query_text,
            n_results=n_results,
            where={"category": category}
        )

    def delete(self, ids: list[str]) -> None:
        """Delete documents by ID."""
        self.collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents")

    def reset(self) -> None:
        """Reset the collection (delete all documents)."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            embedding_function=self.embedding_fn,
            metadata={"description": "Kotak Gold Loan knowledge base"}
        )
        logger.info("Vector store reset")

    def count(self) -> int:
        """Return total document count."""
        return self.collection.count()

    def get_all_ids(self) -> list[str]:
        """Get all document IDs in the collection."""
        result = self.collection.get(include=[])
        return result["ids"]
