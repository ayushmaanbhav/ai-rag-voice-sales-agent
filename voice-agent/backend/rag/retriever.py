"""
Hybrid retriever combining semantic search (ChromaDB) with keyword search (BM25).

Provides better retrieval for exact terms like rates, provider names, and technical terms.
"""

from typing import Optional
from rank_bm25 import BM25Okapi
import re
import logging

from rag.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever combining semantic and keyword search.

    Features:
    - Semantic search via ChromaDB embeddings
    - Keyword search via BM25 for exact matches
    - Score fusion for combined ranking
    - State-aware filtering for conversation context
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_store: ChromaDB vector store instance
            semantic_weight: Weight for semantic search scores (0-1)
            keyword_weight: Weight for keyword search scores (0-1)
        """
        self.vector_store = vector_store
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

        # BM25 index (built lazily)
        self._bm25: Optional[BM25Okapi] = None
        self._doc_ids: list[str] = []
        self._documents: list[str] = []
        self._metadatas: list[dict] = []

    def _build_bm25_index(self) -> None:
        """Build BM25 index from all documents in vector store."""
        # Fetch all documents
        all_ids = self.vector_store.get_all_ids()
        if not all_ids:
            logger.warning("No documents in vector store for BM25 index")
            return

        # Get documents with metadata
        results = self.vector_store.collection.get(
            ids=all_ids,
            include=["documents", "metadatas"]
        )

        self._doc_ids = results["ids"]
        self._documents = results["documents"]
        self._metadatas = results["metadatas"]

        # Tokenize documents for BM25
        tokenized_docs = [self._tokenize(doc) for doc in self._documents]
        self._bm25 = BM25Okapi(tokenized_docs)

        logger.info(f"Built BM25 index with {len(self._doc_ids)} documents")

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text for BM25.

        Handles Hindi/English mixed text by splitting on whitespace
        and removing punctuation.
        """
        # Lowercase and remove punctuation (keep Hindi characters)
        text = text.lower()
        # Split on whitespace and filter empty tokens
        tokens = re.findall(r'[\w\u0900-\u097F]+', text)
        return tokens

    def _keyword_search(
        self,
        query: str,
        n_results: int = 10
    ) -> list[tuple[str, float, str, dict]]:
        """
        Perform BM25 keyword search.

        Args:
            query: Search query
            n_results: Number of results

        Returns:
            List of (doc_id, score, document, metadata) tuples
        """
        if self._bm25 is None:
            self._build_bm25_index()

        if self._bm25 is None or not self._doc_ids:
            return []

        # Get BM25 scores
        query_tokens = self._tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # Get top results
        scored_docs = list(zip(
            self._doc_ids,
            scores,
            self._documents,
            self._metadatas
        ))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs[:n_results]

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        doc_type: Optional[str] = None,
        category: Optional[str] = None,
        language: str = "en"
    ) -> list[dict]:
        """
        Retrieve relevant documents using hybrid search.

        Args:
            query: Search query (can be Hindi or English)
            n_results: Number of results to return
            doc_type: Filter by document type (product, competitor, objection, etc.)
            category: Filter by category (rates, safety, etc.)
            language: Preferred language for content (en/hi)

        Returns:
            List of dicts with id, content, metadata, and score
        """
        # Build filter
        where_filter = None
        if doc_type and category:
            where_filter = {"$and": [{"type": doc_type}, {"category": category}]}
        elif doc_type:
            where_filter = {"type": doc_type}
        elif category:
            where_filter = {"category": category}

        # Semantic search
        semantic_results = self.vector_store.query(
            query_text=query,
            n_results=n_results * 2,  # Get more for fusion
            where=where_filter
        )

        # Keyword search (no filter - applied post-hoc)
        keyword_results = self._keyword_search(query, n_results * 2)

        # Fuse results
        fused = self._fuse_results(
            semantic_results,
            keyword_results,
            doc_type=doc_type,
            category=category
        )

        # Return top n_results
        results = []
        for doc_id, score, document, metadata in fused[:n_results]:
            # Select content based on language preference
            content = document
            if language == "hi" and metadata.get("content_hi"):
                content = metadata.get("content_hi", document)

            results.append({
                "id": doc_id,
                "content": content,
                "metadata": metadata,
                "score": score
            })

        return results

    def _fuse_results(
        self,
        semantic_results: dict,
        keyword_results: list[tuple],
        doc_type: Optional[str] = None,
        category: Optional[str] = None
    ) -> list[tuple[str, float, str, dict]]:
        """
        Fuse semantic and keyword search results using weighted scoring.

        Uses Reciprocal Rank Fusion (RRF) style combination.
        """
        # Normalize semantic scores (distances to similarities)
        semantic_scores = {}
        if semantic_results["ids"]:
            max_dist = max(semantic_results["distances"]) if semantic_results["distances"] else 1
            for i, doc_id in enumerate(semantic_results["ids"]):
                # Convert distance to similarity (lower distance = higher similarity)
                dist = semantic_results["distances"][i]
                sim = 1 - (dist / (max_dist + 1e-6))
                semantic_scores[doc_id] = {
                    "score": sim,
                    "document": semantic_results["documents"][i],
                    "metadata": semantic_results["metadatas"][i]
                }

        # Normalize BM25 scores
        keyword_scores = {}
        if keyword_results:
            max_score = max(r[1] for r in keyword_results) if keyword_results else 1
            for doc_id, score, document, metadata in keyword_results:
                # Apply type/category filter for keyword results
                if doc_type and metadata.get("type") != doc_type:
                    continue
                if category and metadata.get("category") != category:
                    continue

                norm_score = score / (max_score + 1e-6)
                keyword_scores[doc_id] = {
                    "score": norm_score,
                    "document": document,
                    "metadata": metadata
                }

        # Combine scores
        all_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
        fused = []

        for doc_id in all_ids:
            sem_data = semantic_scores.get(doc_id, {"score": 0})
            kw_data = keyword_scores.get(doc_id, {"score": 0})

            # Weighted combination
            combined_score = (
                self.semantic_weight * sem_data.get("score", 0) +
                self.keyword_weight * kw_data.get("score", 0)
            )

            # Get document and metadata from whichever source has it
            document = sem_data.get("document") or kw_data.get("document", "")
            metadata = sem_data.get("metadata") or kw_data.get("metadata", {})

            fused.append((doc_id, combined_score, document, metadata))

        # Sort by combined score
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused

    def retrieve_for_objection(
        self,
        query: str,
        n_results: int = 3,
        language: str = "en"
    ) -> list[dict]:
        """
        Retrieve relevant objection handling content.

        Args:
            query: Customer's objection or concern
            n_results: Number of results
            language: Preferred language

        Returns:
            Relevant objection handling documents
        """
        return self.retrieve(
            query=query,
            n_results=n_results,
            doc_type="objection",
            language=language
        )

    def retrieve_product_info(
        self,
        query: str,
        n_results: int = 3,
        language: str = "en"
    ) -> list[dict]:
        """
        Retrieve product information.

        Args:
            query: Product-related query
            n_results: Number of results
            language: Preferred language

        Returns:
            Relevant product documents
        """
        return self.retrieve(
            query=query,
            n_results=n_results,
            doc_type="product",
            language=language
        )

    def retrieve_competitor_info(
        self,
        query: str,
        n_results: int = 3,
        language: str = "en"
    ) -> list[dict]:
        """
        Retrieve competitor comparison information.

        Args:
            query: Competitor-related query
            n_results: Number of results
            language: Preferred language

        Returns:
            Relevant competitor documents
        """
        return self.retrieve(
            query=query,
            n_results=n_results,
            doc_type="competitor",
            language=language
        )

    def rebuild_index(self) -> None:
        """Rebuild BM25 index (call after adding new documents)."""
        self._bm25 = None
        self._build_bm25_index()
