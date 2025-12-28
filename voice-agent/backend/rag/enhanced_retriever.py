"""
Enhanced Retriever with conversation awareness, query rewriting, and reranking.

Improvements over base HybridRetriever:
- Conversation-aware retrieval (considers last N turns)
- Query rewriting for better semantic matching
- Cross-encoder reranking for relevance
- Stage-based context sizing
- Caching for repeated queries
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from functools import lru_cache
from enum import Enum

from rag.retriever import HybridRetriever
from rag.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class ConversationStage(str, Enum):
    """Stages of the sales conversation."""
    GREETING = "greeting"
    QUALIFICATION = "qualification"
    PITCH = "pitch"
    OBJECTION = "objection"
    DEEPDIVE = "deepdive"
    APPOINTMENT = "appointment"
    CLOSING = "closing"


# Stage-based context configuration
STAGE_CONFIG = {
    ConversationStage.GREETING: {
        "max_docs": 1,
        "doc_types": ["product"],
        "max_chars": 300,
    },
    ConversationStage.QUALIFICATION: {
        "max_docs": 2,
        "doc_types": ["product", "competitor"],
        "max_chars": 500,
    },
    ConversationStage.PITCH: {
        "max_docs": 3,
        "doc_types": ["product", "competitor"],
        "max_chars": 600,
    },
    ConversationStage.OBJECTION: {
        "max_docs": 3,
        "doc_types": ["objection", "faq"],
        "max_chars": 800,
    },
    ConversationStage.DEEPDIVE: {
        "max_docs": 4,
        "doc_types": ["product", "faq", "regulation"],
        "max_chars": 800,
    },
    ConversationStage.APPOINTMENT: {
        "max_docs": 2,
        "doc_types": ["product", "faq"],
        "max_chars": 400,
    },
    ConversationStage.CLOSING: {
        "max_docs": 1,
        "doc_types": ["product"],
        "max_chars": 200,
    },
}

# Query rewriting patterns for Hindi/Hinglish
QUERY_EXPANSIONS = {
    # Safety concerns
    "safe": ["safety", "suraksha", "secure", "vault", "insurance", "bima"],
    "सुरक्षित": ["safe", "vault", "insurance", "security"],
    "chori": ["theft", "steal", "safety", "insurance"],

    # Interest rate
    "rate": ["interest", "byaj", "dar", "percent", "annual"],
    "byaj": ["interest", "rate", "percent", "annual"],
    "ब्याज": ["interest", "rate", "percent"],

    # Competitors
    "muthoot": ["competitor", "NBFC", "transfer", "comparison"],
    "manappuram": ["competitor", "NBFC", "transfer", "comparison"],

    # Process
    "document": ["documentation", "dastavej", "paperwork", "required"],
    "time": ["process", "duration", "kitna", "fast", "quick"],

    # Gold
    "gold": ["sona", "jewellery", "ornament", "purity"],
    "sona": ["gold", "jewellery", "ornament"],
}


@dataclass
class RetrievalResult:
    """Enhanced retrieval result with metadata."""
    documents: List[Dict[str, Any]]
    query_original: str
    query_rewritten: Optional[str]
    stage: ConversationStage
    retrieval_time_ms: int
    reranking_applied: bool


class EnhancedRetriever:
    """
    Enhanced retriever with conversation awareness and advanced features.

    Features:
    - Conversation context integration
    - Query expansion and rewriting
    - Stage-based document selection
    - Cross-encoder reranking (optional)
    - Query caching
    """

    def __init__(
        self,
        base_retriever: HybridRetriever,
        enable_reranking: bool = False,
        enable_query_rewriting: bool = True,
        cache_size: int = 100,
    ):
        """
        Initialize enhanced retriever.

        Args:
            base_retriever: Base hybrid retriever instance
            enable_reranking: Enable cross-encoder reranking
            enable_query_rewriting: Enable query expansion
            cache_size: LRU cache size for queries
        """
        self.base_retriever = base_retriever
        self.enable_reranking = enable_reranking
        self.enable_query_rewriting = enable_query_rewriting
        self.cache_size = cache_size

        # Reranker model (lazy loaded)
        self._reranker = None

        # Query cache
        self._query_cache: Dict[str, List[Dict]] = {}

        logger.info(
            f"[ENHANCED_RETRIEVER] Initialized with reranking={enable_reranking}, "
            f"query_rewriting={enable_query_rewriting}"
        )

    def _load_reranker(self):
        """Lazy load cross-encoder reranker."""
        if self._reranker is not None:
            return self._reranker

        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(
                'cross-encoder/ms-marco-MiniLM-L-6-v2',
                max_length=512
            )
            logger.info("[ENHANCED_RETRIEVER] Loaded cross-encoder reranker")
        except ImportError:
            logger.warning(
                "[ENHANCED_RETRIEVER] sentence-transformers not available, "
                "reranking disabled"
            )
            self.enable_reranking = False

        return self._reranker

    def detect_stage(
        self,
        current_query: str,
        conversation_history: List[Dict[str, str]],
    ) -> ConversationStage:
        """
        Detect current conversation stage from query and history.

        Args:
            current_query: Current user query
            conversation_history: List of previous turns

        Returns:
            Detected conversation stage
        """
        query_lower = current_query.lower()
        turn_count = len(conversation_history)

        # Keywords for stage detection
        greeting_keywords = ["hello", "hi", "namaste", "haan", "हां", "नमस्ते"]
        objection_keywords = [
            "safe", "suraksha", "chori", "auction", "nilaam",
            "trust", "bharosa", "risk", "problem", "सुरक्षित", "चोरी"
        ]
        appointment_keywords = [
            "appointment", "book", "visit", "branch", "kal", "aaj",
            "time", "slot", "अपॉइंटमेंट", "बुक"
        ]
        competitor_keywords = [
            "muthoot", "manappuram", "iifl", "currently", "abhi",
            "transfer", "switch", "मुथूट"
        ]

        # Early stage detection
        if turn_count <= 1:
            if any(kw in query_lower for kw in greeting_keywords):
                return ConversationStage.GREETING

        # Objection detection (high priority)
        if any(kw in query_lower for kw in objection_keywords):
            return ConversationStage.OBJECTION

        # Appointment detection
        if any(kw in query_lower for kw in appointment_keywords):
            return ConversationStage.APPOINTMENT

        # Competitor mention → Qualification/Pitch
        if any(kw in query_lower for kw in competitor_keywords):
            return ConversationStage.QUALIFICATION

        # Default progression based on turn count
        if turn_count <= 2:
            return ConversationStage.GREETING
        elif turn_count <= 4:
            return ConversationStage.QUALIFICATION
        elif turn_count <= 6:
            return ConversationStage.PITCH
        else:
            return ConversationStage.DEEPDIVE

    def rewrite_query(
        self,
        query: str,
        conversation_history: List[Dict[str, str]],
        stage: ConversationStage,
    ) -> str:
        """
        Rewrite query for better retrieval.

        Expands query with synonyms and context from conversation.

        Args:
            query: Original query
            conversation_history: Previous conversation turns
            stage: Current conversation stage

        Returns:
            Expanded/rewritten query
        """
        if not self.enable_query_rewriting:
            return query

        expanded_terms = []
        query_lower = query.lower()

        # Add expansions for known terms
        for term, expansions in QUERY_EXPANSIONS.items():
            if term in query_lower:
                expanded_terms.extend(expansions)

        # Add context from recent history
        if conversation_history:
            # Get last 2 turns for context
            recent_turns = conversation_history[-2:]
            for turn in recent_turns:
                # Extract key terms from assistant responses
                response = turn.get("content", "").lower()
                if "muthoot" in response or "manappuram" in response:
                    expanded_terms.append("competitor comparison")
                if "rate" in response or "percent" in response:
                    expanded_terms.append("interest savings")

        # Add stage-specific terms
        if stage == ConversationStage.OBJECTION:
            expanded_terms.append("objection handling response")
        elif stage == ConversationStage.QUALIFICATION:
            expanded_terms.append("customer qualification")

        # Combine original query with expansions
        if expanded_terms:
            unique_terms = list(set(expanded_terms))[:5]  # Limit expansions
            rewritten = f"{query} {' '.join(unique_terms)}"
            logger.debug(f"[ENHANCED_RETRIEVER] Query rewritten: '{query}' -> '{rewritten}'")
            return rewritten

        return query

    def rerank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: Search query
            documents: Retrieved documents
            top_k: Number of documents to return

        Returns:
            Reranked documents
        """
        if not self.enable_reranking or not documents:
            return documents[:top_k]

        reranker = self._load_reranker()
        if reranker is None:
            return documents[:top_k]

        try:
            # Prepare pairs for cross-encoder
            pairs = [
                (query, doc.get("content", "")[:500])
                for doc in documents
            ]

            # Get reranking scores
            scores = reranker.predict(pairs)

            # Sort by reranking score
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            # Update scores and return top_k
            reranked = []
            for doc, score in scored_docs[:top_k]:
                doc["rerank_score"] = float(score)
                reranked.append(doc)

            logger.debug(f"[ENHANCED_RETRIEVER] Reranked {len(documents)} -> {len(reranked)} docs")
            return reranked

        except Exception as e:
            logger.error(f"[ENHANCED_RETRIEVER] Reranking failed: {e}")
            return documents[:top_k]

    def retrieve(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        stage: Optional[ConversationStage] = None,
        language: str = "hi",
    ) -> RetrievalResult:
        """
        Retrieve documents with conversation awareness.

        Args:
            query: Current user query
            conversation_history: Previous conversation turns
            stage: Explicit stage override (auto-detected if None)
            language: Preferred language

        Returns:
            RetrievalResult with documents and metadata
        """
        start_time = time.time()
        conversation_history = conversation_history or []

        # Detect stage if not provided
        if stage is None:
            stage = self.detect_stage(query, conversation_history)

        # Get stage configuration
        config = STAGE_CONFIG.get(stage, STAGE_CONFIG[ConversationStage.PITCH])

        # Check cache
        cache_key = f"{query}:{stage.value}:{language}"
        if cache_key in self._query_cache:
            cached = self._query_cache[cache_key]
            return RetrievalResult(
                documents=cached,
                query_original=query,
                query_rewritten=None,
                stage=stage,
                retrieval_time_ms=0,
                reranking_applied=False,
            )

        # Rewrite query
        rewritten_query = self.rewrite_query(query, conversation_history, stage)

        # Retrieve from base retriever
        all_docs = []
        for doc_type in config["doc_types"]:
            docs = self.base_retriever.retrieve(
                query=rewritten_query,
                n_results=config["max_docs"],
                doc_type=doc_type,
                language=language,
            )
            all_docs.extend(docs)

        # Rerank if enabled
        reranking_applied = False
        if self.enable_reranking and len(all_docs) > config["max_docs"]:
            all_docs = self.rerank_documents(
                query, all_docs, top_k=config["max_docs"]
            )
            reranking_applied = True
        else:
            # Sort by original score and limit
            all_docs.sort(key=lambda x: x.get("score", 0), reverse=True)
            all_docs = all_docs[:config["max_docs"]]

        # Truncate content to max_chars
        for doc in all_docs:
            if len(doc.get("content", "")) > config["max_chars"]:
                doc["content"] = doc["content"][:config["max_chars"]] + "..."

        # Cache result
        if len(self._query_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]
        self._query_cache[cache_key] = all_docs

        retrieval_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"[ENHANCED_RETRIEVER] Retrieved {len(all_docs)} docs for stage={stage.value} "
            f"in {retrieval_time_ms}ms (reranking={reranking_applied})"
        )

        return RetrievalResult(
            documents=all_docs,
            query_original=query,
            query_rewritten=rewritten_query if rewritten_query != query else None,
            stage=stage,
            retrieval_time_ms=retrieval_time_ms,
            reranking_applied=reranking_applied,
        )

    def retrieve_for_context(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        stage: Optional[ConversationStage] = None,
        language: str = "hi",
    ) -> str:
        """
        Retrieve and format documents as context string for LLM.

        Args:
            query: Current user query
            conversation_history: Previous conversation
            stage: Conversation stage
            language: Preferred language

        Returns:
            Formatted context string
        """
        result = self.retrieve(
            query=query,
            conversation_history=conversation_history,
            stage=stage,
            language=language,
        )

        if not result.documents:
            return ""

        # Format as context
        context_parts = []
        for i, doc in enumerate(result.documents, 1):
            doc_type = doc.get("metadata", {}).get("type", "info")
            content = doc.get("content", "")
            context_parts.append(f"[{doc_type.upper()} {i}]\n{content}")

        return "\n\n".join(context_parts)

    def clear_cache(self):
        """Clear the query cache."""
        self._query_cache.clear()
        logger.info("[ENHANCED_RETRIEVER] Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "cache_size": len(self._query_cache),
            "cache_max_size": self.cache_size,
            "reranking_enabled": self.enable_reranking,
            "query_rewriting_enabled": self.enable_query_rewriting,
        }
