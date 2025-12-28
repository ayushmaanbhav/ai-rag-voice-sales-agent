"""
Tests for RAG components: embeddings, vector store, retriever.
"""
import pytest
from rag.embeddings import OllamaEmbeddingService, OllamaEmbeddingFunction
from rag.vector_store import ChromaVectorStore
from rag.retriever import HybridRetriever


class TestOllamaEmbeddings:
    """Test Ollama embedding service."""

    @pytest.fixture
    def embedding_service(self):
        return OllamaEmbeddingService()

    def test_embed_text(self, embedding_service):
        """Test single text embedding."""
        embedding = embedding_service.embed_text("What is gold loan interest rate?")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    def test_embed_documents(self, embedding_service):
        """Test batch document embedding."""
        texts = [
            "Kotak offers gold loans at 9-12% interest",
            "Muthoot Finance charges 18-24% interest"
        ]
        embeddings = embedding_service.embed_documents(texts)
        assert len(embeddings) == 2
        assert all(len(e) == len(embeddings[0]) for e in embeddings)

    def test_embed_empty_list(self, embedding_service):
        """Test empty list returns empty."""
        embeddings = embedding_service.embed_documents([])
        assert embeddings == []


class TestOllamaEmbeddingFunction:
    """Test ChromaDB-compatible embedding function."""

    @pytest.fixture
    def embedding_fn(self):
        return OllamaEmbeddingFunction()

    def test_name(self, embedding_fn):
        """Test function name."""
        assert "ollama" in embedding_fn.name()

    def test_embed_query(self, embedding_fn):
        """Test query embedding returns wrapped list."""
        result = embedding_fn.embed_query("test query")
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)

    def test_callable(self, embedding_fn):
        """Test __call__ for ChromaDB compatibility."""
        result = embedding_fn(["doc1", "doc2"])
        assert len(result) == 2


class TestVectorStore:
    """Test ChromaDB vector store."""

    @pytest.fixture
    def vector_store(self):
        return ChromaVectorStore(
            persist_dir="./data/chroma",
            collection_name="kotak_knowledge"
        )

    def test_count(self, vector_store):
        """Test document count."""
        count = vector_store.count()
        assert count > 0
        print(f"Vector store contains {count} documents")

    def test_query(self, vector_store):
        """Test semantic query."""
        results = vector_store.query("interest rate", n_results=3)
        # Results is a dict with ids, documents, metadatas, distances
        assert "ids" in results
        assert "documents" in results
        assert "distances" in results
        assert len(results["ids"]) <= 3

    def test_query_with_filter(self, vector_store):
        """Test query with metadata filter."""
        results = vector_store.query(
            "gold loan",
            n_results=5,
            where={"type": "product"}
        )
        # Check metadatas have correct type
        for metadata in results["metadatas"]:
            assert metadata.get("type") == "product"


class TestHybridRetriever:
    """Test hybrid retriever."""

    @pytest.fixture
    def retriever(self):
        vector_store = ChromaVectorStore(
            persist_dir="./data/chroma",
            collection_name="kotak_knowledge"
        )
        return HybridRetriever(vector_store)

    def test_retrieve(self, retriever):
        """Test hybrid retrieval."""
        results = retriever.retrieve("What is Kotak gold loan interest?", n_results=3)
        assert len(results) <= 3
        assert all("content" in r for r in results)

    def test_retrieve_for_objection(self, retriever):
        """Test objection-prioritized retrieval."""
        results = retriever.retrieve_for_objection(
            "Is my gold safe?",
            n_results=3,
            language="en"
        )
        assert len(results) > 0
        # Should prioritize safety-related content
        contents = " ".join(r["content"].lower() for r in results)
        assert any(word in contents for word in ["safe", "security", "vault", "insurance"])

    def test_retrieve_hindi(self, retriever):
        """Test Hindi content retrieval."""
        results = retriever.retrieve(
            "ब्याज दर क्या है?",  # What is the interest rate?
            n_results=3,
            language="hi"
        )
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
