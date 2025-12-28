"""
Embedding service using Ollama's qwen3-embedding:8b-q4_K_M model.

Provides multilingual embeddings for Hindi, English, and other Indian languages.
Uses Ollama API for local inference - no external API calls.
"""

import httpx
import time
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class OllamaEmbeddingService:
    """
    Embedding service using Ollama's qwen3-embedding:8b-q4_K_M model.

    Features:
    - 100+ language support including Hindi, Tamil, Telugu, etc.
    - 2560-dimensional embeddings
    - Local inference via Ollama
    - Async and sync support
    """

    def __init__(
        self,
        model: str = "qwen3-embedding:8b-q4_K_M",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0
    ):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        """Return embedding dimension (2560 for qwen3-embedding:8b-q4_K_M)."""
        if self._dimension is None:
            # Get dimension from a test embedding
            test_emb = self.embed_text("test")
            self._dimension = len(test_emb)
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text (synchronous).

        Uses sync httpx client to avoid async context issues.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding
        """
        start = time.time()
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.model,
                    "input": text
                }
            )
            response.raise_for_status()
            data = response.json()
            elapsed_ms = int((time.time() - start) * 1000)
            logger.debug(f"[EMBED] Single text: {elapsed_ms}ms ({len(text)} chars)")
            return data["embeddings"][0]

    async def aembed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text (async).

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.model,
                    "input": text
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"][0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts (synchronous).

        Uses sync httpx client to avoid async context issues.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        if not texts:
            return []

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.model,
                    "input": texts
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts (async).

        Uses Ollama's batch embedding API for efficiency.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        if not texts:
            return []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.model,
                    "input": texts
                }
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]

    def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a search query (synchronous).

        For qwen3-embedding, query and document embeddings use the same method.

        Args:
            query: Search query text

        Returns:
            Query embedding
        """
        return self.embed_text(query)

    async def aembed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a search query (async).

        Args:
            query: Search query text

        Returns:
            Query embedding
        """
        return await self.aembed_text(query)

    async def health_check(self) -> bool:
        """
        Check if Ollama embedding service is available.

        Returns:
            True if service is healthy
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/embed",
                    json={
                        "model": self.model,
                        "input": "health check"
                    }
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Embedding service health check failed: {e}")
            return False


# Convenience function for ChromaDB integration
class OllamaEmbeddingFunction:
    """
    ChromaDB-compatible embedding function wrapper.

    Usage with ChromaDB:
        embedding_fn = OllamaEmbeddingFunction()
        collection = client.create_collection(
            name="knowledge",
            embedding_function=embedding_fn
        )
    """

    def __init__(
        self,
        model: str = "qwen3-embedding:8b-q4_K_M",
        base_url: str = "http://localhost:11434"
    ):
        self._model = model
        self.service = OllamaEmbeddingService(model=model, base_url=base_url)

    def name(self) -> str:
        """Return embedding function name for ChromaDB."""
        return f"ollama_{self._model}"

    def embed_query(self, input: str) -> list[list[float]]:
        """
        Embed a single query for ChromaDB query operations.

        Args:
            input: Query text to embed

        Returns:
            Query embedding wrapped in a list (ChromaDB requirement)
        """
        return [self.service.embed_text(input)]

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        """
        Embed documents for ChromaDB.

        Args:
            input: List of texts to embed

        Returns:
            List of embeddings
        """
        return self.service.embed_documents(input)

    def __call__(self, input: list[str]) -> list[list[float]]:
        """
        ChromaDB calls this method to get embeddings.

        Args:
            input: List of texts to embed

        Returns:
            List of embeddings
        """
        return self.service.embed_documents(input)
