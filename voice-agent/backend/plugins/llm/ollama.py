"""
Ollama LLM Plugin.

Wraps Ollama for the modular plugin architecture.
"""

import httpx
import time
import logging
from typing import Optional, List, Dict, Any

from core.interfaces import BaseLLMPlugin, LLMResult

logger = logging.getLogger(__name__)


class OllamaLLMPlugin(BaseLLMPlugin):
    """
    Ollama-based LLM plugin.

    Implements the LLMPlugin interface for the modular architecture.
    """

    def __init__(
        self,
        model: str = "qwen3:8b-q4_K_M",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 150,
        timeout: float = 60.0,
        **kwargs,
    ):
        """
        Initialize Ollama LLM plugin.

        Args:
            model: Ollama model name
            base_url: Ollama server URL
            temperature: Default temperature
            max_tokens: Default max tokens
            timeout: Request timeout
        """
        super().__init__()
        self._model = model
        self._base_url = base_url
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def model(self) -> str:
        return self._model

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    async def generate(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResult:
        """
        Generate text using Ollama.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            LLMResult with generated text and metadata
        """
        start_time = time.time()

        # Build messages for Ollama
        ollama_messages = []
        if system_prompt:
            ollama_messages.append({"role": "system", "content": system_prompt})
        ollama_messages.extend(messages)

        payload = {
            "model": self._model,
            "messages": ollama_messages,
            "stream": False,
            "think": False,  # Disable thinking mode for faster responses
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens or self._max_tokens,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            }
        }

        # Log prompt size for debugging
        prompt_chars = sum(len(m.get("content", "")) for m in ollama_messages)
        logger.info(
            f"[OLLAMA] Prompt size: {prompt_chars} chars across "
            f"{len(ollama_messages)} messages"
        )

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                logger.info(f"[OLLAMA] Calling model: {self._model}...")
                response = await client.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()

                result = response.json()
                text = result.get("message", {}).get("content", "")

                duration_ms = int((time.time() - start_time) * 1000)
                eval_count = result.get("eval_count", 0)
                prompt_eval_count = result.get("prompt_eval_count", 0)
                total_tokens = eval_count + prompt_eval_count

                logger.info(
                    f"[OLLAMA] Response: {duration_ms}ms | "
                    f"prompt_tokens:{prompt_eval_count} response_tokens:{eval_count} | "
                    f"text:{text[:50]}..."
                )

                # Warn if slow
                if duration_ms > 5000:
                    logger.warning(
                        f"[OLLAMA] SLOW: {duration_ms}ms for {prompt_eval_count} prompt tokens"
                    )

                # Record metrics
                self._record_call(duration_ms, total_tokens)

                return LLMResult(
                    text=text.strip(),
                    duration_ms=duration_ms,
                    prompt_tokens=prompt_eval_count,
                    completion_tokens=eval_count,
                    total_tokens=total_tokens,
                    provider=self.name,
                    model=self._model,
                    finish_reason=result.get("done_reason"),
                )

        except httpx.ConnectError:
            logger.error("[OLLAMA] Server not running! Start with: ollama serve")
            self._record_error()
            raise ConnectionError("Ollama server not running")

        except httpx.TimeoutException:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(f"[OLLAMA] TIMEOUT after {duration_ms}ms")
            self._record_error()
            raise TimeoutError(f"Ollama timeout after {duration_ms}ms")

        except Exception as e:
            logger.error(f"[OLLAMA] Error: {type(e).__name__}: {e}")
            self._record_error()
            raise

    async def health_check(self) -> bool:
        """Check if Ollama is available and the model is loaded."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self._base_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [m.get("name", "") for m in models]
                    # Check if our model is available
                    if any(self._model in name for name in model_names):
                        return True
                    logger.warning(
                        f"[OLLAMA] Model {self._model} not found in {model_names}"
                    )
                return False
        except Exception as e:
            logger.error(f"[OLLAMA] Health check failed: {e}")
            return False


def register_plugin():
    """Register the Ollama LLM plugin with the registry."""
    from core.registry import get_registry
    registry = get_registry()
    registry.register_llm("ollama", OllamaLLMPlugin)
    logger.info("[OLLAMA] Plugin registered")
