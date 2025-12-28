"""Application settings loaded from environment variables."""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    sarvam_api_key: str = ""
    anthropic_api_key: str = ""

    # Sarvam AI endpoints
    sarvam_base_url: str = "https://api.sarvam.ai"
    sarvam_stt_endpoint: str = "/speech-to-text"
    sarvam_tts_endpoint: str = "/text-to-speech"

    # Claude settings
    claude_model: str = "claude-3-5-haiku-20241022"
    claude_max_tokens: int = 1024

    # Voice agent settings
    default_language: str = "hi"  # Hindi
    supported_languages: list = ["hi", "ta", "te", "kn", "ml", "en"]

    # Audio settings
    sample_rate: int = 16000
    audio_format: str = "wav"

    # Fallback settings
    use_local_asr_fallback: bool = True
    use_browser_tts_fallback: bool = True

    # Speech provider settings (sarvam, whisper, indicconformer, piper, indicf5, parler)
    stt_provider: str = "indicconformer"  # Options: sarvam, whisper, indicconformer
    tts_provider: str = "indicf5"         # Options: sarvam, piper, indicf5, parler

    # Whisper STT settings
    whisper_model: str = "small"   # tiny (fastest), base (fast), small, medium, large-v3
    whisper_device: str = "cpu"    # cpu, cuda
    whisper_compute_type: str = "int8"  # int8 (CPU), float16 (GPU)

    # Piper TTS settings
    piper_models_dir: Optional[str] = None  # Directory for voice models

    # Performance tuning for CPU-only deployments
    use_minimal_prompts: bool = True  # Reduce prompt size for faster inference
    disable_rag: bool = False  # Enable RAG for context-aware responses
    llm_max_tokens: int = 150  # Relaxed response length

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
