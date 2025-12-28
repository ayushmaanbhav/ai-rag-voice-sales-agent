"""
Whisper STT Plugin.

Wraps OpenAI Whisper for the modular plugin architecture.
"""

import base64
import tempfile
import os
import time
import logging
from typing import Optional

from pydub import AudioSegment

from core.interfaces import BaseSTTPlugin, STTResult

logger = logging.getLogger(__name__)

# Language code mapping
LANGUAGE_MAP = {
    "hi": "hi",      # Hindi
    "ta": "ta",      # Tamil
    "te": "te",      # Telugu
    "kn": "kn",      # Kannada
    "ml": "ml",      # Malayalam
    "en": "en",      # English
    "mr": "mr",      # Marathi
    "bn": "bn",      # Bengali
    "gu": "gu",      # Gujarati
    "pa": "pa",      # Punjabi
}

# Language-specific prompts to guide transcription
LANGUAGE_PROMPTS = {
    "hi": "यह हिंदी में बातचीत है।",
    "ta": "இது தமிழில் உரையாடல்.",
    "te": "ఇది తెలుగులో సంభాషణ.",
    "kn": "ಇದು ಕನ್ನಡದಲ್ಲಿ ಸಂಭಾಷಣೆ.",
    "ml": "ഇത് മലയാളത്തിലെ സംഭാഷണമാണ്.",
    "en": "This is a conversation in English.",
    "mr": "ही मराठीतील संभाषण आहे.",
    "bn": "এটি বাংলায় কথোপকথন।",
    "gu": "આ ગુજરાતીમાં વાતચીત છે.",
    "pa": "ਇਹ ਪੰਜਾਬੀ ਵਿੱਚ ਗੱਲਬਾਤ ਹੈ.",
}


class WhisperSTTPlugin(BaseSTTPlugin):
    """
    OpenAI Whisper based Speech-to-Text plugin.

    Implements the STTPlugin interface for the modular architecture.
    """

    def __init__(
        self,
        model: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",  # Kept for API compat
        **kwargs,
    ):
        """
        Initialize Whisper STT plugin.

        Args:
            model: Model size (tiny, base, small, medium, large)
            device: Device to use (cpu, cuda)
            compute_type: Ignored for openai-whisper
        """
        super().__init__()
        self._model_size = model
        self._device = device
        self._model = None

    def _load_model(self):
        """Lazy load the Whisper model."""
        if self._model is None:
            import whisper
            logger.info(f"[WHISPER] Loading model: {self._model_size} on {self._device}")
            start = time.time()
            self._model = whisper.load_model(self._model_size, device=self._device)
            logger.info(f"[WHISPER] Model loaded in {time.time() - start:.2f}s")
        return self._model

    @property
    def name(self) -> str:
        return f"whisper-{self._model_size}"

    @property
    def supported_languages(self) -> list[str]:
        return list(LANGUAGE_MAP.keys())

    def _convert_audio_from_base64(self, audio_base64: str, audio_format: str) -> str:
        """Convert base64 audio to WAV format suitable for Whisper."""
        audio_bytes = base64.b64decode(audio_base64)
        return self._convert_audio_from_bytes(audio_bytes, audio_format)

    def _convert_audio_from_bytes(self, audio_bytes: bytes, audio_format: str) -> str:
        """Convert audio bytes to WAV format suitable for Whisper."""
        with tempfile.NamedTemporaryFile(suffix=f'.{audio_format}', delete=False) as f:
            f.write(audio_bytes)
            input_path = f.name

        # Create a separate output path
        wav_path = tempfile.mktemp(suffix='_converted.wav')

        try:
            audio = AudioSegment.from_file(input_path, format=audio_format)
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(wav_path, format="wav")
            return wav_path
        finally:
            if os.path.exists(input_path) and input_path != wav_path:
                os.remove(input_path)

    async def transcribe(
        self,
        audio_bytes: bytes,
        language: str = "hi",
        audio_format: str = "wav",
    ) -> STTResult:
        """
        Transcribe audio to text using Whisper.

        Args:
            audio_bytes: Raw audio bytes
            language: Language code (hi, ta, te, etc.)
            audio_format: Audio format (webm, wav, mp3)

        Returns:
            STTResult with transcribed text and metadata
        """
        start_time = time.time()

        # Check if input is base64 encoded (for backward compatibility)
        try:
            # Try to decode as base64 first
            if isinstance(audio_bytes, str):
                audio_bytes = base64.b64decode(audio_bytes)
            elif isinstance(audio_bytes, bytes):
                # Check if it looks like base64
                try:
                    decoded = base64.b64decode(audio_bytes)
                    # If successful and significantly smaller, it was base64
                    if len(decoded) < len(audio_bytes) * 0.9:
                        audio_bytes = decoded
                except Exception:
                    pass  # Keep original bytes
        except Exception:
            pass

        # Convert audio to WAV
        logger.info(f"[WHISPER] Converting {audio_format} audio...")
        wav_path = self._convert_audio_from_bytes(audio_bytes, audio_format)

        try:
            # Load model (lazy)
            model = self._load_model()

            # Map language code
            whisper_lang = LANGUAGE_MAP.get(language, "hi")

            # Transcribe with forced language
            logger.info(f"[WHISPER] Transcribing in {whisper_lang}...")

            initial_prompt = LANGUAGE_PROMPTS.get(whisper_lang, "")

            result = model.transcribe(
                wav_path,
                language=whisper_lang,
                task="transcribe",
                fp16=False,  # Use FP32 on CPU
                initial_prompt=initial_prompt,
            )

            text = result.get("text", "").strip()

            duration_ms = int((time.time() - start_time) * 1000)
            logger.info(f"[WHISPER] Transcribed in {duration_ms}ms: {text[:50]}...")

            # Record metrics
            self._record_call(duration_ms)

            return STTResult(
                text=text,
                language=result.get("language", whisper_lang),
                confidence=None,  # openai-whisper doesn't provide confidence
                duration_ms=duration_ms,
                provider=self.name,
                model=self._model_size,
            )

        except Exception as e:
            logger.error(f"[WHISPER] Transcription error: {e}")
            self._record_error()
            raise

        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)

    async def health_check(self) -> bool:
        """Check if Whisper is available."""
        try:
            import whisper
            return True
        except ImportError:
            return False


def register_plugin():
    """Register the Whisper STT plugin with the registry."""
    from core.registry import get_registry
    registry = get_registry()
    registry.register_stt("whisper", WhisperSTTPlugin)
    logger.info("[WHISPER] Plugin registered")
