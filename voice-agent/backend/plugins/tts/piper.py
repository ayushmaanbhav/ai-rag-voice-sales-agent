"""
Piper TTS Plugin.

Wraps Piper for the modular plugin architecture.
"""

import base64
import tempfile
import os
import time
import wave
import logging
from typing import Optional, Dict, List
from pathlib import Path

from core.interfaces import BaseTTSPlugin, TTSResult

logger = logging.getLogger(__name__)

# Voice model configurations
VOICE_MODELS = {
    "hi": {
        "default": "hi_IN-priyamvada-medium",
        "male": "hi_IN-rohan-medium",
        "female": "hi_IN-priyamvada-medium",
        "pratham": "hi_IN-pratham-medium",
    },
    "en": {
        "default": "en_US-amy-medium",
        "male": "en_US-ryan-medium",
        "female": "en_US-amy-medium",
    },
    "ta": {"default": "en_IN-iisc_mit-medium"},
    "te": {"default": "en_IN-iisc_mit-medium"},
    "kn": {"default": "en_IN-iisc_mit-medium"},
    "ml": {"default": "en_IN-iisc_mit-medium"},
}

# Model download base URL
PIPER_VOICES_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"


class PiperTTSPlugin(BaseTTSPlugin):
    """
    Piper-based Text-to-Speech plugin.

    Implements the TTSPlugin interface for the modular architecture.
    """

    def __init__(
        self,
        models_dir: Optional[str] = None,
        sample_rate: int = 22050,
        **kwargs,
    ):
        """
        Initialize Piper TTS plugin.

        Args:
            models_dir: Directory to store voice models
            sample_rate: Output sample rate (default 22050)
        """
        super().__init__()
        self.models_dir = Path(models_dir or os.path.expanduser("~/.piper-models"))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._sample_rate = sample_rate
        self._voices_cache = {}

    @property
    def name(self) -> str:
        return "piper"

    @property
    def supported_languages(self) -> list[str]:
        return list(VOICE_MODELS.keys())

    @property
    def available_voices(self) -> Dict[str, List[str]]:
        return {
            lang: list(voices.keys())
            for lang, voices in VOICE_MODELS.items()
        }

    def _get_model_path(self, voice_name: str) -> tuple[Path, Path]:
        """Get paths for model and config files."""
        parts = voice_name.split("-")
        if len(parts) >= 2:
            lang_region = parts[0]
            lang = lang_region.split("_")[0]
        else:
            lang = "en"
            lang_region = "en_US"

        model_dir = self.models_dir / lang / lang_region
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"{voice_name}.onnx"
        config_path = model_dir / f"{voice_name}.onnx.json"

        return model_path, config_path

    async def _download_model(self, voice_name: str) -> tuple[Path, Path]:
        """Download voice model if not present."""
        model_path, config_path = self._get_model_path(voice_name)

        if model_path.exists() and config_path.exists():
            logger.info(f"[PIPER] Model already exists: {voice_name}")
            return model_path, config_path

        # Parse voice path
        parts = voice_name.split("-")
        lang_region = parts[0]
        speaker = parts[1]
        quality = parts[2] if len(parts) > 2 else "medium"
        lang = lang_region.split("_")[0]

        # Download model
        model_url = f"{PIPER_VOICES_URL}/{lang}/{lang_region}/{speaker}/{quality}/{voice_name}.onnx"
        config_url = f"{model_url}.json"

        logger.info(f"[PIPER] Downloading model: {voice_name}")

        import httpx
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            logger.info(f"[PIPER] Downloading {model_url}")
            resp = await client.get(model_url)
            resp.raise_for_status()
            model_path.write_bytes(resp.content)
            logger.info(f"[PIPER] Model saved: {model_path}")

            logger.info(f"[PIPER] Downloading {config_url}")
            resp = await client.get(config_url)
            resp.raise_for_status()
            config_path.write_bytes(resp.content)
            logger.info(f"[PIPER] Config saved: {config_path}")

        return model_path, config_path

    def _get_voice_for_language(self, language: str, voice: Optional[str] = None) -> str:
        """Get voice model name for language."""
        lang_voices = VOICE_MODELS.get(language, VOICE_MODELS["en"])

        if voice and voice in lang_voices:
            return lang_voices[voice]

        return lang_voices.get(
            "default",
            lang_voices.get("female", list(lang_voices.values())[0])
        )

    async def synthesize(
        self,
        text: str,
        language: str = "hi",
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> TTSResult:
        """
        Synthesize text to speech using Piper.

        Args:
            text: Text to convert to speech
            language: Language code (hi, en, etc.)
            voice: Voice name (default, male, female)
            speed: Speech speed multiplier

        Returns:
            TTSResult with audio data
        """
        start_time = time.time()

        # Get voice model
        voice_name = self._get_voice_for_language(language, voice)
        logger.info(f"[PIPER] Using voice: {voice_name}")

        try:
            # Ensure model is downloaded
            model_path, config_path = await self._download_model(voice_name)

            # Create temp file for output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                output_path = f.name

            try:
                from piper import PiperVoice

                # Load voice (cache for reuse)
                if voice_name not in self._voices_cache:
                    logger.info(f"[PIPER] Loading voice model: {voice_name}")
                    self._voices_cache[voice_name] = PiperVoice.load(
                        str(model_path),
                        str(config_path),
                    )

                piper_voice = self._voices_cache[voice_name]

                # Synthesize to WAV
                with wave.open(output_path, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(piper_voice.config.sample_rate)
                    for audio_chunk in piper_voice.synthesize(text):
                        wav_file.writeframes(audio_chunk.audio_int16_bytes)

                # Read output
                with open(output_path, 'rb') as f:
                    audio_bytes = f.read()

                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

                duration_ms = int((time.time() - start_time) * 1000)
                logger.info(f"[PIPER] Synthesized in {duration_ms}ms: {text[:30]}...")

                # Record metrics
                self._record_call(duration_ms)

                return TTSResult(
                    audio_bytes=audio_bytes,
                    audio_base64=audio_base64,
                    format="wav",
                    sample_rate=piper_voice.config.sample_rate,
                    duration_ms=duration_ms,
                    provider=self.name,
                    voice=voice_name,
                )

            finally:
                if os.path.exists(output_path):
                    os.remove(output_path)

        except Exception as e:
            logger.error(f"[PIPER] Synthesis error: {e}")
            self._record_error()
            raise

    async def health_check(self) -> bool:
        """Check if Piper is available."""
        try:
            from piper import PiperVoice
            return True
        except ImportError:
            return False


def register_plugin():
    """Register the Piper TTS plugin with the registry."""
    from core.registry import get_registry
    registry = get_registry()
    registry.register_tts("piper", PiperTTSPlugin)
    logger.info("[PIPER] Plugin registered")
