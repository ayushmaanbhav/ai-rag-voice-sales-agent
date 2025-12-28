"""
IndicF5 TTS Plugin.

AI4Bharat's IndicF5 model for high-quality Indian language TTS.
Supports 11 Indian languages with natural-sounding speech synthesis.

Model: https://huggingface.co/ai4bharat/IndicF5
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

# Local model path
LOCAL_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "models",
    "IndicF5"
)

# Language code mapping for IndicF5
LANGUAGE_MAP = {
    "hi": "hindi",
    "ta": "tamil",
    "te": "telugu",
    "kn": "kannada",
    "ml": "malayalam",
    "en": "english",
    "mr": "marathi",
    "bn": "bengali",
    "gu": "gujarati",
    "pa": "punjabi",
    "or": "odia",
}

# Default reference audio paths per language (for voice cloning)
# Using Punjabi female voice as default (works well for all Indian languages)
DEFAULT_REFERENCE_AUDIO = {
    "hi": "prompts/PAN_F_HAPPY_00001.wav",
    "ta": "prompts/PAN_F_HAPPY_00001.wav",
    "te": "prompts/PAN_F_HAPPY_00001.wav",
    "kn": "prompts/PAN_F_HAPPY_00001.wav",
    "ml": "prompts/PAN_F_HAPPY_00001.wav",
    "en": "prompts/PAN_F_HAPPY_00001.wav",
    "mr": "prompts/PAN_F_HAPPY_00001.wav",
    "bn": "prompts/PAN_F_HAPPY_00001.wav",
    "gu": "prompts/PAN_F_HAPPY_00001.wav",
    "pa": "prompts/PAN_F_HAPPY_00001.wav",
    "or": "prompts/PAN_F_HAPPY_00001.wav",
}


class IndicF5TTSPlugin(BaseTTSPlugin):
    """
    AI4Bharat IndicF5 Text-to-Speech plugin.

    Uses F5-TTS architecture fine-tuned on Indian languages.
    Supports zero-shot voice cloning with reference audio.
    """

    def __init__(
        self,
        model_name: str = "ai4bharat/IndicF5",
        device: str = "cpu",
        reference_audio: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize IndicF5 TTS plugin.

        Args:
            model_name: HuggingFace model name or path
            device: Device to use (cpu, cuda)
            reference_audio: Path to reference audio for voice cloning
            cache_dir: Directory for model cache
        """
        super().__init__()
        self._model_name = model_name
        self._device = device
        self._reference_audio = reference_audio
        self._cache_dir = Path(cache_dir or os.path.expanduser("~/.cache/indicf5"))
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._vocoder = None

    def _load_model(self):
        """Lazy load the IndicF5 model from local files."""
        if self._model is not None:
            return self._model

        import torch
        from transformers import AutoModel

        # Check for local model first
        local_path = LOCAL_MODEL_PATH
        if os.path.exists(os.path.join(local_path, "model.safetensors")):
            model_path = local_path
            logger.info(f"[INDICF5] Loading local model from: {model_path}")
        else:
            model_path = self._model_name
            logger.info(f"[INDICF5] Loading model: {model_path} on {self._device}")

        start = time.time()

        try:
            # Load using transformers with trust_remote_code for custom model class
            self._model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=os.path.isdir(model_path),
            )

            device = self._device
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("[INDICF5] CUDA not available, falling back to CPU")
                device = "cpu"

            if device == "cuda":
                self._model = self._model.cuda()

            logger.info(f"[INDICF5] Model loaded in {time.time() - start:.2f}s")
            return self._model

        except Exception as e:
            logger.error(f"[INDICF5] Failed to load model: {e}")
            raise

    @property
    def name(self) -> str:
        return "indicf5"

    @property
    def supported_languages(self) -> list[str]:
        return list(LANGUAGE_MAP.keys())

    @property
    def available_voices(self) -> Dict[str, List[str]]:
        return {
            lang: ["default", "custom"]
            for lang in LANGUAGE_MAP.keys()
        }

    def _get_reference_audio(self, language: str) -> Optional[str]:
        """Get reference audio path for voice cloning."""
        if self._reference_audio:
            return self._reference_audio

        # Check for language-specific default (relative to model directory)
        rel_path = DEFAULT_REFERENCE_AUDIO.get(language)
        if rel_path:
            full_path = os.path.join(LOCAL_MODEL_PATH, rel_path)
            if os.path.exists(full_path):
                return full_path

        return None

    async def synthesize(
        self,
        text: str,
        language: str = "hi",
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> TTSResult:
        """
        Synthesize text to speech using IndicF5.

        Args:
            text: Text to convert to speech
            language: Language code (hi, ta, te, etc.)
            voice: Voice name (default, custom with reference audio)
            speed: Speech speed multiplier

        Returns:
            TTSResult with audio data
        """
        import numpy as np
        import soundfile as sf

        start_time = time.time()

        # Map language code
        indic_lang = LANGUAGE_MAP.get(language, "hindi")
        logger.info(f"[INDICF5] Synthesizing in {indic_lang}: {text[:50]}...")

        try:
            # Load model
            model = self._load_model()

            # Get reference audio for voice cloning
            ref_audio_path = self._get_reference_audio(language)

            # If no reference audio, we need to use a default or synthesize without it
            if ref_audio_path is None:
                # Use a bundled default reference audio or generate a placeholder
                ref_audio_path = self._get_or_create_default_reference(language)

            # Reference text (should match the reference audio)
            ref_text = self._get_reference_text(language)

            # Create temp file for output
            output_path = tempfile.mktemp(suffix='.wav')

            try:
                # Call model forward (synthesize)
                audio_array = model(
                    text=text,
                    ref_audio_path=ref_audio_path,
                    ref_text=ref_text,
                )

                # Convert to float32 if needed
                if audio_array.dtype == np.int16:
                    audio_array = audio_array.astype(np.float32) / 32768.0

                # Save to WAV
                sf.write(output_path, np.array(audio_array, dtype=np.float32), 24000)

                # Read output
                with open(output_path, 'rb') as f:
                    audio_bytes = f.read()

                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

                duration_ms = int((time.time() - start_time) * 1000)
                logger.info(f"[INDICF5] Synthesized in {duration_ms}ms")

                # Record metrics
                self._record_call(duration_ms)

                return TTSResult(
                    audio_bytes=audio_bytes,
                    audio_base64=audio_base64,
                    format="wav",
                    sample_rate=24000,
                    duration_ms=duration_ms,
                    provider=self.name,
                    voice=f"{indic_lang}-default",
                    model=self._model_name,
                )

            finally:
                if os.path.exists(output_path):
                    os.remove(output_path)

        except Exception as e:
            logger.error(f"[INDICF5] Synthesis error: {e}")
            self._record_error()
            raise

    def _get_or_create_default_reference(self, language: str) -> str:
        """Get or create a default reference audio file."""
        # Check for bundled reference audio
        ref_dir = os.path.join(LOCAL_MODEL_PATH, "reference_audio")
        ref_path = os.path.join(ref_dir, f"{language}_default.wav")

        if os.path.exists(ref_path):
            return ref_path

        # Create a simple reference audio as placeholder
        # In production, you'd want proper reference audio files for each language
        os.makedirs(ref_dir, exist_ok=True)

        # Generate a simple sine wave as placeholder
        import numpy as np
        import soundfile as sf

        duration = 2.0  # seconds
        sample_rate = 24000
        frequency = 440  # Hz
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = 0.3 * np.sin(2 * np.pi * frequency * t)
        sf.write(ref_path, audio.astype(np.float32), sample_rate)

        logger.warning(f"[INDICF5] Created placeholder reference audio for {language}. For better quality, provide actual reference audio.")
        return ref_path

    def _get_reference_text(self, language: str) -> str:
        """Get reference text for the language.

        Note: The reference text should match the content of the reference audio.
        Using Punjabi reference audio (PAN_F_HAPPY_00001.wav) with matching text.
        """
        # Reference text that matches the Punjabi reference audio
        ref_text = "ਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ ਹਨ।"
        return ref_text

    async def health_check(self) -> bool:
        """Check if IndicF5 is available."""
        try:
            from indicf5 import IndicF5
            return True
        except ImportError:
            pass

        try:
            from transformers import AutoProcessor, AutoModel
            return True
        except ImportError:
            return False


def register_plugin():
    """Register the IndicF5 TTS plugin with the registry."""
    from core.registry import get_registry
    registry = get_registry()
    registry.register_tts("indicf5", IndicF5TTSPlugin)
    logger.info("[INDICF5] Plugin registered")
