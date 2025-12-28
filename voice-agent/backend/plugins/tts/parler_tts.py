"""
Indic Parler-TTS Plugin.

AI4Bharat's Indic Parler-TTS for high-quality Indian language TTS.
Supports 21 Indian languages with natural-sounding speech synthesis.
No reference audio required - uses text descriptions to control voice.

Model: https://huggingface.co/ai4bharat/indic-parler-tts
"""

import base64
import tempfile
import os
import time
import logging
from typing import Optional, Dict, List

from core.interfaces import BaseTTSPlugin, TTSResult

logger = logging.getLogger(__name__)

# Language code mapping for Indic Parler-TTS
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
    "as": "assamese",
    "ur": "urdu",
    "ne": "nepali",
    "sa": "sanskrit",
    "sd": "sindhi",
    "ks": "kashmiri",
    "doi": "dogri",
    "kok": "konkani",
    "mai": "maithili",
    "mni": "manipuri",
}

# Voice descriptions for different speaking styles
VOICE_DESCRIPTIONS = {
    "default": "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch.",
    "calm": "A female speaker speaks in a calm and measured tone with clear pronunciation.",
    "expressive": "A female speaker delivers an expressive and animated speech with varied intonation.",
    "professional": "A female speaker speaks in a clear, professional manner at a moderate pace.",
    "male_default": "A male speaker delivers a clear and steady speech with moderate speed.",
}


class IndicParlerTTSPlugin(BaseTTSPlugin):
    """
    AI4Bharat Indic Parler-TTS plugin.

    Uses Parler-TTS architecture for text-to-speech synthesis.
    Supports 21 Indian languages with controllable voice characteristics.
    """

    def __init__(
        self,
        model_name: str = "ai4bharat/indic-parler-tts",
        device: str = "cpu",
        **kwargs,
    ):
        """
        Initialize Indic Parler-TTS plugin.

        Args:
            model_name: HuggingFace model name
            device: Device to use (cpu, cuda)
        """
        super().__init__()
        self._model_name = model_name
        self._device = device
        self._model = None
        self._tokenizer = None
        self._description_tokenizer = None

    def _load_model(self):
        """Lazy load the Parler-TTS model."""
        if self._model is not None:
            return self._model

        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        logger.info(f"[PARLER-TTS] Loading model: {self._model_name} on {self._device}")
        start = time.time()

        try:
            device = self._device
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("[PARLER-TTS] CUDA not available, falling back to CPU")
                device = "cpu"

            self._model = ParlerTTSForConditionalGeneration.from_pretrained(
                self._model_name
            ).to(device)

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._description_tokenizer = AutoTokenizer.from_pretrained(
                self._model.config.text_encoder._name_or_path
            )

            logger.info(f"[PARLER-TTS] Model loaded in {time.time() - start:.2f}s")
            return self._model

        except Exception as e:
            logger.error(f"[PARLER-TTS] Failed to load model: {e}")
            raise

    @property
    def name(self) -> str:
        return "indic-parler-tts"

    @property
    def supported_languages(self) -> list[str]:
        return list(LANGUAGE_MAP.keys())

    @property
    def available_voices(self) -> Dict[str, List[str]]:
        return {
            lang: list(VOICE_DESCRIPTIONS.keys())
            for lang in LANGUAGE_MAP.keys()
        }

    async def synthesize(
        self,
        text: str,
        language: str = "hi",
        voice: Optional[str] = None,
        speed: float = 1.0,
    ) -> TTSResult:
        """
        Synthesize text to speech using Indic Parler-TTS.

        Args:
            text: Text to convert to speech
            language: Language code (hi, ta, te, etc.)
            voice: Voice style (default, calm, expressive, professional)
            speed: Speech speed multiplier (not directly supported, ignored)

        Returns:
            TTSResult with audio data
        """
        import torch
        import soundfile as sf

        start_time = time.time()

        # Map language and voice
        indic_lang = LANGUAGE_MAP.get(language, "hindi")
        voice_style = voice or "default"
        description = VOICE_DESCRIPTIONS.get(voice_style, VOICE_DESCRIPTIONS["default"])

        logger.info(f"[PARLER-TTS] Synthesizing in {indic_lang}: {text[:50]}...")

        try:
            # Load model
            model = self._load_model()
            device = next(model.parameters()).device

            # Prepare inputs
            description_input_ids = self._description_tokenizer(
                description, return_tensors="pt"
            ).to(device)

            prompt_input_ids = self._tokenizer(
                text, return_tensors="pt"
            ).to(device)

            # Generate audio
            with torch.no_grad():
                generation = model.generate(
                    input_ids=description_input_ids.input_ids,
                    attention_mask=description_input_ids.attention_mask,
                    prompt_input_ids=prompt_input_ids.input_ids,
                    prompt_attention_mask=prompt_input_ids.attention_mask,
                )

            audio_arr = generation.cpu().numpy().squeeze()

            # Save to temp file
            output_path = tempfile.mktemp(suffix='.wav')
            sample_rate = model.config.sampling_rate

            try:
                sf.write(output_path, audio_arr, sample_rate)

                # Read back as bytes
                with open(output_path, 'rb') as f:
                    audio_bytes = f.read()

                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

                duration_ms = int((time.time() - start_time) * 1000)
                logger.info(f"[PARLER-TTS] Synthesized in {duration_ms}ms")

                # Record metrics
                self._record_call(duration_ms)

                return TTSResult(
                    audio_bytes=audio_bytes,
                    audio_base64=audio_base64,
                    format="wav",
                    sample_rate=sample_rate,
                    duration_ms=duration_ms,
                    provider=self.name,
                    voice=f"{indic_lang}-{voice_style}",
                    model=self._model_name,
                )

            finally:
                if os.path.exists(output_path):
                    os.remove(output_path)

        except Exception as e:
            logger.error(f"[PARLER-TTS] Synthesis error: {e}")
            self._record_error()
            raise

    async def health_check(self) -> bool:
        """Check if Parler-TTS is available."""
        try:
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer
            return True
        except ImportError:
            return False


def register_plugin():
    """Register the Indic Parler-TTS plugin with the registry."""
    from core.registry import get_registry
    registry = get_registry()
    registry.register_tts("parler", IndicParlerTTSPlugin)
    logger.info("[PARLER-TTS] Plugin registered")
