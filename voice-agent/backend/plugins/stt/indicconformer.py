"""
IndicConformer STT Plugin.

AI4Bharat's IndicConformer model for accurate Indian language ASR.
Supports 22 scheduled Indian languages with state-of-the-art accuracy.

Model: https://huggingface.co/ai4bharat/indicconformer-hi
"""

import tempfile
import os
import time
import logging
from typing import Optional

from core.interfaces import BaseSTTPlugin, STTResult

logger = logging.getLogger(__name__)

# Language code mapping for IndicConformer
# Maps ISO codes to IndicConformer language codes
LANGUAGE_MAP = {
    "hi": "hi",       # Hindi
    "ta": "ta",       # Tamil
    "te": "te",       # Telugu
    "kn": "kn",       # Kannada
    "ml": "ml",       # Malayalam
    "en": "en",       # English
    "mr": "mr",       # Marathi
    "bn": "bn",       # Bengali
    "gu": "gu",       # Gujarati
    "pa": "pa",       # Punjabi
    "or": "or",       # Odia
    "as": "as",       # Assamese
    "ur": "ur",       # Urdu
    "ne": "ne",       # Nepali
    "sa": "sa",       # Sanskrit
    "sd": "sd",       # Sindhi
    "ks": "ks",       # Kashmiri
    "doi": "doi",     # Dogri
    "kok": "kok",     # Konkani
    "mai": "mai",     # Maithili
    "mni": "mni",     # Manipuri
    "sat": "sat",     # Santali
    "bodo": "bodo",   # Bodo
}

# Local model path (downloaded from e2enetworks)
# Supports all 22 Indian languages with NeMo framework
import os
LOCAL_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "models",
    "indicconformer_multi_600m.nemo"
)


class IndicConformerSTTPlugin(BaseSTTPlugin):
    """
    AI4Bharat IndicConformer Speech-to-Text plugin.

    Uses NeMo framework for inference with IndicConformer models.
    Provides state-of-the-art accuracy for Indian languages.
    """

    def __init__(
        self,
        model_name: str = "ai4bharat/indicconformer-hi",
        device: str = "cpu",
        batch_size: int = 1,
        **kwargs,
    ):
        """
        Initialize IndicConformer STT plugin.

        Args:
            model_name: HuggingFace model name or path
            device: Device to use (cpu, cuda)
            batch_size: Batch size for inference
        """
        super().__init__()
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._model = None
        self._processor = None

    def _load_model(self, language: str = "hi"):
        """Lazy load the IndicConformer model from local .nemo file."""
        if self._model is not None:
            return self._model

        import torch
        import nemo.collections.asr as nemo_asr

        logger.info(f"[INDICCONFORMER] Loading model from: {LOCAL_MODEL_PATH}")
        start = time.time()

        try:
            # Load the local .nemo model
            self._model = nemo_asr.models.ASRModel.restore_from(LOCAL_MODEL_PATH)

            # Set device
            if self._device == "cuda" and torch.cuda.is_available():
                self._model = self._model.cuda()
            else:
                self._model = self._model.cpu()

            # Freeze model for inference
            self._model.freeze()

            logger.info(f"[INDICCONFORMER] Model loaded in {time.time() - start:.2f}s")
            return self._model

        except Exception as e:
            logger.error(f"[INDICCONFORMER] Failed to load model: {e}")
            raise

    @property
    def name(self) -> str:
        return "indicconformer"

    @property
    def supported_languages(self) -> list[str]:
        return list(LANGUAGE_MAP.keys())

    def _save_audio_to_temp(self, audio_bytes: bytes, audio_format: str) -> str:
        """Save audio bytes to temporary file."""
        with tempfile.NamedTemporaryFile(
            suffix=f'.{audio_format}',
            delete=False
        ) as f:
            f.write(audio_bytes)
            return f.name

    def _convert_to_wav(self, input_path: str) -> str:
        """Convert audio to WAV format if needed."""
        from pydub import AudioSegment

        output_path = tempfile.mktemp(suffix='_converted.wav')

        try:
            # Determine format from extension
            ext = os.path.splitext(input_path)[1].lstrip('.')
            audio = AudioSegment.from_file(input_path, format=ext)
            # Convert to 16kHz mono for ASR
            audio = audio.set_channels(1).set_frame_rate(16000)
            audio.export(output_path, format="wav")
            return output_path
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)

    async def transcribe(
        self,
        audio_base64: str,
        language: str = "hi",
        audio_format: str = "wav",
    ) -> STTResult:
        """
        Transcribe audio to text using IndicConformer.

        Args:
            audio_base64: Base64 encoded audio string
            language: Language code (hi, ta, te, etc.)
            audio_format: Audio format (webm, wav, mp3)

        Returns:
            STTResult with transcribed text and metadata
        """
        import base64
        start_time = time.time()

        # Decode base64 to bytes
        audio_bytes = base64.b64decode(audio_base64)

        # Save audio to temp file
        temp_path = self._save_audio_to_temp(audio_bytes, audio_format)

        try:
            # Convert to WAV if needed
            if audio_format != "wav":
                wav_path = self._convert_to_wav(temp_path)
            else:
                wav_path = temp_path
                temp_path = None  # Don't delete twice

            # Load model
            model = self._load_model(language)

            # Map language code
            indic_lang = LANGUAGE_MAP.get(language, "hi")

            logger.info(f"[INDICCONFORMER] Transcribing in {indic_lang}...")

            # NeMo model transcription
            # Set the decoder to CTC for faster inference
            if hasattr(model, 'cur_decoder'):
                model.cur_decoder = "ctc"

            # Transcribe using NeMo's transcribe method
            # It accepts file paths directly
            transcription = model.transcribe(
                [wav_path],
                batch_size=1,
                language_id=indic_lang
            )

            if isinstance(transcription, list):
                text = transcription[0] if transcription else ""
            else:
                text = str(transcription)
            text = text.strip()

            duration_ms = int((time.time() - start_time) * 1000)
            logger.info(f"[INDICCONFORMER] Transcribed in {duration_ms}ms: {text[:50]}...")

            # Record metrics
            self._record_call(duration_ms)

            return STTResult(
                text=text,
                language=indic_lang,
                confidence=None,  # IndicConformer doesn't provide confidence scores
                duration_ms=duration_ms,
                provider=self.name,
                model=self._model_name,
            )

        except Exception as e:
            logger.error(f"[INDICCONFORMER] Transcription error: {e}")
            self._record_error()
            raise

        finally:
            # Cleanup temp files
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            if 'wav_path' in locals() and wav_path != temp_path and os.path.exists(wav_path):
                os.remove(wav_path)

    async def health_check(self) -> bool:
        """Check if IndicConformer is available."""
        try:
            import nemo.collections.asr as nemo_asr
            # Also check if model file exists
            return os.path.exists(LOCAL_MODEL_PATH)
        except ImportError:
            return False


def register_plugin():
    """Register the IndicConformer STT plugin with the registry."""
    from core.registry import get_registry
    registry = get_registry()
    registry.register_stt("indicconformer", IndicConformerSTTPlugin)
    logger.info("[INDICCONFORMER] Plugin registered")
