"""
IndicTrans2 Translation Plugin.

AI4Bharat's IndicTrans2 model for high-quality Indian language translation.
Supports translation between English and 22 Indian languages.

Model: https://huggingface.co/ai4bharat/indictrans2-indic-en-dist-200M
"""

import time
import logging
from typing import Optional, List, Tuple

from core.interfaces import BaseTranslationPlugin, TranslationResult

logger = logging.getLogger(__name__)

# Language code mapping for IndicTrans2
# Maps ISO codes to IndicTrans2 language codes
LANGUAGE_MAP = {
    "hi": "hin_Deva",    # Hindi
    "ta": "tam_Taml",    # Tamil
    "te": "tel_Telu",    # Telugu
    "kn": "kan_Knda",    # Kannada
    "ml": "mal_Mlym",    # Malayalam
    "en": "eng_Latn",    # English
    "mr": "mar_Deva",    # Marathi
    "bn": "ben_Beng",    # Bengali
    "gu": "guj_Gujr",    # Gujarati
    "pa": "pan_Guru",    # Punjabi
    "or": "ory_Orya",    # Odia
    "as": "asm_Beng",    # Assamese
    "ur": "urd_Arab",    # Urdu
    "ne": "npi_Deva",    # Nepali
    "sa": "san_Deva",    # Sanskrit
    "sd": "snd_Arab",    # Sindhi
    "ks": "kas_Arab",    # Kashmiri
    "doi": "doi_Deva",   # Dogri
    "kok": "kok_Deva",   # Konkani
    "mai": "mai_Deva",   # Maithili
    "mni": "mni_Beng",   # Manipuri
    "sat": "sat_Olck",   # Santali
    "bodo": "brx_Deva",  # Bodo
}

# Model variants
MODELS = {
    "indic-en": "ai4bharat/indictrans2-indic-en-dist-200M",  # Indic → English
    "en-indic": "ai4bharat/indictrans2-en-indic-dist-200M",  # English → Indic
    "indic-indic": "ai4bharat/indictrans2-indic-indic-dist-320M",  # Indic ↔ Indic
}


class IndicTrans2Plugin(BaseTranslationPlugin):
    """
    AI4Bharat IndicTrans2 Translation plugin.

    Uses distilled 200M parameter model for fast inference.
    Supports translation between English and 22 Indian languages.
    """

    def __init__(
        self,
        model_size: str = "distilled-200M",
        device: str = "cpu",
        batch_size: int = 1,
        **kwargs,
    ):
        """
        Initialize IndicTrans2 plugin.

        Args:
            model_size: Model variant (distilled-200M, full-1B)
            device: Device to use (cpu, cuda)
            batch_size: Batch size for inference
        """
        super().__init__()
        self._model_size = model_size
        self._device = device
        self._batch_size = batch_size

        # Model instances (lazy loaded)
        self._indic_en_model = None
        self._en_indic_model = None
        self._tokenizer_indic_en = None
        self._tokenizer_en_indic = None

    def _load_model(self, direction: str):
        """
        Lazy load the translation model for given direction.

        Args:
            direction: 'indic-en' or 'en-indic'
        """
        if direction == "indic-en":
            if self._indic_en_model is not None:
                return self._indic_en_model, self._tokenizer_indic_en

            model_name = MODELS["indic-en"]
        else:
            if self._en_indic_model is not None:
                return self._en_indic_model, self._tokenizer_en_indic

            model_name = MODELS["en-indic"]

        logger.info(f"[INDICTRANS2] Loading model: {model_name} on {self._device}")
        start = time.time()

        try:
            # Try using the official IndicTrans2 package
            from IndicTransToolkit import IndicProcessor
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch

            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )

            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True,
            )

            if self._device == "cuda" and torch.cuda.is_available():
                model = model.cuda()

            # Store models
            if direction == "indic-en":
                self._indic_en_model = model
                self._tokenizer_indic_en = tokenizer
            else:
                self._en_indic_model = model
                self._tokenizer_en_indic = tokenizer

            logger.info(f"[INDICTRANS2] Model loaded in {time.time() - start:.2f}s")
            return model, tokenizer

        except ImportError as e:
            logger.error(f"[INDICTRANS2] Missing dependencies: {e}")
            raise

    @property
    def name(self) -> str:
        return "indictrans2"

    @property
    def supported_pairs(self) -> List[Tuple[str, str]]:
        """Return all supported translation pairs."""
        pairs = []
        indic_langs = [k for k in LANGUAGE_MAP.keys() if k != "en"]

        # English ↔ Indic pairs
        for lang in indic_langs:
            pairs.append(("en", lang))  # English → Indic
            pairs.append((lang, "en"))  # Indic → English

        return pairs

    def _get_direction(self, source_lang: str, target_lang: str) -> str:
        """Determine translation direction."""
        if source_lang == "en":
            return "en-indic"
        elif target_lang == "en":
            return "indic-en"
        else:
            return "indic-indic"

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> TranslationResult:
        """
        Translate text using IndicTrans2.

        Args:
            text: Text to translate
            source_language: Source language code (hi, en, etc.)
            target_language: Target language code

        Returns:
            TranslationResult with translated text
        """
        start_time = time.time()

        # Map language codes
        src_lang = LANGUAGE_MAP.get(source_language, source_language)
        tgt_lang = LANGUAGE_MAP.get(target_language, target_language)

        logger.info(f"[INDICTRANS2] Translating {src_lang} → {tgt_lang}: {text[:50]}...")

        try:
            # Determine direction and load appropriate model
            direction = self._get_direction(source_language, target_language)

            if direction == "indic-indic":
                # For Indic-Indic, we need two-step translation
                # First translate to English, then to target
                result = await self._translate_via_english(text, src_lang, tgt_lang)
                duration_ms = int((time.time() - start_time) * 1000)
                return TranslationResult(
                    translated_text=result,
                    source_language=source_language,
                    target_language=target_language,
                    duration_ms=duration_ms,
                    provider=self.name,
                    model=f"{self._model_size}-pivot",
                )

            model, tokenizer = self._load_model(direction)

            # Prepare input with language tags
            input_text = f">>{tgt_lang}<< {text}"

            # Tokenize
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )

            if self._device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Generate translation
            import torch
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=5,
                    num_return_sequences=1,
                    early_stopping=True,
                )

            # Decode
            translated_text = tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            ).strip()

            duration_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"[INDICTRANS2] Translated in {duration_ms}ms: {translated_text[:50]}..."
            )

            # Record metrics
            self._record_call(duration_ms)

            return TranslationResult(
                translated_text=translated_text,
                source_language=source_language,
                target_language=target_language,
                duration_ms=duration_ms,
                provider=self.name,
                model=self._model_size,
            )

        except Exception as e:
            logger.error(f"[INDICTRANS2] Translation error: {e}")
            self._record_error()
            raise

    async def _translate_via_english(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
    ) -> str:
        """
        Translate between two Indic languages via English pivot.

        Args:
            text: Source text
            src_lang: Source language code
            tgt_lang: Target language code

        Returns:
            Translated text
        """
        # First: Indic → English
        model_ie, tokenizer_ie = self._load_model("indic-en")
        input_text = f">>eng_Latn<< {text}"

        inputs = tokenizer_ie(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )

        if self._device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        import torch
        with torch.no_grad():
            outputs = model_ie.generate(**inputs, max_length=256, num_beams=5)

        english_text = tokenizer_ie.decode(outputs[0], skip_special_tokens=True).strip()

        # Second: English → Target Indic
        model_ei, tokenizer_ei = self._load_model("en-indic")
        input_text = f">>{tgt_lang}<< {english_text}"

        inputs = tokenizer_ei(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )

        if self._device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model_ei.generate(**inputs, max_length=256, num_beams=5)

        translated_text = tokenizer_ei.decode(outputs[0], skip_special_tokens=True).strip()

        return translated_text

    async def health_check(self) -> bool:
        """Check if IndicTrans2 is available."""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            return True
        except ImportError:
            return False


class NoOpTranslationPlugin(BaseTranslationPlugin):
    """
    No-operation translation plugin.

    Returns input text unchanged. Used for native language pipeline
    where no translation is needed.
    """

    @property
    def name(self) -> str:
        return "noop"

    @property
    def supported_pairs(self) -> List[Tuple[str, str]]:
        return []  # Supports all pairs (pass-through)

    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> TranslationResult:
        """Pass through text unchanged."""
        return TranslationResult(
            translated_text=text,
            source_language=source_language,
            target_language=target_language,
            duration_ms=0,
            provider=self.name,
        )

    async def health_check(self) -> bool:
        return True


def register_plugin():
    """Register the IndicTrans2 translation plugin with the registry."""
    from core.registry import get_registry
    registry = get_registry()
    registry.register_translation("indictrans2", IndicTrans2Plugin)
    registry.register_translation("noop", NoOpTranslationPlugin)
    logger.info("[INDICTRANS2] Plugins registered")
