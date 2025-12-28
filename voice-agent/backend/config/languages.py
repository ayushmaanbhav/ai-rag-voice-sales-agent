"""Language configuration for supported Indian languages."""

LANGUAGE_CONFIG = {
    "hi": {
        "name": "Hindi",
        "native_name": "हिंदी",
        "sarvam_code": "hi-IN",
        "tts_voice": "meera",  # Female voice
        "tts_voice_alt": "arvind",  # Male voice
    },
    "ta": {
        "name": "Tamil",
        "native_name": "தமிழ்",
        "sarvam_code": "ta-IN",
        "tts_voice": "tamil_female",
        "tts_voice_alt": "tamil_male",
    },
    "te": {
        "name": "Telugu",
        "native_name": "తెలుగు",
        "sarvam_code": "te-IN",
        "tts_voice": "telugu_female",
        "tts_voice_alt": "telugu_male",
    },
    "kn": {
        "name": "Kannada",
        "native_name": "ಕನ್ನಡ",
        "sarvam_code": "kn-IN",
        "tts_voice": "kannada_female",
        "tts_voice_alt": "kannada_male",
    },
    "ml": {
        "name": "Malayalam",
        "native_name": "മലയാളം",
        "sarvam_code": "ml-IN",
        "tts_voice": "malayalam_female",
        "tts_voice_alt": "malayalam_male",
    },
    "en": {
        "name": "English",
        "native_name": "English",
        "sarvam_code": "en-IN",
        "tts_voice": "english_female",
        "tts_voice_alt": "english_male",
    },
}


def get_language_config(lang_code: str) -> dict:
    """Get language configuration by code."""
    return LANGUAGE_CONFIG.get(lang_code, LANGUAGE_CONFIG["en"])


def get_supported_languages() -> list[str]:
    """Get list of supported language codes."""
    return list(LANGUAGE_CONFIG.keys())
