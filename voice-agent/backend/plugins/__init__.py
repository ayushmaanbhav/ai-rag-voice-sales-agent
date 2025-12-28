"""
Plugins package for Voice Agent.

Provides auto-registration of all available plugins.
"""

import logging

logger = logging.getLogger(__name__)


def register_all_plugins():
    """
    Register all available plugins with the registry.

    This function should be called during application startup
    before initializing the factory.
    """
    logger.info("[PLUGINS] Registering all plugins...")

    # =========================================================================
    # STT Plugins
    # =========================================================================
    try:
        from plugins.stt.whisper import register_plugin as register_whisper
        register_whisper()
    except ImportError as e:
        logger.warning(f"[PLUGINS] Failed to register Whisper STT: {e}")

    try:
        from plugins.stt.indicconformer import register_plugin as register_indicconformer
        register_indicconformer()
    except ImportError as e:
        logger.debug(f"[PLUGINS] IndicConformer STT not available: {e}")

    # =========================================================================
    # TTS Plugins
    # =========================================================================
    try:
        from plugins.tts.piper import register_plugin as register_piper
        register_piper()
    except ImportError as e:
        logger.warning(f"[PLUGINS] Failed to register Piper TTS: {e}")

    try:
        from plugins.tts.indicf5 import register_plugin as register_indicf5
        register_indicf5()
    except ImportError as e:
        logger.debug(f"[PLUGINS] IndicF5 TTS not available: {e}")

    # =========================================================================
    # Translation Plugins
    # =========================================================================
    try:
        from plugins.translation.indictrans2 import register_plugin as register_indictrans2
        register_indictrans2()
    except ImportError as e:
        logger.debug(f"[PLUGINS] IndicTrans2 not available: {e}")

    # =========================================================================
    # LLM Plugins
    # =========================================================================
    try:
        from plugins.llm.ollama import register_plugin as register_ollama
        register_ollama()
    except ImportError as e:
        logger.warning(f"[PLUGINS] Failed to register Ollama LLM: {e}")

    logger.info("[PLUGINS] Plugin registration complete")


def get_available_plugins() -> dict:
    """
    Get a summary of available plugins.

    Returns:
        Dict with plugin types and their registered implementations
    """
    from core.registry import get_registry
    registry = get_registry()

    return {
        "stt": registry.list_stt_plugins(),
        "tts": registry.list_tts_plugins(),
        "translation": registry.list_translation_plugins(),
        "llm": registry.list_llm_plugins(),
    }
