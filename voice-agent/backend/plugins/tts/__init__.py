"""TTS plugins package."""

# Import plugins - these may fail if dependencies aren't installed
try:
    from plugins.tts.piper import PiperTTSPlugin
except ImportError:
    PiperTTSPlugin = None

try:
    from plugins.tts.indicf5 import IndicF5TTSPlugin
except ImportError:
    IndicF5TTSPlugin = None

__all__ = [
    "PiperTTSPlugin",
    "IndicF5TTSPlugin",
]
