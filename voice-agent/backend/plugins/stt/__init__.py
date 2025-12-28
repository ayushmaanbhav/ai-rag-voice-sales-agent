"""STT plugins package."""

# Import plugins - these may fail if dependencies aren't installed
try:
    from plugins.stt.whisper import WhisperSTTPlugin
except ImportError:
    WhisperSTTPlugin = None

try:
    from plugins.stt.indicconformer import IndicConformerSTTPlugin
except ImportError:
    IndicConformerSTTPlugin = None

__all__ = [
    "WhisperSTTPlugin",
    "IndicConformerSTTPlugin",
]
