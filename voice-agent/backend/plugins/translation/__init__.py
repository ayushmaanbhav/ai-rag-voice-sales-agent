"""Translation plugins package."""

# Import plugins - these may fail if dependencies aren't installed
try:
    from plugins.translation.indictrans2 import IndicTrans2Plugin, NoOpTranslationPlugin
except ImportError:
    IndicTrans2Plugin = None
    NoOpTranslationPlugin = None

__all__ = [
    "IndicTrans2Plugin",
    "NoOpTranslationPlugin",
]
