"""
Configuration module for Voice Agent.

Provides unified access to all configuration:
- settings: Environment-based settings (API keys, providers, etc.)
- languages: Language configuration
- constants: Application-wide constants
- features: Feature flags (loaded from features.yaml)
"""

import yaml
from pathlib import Path

from config.settings import Settings, settings
from config.languages import LANGUAGE_CONFIG, get_language_config, get_supported_languages
from config.constants import (
    KOTAK_GOLD_LOAN,
    COMPETITOR_RATES,
    SEGMENTS,
    CONVERSATION_LIMITS,
    RAG_CONFIG,
)


def load_features() -> dict:
    """Load feature flags from features.yaml."""
    config_path = Path(__file__).parent / "features.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


# Load features at import time
features = load_features()


__all__ = [
    # Settings
    "Settings",
    "settings",
    # Languages
    "LANGUAGE_CONFIG",
    "get_language_config",
    "get_supported_languages",
    # Constants
    "KOTAK_GOLD_LOAN",
    "COMPETITOR_RATES",
    "SEGMENTS",
    "CONVERSATION_LIMITS",
    "RAG_CONFIG",
    # Features
    "features",
    "load_features",
]
