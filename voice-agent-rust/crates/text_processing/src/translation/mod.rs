//! Translation module with script detection
//!
//! Supports the Translate-Think-Translate pattern for LLM reasoning.

mod detect;
mod noop;

pub use detect::ScriptDetector;
pub use noop::NoopTranslator;

use voice_agent_core::{Translator, Language};
use std::sync::Arc;
use serde::{Deserialize, Serialize};

/// Translation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationConfig {
    /// Which provider to use
    pub provider: TranslationProvider,
    /// gRPC endpoint for fallback
    #[serde(default = "default_grpc_endpoint")]
    pub grpc_endpoint: String,
    /// Whether to fall back to gRPC if ONNX fails
    #[serde(default = "default_true")]
    pub fallback_to_grpc: bool,
}

fn default_grpc_endpoint() -> String {
    "http://localhost:50051".to_string()
}

fn default_true() -> bool {
    true
}

/// Translation providers
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TranslationProvider {
    /// gRPC-based translation (Python sidecar)
    Grpc,
    /// Disabled (pass-through)
    #[default]
    Disabled,
}

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            provider: TranslationProvider::Disabled,
            grpc_endpoint: default_grpc_endpoint(),
            fallback_to_grpc: true,
        }
    }
}

/// Create translator based on config
pub fn create_translator(config: &TranslationConfig) -> Arc<dyn Translator> {
    match config.provider {
        TranslationProvider::Grpc => {
            // TODO: Implement gRPC translator when needed
            tracing::warn!("gRPC translator not yet implemented, using noop");
            Arc::new(NoopTranslator::new())
        }
        TranslationProvider::Disabled => Arc::new(NoopTranslator::new()),
    }
}

/// Supported translation pairs
pub fn supported_pairs() -> Vec<(Language, Language)> {
    vec![
        // Indic to English
        (Language::Hindi, Language::English),
        (Language::Tamil, Language::English),
        (Language::Telugu, Language::English),
        (Language::Bengali, Language::English),
        (Language::Marathi, Language::English),
        (Language::Gujarati, Language::English),
        (Language::Kannada, Language::English),
        (Language::Malayalam, Language::English),
        (Language::Punjabi, Language::English),
        (Language::Odia, Language::English),
        // English to Indic
        (Language::English, Language::Hindi),
        (Language::English, Language::Tamil),
        (Language::English, Language::Telugu),
        (Language::English, Language::Bengali),
        (Language::English, Language::Marathi),
        (Language::English, Language::Gujarati),
        (Language::English, Language::Kannada),
        (Language::English, Language::Malayalam),
        (Language::English, Language::Punjabi),
        (Language::English, Language::Odia),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TranslationConfig::default();
        assert!(matches!(config.provider, TranslationProvider::Disabled));
    }

    #[test]
    fn test_supported_pairs() {
        let pairs = supported_pairs();
        assert!(pairs.contains(&(Language::Hindi, Language::English)));
        assert!(pairs.contains(&(Language::English, Language::Hindi)));
    }
}
