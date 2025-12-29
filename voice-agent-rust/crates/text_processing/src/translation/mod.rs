//! Translation module with script detection
//!
//! Supports the Translate-Think-Translate pattern for LLM reasoning.
//!
//! P3 FIX: Added IndicTrans2 ONNX-based translation for native Indic language support.

mod detect;
mod noop;
mod grpc;
mod indictrans2;

pub use detect::ScriptDetector;
pub use noop::NoopTranslator;
pub use grpc::{GrpcTranslator, GrpcTranslatorConfig, FallbackTranslator};
pub use indictrans2::{IndicTrans2Translator, IndicTrans2Config};

use voice_agent_core::{Translator, Language};
use std::sync::Arc;
use std::path::PathBuf;
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
    /// P3 FIX: IndicTrans2 model path (for ONNX provider)
    #[serde(default)]
    pub indictrans2_model_path: Option<PathBuf>,
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
    /// P3 FIX: ONNX-based IndicTrans2 translation (native, fastest)
    #[serde(alias = "onnx")]
    IndicTrans2,
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
            indictrans2_model_path: None,
        }
    }
}

/// Create translator based on config
pub fn create_translator(config: &TranslationConfig) -> Arc<dyn Translator> {
    match config.provider {
        TranslationProvider::IndicTrans2 => {
            // P3 FIX: Create IndicTrans2 ONNX translator
            let indictrans2_config = if let Some(ref model_path) = config.indictrans2_model_path {
                IndicTrans2Config {
                    encoder_path: model_path.join("encoder.onnx"),
                    decoder_path: model_path.join("decoder.onnx"),
                    tokenizer_path: model_path.join("tokenizer"),
                    ..Default::default()
                }
            } else {
                IndicTrans2Config::default()
            };

            match IndicTrans2Translator::new(indictrans2_config) {
                Ok(translator) => {
                    let primary = Arc::new(translator);

                    // Wrap with fallback if enabled
                    if config.fallback_to_grpc {
                        tracing::info!("Using IndicTrans2 with gRPC fallback");
                        let grpc_config = GrpcTranslatorConfig {
                            endpoint: config.grpc_endpoint.clone(),
                            ..Default::default()
                        };
                        let fallback = Arc::new(GrpcTranslator::new(grpc_config));
                        Arc::new(FallbackTranslator::new(primary, fallback))
                    } else {
                        tracing::info!("Using IndicTrans2 (no fallback)");
                        primary
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "Failed to load IndicTrans2, falling back to gRPC"
                    );
                    // Fall back to gRPC if ONNX fails to load
                    let grpc_config = GrpcTranslatorConfig {
                        endpoint: config.grpc_endpoint.clone(),
                        ..Default::default()
                    };
                    Arc::new(GrpcTranslator::new(grpc_config))
                }
            }
        }
        TranslationProvider::Grpc => {
            let grpc_config = GrpcTranslatorConfig {
                endpoint: config.grpc_endpoint.clone(),
                ..Default::default()
            };
            let grpc_translator = Arc::new(GrpcTranslator::new(grpc_config));

            if config.fallback_to_grpc {
                tracing::info!(
                    endpoint = %config.grpc_endpoint,
                    "Using gRPC translator with fallback enabled"
                );
            } else {
                tracing::info!(
                    endpoint = %config.grpc_endpoint,
                    "Using gRPC translator (fallback disabled)"
                );
            }
            grpc_translator
        }
        TranslationProvider::Disabled => Arc::new(NoopTranslator::new()),
    }
}

/// Create a fallback translator that tries ONNX first, then gRPC
pub fn create_fallback_translator(
    primary: Arc<dyn Translator>,
    config: &TranslationConfig,
) -> Arc<dyn Translator> {
    if config.fallback_to_grpc && matches!(config.provider, TranslationProvider::Grpc) {
        let grpc_config = GrpcTranslatorConfig {
            endpoint: config.grpc_endpoint.clone(),
            ..Default::default()
        };
        let fallback = Arc::new(GrpcTranslator::new(grpc_config));
        Arc::new(FallbackTranslator::new(primary, fallback))
    } else {
        primary
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
