//! Main settings module

use config::{Config, Environment, File};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::{AgentConfig, ConfigError, GoldLoanConfig, PipelineConfig};

/// Main application settings
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct Settings {
    /// Server configuration
    #[serde(default)]
    pub server: ServerConfig,

    /// Pipeline configuration
    #[serde(default)]
    pub pipeline: PipelineConfig,

    /// Agent configuration
    #[serde(default)]
    pub agent: AgentConfig,

    /// Gold loan business configuration
    #[serde(default)]
    pub gold_loan: GoldLoanConfig,

    /// Model paths
    #[serde(default)]
    pub models: ModelPaths,

    /// Observability configuration
    #[serde(default)]
    pub observability: ObservabilityConfig,

    /// Feature flags
    #[serde(default)]
    pub features: FeatureFlags,
}

impl Settings {
    /// Create default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate settings
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Validate model paths exist
        if !self.models.vad.is_empty() && !Path::new(&self.models.vad).exists() {
            tracing::warn!("VAD model not found: {}", self.models.vad);
        }

        // Validate latency budget
        if self.pipeline.latency_budget_ms < 200 {
            return Err(ConfigError::InvalidValue {
                field: "pipeline.latency_budget_ms".to_string(),
                message: "Latency budget too low (minimum 200ms)".to_string(),
            });
        }

        Ok(())
    }
}


/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// HTTP server host
    #[serde(default = "default_host")]
    pub host: String,

    /// HTTP server port
    #[serde(default = "default_port")]
    pub port: u16,

    /// WebSocket path
    #[serde(default = "default_ws_path")]
    pub ws_path: String,

    /// Maximum concurrent connections
    #[serde(default = "default_max_connections")]
    pub max_connections: usize,

    /// Request timeout in seconds
    #[serde(default = "default_timeout")]
    pub timeout_seconds: u64,

    /// Enable CORS
    #[serde(default = "default_true")]
    pub cors_enabled: bool,

    /// CORS allowed origins
    #[serde(default)]
    pub cors_origins: Vec<String>,

    /// Rate limiting configuration
    #[serde(default)]
    pub rate_limit: RateLimitConfig,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Enable rate limiting
    #[serde(default = "default_true")]
    pub enabled: bool,

    /// Maximum messages per second per connection
    #[serde(default = "default_messages_per_second")]
    pub messages_per_second: u32,

    /// Maximum audio bytes per second per connection
    #[serde(default = "default_audio_bytes_per_second")]
    pub audio_bytes_per_second: u32,

    /// Burst allowance (multiple of rate limit)
    #[serde(default = "default_burst_multiplier")]
    pub burst_multiplier: f32,
}

fn default_messages_per_second() -> u32 {
    100 // 100 messages/sec should be plenty for voice
}

fn default_audio_bytes_per_second() -> u32 {
    64000 // 16kHz * 2 bytes * 2 (some headroom) = 64KB/s
}

fn default_burst_multiplier() -> f32 {
    2.0
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            messages_per_second: default_messages_per_second(),
            audio_bytes_per_second: default_audio_bytes_per_second(),
            burst_multiplier: default_burst_multiplier(),
        }
    }
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}
fn default_port() -> u16 {
    8080
}
fn default_ws_path() -> String {
    "/ws/conversation".to_string()
}
fn default_max_connections() -> usize {
    1000
}
fn default_timeout() -> u64 {
    30
}
fn default_true() -> bool {
    true
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            ws_path: default_ws_path(),
            max_connections: default_max_connections(),
            timeout_seconds: default_timeout(),
            cors_enabled: default_true(),
            // SECURITY: Empty by default - must be explicitly configured for production
            // Use ["http://localhost:3000"] for local dev, or specific domains for production
            cors_origins: Vec::new(),
            rate_limit: RateLimitConfig::default(),
        }
    }
}

/// Model file paths
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPaths {
    /// VAD model path
    #[serde(default = "default_vad_path")]
    pub vad: String,

    /// Turn detection model path
    #[serde(default = "default_turn_detection_path")]
    pub turn_detection: String,

    /// Turn detection tokenizer path
    #[serde(default = "default_turn_tokenizer_path")]
    pub turn_detection_tokenizer: String,

    /// STT model path
    #[serde(default = "default_stt_path")]
    pub stt: String,

    /// STT tokens path
    #[serde(default = "default_stt_tokens_path")]
    pub stt_tokens: String,

    /// TTS model path
    #[serde(default = "default_tts_path")]
    pub tts: String,

    /// Cross-encoder model path
    #[serde(default = "default_reranker_path")]
    pub reranker: String,

    /// Embedding model path
    #[serde(default = "default_embeddings_path")]
    pub embeddings: String,
}

fn default_vad_path() -> String {
    "models/vad/silero_vad.onnx".to_string()
}
fn default_turn_detection_path() -> String {
    "models/turn_detection/smollm2-135m.onnx".to_string()
}
fn default_turn_tokenizer_path() -> String {
    "models/turn_detection/tokenizer.json".to_string()
}
fn default_stt_path() -> String {
    "models/stt/indicconformer.onnx".to_string()
}
fn default_stt_tokens_path() -> String {
    "models/stt/tokens.txt".to_string()
}
fn default_tts_path() -> String {
    "models/tts/indicf5.onnx".to_string()
}
fn default_reranker_path() -> String {
    "models/reranker/bge-reranker-v2-m3.onnx".to_string()
}
fn default_embeddings_path() -> String {
    "models/embeddings/e5-multilingual.onnx".to_string()
}

impl Default for ModelPaths {
    fn default() -> Self {
        Self {
            vad: default_vad_path(),
            turn_detection: default_turn_detection_path(),
            turn_detection_tokenizer: default_turn_tokenizer_path(),
            stt: default_stt_path(),
            stt_tokens: default_stt_tokens_path(),
            tts: default_tts_path(),
            reranker: default_reranker_path(),
            embeddings: default_embeddings_path(),
        }
    }
}

/// Observability configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    /// Log level
    #[serde(default = "default_log_level")]
    pub log_level: String,

    /// Enable JSON logging
    #[serde(default)]
    pub log_json: bool,

    /// Enable tracing
    #[serde(default = "default_true")]
    pub tracing_enabled: bool,

    /// OTLP endpoint for traces
    #[serde(default)]
    pub otlp_endpoint: Option<String>,

    /// Enable metrics
    #[serde(default = "default_true")]
    pub metrics_enabled: bool,

    /// Metrics port
    #[serde(default = "default_metrics_port")]
    pub metrics_port: u16,
}

fn default_log_level() -> String {
    "info".to_string()
}
fn default_metrics_port() -> u16 {
    9090
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            log_level: default_log_level(),
            log_json: false,
            tracing_enabled: true,
            otlp_endpoint: None,
            metrics_enabled: true,
            metrics_port: default_metrics_port(),
        }
    }
}

/// Feature flags for experimentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureFlags {
    /// Enable semantic turn detection
    #[serde(default = "default_true")]
    pub semantic_turn_detection: bool,

    /// Enable speculative LLM execution
    #[serde(default = "default_true")]
    pub speculative_llm: bool,

    /// Enable early-exit cross-encoder
    #[serde(default = "default_true")]
    pub early_exit_reranker: bool,

    /// Enable RAG prefetch on partial transcript
    #[serde(default = "default_true")]
    pub rag_prefetch: bool,

    /// Enable word-level TTS
    #[serde(default = "default_true")]
    pub word_level_tts: bool,

    /// Enable barge-in handling
    #[serde(default = "default_true")]
    pub barge_in_enabled: bool,
}

impl Default for FeatureFlags {
    fn default() -> Self {
        Self {
            semantic_turn_detection: true,
            speculative_llm: true,
            early_exit_reranker: true,
            rag_prefetch: true,
            word_level_tts: true,
            barge_in_enabled: true,
        }
    }
}

/// Load settings from files and environment
///
/// Priority (highest to lowest):
/// 1. Environment variables (VOICE_AGENT_ prefix)
/// 2. config/{env}.yaml (if env specified)
/// 3. config/default.yaml
pub fn load_settings(env: Option<&str>) -> Result<Settings, ConfigError> {
    let mut builder = Config::builder();

    // Load default config
    builder = builder.add_source(
        File::with_name("config/default")
            .required(false)
    );

    // Load environment-specific config
    if let Some(env_name) = env {
        builder = builder.add_source(
            File::with_name(&format!("config/{}", env_name))
                .required(false)
        );
    }

    // Load from environment variables
    builder = builder.add_source(
        Environment::with_prefix("VOICE_AGENT")
            .separator("__")
            .try_parsing(true)
    );

    let config = builder.build()?;
    let settings: Settings = config.try_deserialize()?;

    // Validate
    settings.validate()?;

    Ok(settings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_settings() {
        let settings = Settings::default();
        assert_eq!(settings.server.port, 8080);
        assert!(settings.features.semantic_turn_detection);
    }

    #[test]
    fn test_settings_validation() {
        let mut settings = Settings::default();
        settings.pipeline.latency_budget_ms = 100; // Too low
        assert!(settings.validate().is_err());

        settings.pipeline.latency_budget_ms = 500;
        assert!(settings.validate().is_ok());
    }
}
