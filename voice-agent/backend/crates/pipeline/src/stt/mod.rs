//! Streaming Speech-to-Text
//!
//! Supports multiple STT backends with enhanced decoding:
//! - Whisper (via ONNX)
//! - IndicConformer (for Indian languages)

mod decoder;
mod indicconformer;
mod streaming;
mod vocab;

pub use decoder::{DecoderConfig, EnhancedDecoder};
pub use indicconformer::{IndicConformerConfig, IndicConformerStt, MelFilterbank};
pub use streaming::{StreamingStt, SttConfig, SttEngine};
pub use vocab::{load_domain_vocab, load_vocabulary, Vocabulary};

use crate::PipelineError;
use voice_agent_core::TranscriptResult;

/// STT backend trait
#[async_trait::async_trait]
pub trait SttBackend: Send + Sync {
    /// Process audio chunk and return partial transcript
    async fn process_chunk(
        &mut self,
        audio: &[f32],
    ) -> Result<Option<TranscriptResult>, PipelineError>;

    /// Finalize and return final transcript
    async fn finalize(&mut self) -> Result<TranscriptResult, PipelineError>;

    /// Reset state
    fn reset(&mut self);

    /// Get current partial transcript
    fn partial(&self) -> Option<&TranscriptResult>;
}
