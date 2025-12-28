//! Voice Activity Detection
//!
//! MagicNet-inspired VAD with 10ms frames and no future lookahead.
//! Architecture: Causal depth-separable convolutions + GRU

mod magicnet;

pub use magicnet::{VoiceActivityDetector, VadConfig, VadState, VadResult};

use crate::PipelineError;
use voice_agent_core::AudioFrame;

/// VAD engine trait for pluggable implementations
#[async_trait::async_trait]
pub trait VadEngine: Send + Sync {
    /// Process a single audio frame
    fn process_frame(&mut self, frame: &mut AudioFrame) -> Result<VadResult, PipelineError>;

    /// Reset VAD state
    fn reset(&mut self);

    /// Get current state
    fn state(&self) -> VadState;
}
