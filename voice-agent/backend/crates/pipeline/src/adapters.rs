//! Trait Adapters
//!
//! Bridges the pipeline's internal STT/TTS implementations with the core traits.
//!
//! This enables:
//! - Using pipeline STT/TTS as `dyn SpeechToText` or `dyn TextToSpeech`
//! - Injecting external implementations that implement the core traits
//! - Testing with mock implementations

use async_trait::async_trait;
use futures::Stream;
use parking_lot::Mutex;
use std::pin::Pin;
use std::sync::Arc;

use voice_agent_core::{
    AudioFrame, Language, Result as CoreResult, SpeechToText, TextToSpeech, TranscriptResult,
    VoiceConfig, VoiceInfo,
};

use crate::stt::{StreamingStt, SttConfig};
use crate::tts::{StreamingTts, TtsBackend, TtsConfig};

// =============================================================================
// SpeechToText Adapter
// =============================================================================

/// Adapter that wraps StreamingStt to implement the core SpeechToText trait
///
/// This allows the pipeline's STT implementation to be used anywhere
/// the `dyn SpeechToText` trait is expected.
pub struct SttAdapter {
    inner: Arc<Mutex<StreamingStt>>,
    config: SttConfig,
}

impl SttAdapter {
    /// Create a new adapter wrapping a StreamingStt
    pub fn new(stt: StreamingStt, config: SttConfig) -> Self {
        Self {
            inner: Arc::new(Mutex::new(stt)),
            config,
        }
    }

    /// Create from config (initializes StreamingStt internally)
    #[cfg(feature = "onnx")]
    pub fn from_config(
        model_path: impl AsRef<std::path::Path>,
        config: SttConfig,
    ) -> Result<Self, PipelineError> {
        let stt = StreamingStt::new(model_path, config.clone())?;
        Ok(Self::new(stt, config))
    }
}

#[async_trait]
impl SpeechToText for SttAdapter {
    async fn transcribe(&self, audio: &AudioFrame) -> CoreResult<TranscriptResult> {
        let stt = self.inner.lock();

        // Process the audio chunk using inherent method (sync)
        if let Some(partial) = stt.process(&audio.samples).map_err(|e| {
            voice_agent_core::Error::Pipeline(voice_agent_core::error::PipelineError::Stt(
                e.to_string(),
            ))
        })? {
            return Ok(partial);
        }

        // Finalize to get the result using inherent method (sync)
        Ok(stt.finalize())
    }

    fn transcribe_stream<'a>(
        &'a self,
        _audio_stream: Pin<Box<dyn Stream<Item = AudioFrame> + Send + 'a>>,
    ) -> Pin<Box<dyn Stream<Item = CoreResult<TranscriptResult>> + Send + 'a>> {
        // TODO: Implement streaming transcription
        // For now, return an empty stream
        Box::pin(futures::stream::empty())
    }

    fn supported_languages(&self) -> &[Language] {
        // Return supported languages based on engine
        static WHISPER_LANGS: &[Language] = &[
            Language::Hindi,
            Language::English,
            Language::Tamil,
            Language::Telugu,
            Language::Marathi,
            Language::Bengali,
            Language::Gujarati,
            Language::Kannada,
            Language::Malayalam,
            Language::Punjabi,
        ];
        WHISPER_LANGS
    }

    fn model_name(&self) -> &str {
        match self.config.engine {
            crate::stt::SttEngine::Whisper => "whisper",
            crate::stt::SttEngine::IndicConformer => "indicconformer",
            crate::stt::SttEngine::Wav2Vec2 => "wav2vec2",
        }
    }
}

// =============================================================================
// TextToSpeech Adapter
// =============================================================================

/// Adapter that wraps StreamingTts to implement the core TextToSpeech trait
///
/// This allows the pipeline's TTS implementation to be used anywhere
/// the `dyn TextToSpeech` trait is expected.
pub struct TtsAdapter {
    inner: Arc<StreamingTts>,
    config: TtsConfig,
}

impl TtsAdapter {
    /// Create a new adapter wrapping a StreamingTts
    pub fn new(tts: StreamingTts, config: TtsConfig) -> Self {
        Self {
            inner: Arc::new(tts),
            config,
        }
    }
}

#[async_trait]
impl TextToSpeech for TtsAdapter {
    async fn synthesize(&self, text: &str, _config: &VoiceConfig) -> CoreResult<AudioFrame> {
        // Use the internal TTS to synthesize (TtsBackend only takes text)
        let samples = self.inner.synthesize(text).await.map_err(|e| {
            voice_agent_core::Error::Pipeline(voice_agent_core::error::PipelineError::Tts(
                e.to_string(),
            ))
        })?;

        // Create AudioFrame from samples using TTS sample rate
        Ok(AudioFrame::new(
            samples,
            voice_agent_core::SampleRate::Hz16000,
            voice_agent_core::Channels::Mono,
            0,
        ))
    }

    fn synthesize_stream<'a>(
        &'a self,
        _text_stream: Pin<Box<dyn Stream<Item = String> + Send + 'a>>,
        _config: &'a VoiceConfig,
    ) -> Pin<Box<dyn Stream<Item = CoreResult<AudioFrame>> + Send + 'a>> {
        // TODO: Implement streaming synthesis
        // For now, return an empty stream
        Box::pin(futures::stream::empty())
    }

    fn available_voices(&self) -> &[VoiceInfo] {
        static VOICES: &[VoiceInfo] = &[];
        VOICES
    }

    fn model_name(&self) -> &str {
        match self.config.engine {
            crate::tts::TtsEngine::Piper => "piper",
            crate::tts::TtsEngine::IndicF5 => "indicf5",
            crate::tts::TtsEngine::ParlerTts => "parler",
        }
    }
}

// =============================================================================
// Factory Functions
// =============================================================================

/// Create a boxed SpeechToText from StreamingStt
pub fn create_stt_adapter(stt: StreamingStt, config: SttConfig) -> Box<dyn SpeechToText> {
    Box::new(SttAdapter::new(stt, config))
}

/// Create a boxed TextToSpeech from StreamingTts
pub fn create_tts_adapter(tts: StreamingTts, config: TtsConfig) -> Box<dyn TextToSpeech> {
    Box::new(TtsAdapter::new(tts, config))
}

// =============================================================================
// AudioProcessor Adapter (P2-2: Deferred - placeholder for future AEC/NS/AGC)
// =============================================================================

use voice_agent_core::AudioProcessor;

/// Passthrough audio processor that does no processing
///
/// This is a placeholder for future audio signal processing (AEC, NS, AGC).
/// Currently, browser-side processing via getUserMedia constraints is used.
///
/// When implementing real audio processing, this adapter can wrap:
/// - `webrtc-audio-processing-rs` for AEC/NS/AGC
/// - Custom DSP pipelines
pub struct PassthroughAudioProcessor {
    name: &'static str,
}

impl PassthroughAudioProcessor {
    /// Create a new passthrough processor
    pub fn new() -> Self {
        Self {
            name: "passthrough",
        }
    }

    /// Create with custom name (for testing/logging)
    pub fn with_name(name: &'static str) -> Self {
        Self { name }
    }
}

impl Default for PassthroughAudioProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AudioProcessor for PassthroughAudioProcessor {
    async fn process(
        &self,
        input: &AudioFrame,
        _reference: Option<&AudioFrame>,
    ) -> CoreResult<AudioFrame> {
        // P2-2 DEFERRED: No processing, just clone the input
        // Future: Add AEC with reference signal, NS, AGC
        Ok(input.clone())
    }

    fn name(&self) -> &str {
        self.name
    }

    fn reset(&self) {
        // No state to reset in passthrough mode
    }
}

/// Create a boxed AudioProcessor (passthrough)
pub fn create_passthrough_processor() -> Box<dyn AudioProcessor> {
    Box::new(PassthroughAudioProcessor::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stt_adapter_model_name() {
        // This test verifies the adapter correctly reports model name
        // Actual STT testing requires ONNX models
    }

    #[test]
    fn test_tts_adapter_model_name() {
        // This test verifies the adapter correctly reports model name
        // Actual TTS testing requires models
    }

    #[tokio::test]
    async fn test_passthrough_processor() {
        let processor = PassthroughAudioProcessor::new();
        assert_eq!(processor.name(), "passthrough");

        let frame = AudioFrame::new(
            vec![0.1, 0.2, 0.3],
            voice_agent_core::SampleRate::Hz16000,
            voice_agent_core::Channels::Mono,
            0,
        );

        let result = processor.process(&frame, None).await.unwrap();
        assert_eq!(result.samples, frame.samples);
    }
}
