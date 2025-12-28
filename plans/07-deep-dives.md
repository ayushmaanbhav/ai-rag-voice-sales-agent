# Deep Dives & Resolved Questions

This document addresses the questions raised during code review with solutions and architecture designs.

---

## Q9: Mutex Contention in VAD Hot Path - SOLUTION

### Problem Analysis

Current code in `vad/magicnet.rs:99-103`:
```rust
pub struct VoiceActivityDetector {
    gru_state: Mutex<Array2<f32>>,     // Lock 1
    state: Mutex<VadState>,             // Lock 2
    speech_frames: Mutex<usize>,        // Lock 3
    silence_frames: Mutex<usize>,       // Lock 4
}
```

At 100 frames/second (10ms each), each `process_frame()` call acquires **4 separate locks**:
- `update_state()` acquires 3 locks (lines 241-243)
- `infer()` acquires 1 lock (line 205)

### Why This Is Problematic

1. **Lock Acquisition Overhead**: `parking_lot::Mutex` is fast (~15-20ns uncontended), but 4 locks × 100 fps = 400 lock ops/second
2. **False Sharing Risk**: Adjacent Mutex fields may share cache lines, causing cache invalidation
3. **Semantic Confusion**: State is split across 4 fields that should be atomic together

### Recommended Fix

Consolidate into a single lock with a state struct:

```rust
// vad/magicnet.rs - FIXED VERSION

/// All mutable VAD state in one struct
struct VadMutableState {
    gru_state: Array2<f32>,
    state: VadState,
    speech_frames: usize,
    silence_frames: usize,
}

pub struct VoiceActivityDetector {
    #[cfg(feature = "onnx")]
    session: Session,
    #[cfg(feature = "onnx")]
    mel_filterbank: MelFilterbank,
    config: VadConfig,
    mutable: Mutex<VadMutableState>,  // Single lock instead of 4
}

impl VoiceActivityDetector {
    pub fn process_frame(&self, frame: &mut AudioFrame) -> Result<(VadState, f32), PipelineError> {
        // Quick energy check (no lock needed)
        if frame.energy_db < self.config.energy_floor_db {
            frame.vad_probability = Some(0.0);
            frame.is_speech = false;

            let mut state = self.mutable.lock();
            return self.update_state_inner(&mut state, false, 0.0);
        }

        // Single lock for entire frame processing
        let mut state = self.mutable.lock();
        let speech_prob = self.compute_probability_inner(&mut state, frame)?;

        frame.vad_probability = Some(speech_prob);
        let is_speech = speech_prob >= self.config.threshold;
        frame.is_speech = is_speech;

        self.update_state_inner(&mut state, is_speech, speech_prob)
    }

    #[cfg(feature = "onnx")]
    fn compute_probability_inner(
        &self,
        state: &mut VadMutableState,
        frame: &AudioFrame,
    ) -> Result<f32, PipelineError> {
        let mel_features = self.mel_filterbank.compute(&frame.samples)?;

        // ONNX inference with direct state access (no lock)
        let input = Array3::from_shape_vec(
            (1, 1, self.config.n_mels),
            mel_features.to_vec(),
        )?;

        let outputs = self.session.run(ort::inputs![
            "mel_input" => input.view(),
            "gru_state_in" => state.gru_state.view(),
        ]?)?;

        // Update GRU state in place
        if let Some(new_state) = outputs.get("gru_state_out") {
            state.gru_state.assign(&new_state.try_extract_tensor()?);
        }

        Ok(outputs.get("speech_prob")?.try_extract_scalar()?)
    }

    fn update_state_inner(
        &self,
        state: &mut VadMutableState,
        is_speech: bool,
        probability: f32,
    ) -> Result<(VadState, f32), PipelineError> {
        // Direct field access, no locking
        match (state.state, is_speech) {
            (VadState::Silence, true) => {
                state.state = VadState::SpeechStart;
                state.speech_frames = 1;
                state.silence_frames = 0;
            }
            // ... rest of state machine
        }
        Ok((state.state, probability))
    }
}
```

### Performance Improvement

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Lock acquisitions/frame | 4 | 1 | **75% reduction** |
| Lock overhead @ 100fps | ~80ns | ~20ns | **60ns saved/frame** |
| Cache line contention | Possible | Eliminated | - |

### Alternative: Lock-Free Atomics (If Needed)

For even higher performance, counters can use atomics:

```rust
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

pub struct VoiceActivityDetector {
    session: Session,
    config: VadConfig,
    gru_state: Mutex<Array2<f32>>,        // Only for GRU (complex type)
    state: AtomicU32,                      // VadState as u32
    speech_frames: AtomicUsize,
    silence_frames: AtomicUsize,
}
```

This eliminates locks for simple counters but adds complexity. **Recommended only if profiling shows contention.**

---

## Q3 & Q4: Model Integration Architecture

### Research Findings

**IndicConformer** ([AI4Bharat](https://ai4bharat.iitm.ac.in/areas/model/ASR/IndicConformer)):
- Built on NVIDIA NeMo framework
- 120M parameter Hybrid CTC-RNNT model
- Supports all 22 scheduled Indian languages
- No official ONNX export available; requires NeMo export utilities
- Community requesting low-resource/ONNX versions ([HuggingFace discussion](https://huggingface.co/ai4bharat/IndicF5/discussions/2))

**IndicTrans2** ([GitHub](https://github.com/AI4Bharat/IndicTrans2)):
- Transformer-based NMT for 22 languages
- Available on HuggingFace: `ai4bharat/indictrans2-indic-en-1B`
- Distilled version (~211M) available for lighter deployment
- No official Rust/ONNX bindings

### Recommended Architecture: Pluggable Model Interface

```rust
// crates/pipeline/src/models/mod.rs

/// Trait for pluggable STT backends
#[async_trait]
pub trait SttBackend: Send + Sync {
    /// Process audio chunk, return partial/final transcription
    async fn transcribe(&self, audio: &[f32], is_final: bool) -> Result<TranscriptChunk, SttError>;

    /// Get supported languages
    fn supported_languages(&self) -> &[LanguageCode];

    /// Reset internal state for new utterance
    fn reset(&self);
}

/// Trait for pluggable translation backends
#[async_trait]
pub trait TranslationBackend: Send + Sync {
    /// Translate text between languages
    async fn translate(
        &self,
        text: &str,
        source: LanguageCode,
        target: LanguageCode,
    ) -> Result<String, TranslationError>;

    /// Get supported language pairs
    fn supported_pairs(&self) -> &[(LanguageCode, LanguageCode)];
}

/// Trait for pluggable TTS backends
#[async_trait]
pub trait TtsBackend: Send + Sync {
    /// Synthesize speech from text
    async fn synthesize(
        &self,
        text: &str,
        voice: &VoiceConfig,
    ) -> Result<AudioStream, TtsError>;
}
```

### Implementation Options Matrix

| Model | Rust Integration Path | Latency | Complexity |
|-------|----------------------|---------|------------|
| **IndicConformer** | ONNX via NeMo export → `ort` | ~100ms | High (export required) |
| **Whisper** | `whisper-rs` (cpp bindings) | ~150ms | Low (ready to use) |
| **IndicTrans2** | ONNX export → `ort` | ~50ms | Medium |
| **IndicTrans2** | Python subprocess (gRPC) | ~80ms | Low (quick to integrate) |
| **IndicF5 TTS** | ONNX export → `ort` | ~80ms | High (phoneme preprocessing) |
| **Piper TTS** | Native ONNX | ~50ms | Low (ready to use) |

### Hybrid Architecture (Recommended)

For fastest time-to-production:

```
┌─────────────────────────────────────────────────────────────┐
│                    Model Orchestrator                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ STT Backend │    │ Translation │    │ TTS Backend │     │
│  │             │    │   Backend   │    │             │     │
│  ├─────────────┤    ├─────────────┤    ├─────────────┤     │
│  │ Option A:   │    │ Option A:   │    │ Option A:   │     │
│  │ Whisper-rs  │    │ IndicTrans2 │    │ Piper ONNX  │     │
│  │ (embedded)  │    │ (gRPC svc)  │    │ (embedded)  │     │
│  ├─────────────┤    ├─────────────┤    ├─────────────┤     │
│  │ Option B:   │    │ Option B:   │    │ Option B:   │     │
│  │ IndicConf.  │    │ IndicTrans2 │    │ IndicF5     │     │
│  │ (ONNX)      │    │ (ONNX)      │    │ (ONNX)      │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Configuration Schema

```yaml
# config/models.yaml

stt:
  backend: "whisper"  # or "indicconformer", "external"
  whisper:
    model_path: "models/whisper-small.onnx"
    language: "hi"
  indicconformer:
    model_path: "models/indicconformer-large.onnx"
    vocab_path: "models/indicconformer.vocab"
  external:
    endpoint: "grpc://localhost:50051"
    timeout_ms: 200

translation:
  backend: "indictrans2"  # or "none", "external"
  indictrans2:
    model_path: "models/indictrans2-dist.onnx"
  external:
    endpoint: "http://localhost:8080/translate"

tts:
  backend: "piper"  # or "indicf5", "external"
  piper:
    model_path: "models/piper-hi.onnx"
    config_path: "models/piper-hi.json"
  indicf5:
    model_path: "models/indicf5.onnx"
    phonemizer: "espeak"  # or "custom"
```

### Model Download Strategy

Create `scripts/download_models.sh`:

```bash
#!/bin/bash
# Download production models

MODELS_DIR="models"
mkdir -p $MODELS_DIR

# STT - Whisper (fallback)
wget -O $MODELS_DIR/whisper-small.onnx \
  "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin"

# TTS - Piper Hindi
wget -O $MODELS_DIR/piper-hi.onnx \
  "https://github.com/rhasspy/piper/releases/download/v0.0.2/voice-hi-swara-medium.onnx"

# IndicConformer - requires manual NeMo export
echo "IndicConformer requires manual export from NeMo. See docs/model-export.md"

# IndicTrans2 - requires manual export
echo "IndicTrans2 requires manual export. See docs/model-export.md"
```

---

## Q5: Configurable Agentic RAG Architecture

### Current State
Single-shot retrieval in `retriever.rs:148`

### Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Agentic RAG Controller                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐                                           │
│  │ 1. Intent    │ → FAQ / Product / Complaint / General    │
│  │    Classify  │                                           │
│  └──────┬───────┘                                           │
│         ▼                                                   │
│  ┌──────────────┐                                           │
│  │ 2. Initial   │ → Hybrid (Dense + Sparse + RRF)          │
│  │    Retrieve  │                                           │
│  └──────┬───────┘                                           │
│         ▼                                                   │
│  ┌──────────────┐    ┌─────────────────┐                   │
│  │ 3. Suffic-   │───▶│ If score < 0.7  │                   │
│  │    iency     │    │ → Query Rewrite │                   │
│  │    Check     │    └────────┬────────┘                   │
│  └──────┬───────┘             │                             │
│         │                     ▼                             │
│         │           ┌─────────────────┐                     │
│         │           │ 4. Re-retrieve  │                     │
│         │           │    (max 3 iters)│                     │
│         │           └────────┬────────┘                     │
│         ▼                    │                              │
│  ┌──────────────┐            │                              │
│  │ 5. Rerank &  │◀───────────┘                              │
│  │    Return    │                                           │
│  └──────────────┘                                           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Implementation

```rust
// crates/rag/src/agentic.rs

use crate::{HybridRetriever, EarlyExitReranker, SearchResult, RagError};

#[derive(Debug, Clone)]
pub struct AgenticRagConfig {
    /// Minimum sufficiency score to skip rewrite (0.0-1.0)
    pub sufficiency_threshold: f32,

    /// Maximum query rewrite iterations
    pub max_iterations: usize,

    /// Enable/disable agentic flow (fallback to single-shot)
    pub enabled: bool,

    /// Intent classification model path (optional)
    pub intent_model: Option<String>,
}

impl Default for AgenticRagConfig {
    fn default() -> Self {
        Self {
            sufficiency_threshold: 0.7,
            max_iterations: 3,
            enabled: true,
            intent_model: None,
        }
    }
}

pub struct AgenticRetriever {
    config: AgenticRagConfig,
    retriever: HybridRetriever,
    reranker: EarlyExitReranker,
    query_rewriter: Option<QueryRewriter>,
    sufficiency_checker: SufficiencyChecker,
}

impl AgenticRetriever {
    /// Multi-step retrieval with configurable complexity
    pub async fn search(
        &self,
        query: &str,
        context: &ConversationContext,
    ) -> Result<Vec<SearchResult>, RagError> {
        // Fast path: single-shot if agentic disabled
        if !self.config.enabled {
            return self.retriever.search(query, 5).await;
        }

        // Step 1: Initial retrieval
        let mut results = self.retriever.search(query, 10).await?;
        let mut current_query = query.to_string();

        // Step 2-4: Iterative refinement
        for iteration in 0..self.config.max_iterations {
            // Check sufficiency
            let score = self.sufficiency_checker.score(&results, &current_query).await?;

            if score >= self.config.sufficiency_threshold {
                tracing::debug!(
                    iteration,
                    score,
                    "Sufficiency threshold met, stopping iteration"
                );
                break;
            }

            // Rewrite query if we have a rewriter
            if let Some(rewriter) = &self.query_rewriter {
                current_query = rewriter.rewrite(&current_query, &results, context).await?;
                tracing::debug!(
                    iteration,
                    new_query = %current_query,
                    "Query rewritten"
                );

                // Re-retrieve with new query
                results = self.retriever.search(&current_query, 10).await?;
            } else {
                break; // No rewriter, can't improve
            }
        }

        // Step 5: Final reranking
        let reranked = self.reranker.rerank(&current_query, results, 5).await?;

        Ok(reranked)
    }
}

/// Checks if retrieved results sufficiently answer the query
pub struct SufficiencyChecker {
    /// Cross-encoder model for relevance scoring
    model: Option<CrossEncoder>,
}

impl SufficiencyChecker {
    pub async fn score(
        &self,
        results: &[SearchResult],
        query: &str,
    ) -> Result<f32, RagError> {
        if results.is_empty() {
            return Ok(0.0);
        }

        match &self.model {
            Some(encoder) => {
                // Use cross-encoder for semantic relevance
                let scores: Vec<f32> = results
                    .iter()
                    .take(3)
                    .map(|r| encoder.score(query, &r.content))
                    .collect::<Result<Vec<_>, _>>()?;

                Ok(scores.iter().sum::<f32>() / scores.len() as f32)
            }
            None => {
                // Fallback: use retriever scores
                Ok(results.iter().take(3).map(|r| r.score).sum::<f32>() / 3.0)
            }
        }
    }
}

/// Rewrites queries for better retrieval
pub struct QueryRewriter {
    llm: Arc<dyn LlmBackend>,
}

impl QueryRewriter {
    pub async fn rewrite(
        &self,
        query: &str,
        results: &[SearchResult],
        context: &ConversationContext,
    ) -> Result<String, RagError> {
        let prompt = format!(
            r#"The following query did not retrieve sufficient information:
Query: {query}

Top results retrieved (insufficient):
{results}

Conversation context:
{context}

Rewrite the query to be more specific and likely to find relevant information.
Only output the rewritten query, nothing else."#,
            query = query,
            results = results.iter().take(3).map(|r| &r.content).collect::<Vec<_>>().join("\n"),
            context = context.summary,
        );

        let response = self.llm.generate(&[Message::user(&prompt)]).await?;
        Ok(response.text.trim().to_string())
    }
}
```

### Configuration

```yaml
# config/rag.yaml

agentic:
  enabled: true                    # Set false for simple single-shot
  sufficiency_threshold: 0.7       # 0.0-1.0
  max_iterations: 3                # 1-5 recommended

  # Optional: disable components for simpler flow
  query_rewriting: true
  intent_classification: false     # Enable when model available
```

---

## Q11: Error Recovery & Fallback Strategy

### Design Principles

1. **Graceful Degradation**: Always provide a response, even if degraded
2. **User Transparency**: Inform user when operating in degraded mode
3. **Retry with Backoff**: Transient errors get retries
4. **Circuit Breaker**: Persistent failures trigger fallback mode

### Error Categories

| Category | Examples | Strategy |
|----------|----------|----------|
| **Transient** | Network timeout, rate limit | Retry 3x with exponential backoff |
| **Recoverable** | Model OOM, parsing error | Fallback to simpler model/method |
| **Permanent** | Invalid config, missing model | Fail fast with clear error |
| **Partial** | LLM returns incomplete | Prompt for completion or use partial |

### Implementation

```rust
// crates/core/src/error_recovery.rs

use std::time::Duration;
use tokio::time::sleep;

#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_retries: usize,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 100,
            max_delay_ms: 5000,
            backoff_multiplier: 2.0,
        }
    }
}

/// Retry with exponential backoff
pub async fn retry_with_backoff<T, E, F, Fut>(
    config: &RetryConfig,
    operation: F,
) -> Result<T, E>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
    E: std::fmt::Debug,
{
    let mut delay = config.initial_delay_ms;
    let mut last_error = None;

    for attempt in 0..=config.max_retries {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(e) => {
                tracing::warn!(
                    attempt,
                    max_retries = config.max_retries,
                    error = ?e,
                    "Operation failed, retrying"
                );
                last_error = Some(e);

                if attempt < config.max_retries {
                    sleep(Duration::from_millis(delay)).await;
                    delay = (delay as f64 * config.backoff_multiplier)
                        .min(config.max_delay_ms as f64) as u64;
                }
            }
        }
    }

    Err(last_error.unwrap())
}

/// Circuit breaker for persistent failures
pub struct CircuitBreaker {
    failure_count: AtomicUsize,
    last_failure: AtomicU64,
    config: CircuitBreakerConfig,
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: usize,
    pub reset_timeout_ms: u64,
}

impl CircuitBreaker {
    pub fn is_open(&self) -> bool {
        let failures = self.failure_count.load(Ordering::Relaxed);
        if failures < self.config.failure_threshold {
            return false;
        }

        // Check if enough time has passed to retry
        let last = self.last_failure.load(Ordering::Relaxed);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        now - last < self.config.reset_timeout_ms
    }

    pub fn record_success(&self) {
        self.failure_count.store(0, Ordering::Relaxed);
    }

    pub fn record_failure(&self) {
        self.failure_count.fetch_add(1, Ordering::Relaxed);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        self.last_failure.store(now, Ordering::Relaxed);
    }
}
```

### Component-Specific Fallbacks

```rust
// crates/agent/src/fallback.rs

/// Fallback responses when LLM fails
pub struct FallbackResponses {
    responses: HashMap<Intent, Vec<String>>,
}

impl FallbackResponses {
    pub fn get_fallback(&self, intent: Intent, language: LanguageCode) -> String {
        // Predefined responses for common intents
        match intent {
            Intent::Greeting => match language {
                LanguageCode::Hindi => "नमस्ते! मैं आपकी मदद कर सकता हूं।".to_string(),
                _ => "Hello! I can help you with gold loans.".to_string(),
            },
            Intent::GoldLoanEnquiry => {
                "I can help you with gold loan information. Could you tell me how much gold you have?".to_string()
            },
            Intent::Unknown => {
                "I'm having trouble understanding. Could you please rephrase that?".to_string()
            },
            _ => "Let me transfer you to a human agent for assistance.".to_string(),
        }
    }
}

/// STT fallback chain
pub struct SttFallbackChain {
    primary: Box<dyn SttBackend>,       // IndicConformer
    secondary: Box<dyn SttBackend>,     // Whisper
    circuit_breaker: CircuitBreaker,
}

impl SttFallbackChain {
    pub async fn transcribe(&self, audio: &[f32]) -> Result<String, SttError> {
        // Try primary if circuit is closed
        if !self.circuit_breaker.is_open() {
            match self.primary.transcribe(audio, true).await {
                Ok(result) => {
                    self.circuit_breaker.record_success();
                    return Ok(result.text);
                }
                Err(e) => {
                    self.circuit_breaker.record_failure();
                    tracing::warn!(error = ?e, "Primary STT failed, falling back");
                }
            }
        }

        // Fallback to secondary
        self.secondary.transcribe(audio, true).await.map(|r| r.text)
    }
}
```

### Error Response Format

```rust
/// User-facing error responses
#[derive(Debug)]
pub enum UserFacingError {
    /// Temporary issue, will retry
    Temporary { message: String },
    /// Need user action
    NeedsInput { message: String, prompt: String },
    /// Transferring to human
    Escalation { message: String, reason: String },
}

impl UserFacingError {
    pub fn to_tts_text(&self, language: LanguageCode) -> String {
        match (self, language) {
            (Self::Temporary { .. }, LanguageCode::Hindi) => {
                "एक छोटी सी समस्या है। कृपया दोबारा बोलें।".to_string()
            }
            (Self::Temporary { .. }, _) => {
                "I had a small issue. Could you please repeat that?".to_string()
            }
            (Self::NeedsInput { prompt, .. }, _) => prompt.clone(),
            (Self::Escalation { .. }, LanguageCode::Hindi) => {
                "मैं आपको हमारी टीम से जोड़ रहा हूं।".to_string()
            }
            (Self::Escalation { .. }, _) => {
                "Let me connect you with our team.".to_string()
            }
        }
    }
}
```

---

## Summary of Resolved Questions

| Question | Resolution |
|----------|------------|
| Q1: Latency 450-550ms | Achievable - reduce SLM timeout to 200ms |
| Q2: Model deployment | Need download script + NeMo export guide |
| Q3: IndicConformer | Use NeMo export to ONNX, Whisper as fallback |
| Q4: Translation layer | Pluggable trait design with gRPC/ONNX options |
| Q5: Agentic RAG | Configurable multi-step with sufficiency threshold |
| Q6: WebRTC | Planned, critical for 500ms target |
| Q7: Gold price API | Future enhancement, use static for now |
| Q8: Competitor rates | Static config, database later |
| Q9: Mutex contention | Consolidate to single lock (see above) |
| Q10: Integration tests | Add after implementation complete |
| Q11: Error recovery | Retry + circuit breaker + fallback chain |

---

*Last Updated: 2024-12-27*
