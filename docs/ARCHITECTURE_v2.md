# Voice Sales Agent Architecture v2.0

> Production-grade, domain-agnostic voice agent for customer acquisition
>
> **Target:** Sub-800ms latency | 22 Indian languages | On-premise banking deployment

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Design Philosophy](#design-philosophy)
3. [System Overview](#system-overview)
4. [Technology Stack](#technology-stack)
5. [Core Architecture](#core-architecture)
6. [Pipeline Design](#pipeline-design)
7. [Text Processing Pipeline](#text-processing-pipeline)
8. [RAG Strategy](#rag-strategy)
9. [Conversation Management](#conversation-management)
10. [Personalization Engine](#personalization-engine)
11. [Experiment Framework](#experiment-framework)
12. [Deployment Architecture](#deployment-architecture)
13. [Risk Assessment](#risk-assessment)
14. [Implementation Phases](#implementation-phases)
15. [Appendices](#appendices)

---

## Executive Summary

### The Challenge

Build a voice agent that:
- Acquires gold loan customers from competitors (Muthoot, Manappuram, IIFL)
- Speaks naturally in 22 Indian languages
- Responds in under 800ms
- Runs on-premise in bank infrastructure
- Complies with banking regulations
- Adapts to different customer segments
- Handles objections intelligently

### The Solution

A **Pure Rust** voice agent with:

| Capability | Implementation |
|------------|----------------|
| **Voice Processing** | sherpa-rs + ONNX models |
| **Text Processing** | Grammar correction, translation, PII redaction |
| **Intelligence** | Agentic RAG with configurable timing |
| **Orchestration** | Event-driven FSM with Tokio |
| **Personalization** | Segment-based messaging with psychology guardrails |
| **Extensibility** | Trait-driven, config-based domains |

### Key Innovations

1. **Sentence-by-Sentence Streaming** - TTS starts before LLM finishes
2. **Configurable RAG Timing** - Prefetch/parallel/sequential experiments
3. **LLM-as-Grammar-Corrector** - Domain-aware transcription fixing
4. **Translate-Think-Translate** - LLM reasons in English for accuracy
5. **Natural AI Disclosure** - Weave identity into conversation flow

---

## Design Philosophy

### Core Principles

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DESIGN PRINCIPLES                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. CONFIGURABILITY OVER CODE                                           │
│     └── Domain logic lives in TOML/YAML, not Rust                       │
│     └── New vertical = new config directory, zero code changes          │
│                                                                         │
│  2. STREAMING BY DEFAULT                                                │
│     └── Every pipeline stage operates sentence-by-sentence              │
│     └── Latency = time-to-first-byte, not time-to-completion            │
│                                                                         │
│  3. EXPERIMENT EVERYTHING                                               │
│     └── Every decision is an A/B testable hypothesis                    │
│     └── Metrics drive iteration, not intuition                          │
│                                                                         │
│  4. FAIL GRACEFULLY                                                     │
│     └── Every component has a fallback                                  │
│     └── Degraded service > no service                                   │
│                                                                         │
│  5. PRIVACY BY DESIGN                                                   │
│     └── On-premise first, cloud optional                                │
│     └── PII never leaves the system unredacted                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why Pure Rust?

| Reason | Explanation |
|--------|-------------|
| **No GIL** | True parallelism for pipeline stages |
| **Memory Safety** | No segfaults in production |
| **Performance** | C-level speed with zero-cost abstractions |
| **Type Safety** | Compile-time guarantees for complex state |
| **Single Binary** | Easy on-premise deployment |
| **WASM Future** | Browser/edge deployment possible |

**Acknowledged Trade-offs:**
- Steeper learning curve
- Smaller AI ecosystem (mitigated by ONNX)
- Longer development time (mitigated by good design)

### Why NOT Python?

Research shows Python is excellent for prototyping but faces challenges at scale:

> "When scaling to 500 agents on a 64-core machine, the Rust version would continue to scale, while the Python version would not."
> — [Red Hat Developer](https://developers.redhat.com/articles/2025/09/15/why-some-agentic-ai-developers-are-moving-code-python-rust)

However, we maintain Python fallback paths (gRPC sidecars) for:
- IndicTrans2 translation (if ONNX export fails)
- Complex NLP tasks without Rust libraries

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VOICE AGENT SYSTEM                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    WebSocket/    ┌──────────────────────────────────────┐ │
│  │   CLIENT    │    WebRTC        │            RUST BACKEND               │ │
│  │             │◄─────────────────►│                                      │ │
│  │ • Browser   │                  │  ┌────────────────────────────────┐  │ │
│  │ • Mobile    │                  │  │       AUDIO PIPELINE           │  │ │
│  │ • SIP/PSTN  │                  │  │  Audio → VAD → STT → ...       │  │ │
│  └─────────────┘                  │  └────────────────────────────────┘  │ │
│                                   │                 │                     │ │
│                                   │                 ▼                     │ │
│                                   │  ┌────────────────────────────────┐  │ │
│                                   │  │    TEXT PROCESSING PIPELINE    │  │ │
│                                   │  │  Grammar → Translate → ...     │  │ │
│                                   │  └────────────────────────────────┘  │ │
│                                   │                 │                     │ │
│                                   │                 ▼                     │ │
│                                   │  ┌────────────────────────────────┐  │ │
│                                   │  │      INTELLIGENCE LAYER        │  │ │
│                                   │  │  RAG │ LLM │ Tools │ Memory    │  │ │
│                                   │  └────────────────────────────────┘  │ │
│                                   │                 │                     │ │
│                                   │                 ▼                     │ │
│                                   │  ┌────────────────────────────────┐  │ │
│                                   │  │    OUTPUT PROCESSING PIPELINE  │  │ │
│                                   │  │  ... → Comply → PII → TTS      │  │ │
│                                   │  └────────────────────────────────┘  │ │
│                                   │                                      │ │
│                                   └──────────────────────────────────────┘ │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        SUPPORTING SERVICES                           │   │
│  │                                                                      │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │   │
│  │  │  Qdrant     │  │  Python     │  │  Metrics    │  │  Config    │ │   │
│  │  │  (Vectors)  │  │  Sidecar    │  │  (Prometheus│  │  Store     │ │   │
│  │  │             │  │  (Fallback) │  │  /Grafana)  │  │            │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow (Single Conversation Turn)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         CONVERSATION TURN DATA FLOW                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Time    0ms        200ms       300ms       500ms       700ms       900ms   │
│    │       │           │           │           │           │           │     │
│    ▼       ▼           ▼           ▼           ▼           ▼           ▼     │
│                                                                              │
│  ┌─────┐ ┌─────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────┐ ┌─────────┐   │
│  │Audio│→│ VAD │→│   STT   │→│ Grammar │→│Translate│→│ RAG │→│   LLM   │   │
│  │ In  │ │     │ │         │ │  Fix    │ │  IN→EN  │ │     │ │(stream) │   │
│  └─────┘ └─────┘ └─────────┘ └─────────┘ └─────────┘ └─────┘ └────┬────┘   │
│                                                                    │        │
│                        STREAMING SENTENCE-BY-SENTENCE              │        │
│                                    ┌───────────────────────────────┘        │
│                                    ▼                                        │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │Translate│→│Compliance│→│  PII   │→│Simplify │→│   TTS   │→│  Audio  │   │
│  │  EN→IN  │ │  Check   │ │ Redact │ │         │ │(stream) │ │   Out   │   │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
│                                                                              │
│  TOTAL TARGET LATENCY: < 800ms (Time to First Audio Byte)                   │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

### Core Dependencies

| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| **Async Runtime** | tokio | 1.x | Async I/O, channels, tasks |
| **Web Framework** | axum | 0.7.x | HTTP/WebSocket server |
| **ONNX Runtime** | ort | 2.x | ML model inference |
| **Speech** | sherpa-rs | 0.6.x | STT/TTS via sherpa-onnx |
| **Vector Store** | qdrant-client | 1.x | Semantic search |
| **NER/NLP** | rust-bert | 0.22.x | PII detection |
| **LLM** | kalosm | 0.4.x | Local LLM inference |
| **Config** | config + serde | - | TOML/YAML configuration |
| **Telemetry** | tracing + opentelemetry | - | Distributed tracing |
| **Serialization** | serde + serde_json | - | JSON/TOML/YAML |

### Model Stack

| Model | Size | Purpose | Fallback |
|-------|------|---------|----------|
| **IndicConformer** | 600M | Indian STT | Whisper |
| **Whisper** | 244M | Fallback STT | - |
| **IndicF5** | ~500M | Indian TTS | Piper |
| **Piper** | ~50M | Fallback TTS | - |
| **IndicTrans2** | 1.1B | Translation | Python gRPC |
| **Qwen2.5** | 7B-Q4 | Reasoning | Ollama API |
| **E5-multilingual** | 278M | Embeddings | - |

### Rust Crate Structure

```
voice-agent/
├── Cargo.toml                    # Workspace definition
│
├── crates/
│   ├── core/                     # Traits, types, errors
│   │   └── src/
│   │       ├── traits/           # All component interfaces
│   │       ├── types/            # Core data structures
│   │       └── error.rs          # Error types
│   │
│   ├── config/                   # Configuration management
│   │   └── src/
│   │       ├── settings.rs       # Global settings
│   │       ├── domain.rs         # Domain loader
│   │       └── experiments.rs    # A/B test config
│   │
│   ├── pipeline/                 # Event-driven pipeline
│   │   └── src/
│   │       ├── audio/            # Audio pipeline
│   │       ├── text/             # Text processing pipeline
│   │       └── streaming.rs      # Sentence streaming
│   │
│   ├── speech/                   # STT/TTS implementations
│   │   └── src/
│   │       ├── stt/              # Speech-to-text
│   │       └── tts/              # Text-to-speech
│   │
│   ├── text_processing/          # Grammar, translation, PII
│   │   └── src/
│   │       ├── grammar/          # Grammar correction
│   │       ├── translation/      # Language translation
│   │       ├── compliance/       # Regulatory checks
│   │       └── pii/              # PII detection/redaction
│   │
│   ├── rag/                      # Retrieval system
│   │   └── src/
│   │       ├── retriever.rs      # Hybrid retriever
│   │       ├── agentic.rs        # Multi-step RAG
│   │       └── timing.rs         # Timing strategies
│   │
│   ├── agent/                    # Conversation agent
│   │   └── src/
│   │       ├── fsm/              # State machine
│   │       ├── tools/            # Function tools
│   │       └── prompts/          # Prompt building
│   │
│   ├── personalization/          # Customer personalization
│   │   └── src/
│   │       ├── segments.rs       # Segment detection
│   │       ├── strategy.rs       # Persuasion strategy
│   │       └── disclosure.rs     # AI identity handling
│   │
│   ├── llm/                      # LLM providers
│   │   └── src/
│   │       ├── providers/        # Ollama, Claude, etc.
│   │       ├── router.rs         # Model routing
│   │       └── cache.rs          # Semantic caching
│   │
│   ├── experiments/              # A/B testing framework
│   │   └── src/
│   │       ├── variants.rs       # Variant selection
│   │       ├── metrics.rs        # Funnel tracking
│   │       └── sentiment.rs      # Sentiment analysis
│   │
│   └── server/                   # API server
│       └── src/
│           ├── main.rs           # Entry point
│           ├── routes/           # HTTP/WS handlers
│           └── middleware/       # Auth, metrics
│
├── domains/                      # Domain configurations
│   └── gold_loan/
│       ├── knowledge/            # YAML knowledge base
│       ├── prompts/              # Tera templates
│       ├── segments.toml         # Customer segments
│       ├── tools.toml            # Available tools
│       ├── compliance.toml       # Regulatory rules
│       └── experiments.toml      # A/B tests
│
├── models/                       # ONNX models (git-lfs)
│   ├── stt/
│   ├── tts/
│   ├── translation/
│   └── embeddings/
│
└── docs/                         # Documentation
    ├── interfaces/
    ├── pipeline/
    ├── rag/
    └── deployment/
```

---

## Core Architecture

### Trait-Driven Design

All components implement Rust traits for maximum flexibility:

```rust
// crates/core/src/traits/mod.rs

/// Speech-to-Text interface
#[async_trait]
pub trait SpeechToText: Send + Sync + 'static {
    /// Transcribe audio to text
    async fn transcribe(&self, audio: &AudioFrame) -> Result<TranscriptFrame>;

    /// Stream transcription as audio arrives
    fn transcribe_stream(
        &self,
        audio_stream: impl Stream<Item = AudioFrame> + Send,
    ) -> impl Stream<Item = Result<TranscriptFrame>> + Send;

    /// Get supported languages
    fn supported_languages(&self) -> &[Language];
}

/// Text-to-Speech interface
#[async_trait]
pub trait TextToSpeech: Send + Sync + 'static {
    /// Synthesize text to audio
    async fn synthesize(&self, text: &str, config: &VoiceConfig) -> Result<AudioFrame>;

    /// Stream synthesis sentence-by-sentence
    fn synthesize_stream(
        &self,
        text_stream: impl Stream<Item = String> + Send,
        config: &VoiceConfig,
    ) -> impl Stream<Item = Result<AudioFrame>> + Send;

    /// Get available voices
    fn available_voices(&self) -> &[VoiceInfo];
}

/// Language Model interface
#[async_trait]
pub trait LanguageModel: Send + Sync + 'static {
    /// Generate completion
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse>;

    /// Stream tokens as generated
    fn generate_stream(
        &self,
        request: GenerateRequest,
    ) -> impl Stream<Item = Result<StreamChunk>> + Send;

    /// Generate with tool use
    async fn generate_with_tools(
        &self,
        request: GenerateRequest,
        tools: &[ToolDefinition],
    ) -> Result<GenerateResponse>;
}

/// Retriever interface
#[async_trait]
pub trait Retriever: Send + Sync + 'static {
    /// Retrieve relevant documents
    async fn retrieve(&self, query: &str, options: &RetrieveOptions) -> Result<Vec<Document>>;

    /// Agentic multi-step retrieval
    async fn retrieve_agentic(
        &self,
        query: &str,
        context: &ConversationContext,
        max_iterations: usize,
    ) -> Result<Vec<Document>>;
}

/// Grammar Corrector interface
#[async_trait]
pub trait GrammarCorrector: Send + Sync + 'static {
    /// Correct grammar with domain context
    async fn correct(&self, text: &str, context: &DomainContext) -> Result<String>;

    /// Stream corrections
    fn correct_stream(
        &self,
        text_stream: impl Stream<Item = String> + Send,
        context: &DomainContext,
    ) -> impl Stream<Item = Result<String>> + Send;
}

/// Translator interface
#[async_trait]
pub trait Translator: Send + Sync + 'static {
    /// Translate text
    async fn translate(&self, text: &str, from: Language, to: Language) -> Result<String>;

    /// Detect language
    async fn detect_language(&self, text: &str) -> Result<Language>;

    /// Stream translation
    fn translate_stream(
        &self,
        text_stream: impl Stream<Item = String> + Send,
        from: Language,
        to: Language,
    ) -> impl Stream<Item = Result<String>> + Send;
}

/// PII Redactor interface
#[async_trait]
pub trait PIIRedactor: Send + Sync + 'static {
    /// Detect PII entities
    async fn detect(&self, text: &str) -> Result<Vec<PIIEntity>>;

    /// Redact PII from text
    async fn redact(&self, text: &str, strategy: &RedactionStrategy) -> Result<String>;
}

/// Compliance Checker interface
#[async_trait]
pub trait ComplianceChecker: Send + Sync + 'static {
    /// Check for compliance violations
    async fn check(&self, text: &str) -> Result<ComplianceResult>;

    /// Make text compliant
    async fn make_compliant(&self, text: &str) -> Result<String>;
}
```

### Core Types

```rust
// crates/core/src/types/mod.rs

/// Audio frame with metadata
#[derive(Debug, Clone)]
pub struct AudioFrame {
    pub data: Vec<i16>,
    pub sample_rate: u32,
    pub channels: u8,
    pub timestamp_ms: u64,
}

/// Transcript with confidence
#[derive(Debug, Clone)]
pub struct TranscriptFrame {
    pub text: String,
    pub language: Language,
    pub confidence: f32,
    pub is_final: bool,
    pub words: Vec<WordTiming>,
}

/// Supported languages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Language {
    English,
    Hindi,
    Tamil,
    Telugu,
    Kannada,
    Malayalam,
    Bengali,
    Marathi,
    Gujarati,
    Punjabi,
    Odia,
    Assamese,
    // ... all 22 scheduled languages
}

/// Voice configuration
#[derive(Debug, Clone)]
pub struct VoiceConfig {
    pub language: Language,
    pub voice_id: String,
    pub speed: f32,
    pub pitch: f32,
}

/// Conversation context
#[derive(Debug, Clone)]
pub struct ConversationContext {
    pub session_id: String,
    pub customer_id: Option<String>,
    pub customer_profile: Option<CustomerProfile>,
    pub segment: Option<CustomerSegment>,
    pub language: Language,
    pub history: Vec<Turn>,
    pub state: ConversationState,
    pub metadata: HashMap<String, Value>,
}

/// Customer segment
#[derive(Debug, Clone, Copy)]
pub enum CustomerSegment {
    P1HighValue,      // MSME, 5-25L loans
    P2TrustSeeker,    // Safety-focused, 40-55y
    P3Shakti,         // Women entrepreneurs
    P4YoungPro,       // Digital-native, 21-35y
    Unknown,
}

/// Conversation state
#[derive(Debug, Clone)]
pub enum ConversationState {
    Idle,
    Greeting,
    Discovery,
    NeedsAnalysis,
    Pitch,
    Comparison,
    ObjectionHandling { objection_type: String },
    Closing,
    // End states
    Converted { appointment: Option<String> },
    FollowUp { reason: String, scheduled: Option<String> },
    Declined { reason: String },
    Escalated { to: String },
}
```

---

## Pipeline Design

### Event-Driven Architecture

```rust
// crates/pipeline/src/lib.rs

/// Pipeline events (frames)
#[derive(Debug, Clone)]
pub enum Frame {
    // Audio frames
    AudioInput(AudioFrame),
    AudioOutput(AudioFrame),

    // Speech frames
    TranscriptPartial(TranscriptFrame),
    TranscriptFinal(TranscriptFrame),

    // Text processing frames
    GrammarCorrected(String),
    Translated(String, Language, Language),
    ComplianceChecked(String, ComplianceResult),
    PIIRedacted(String),

    // LLM frames
    LLMChunk(String),
    LLMComplete(String),
    ToolCall(ToolCall),
    ToolResult(ToolResult),

    // Control frames
    UserSpeaking,
    UserSilence(Duration),
    BargeIn,
    EndOfTurn,

    // System frames
    StateChange(ConversationState),
    Error(PipelineError),
    Metrics(MetricsEvent),
}

/// Frame processor trait
#[async_trait]
pub trait FrameProcessor: Send + Sync + 'static {
    /// Process a frame and emit zero or more output frames
    async fn process(
        &self,
        frame: Frame,
        context: &mut ProcessorContext,
    ) -> Result<Vec<Frame>>;

    /// Get processor name for tracing
    fn name(&self) -> &'static str;
}

/// Pipeline orchestrator
pub struct Pipeline {
    processors: Vec<Box<dyn FrameProcessor>>,
    input_tx: mpsc::Sender<Frame>,
    output_rx: mpsc::Receiver<Frame>,
}

impl Pipeline {
    pub async fn run(&mut self) -> Result<()> {
        // Create channels between processors
        let mut channels = Vec::new();
        for _ in 0..self.processors.len() {
            channels.push(mpsc::channel::<Frame>(100));
        }

        // Spawn each processor as a task
        for (i, processor) in self.processors.iter().enumerate() {
            let name = processor.name();
            let processor = processor.clone();
            let rx = if i == 0 {
                self.input_tx.subscribe()
            } else {
                channels[i - 1].1.clone()
            };
            let tx = channels[i].0.clone();

            tokio::spawn(async move {
                let span = tracing::span!(Level::INFO, "processor", name);
                let _guard = span.enter();

                let mut context = ProcessorContext::new();

                while let Some(frame) = rx.recv().await {
                    let start = Instant::now();

                    match processor.process(frame, &mut context).await {
                        Ok(output_frames) => {
                            for output in output_frames {
                                tx.send(output).await?;
                            }
                        }
                        Err(e) => {
                            tx.send(Frame::Error(e.into())).await?;
                        }
                    }

                    tracing::debug!(
                        processor = name,
                        duration_ms = start.elapsed().as_millis(),
                        "frame processed"
                    );
                }

                Ok::<_, Error>(())
            });
        }

        Ok(())
    }
}
```

### Sentence Streaming

The key innovation for low latency is sentence-by-sentence streaming:

```rust
// crates/pipeline/src/streaming.rs

/// Detects sentence boundaries across languages
pub struct SentenceDetector {
    terminators: HashSet<char>,
}

impl SentenceDetector {
    pub fn new() -> Self {
        Self {
            terminators: [
                '.', '!', '?',           // English
                '।',                      // Hindi (Devanagari Danda)
                '॥',                      // Sanskrit (Double Danda)
                '।',                      // Other Indic scripts
            ].into_iter().collect(),
        }
    }

    /// Find sentence boundary in text
    pub fn find_boundary(&self, text: &str) -> Option<usize> {
        text.char_indices()
            .find(|(_, c)| self.terminators.contains(c))
            .map(|(i, c)| i + c.len_utf8())
    }
}

/// Accumulates text and emits complete sentences
pub struct SentenceAccumulator {
    buffer: String,
    detector: SentenceDetector,
}

impl SentenceAccumulator {
    /// Add chunk and get any complete sentences
    pub fn add(&mut self, chunk: &str) -> Vec<String> {
        self.buffer.push_str(chunk);

        let mut sentences = Vec::new();

        while let Some(boundary) = self.detector.find_boundary(&self.buffer) {
            let sentence = self.buffer[..boundary].trim().to_string();
            if !sentence.is_empty() {
                sentences.push(sentence);
            }
            self.buffer = self.buffer[boundary..].to_string();
        }

        sentences
    }

    /// Flush remaining buffer
    pub fn flush(&mut self) -> Option<String> {
        let remaining = std::mem::take(&mut self.buffer);
        let trimmed = remaining.trim();
        if !trimmed.is_empty() {
            Some(trimmed.to_string())
        } else {
            None
        }
    }
}

/// LLM to TTS streaming processor
pub struct LLMToTTSStreamer {
    accumulator: SentenceAccumulator,
    tts: Arc<dyn TextToSpeech>,
    voice_config: VoiceConfig,
}

impl FrameProcessor for LLMToTTSStreamer {
    async fn process(
        &self,
        frame: Frame,
        context: &mut ProcessorContext,
    ) -> Result<Vec<Frame>> {
        match frame {
            Frame::LLMChunk(chunk) => {
                let sentences = self.accumulator.add(&chunk);
                let mut outputs = Vec::new();

                for sentence in sentences {
                    // Synthesize each sentence immediately
                    let audio = self.tts
                        .synthesize(&sentence, &self.voice_config)
                        .await?;
                    outputs.push(Frame::AudioOutput(audio));
                }

                Ok(outputs)
            }
            Frame::LLMComplete(_) => {
                // Flush any remaining text
                let mut outputs = Vec::new();
                if let Some(remaining) = self.accumulator.flush() {
                    let audio = self.tts
                        .synthesize(&remaining, &self.voice_config)
                        .await?;
                    outputs.push(Frame::AudioOutput(audio));
                }
                outputs.push(Frame::EndOfTurn);
                Ok(outputs)
            }
            _ => Ok(vec![frame]),
        }
    }

    fn name(&self) -> &'static str {
        "llm_to_tts_streamer"
    }
}
```

### Interrupt Handling (Barge-In)

```rust
// crates/pipeline/src/interrupt.rs

/// Interrupt detection and handling
pub struct InterruptHandler {
    config: InterruptConfig,
    state: InterruptState,
    vad: Arc<dyn VoiceActivityDetector>,
}

#[derive(Debug, Clone)]
pub struct InterruptConfig {
    /// How to handle interrupts
    pub mode: InterruptMode,
    /// VAD sensitivity (0.0 - 1.0)
    pub vad_sensitivity: f32,
    /// Minimum speech duration to trigger interrupt
    pub min_speech_duration_ms: u64,
    /// Silence timeout after speech
    pub silence_timeout_ms: u64,
}

#[derive(Debug, Clone, Copy)]
pub enum InterruptMode {
    /// Stop at sentence boundary
    SentenceBoundary,
    /// Stop immediately (may clip)
    Immediate,
    /// Stop at word boundary
    WordBoundary,
}

#[derive(Debug)]
enum InterruptState {
    Idle,
    AgentSpeaking {
        start_time: Instant,
        current_sentence: String,
    },
    UserInterrupting {
        speech_start: Instant,
        accumulated_duration: Duration,
    },
}

impl InterruptHandler {
    pub async fn process_audio_during_agent_speech(
        &mut self,
        audio: &AudioFrame,
    ) -> Option<InterruptAction> {
        // Run VAD on incoming audio
        let is_speech = self.vad
            .detect(audio, self.config.vad_sensitivity)
            .await;

        match (&mut self.state, is_speech) {
            // Agent speaking, user starts talking
            (InterruptState::AgentSpeaking { .. }, true) => {
                self.state = InterruptState::UserInterrupting {
                    speech_start: Instant::now(),
                    accumulated_duration: Duration::ZERO,
                };
                None // Wait for minimum duration
            }

            // User continues speaking
            (InterruptState::UserInterrupting { speech_start, accumulated_duration }, true) => {
                *accumulated_duration = speech_start.elapsed();

                if *accumulated_duration >= Duration::from_millis(self.config.min_speech_duration_ms) {
                    // Trigger interrupt based on mode
                    Some(match self.config.mode {
                        InterruptMode::Immediate => InterruptAction::StopNow,
                        InterruptMode::SentenceBoundary => InterruptAction::StopAtSentence,
                        InterruptMode::WordBoundary => InterruptAction::StopAtWord,
                    })
                } else {
                    None
                }
            }

            // User stopped speaking (false positive or gave up)
            (InterruptState::UserInterrupting { .. }, false) => {
                self.state = InterruptState::AgentSpeaking {
                    start_time: Instant::now(),
                    current_sentence: String::new(),
                };
                None
            }

            _ => None,
        }
    }
}

pub enum InterruptAction {
    StopNow,
    StopAtSentence,
    StopAtWord,
}
```

---

## Text Processing Pipeline

### Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                       TEXT PROCESSING PIPELINE                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT PROCESSING (Post-STT):                                                │
│                                                                              │
│  ┌─────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│  │   STT   │──►│   Grammar   │──►│  Translate  │──►│  To LLM     │          │
│  │ Output  │   │   Correct   │   │   IN → EN   │   │             │          │
│  └─────────┘   └─────────────┘   └─────────────┘   └─────────────┘          │
│                      │                  │                                    │
│                      ▼                  ▼                                    │
│               ┌─────────────┐    ┌─────────────┐                            │
│               │   Domain    │    │  Language   │                            │
│               │   Context   │    │  Detection  │                            │
│               └─────────────┘    └─────────────┘                            │
│                                                                              │
│  OUTPUT PROCESSING (Pre-TTS):                                                │
│                                                                              │
│  ┌─────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐          │
│  │   LLM   │──►│  Translate  │──►│ Compliance  │──►│    PII      │          │
│  │ Output  │   │   EN → IN   │   │   Check     │   │   Redact    │          │
│  └─────────┘   └─────────────┘   └─────────────┘   └─────────────┘          │
│                                         │                  │                 │
│                                         ▼                  ▼                 │
│                                  ┌─────────────┐   ┌─────────────┐          │
│                                  │  Simplify   │──►│    TTS      │          │
│                                  │  For TTS    │   │             │          │
│                                  └─────────────┘   └─────────────┘          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Grammar Correction

**Problem:** STT is error-prone, especially for Indian languages.

**Solution:** LLM-based grammar correction with domain context.

```rust
// crates/text_processing/src/grammar/llm_corrector.rs

pub struct LLMGrammarCorrector {
    llm: Arc<dyn LanguageModel>,
    domain_context: DomainContext,
}

#[derive(Debug, Clone)]
pub struct DomainContext {
    /// Domain-specific vocabulary
    pub vocabulary: Vec<String>,
    /// Common phrases in this domain
    pub phrases: Vec<String>,
    /// Entity types to preserve (names, numbers, etc.)
    pub preserve_entities: Vec<String>,
}

impl LLMGrammarCorrector {
    /// Build grammar correction prompt
    fn build_prompt(&self, text: &str) -> String {
        format!(r#"
You are a speech-to-text error corrector for a gold loan sales conversation.

DOMAIN VOCABULARY:
{vocabulary}

COMMON PHRASES:
{phrases}

RULES:
1. Fix obvious transcription errors
2. Preserve proper nouns and numbers exactly
3. Keep the meaning identical
4. Output ONLY the corrected text, nothing else
5. If text is already correct, output it unchanged

INPUT: {text}
CORRECTED:"#,
            vocabulary = self.domain_context.vocabulary.join(", "),
            phrases = self.domain_context.phrases.join("\n"),
            text = text,
        )
    }
}

#[async_trait]
impl GrammarCorrector for LLMGrammarCorrector {
    async fn correct(&self, text: &str, context: &DomainContext) -> Result<String> {
        let prompt = self.build_prompt(text);

        let response = self.llm.generate(GenerateRequest {
            prompt,
            max_tokens: (text.len() * 2) as u32,
            temperature: 0.1,  // Low temperature for accuracy
            ..Default::default()
        }).await?;

        Ok(response.text.trim().to_string())
    }

    fn correct_stream(
        &self,
        text_stream: impl Stream<Item = String> + Send,
        context: &DomainContext,
    ) -> impl Stream<Item = Result<String>> + Send {
        // For streaming, we accumulate sentences and correct each
        let accumulator = SentenceAccumulator::new();
        let llm = self.llm.clone();
        let ctx = context.clone();

        stream! {
            let mut acc = accumulator;

            pin_mut!(text_stream);
            while let Some(chunk) = text_stream.next().await {
                for sentence in acc.add(&chunk) {
                    match self.correct(&sentence, &ctx).await {
                        Ok(corrected) => yield Ok(corrected),
                        Err(e) => yield Err(e),
                    }
                }
            }

            // Flush remaining
            if let Some(remaining) = acc.flush() {
                match self.correct(&remaining, &ctx).await {
                    Ok(corrected) => yield Ok(corrected),
                    Err(e) => yield Err(e),
                }
            }
        }
    }
}
```

### Translation (Translate-Think-Translate)

**Rationale:** LLMs reason better in English. We translate Indian languages to English for LLM processing, then translate back.

```rust
// crates/text_processing/src/translation/indictrans.rs

/// IndicTrans2 translator (ONNX or gRPC fallback)
pub struct IndicTranslator {
    onnx_session: Option<ort::Session>,
    grpc_client: Option<IndicTransGrpcClient>,
    tokenizer: IndicTransTokenizer,
}

impl IndicTranslator {
    pub async fn new(config: &TranslationConfig) -> Result<Self> {
        // Try ONNX first
        if let Some(onnx_path) = &config.onnx_model_path {
            match Self::load_onnx(onnx_path).await {
                Ok(session) => {
                    return Ok(Self {
                        onnx_session: Some(session),
                        grpc_client: None,
                        tokenizer: IndicTransTokenizer::new()?,
                    });
                }
                Err(e) => {
                    tracing::warn!("ONNX load failed, falling back to gRPC: {}", e);
                }
            }
        }

        // Fallback to gRPC
        let client = IndicTransGrpcClient::connect(&config.grpc_endpoint).await?;
        Ok(Self {
            onnx_session: None,
            grpc_client: Some(client),
            tokenizer: IndicTransTokenizer::new()?,
        })
    }
}

#[async_trait]
impl Translator for IndicTranslator {
    async fn translate(&self, text: &str, from: Language, to: Language) -> Result<String> {
        if from == to {
            return Ok(text.to_string());
        }

        if let Some(session) = &self.onnx_session {
            // ONNX inference
            let tokens = self.tokenizer.encode(text, from)?;
            let input = ort::Value::from_array(tokens)?;

            let outputs = session.run(vec![input])?;
            let output_tokens = outputs[0].try_extract()?;

            Ok(self.tokenizer.decode(&output_tokens, to)?)
        } else if let Some(client) = &self.grpc_client {
            // gRPC fallback
            client.translate(text, from, to).await
        } else {
            Err(Error::NoTranslatorAvailable)
        }
    }

    async fn detect_language(&self, text: &str) -> Result<Language> {
        // Use character analysis for Indic scripts
        let scripts = analyze_scripts(text);

        // Map script to most likely language
        match scripts.dominant_script() {
            Script::Devanagari => Ok(Language::Hindi),
            Script::Tamil => Ok(Language::Tamil),
            Script::Telugu => Ok(Language::Telugu),
            Script::Kannada => Ok(Language::Kannada),
            Script::Malayalam => Ok(Language::Malayalam),
            Script::Bengali => Ok(Language::Bengali),
            Script::Latin => Ok(Language::English),
            _ => Ok(Language::Hindi), // Default
        }
    }

    fn translate_stream(
        &self,
        text_stream: impl Stream<Item = String> + Send,
        from: Language,
        to: Language,
    ) -> impl Stream<Item = Result<String>> + Send {
        let translator = self.clone();

        stream! {
            let mut accumulator = SentenceAccumulator::new();

            pin_mut!(text_stream);
            while let Some(chunk) = text_stream.next().await {
                for sentence in accumulator.add(&chunk) {
                    yield translator.translate(&sentence, from, to).await;
                }
            }

            if let Some(remaining) = accumulator.flush() {
                yield translator.translate(&remaining, from, to).await;
            }
        }
    }
}
```

### PII Detection and Redaction

```rust
// crates/text_processing/src/pii/detector.rs

/// PII types specific to India
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PIIType {
    // Standard
    PersonName,
    PhoneNumber,
    Email,
    Address,

    // India-specific
    Aadhaar,      // 12-digit
    PAN,          // ABCDE1234F
    VoterId,
    DrivingLicense,
    Passport,
    BankAccount,
    IFSC,

    // Financial
    LoanAmount,
    InterestRate,
    CompetitorName,
}

/// Detected PII entity
#[derive(Debug, Clone)]
pub struct PIIEntity {
    pub pii_type: PIIType,
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub confidence: f32,
}

/// Redaction strategy
#[derive(Debug, Clone)]
pub enum RedactionStrategy {
    /// Replace with [REDACTED]
    Mask,
    /// Replace with type: [PHONE]
    TypeMask,
    /// Replace with asterisks: 98****1234
    PartialMask { visible_chars: usize },
    /// Remove entirely
    Remove,
    /// Replace with fake data
    Synthesize,
}

/// Hybrid PII detector (rust-bert NER + regex)
pub struct HybridPIIDetector {
    ner_model: Arc<NERModel>,
    regex_patterns: HashMap<PIIType, Regex>,
}

impl HybridPIIDetector {
    pub fn new() -> Result<Self> {
        let ner_model = NERModel::new(Default::default())?;

        let mut patterns = HashMap::new();

        // Aadhaar: 12 digits, often with spaces
        patterns.insert(
            PIIType::Aadhaar,
            Regex::new(r"\b\d{4}\s?\d{4}\s?\d{4}\b")?,
        );

        // PAN: 5 letters, 4 digits, 1 letter
        patterns.insert(
            PIIType::PAN,
            Regex::new(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")?,
        );

        // Indian phone: +91 or 0 followed by 10 digits
        patterns.insert(
            PIIType::PhoneNumber,
            Regex::new(r"(\+91[\-\s]?)?[0]?[6-9]\d{9}")?,
        );

        // Email
        patterns.insert(
            PIIType::Email,
            Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")?,
        );

        // IFSC: 4 letters, 0, 6 alphanumeric
        patterns.insert(
            PIIType::IFSC,
            Regex::new(r"\b[A-Z]{4}0[A-Z0-9]{6}\b")?,
        );

        Ok(Self {
            ner_model: Arc::new(ner_model),
            regex_patterns: patterns,
        })
    }
}

#[async_trait]
impl PIIRedactor for HybridPIIDetector {
    async fn detect(&self, text: &str) -> Result<Vec<PIIEntity>> {
        let mut entities = Vec::new();

        // 1. Regex-based detection (fast, precise patterns)
        for (pii_type, regex) in &self.regex_patterns {
            for capture in regex.find_iter(text) {
                entities.push(PIIEntity {
                    pii_type: *pii_type,
                    text: capture.as_str().to_string(),
                    start: capture.start(),
                    end: capture.end(),
                    confidence: 0.99, // Regex matches are high confidence
                });
            }
        }

        // 2. NER-based detection (names, addresses)
        let ner_entities = self.ner_model.predict(&[text])?;
        for entity in ner_entities.into_iter().flatten() {
            let pii_type = match entity.label.as_str() {
                "PER" => Some(PIIType::PersonName),
                "LOC" => Some(PIIType::Address),
                _ => None,
            };

            if let Some(pii_type) = pii_type {
                entities.push(PIIEntity {
                    pii_type,
                    text: entity.word.clone(),
                    start: entity.offset.0,
                    end: entity.offset.1,
                    confidence: entity.score,
                });
            }
        }

        // Sort by position and deduplicate overlaps
        entities.sort_by_key(|e| e.start);
        Ok(deduplicate_overlapping(entities))
    }

    async fn redact(&self, text: &str, strategy: &RedactionStrategy) -> Result<String> {
        let entities = self.detect(text).await?;

        let mut result = text.to_string();

        // Apply redactions in reverse order to preserve indices
        for entity in entities.into_iter().rev() {
            let replacement = match strategy {
                RedactionStrategy::Mask => "[REDACTED]".to_string(),
                RedactionStrategy::TypeMask => format!("[{:?}]", entity.pii_type),
                RedactionStrategy::PartialMask { visible_chars } => {
                    partial_mask(&entity.text, *visible_chars)
                }
                RedactionStrategy::Remove => String::new(),
                RedactionStrategy::Synthesize => synthesize_fake(&entity.pii_type),
            };

            result.replace_range(entity.start..entity.end, &replacement);
        }

        Ok(result)
    }
}

fn partial_mask(text: &str, visible: usize) -> String {
    if text.len() <= visible * 2 {
        return "*".repeat(text.len());
    }

    let prefix: String = text.chars().take(visible).collect();
    let suffix: String = text.chars().rev().take(visible).collect::<String>().chars().rev().collect();
    let middle = "*".repeat(text.len() - visible * 2);

    format!("{}{}{}", prefix, middle, suffix)
}
```

### Compliance Checking

```rust
// crates/text_processing/src/compliance/checker.rs

/// Compliance rules for banking conversations
#[derive(Debug, Clone, Deserialize)]
pub struct ComplianceRules {
    /// Phrases that must never be spoken
    pub forbidden_phrases: Vec<String>,
    /// Claims that require disclaimers
    pub claims_requiring_disclaimer: Vec<ClaimRule>,
    /// Required disclosures
    pub required_disclosures: Vec<Disclosure>,
    /// Rate/fee accuracy rules
    pub rate_rules: RateRules,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClaimRule {
    pub pattern: String,
    pub disclaimer: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Disclosure {
    pub trigger: String,
    pub text: String,
    pub timing: DisclosureTiming,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub enum DisclosureTiming {
    Immediately,
    EndOfSentence,
    EndOfTurn,
}

/// Compliance check result
#[derive(Debug, Clone)]
pub struct ComplianceResult {
    pub is_compliant: bool,
    pub violations: Vec<ComplianceViolation>,
    pub required_additions: Vec<String>,
    pub suggested_rewrites: Vec<SuggestedRewrite>,
}

#[derive(Debug, Clone)]
pub struct ComplianceViolation {
    pub rule_id: String,
    pub description: String,
    pub severity: Severity,
    pub text_span: (usize, usize),
}

#[derive(Debug, Clone, Copy)]
pub enum Severity {
    Warning,
    Error,
    Critical, // Must not proceed
}

/// Rule-based compliance checker
pub struct RuleBasedComplianceChecker {
    rules: ComplianceRules,
    forbidden_patterns: Vec<Regex>,
}

impl RuleBasedComplianceChecker {
    pub fn from_config(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let rules: ComplianceRules = toml::from_str(&content)?;

        let forbidden_patterns = rules.forbidden_phrases
            .iter()
            .map(|p| Regex::new(&format!(r"(?i)\b{}\b", regex::escape(p))))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self { rules, forbidden_patterns })
    }
}

#[async_trait]
impl ComplianceChecker for RuleBasedComplianceChecker {
    async fn check(&self, text: &str) -> Result<ComplianceResult> {
        let mut violations = Vec::new();
        let mut required_additions = Vec::new();

        // Check forbidden phrases
        for (i, pattern) in self.forbidden_patterns.iter().enumerate() {
            if let Some(m) = pattern.find(text) {
                violations.push(ComplianceViolation {
                    rule_id: format!("FORBIDDEN_{}", i),
                    description: format!(
                        "Forbidden phrase detected: '{}'",
                        &self.rules.forbidden_phrases[i]
                    ),
                    severity: Severity::Critical,
                    text_span: (m.start(), m.end()),
                });
            }
        }

        // Check claims requiring disclaimers
        for rule in &self.rules.claims_requiring_disclaimer {
            let pattern = Regex::new(&rule.pattern)?;
            if pattern.is_match(text) && !text.contains(&rule.disclaimer) {
                required_additions.push(rule.disclaimer.clone());
            }
        }

        // Check rate accuracy (example)
        if let Some(rate_match) = Regex::new(r"(\d+(?:\.\d+)?)\s*%")?.find(text) {
            let rate: f32 = rate_match.as_str()
                .trim_end_matches('%')
                .parse()
                .unwrap_or(0.0);

            if rate < self.rules.rate_rules.min_rate || rate > self.rules.rate_rules.max_rate {
                violations.push(ComplianceViolation {
                    rule_id: "RATE_ACCURACY".to_string(),
                    description: format!(
                        "Rate {}% outside valid range ({}-{}%)",
                        rate,
                        self.rules.rate_rules.min_rate,
                        self.rules.rate_rules.max_rate
                    ),
                    severity: Severity::Error,
                    text_span: (rate_match.start(), rate_match.end()),
                });
            }
        }

        Ok(ComplianceResult {
            is_compliant: violations.iter().all(|v| v.severity != Severity::Critical),
            violations,
            required_additions,
            suggested_rewrites: Vec::new(),
        })
    }

    async fn make_compliant(&self, text: &str) -> Result<String> {
        let result = self.check(text).await?;

        if result.is_compliant && result.required_additions.is_empty() {
            return Ok(text.to_string());
        }

        let mut compliant_text = text.to_string();

        // Remove critical violations
        for violation in result.violations.iter().filter(|v| v.severity == Severity::Critical) {
            // Replace with safe alternative or remove
            compliant_text.replace_range(
                violation.text_span.0..violation.text_span.1,
                "[content removed]",
            );
        }

        // Add required disclaimers
        for addition in &result.required_additions {
            compliant_text.push_str(" ");
            compliant_text.push_str(addition);
        }

        Ok(compliant_text)
    }
}
```

### Configuration

```toml
# domains/gold_loan/text_processing.toml

[input_pipeline]
enabled = true
order = ["grammar_correction", "translation"]

[input_pipeline.grammar_correction]
provider = "llm"              # "llm" | "nlprule" | "disabled"
enabled = true
model = "qwen2.5:7b-q4"
temperature = 0.1
max_tokens = 256

[input_pipeline.grammar_correction.domain_context]
vocabulary = [
    "gold loan", "Kotak", "Muthoot", "Manappuram", "IIFL",
    "LTV", "per gram", "interest rate", "processing fee",
    "balance transfer", "top-up", "foreclosure"
]
phrases = [
    "Kotak Bank se baat kar rahe hain",
    "gold loan balance transfer",
    "kam interest rate"
]

[input_pipeline.translation]
provider = "onnx"             # "onnx" | "grpc" | "disabled"
enabled = true
target_language = "en"
fallback_provider = "grpc"
grpc_endpoint = "http://localhost:50051"
onnx_model_path = "models/translation/indictrans2.onnx"

[output_pipeline]
enabled = true
order = ["translation", "compliance", "pii_redaction", "simplification"]

[output_pipeline.translation]
provider = "onnx"
source_language = "en"
target_language = "auto"      # Match customer's input language

[output_pipeline.compliance]
provider = "rule_based"       # "rule_based" | "llm" | "hybrid"
rules_file = "compliance.toml"
strict_mode = true            # Block on critical violations

[output_pipeline.pii_redaction]
provider = "hybrid"           # "rustbert" | "regex" | "hybrid"
enabled = true
entities = ["Aadhaar", "PAN", "PhoneNumber", "Email", "BankAccount"]
strategy = "partial_mask"
visible_chars = 4

[output_pipeline.simplification]
provider = "rule_based"
enabled = true
expand_abbreviations = true
normalize_numbers = true      # "5 lakh" → "five lakh"
max_sentence_length = 25      # Words
```

---

## RAG Strategy

### Agentic RAG Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           AGENTIC RAG WORKFLOW                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 1: INTENT CLASSIFICATION                                        │    │
│  │                                                                      │    │
│  │  Query: "Muthoot se kitna kam interest milega?"                      │    │
│  │                                                                      │    │
│  │  → Intent: COMPETITOR_COMPARISON                                     │    │
│  │  → Entities: [Muthoot, interest_rate]                               │    │
│  │  → Required docs: [competitors, rates, savings_calculator]          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                           │                                                  │
│                           ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 2: PARALLEL RETRIEVAL                                          │    │
│  │                                                                      │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │    │
│  │  │  Semantic   │  │    BM25     │  │   Stage-    │                  │    │
│  │  │   Search    │  │   Search    │  │   Aware     │                  │    │
│  │  │  (Qdrant)   │  │  (tantivy)  │  │   Filter    │                  │    │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                  │    │
│  │         │                │                │                          │    │
│  │         └────────────────┼────────────────┘                          │    │
│  │                          ▼                                           │    │
│  │                   ┌─────────────┐                                    │    │
│  │                   │   Fusion    │                                    │    │
│  │                   │  (RRF/RFF)  │                                    │    │
│  │                   └─────────────┘                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                           │                                                  │
│                           ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 3: SUFFICIENCY CHECK                                           │    │
│  │                                                                      │    │
│  │  LLM evaluates: "Do these docs answer the query?"                   │    │
│  │                                                                      │    │
│  │  IF insufficient:                                                    │    │
│  │    → Rewrite query with more specific terms                         │    │
│  │    → Go back to STEP 2 (max 3 iterations)                           │    │
│  │                                                                      │    │
│  │  IF sufficient:                                                      │    │
│  │    → Proceed to reranking                                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                           │                                                  │
│                           ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 4: RERANKING                                                    │    │
│  │                                                                      │    │
│  │  Cross-encoder reranks fused results                                │    │
│  │  Semantic similarity + relevance score                              │    │
│  │                                                                      │    │
│  │  Output: Top-K documents with scores                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                           │                                                  │
│                           ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 5: CONTEXT SIZING                                              │    │
│  │                                                                      │    │
│  │  Stage: PITCH → Generous context (2000 tokens)                      │    │
│  │  Stage: CLOSING → Minimal context (500 tokens)                      │    │
│  │                                                                      │    │
│  │  Compress if exceeds limit (summarize older context)                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### RAG Timing Strategies

Three configurable timing modes for experimentation:

```rust
// crates/rag/src/timing.rs

#[derive(Debug, Clone, Copy, Deserialize)]
pub enum RAGTimingMode {
    /// Retrieve before LLM call (simplest, adds latency)
    Sequential,

    /// Start retrieval on speech detection, ready before LLM
    PrefetchAsync,

    /// Retrieve in parallel with LLM, inject context mid-stream
    ParallelInject,
}

/// Sequential RAG timing
pub struct SequentialRAG {
    retriever: Arc<dyn Retriever>,
}

impl SequentialRAG {
    pub async fn process(&self, query: &str, context: &ConversationContext) -> Result<RAGResult> {
        // Simple: retrieve, then return
        let docs = self.retriever.retrieve(query, &RetrieveOptions::default()).await?;
        Ok(RAGResult { documents: docs, timing_ms: 0 })
    }
}

/// Prefetch RAG - starts on speech detection
pub struct PrefetchRAG {
    retriever: Arc<dyn Retriever>,
    prefetch_handle: Mutex<Option<JoinHandle<Result<Vec<Document>>>>>,
}

impl PrefetchRAG {
    /// Called when VAD detects user speaking
    pub fn start_prefetch(&self, partial_transcript: &str) {
        let retriever = self.retriever.clone();
        let query = partial_transcript.to_string();

        let handle = tokio::spawn(async move {
            retriever.retrieve(&query, &RetrieveOptions::default()).await
        });

        *self.prefetch_handle.lock().unwrap() = Some(handle);
    }

    /// Called when transcript is final
    pub async fn get_results(&self, final_query: &str) -> Result<Vec<Document>> {
        if let Some(handle) = self.prefetch_handle.lock().unwrap().take() {
            // Wait for prefetch to complete
            let docs = handle.await??;

            // If query changed significantly, re-retrieve
            if needs_reretrieval(&docs, final_query) {
                return self.retriever.retrieve(final_query, &RetrieveOptions::default()).await;
            }

            Ok(docs)
        } else {
            // No prefetch, do synchronous retrieval
            self.retriever.retrieve(final_query, &RetrieveOptions::default()).await
        }
    }
}

/// Parallel inject RAG - retrieves during LLM generation
pub struct ParallelInjectRAG {
    retriever: Arc<dyn Retriever>,
    llm: Arc<dyn LanguageModel>,
}

impl ParallelInjectRAG {
    pub async fn generate_with_dynamic_context(
        &self,
        query: &str,
        initial_context: &str,
    ) -> impl Stream<Item = Result<String>> {
        let retriever = self.retriever.clone();
        let llm = self.llm.clone();
        let query = query.to_string();

        stream! {
            // Start LLM with initial context
            let mut request = GenerateRequest {
                prompt: format!("{}\n\nQuery: {}", initial_context, query),
                stream: true,
                ..Default::default()
            };

            // Start retrieval in parallel
            let retrieval_future = retriever.retrieve(&query, &RetrieveOptions::default());
            let mut retrieval_done = false;
            let mut additional_context = None;

            pin_mut!(retrieval_future);

            let llm_stream = llm.generate_stream(request);
            pin_mut!(llm_stream);

            loop {
                tokio::select! {
                    // Check for LLM output
                    Some(chunk) = llm_stream.next() => {
                        yield chunk;
                    }

                    // Check for retrieval completion
                    docs = &mut retrieval_future, if !retrieval_done => {
                        retrieval_done = true;
                        if let Ok(docs) = docs {
                            additional_context = Some(format_docs(&docs));
                            // Note: In practice, we'd need to inject this context
                            // This is a simplified example
                        }
                    }

                    else => break,
                }
            }
        }
    }
}
```

### Stage-Aware Context Sizing

```rust
// crates/rag/src/context.rs

/// Context budget based on conversation stage
pub fn get_context_budget(state: &ConversationState) -> ContextBudget {
    match state {
        ConversationState::Greeting => ContextBudget {
            max_tokens: 200,
            doc_limit: 1,
            history_turns: 0,
        },

        ConversationState::Discovery => ContextBudget {
            max_tokens: 800,
            doc_limit: 3,
            history_turns: 2,
        },

        ConversationState::Pitch => ContextBudget {
            max_tokens: 2000,
            doc_limit: 5,
            history_turns: 4,
        },

        ConversationState::ObjectionHandling { .. } => ContextBudget {
            max_tokens: 1500,
            doc_limit: 4,
            history_turns: 3,
        },

        ConversationState::Comparison => ContextBudget {
            max_tokens: 1800,
            doc_limit: 5,
            history_turns: 2,
        },

        ConversationState::Closing => ContextBudget {
            max_tokens: 500,
            doc_limit: 2,
            history_turns: 5,
        },

        _ => ContextBudget::default(),
    }
}

#[derive(Debug, Clone)]
pub struct ContextBudget {
    pub max_tokens: usize,
    pub doc_limit: usize,
    pub history_turns: usize,
}
```

---

*[Document continues in next file due to length...]*

---

## References

### Voice Agent Architecture
- [Cresta - Engineering for Real-Time Voice Agent Latency](https://cresta.com/blog/engineering-for-real-time-voice-agent-latency)
- [Pipecat Framework](https://github.com/pipecat-ai/pipecat)
- [AssemblyAI - Voice AI Stack 2025](https://www.assemblyai.com/blog/the-voice-ai-stack-for-building-agents)

### Agentic RAG
- [NVIDIA - Traditional vs Agentic RAG](https://developer.nvidia.com/blog/traditional-rag-vs-agentic-rag-why-ai-agents-need-dynamic-knowledge-to-get-smarter/)
- [Weaviate - What is Agentic RAG](https://weaviate.io/blog/what-is-agentic-rag)

### Rust for AI
- [Red Hat - Why Agentic AI Developers Moving to Rust](https://developers.redhat.com/articles/2025/09/15/why-some-agentic-ai-developers-are-moving-code-python-rust)
- [ADK-Rust](https://github.com/zavora-ai/adk-rust)
- [Kalosm](https://lib.rs/crates/kalosm)
- [sherpa-rs](https://github.com/thewh1teagle/sherpa-rs)

### Indian Language Support
- [AI4Bharat IndicConformer](https://github.com/AI4Bharat/IndicConformerASR)
- [AI4Bharat IndicTrans2](https://github.com/AI4Bharat/IndicTrans2)
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)

### Context Management
- [Letta - Memory Blocks](https://www.letta.com/blog/memory-blocks)
- [JetBrains - Efficient Context Management](https://blog.jetbrains.com/research/2025/12/efficient-context-management/)

### Sales Psychology & Ethics
- [Regal AI - AI Agent Ethics & Disclosure Timing](https://www.regal.ai/blog/ai-agent-ethics-disclosure-timing)
- [Building Trust in AI-Powered Sales](https://spiky.ai/en/blog/ethical-ai-in-2025)

### Interrupt Handling
- [Gnani AI - Real-Time Barge-In](https://www.gnani.ai/resources/blogs/real-time-barge-in-ai-for-voice-conversations/)
- [Telnyx - How to Build Voice AI That Works](https://telnyx.com/resources/how-to-build-a-voice-ai-product-that-does-not-fall-apart-on-real-calls)
