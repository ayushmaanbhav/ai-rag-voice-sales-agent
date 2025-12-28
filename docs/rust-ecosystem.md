# Rust Ecosystem for Voice Agent

> Library decisions, alternatives, risks, and fallback strategies
>
> **Last Updated:** December 2025

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Why Rust?](#why-rust)
3. [Speech Processing](#speech-processing)
4. [Language Models](#language-models)
5. [Text Processing](#text-processing)
6. [Vector Search & RAG](#vector-search--rag)
7. [Async Runtime & Networking](#async-runtime--networking)
8. [Configuration & Serialization](#configuration--serialization)
9. [Observability](#observability)
10. [Risk Matrix](#risk-matrix)
11. [Fallback Strategies](#fallback-strategies)
12. [Cargo.toml Template](#cargotoml-template)

---

## Executive Summary

### Library Stack at a Glance

| Component | Primary Library | Fallback | Risk Level |
|-----------|-----------------|----------|------------|
| **Async Runtime** | tokio | - | Low |
| **Web Framework** | axum | - | Low |
| **STT** | sherpa-rs | Whisper ONNX | Medium |
| **TTS** | sherpa-rs | Piper ONNX | Medium |
| **Indian STT** | IndicConformer ONNX | Whisper | **High** |
| **Indian TTS** | IndicF5 ONNX | Piper | **High** |
| **Translation** | ort (IndicTrans2) | Python gRPC | **High** |
| **LLM (Local)** | kalosm | Ollama API | Medium |
| **LLM (API)** | reqwest | - | Low |
| **Grammar** | LLM-based | nlprule (EN only) | Medium |
| **NER/PII** | rust-bert | Regex | Medium |
| **Vector Store** | qdrant-client | - | Low |
| **BM25** | tantivy | - | Low |
| **Config** | config + serde | - | Low |
| **Tracing** | tracing + opentelemetry | - | Low |

### Risk Summary

```
HIGH RISK (requires validation):
├── IndicConformer ONNX export (never verified)
├── IndicF5 ONNX export (never verified)
└── IndicTrans2 ONNX export (fairseq complexity)

MEDIUM RISK (have fallbacks):
├── sherpa-rs stability for production load
├── Kalosm streaming reliability
└── rust-bert Indian language NER accuracy

LOW RISK (battle-tested):
├── tokio, axum, qdrant-client
├── serde, config
└── tracing, opentelemetry
```

---

## Why Rust?

### The Case FOR Rust

| Advantage | Explanation | Impact |
|-----------|-------------|--------|
| **No GIL** | True parallelism across cores | Pipeline stages run concurrently |
| **Memory Safety** | No segfaults, no data races | Production stability |
| **Performance** | Zero-cost abstractions | <800ms latency achievable |
| **Type Safety** | Compile-time guarantees | Fewer runtime errors |
| **Single Binary** | No dependency hell | Easy on-premise deployment |
| **Async Native** | First-class async/await | Efficient I/O handling |

### Research Evidence

> "When scaling to 500 agents on a 64-core machine, the Rust version would continue to scale, while the Python version would not."
> — [Red Hat Developer Blog](https://developers.redhat.com/articles/2025/09/15/why-some-agentic-ai-developers-are-moving-code-python-rust)

> "Rust beats Python for agentic AI with fearless concurrency, tiny latency, and portable WASM."
> — [Vision on Edge](https://visiononedge.com/rise-of-rust-in-agentic-ai-systems/)

### The Case AGAINST Rust

| Disadvantage | Mitigation |
|--------------|------------|
| **Ecosystem maturity** | Use ONNX for ML models, Python fallbacks |
| **Development speed** | Good abstractions, traits, macros |
| **Learning curve** | Documentation, pair programming |
| **Indian language support** | ONNX export + sherpa-rs |

### Decision Rationale

We choose Rust because:
1. **On-premise deployment** requires single-binary simplicity
2. **Sub-800ms latency** requires parallel processing
3. **Banking environment** requires memory safety
4. **Long-running processes** require no GIL bottleneck

We accept the risk of:
1. Unverified ONNX exports (with Python fallback)
2. Longer initial development time
3. Smaller talent pool

---

## Speech Processing

### Primary: sherpa-rs

**Repository:** https://github.com/thewh1teagle/sherpa-rs

**What it is:** Rust bindings to [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), a comprehensive speech toolkit.

**Features:**
- Speech-to-Text (streaming and offline)
- Text-to-Speech
- Voice Activity Detection
- Speaker Diarization
- Keyword Spotting

**Supported Platforms:**
- Windows, Linux, macOS
- Android, iOS
- WebAssembly (experimental)

**Installation:**
```toml
[dependencies]
sherpa-rs = { version = "0.6", features = ["tts", "download-binaries"] }
```

**Feature Flags:**
- `cuda` - CUDA GPU acceleration
- `directml` - DirectML (Windows)
- `tts` - Enable TTS support
- `download-binaries` - Auto-download sherpa-onnx libs
- `static` - Static linking

**Example Usage:**
```rust
use sherpa_rs::{
    online_recognizer::{OnlineRecognizer, OnlineRecognizerConfig},
    offline_tts::{OfflineTts, OfflineTtsConfig},
};

// STT
let config = OnlineRecognizerConfig {
    model_path: "models/stt/encoder.onnx".into(),
    tokens_path: "models/stt/tokens.txt".into(),
    sample_rate: 16000,
    ..Default::default()
};
let recognizer = OnlineRecognizer::new(config)?;

// TTS
let tts_config = OfflineTtsConfig {
    model_path: "models/tts/model.onnx".into(),
    lexicon_path: "models/tts/lexicon.txt".into(),
    ..Default::default()
};
let tts = OfflineTts::new(tts_config)?;
```

**Risk Assessment:**
| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Incomplete bindings | Low | Medium | Contribute upstream |
| Build issues | Medium | Low | Use prebuilt binaries |
| Performance issues | Low | High | Profile early |

### Indian Language Models

#### IndicConformer (STT)

**Source:** https://github.com/AI4Bharat/IndicConformerASR

**Challenge:** Native format is NeMo (.nemo), not ONNX.

**Export Path:**
```python
# Python script to export IndicConformer to ONNX
import nemo.collections.asr as nemo_asr

# Load model
model = nemo_asr.models.ASRModel.from_pretrained(
    "ai4bharat/indicconformer_stt_hi_hybrid_rnnt_large"
)

# Export to ONNX
model.export("indicconformer_hi.onnx")
```

**Sherpa-ONNX Compatibility:**
See [sherpa-onnx NeMo export guide](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/nemo/how-to-export.html)

**Validation Steps:**
1. Export model using NeMo
2. Test with sherpa-onnx Python bindings
3. Test with sherpa-rs Rust bindings
4. Benchmark accuracy vs original

**Fallback:** Whisper (less accurate for Indian languages)

#### IndicF5 / Indic-TTS (TTS)

**Source:** https://github.com/AI4Bharat/Indic-TTS

**Challenge:** Uses FastPitch + HiFi-GAN, requires custom ONNX export.

**Export Path:**
```python
# Export FastPitch
import torch
from nemo.collections.tts.models import FastPitchModel

model = FastPitchModel.from_pretrained("ai4bharat/indic-tts-hi-female")
model.export("fastpitch_hi.onnx")

# Export HiFi-GAN
from nemo.collections.tts.models import HifiGanModel
vocoder = HifiGanModel.from_pretrained("ai4bharat/hifigan-hi")
vocoder.export("hifigan_hi.onnx")
```

**Fallback:** Piper TTS (less natural for Indian languages)

---

## Language Models

### Primary: Kalosm

**Repository:** https://lib.rs/crates/kalosm

**What it is:** Pure Rust framework for local AI inference (LLM, audio, vision).

**Why Kalosm:**
- Pure Rust (uses Candle for inference)
- Streaming support
- Embedding support
- Structured generation

**Installation:**
```toml
[dependencies]
kalosm = { version = "0.4", features = ["language"] }
# For GPU: features = ["language", "cuda"] or "metal"
```

**Example Usage:**
```rust
use kalosm::language::*;

// Load model
let model = Llama::builder()
    .with_source(LlamaSource::qwen_2_5_7b_instruct_q4())
    .build()
    .await?;

// Generate with streaming
let mut stream = model.chat("Hello, how can I help?").await?;
while let Some(token) = stream.next().await {
    print!("{}", token);
}
```

**Supported Models:**
- Llama family (Llama 3, Qwen)
- Mistral family
- Phi family
- Custom GGUF models

**Risk Assessment:**
| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Model compatibility | Medium | Medium | Test target models early |
| Streaming issues | Low | High | Have Ollama fallback |
| GPU support gaps | Medium | Medium | CPU fallback |

### Fallback: Ollama API

**What:** External LLM server with HTTP API

**Why Fallback:**
- Battle-tested
- Easy model management
- Wide model support

**Installation:**
```toml
[dependencies]
ollama-rs = "0.1"  # Or use reqwest directly
```

**Example:**
```rust
use ollama_rs::Ollama;

let ollama = Ollama::default();
let response = ollama
    .generate(GenerationRequest::new("qwen2.5:7b", prompt))
    .await?;
```

### Cloud API: Claude/OpenAI

**Use case:** Fallback for complex reasoning, or when local resources insufficient.

```toml
[dependencies]
reqwest = { version = "0.11", features = ["json"] }
```

---

## Text Processing

### Grammar Correction: LLM-Based

**Rationale:** No mature Rust library for Indian language grammar.

**Approach:** Use small LLM (Qwen 2.5 7B) with domain-specific prompt.

**Alternative for English:** nlprule

```toml
[dependencies]
nlprule = "0.6"  # English, German, Spanish only
```

### Translation: IndicTrans2 via ONNX

**Challenge:** IndicTrans2 is PyTorch/fairseq, no official ONNX.

**Export Complexity:**
- Fairseq transformers have complex ONNX export
- Encoder-decoder architecture needs special handling
- Tokenizer must be ported separately

**ONNX Runtime in Rust:**

```toml
[dependencies]
ort = { version = "2", features = ["cuda"] }  # ONNX Runtime
```

**Tokenizer Option:**
```toml
[dependencies]
tokenizers = "0.15"  # HuggingFace tokenizers in Rust
```

**Fallback: Python gRPC Sidecar**

If ONNX export fails, run IndicTrans2 as a Python service:

```python
# translation_server.py
import grpc
from indicTrans import translate

class TranslationService(translation_pb2_grpc.TranslatorServicer):
    def Translate(self, request, context):
        result = translate(request.text, request.src_lang, request.tgt_lang)
        return translation_pb2.TranslationResponse(text=result)
```

```rust
// Rust client
use tonic::transport::Channel;

let mut client = TranslatorClient::connect("http://localhost:50051").await?;
let response = client.translate(TranslateRequest {
    text: "Hello".into(),
    src_lang: "en".into(),
    tgt_lang: "hi".into(),
}).await?;
```

### NER/PII: rust-bert

**Repository:** https://github.com/guillaume-be/rust-bert

**What it is:** Rust-native NLP pipelines using ONNX models.

**Installation:**
```toml
[dependencies]
rust-bert = "0.22"
```

**Features:**
- Named Entity Recognition
- Sentiment Analysis
- Question Answering
- Text Classification

**Example:**
```rust
use rust_bert::pipelines::ner::NERModel;

let model = NERModel::new(Default::default())?;
let entities = model.predict(&["Raj's Aadhaar is 1234 5678 9012"])?;
// [Entity { word: "Raj", label: "PER" }, ...]
```

**Limitation:** Best accuracy for English. Indian language NER may need fine-tuned models.

**Fallback:** Regex patterns for structured PII (Aadhaar, PAN, phone).

---

## Vector Search & RAG

### Vector Store: Qdrant

**Repository:** https://github.com/qdrant/rust-client

**Why Qdrant:**
- Rust-native server
- Excellent Rust client
- Fast, production-ready
- Filtering support

**Installation:**
```toml
[dependencies]
qdrant-client = "1"
```

**Example:**
```rust
use qdrant_client::prelude::*;
use qdrant_client::qdrant::vectors_config::Config;

let client = QdrantClient::from_url("http://localhost:6333").build()?;

// Create collection
client.create_collection(&CreateCollection {
    collection_name: "knowledge".into(),
    vectors_config: Some(VectorsConfig {
        config: Some(Config::Params(VectorParams {
            size: 384,
            distance: Distance::Cosine.into(),
            ..Default::default()
        })),
    }),
    ..Default::default()
}).await?;

// Search
let results = client.search_points(&SearchPoints {
    collection_name: "knowledge".into(),
    vector: query_embedding,
    limit: 5,
    ..Default::default()
}).await?;
```

### BM25: Tantivy

**Repository:** https://github.com/quickwit-oss/tantivy

**Why Tantivy:**
- Pure Rust
- Fast full-text search
- BM25 ranking
- Production-ready

**Installation:**
```toml
[dependencies]
tantivy = "0.22"
```

**Example:**
```rust
use tantivy::{Index, schema::*, collector::TopDocs, query::QueryParser};

// Create schema
let mut schema_builder = Schema::builder();
let content = schema_builder.add_text_field("content", TEXT | STORED);
let schema = schema_builder.build();

// Create index
let index = Index::create_in_ram(schema.clone());

// Search
let reader = index.reader()?;
let searcher = reader.searcher();
let query_parser = QueryParser::for_index(&index, vec![content]);
let query = query_parser.parse_query("gold loan")?;
let top_docs = searcher.search(&query, &TopDocs::with_limit(10))?;
```

### Embeddings

**Option 1: Kalosm Embeddings**
```rust
use kalosm::language::*;

let embedder = Bert::builder()
    .with_source(BertSource::e5_multilingual_base())
    .build()
    .await?;

let embedding = embedder.embed("Gold loan query").await?;
```

**Option 2: ort (ONNX Runtime)**
```rust
use ort::Session;

let session = Session::builder()?
    .with_model("models/embeddings/e5-multilingual.onnx")?
    .build()?;

let embeddings = session.run(inputs)?;
```

---

## Async Runtime & Networking

### Tokio

**Why:** De facto standard for async Rust.

```toml
[dependencies]
tokio = { version = "1", features = ["full"] }
```

**Key Features Used:**
- `mpsc` channels - Pipeline communication
- `spawn` - Concurrent task execution
- `select!` - Multiple async operations
- `JoinSet` - Parallel tool execution

### Axum

**Why:** Fast, ergonomic, tower-based web framework.

```toml
[dependencies]
axum = { version = "0.7", features = ["ws"] }
tower = "0.4"
tower-http = { version = "0.5", features = ["cors", "trace"] }
```

**Example:**
```rust
use axum::{Router, routing::get, extract::ws::WebSocket};

async fn websocket_handler(ws: WebSocket) {
    // Handle voice conversation
}

let app = Router::new()
    .route("/ws/conversation/:id", get(websocket_handler))
    .layer(TraceLayer::new_for_http());

axum::serve(listener, app).await?;
```

### Streaming

```toml
[dependencies]
tokio-stream = "0.1"
async-stream = "0.3"
futures = "0.3"
```

**Stream Creation:**
```rust
use async_stream::stream;

fn process_stream() -> impl Stream<Item = Result<String, Error>> {
    stream! {
        yield Ok("First".to_string());
        yield Ok("Second".to_string());
    }
}
```

---

## Configuration & Serialization

### Config Loading

```toml
[dependencies]
config = "0.14"
```

**Example:**
```rust
use config::{Config, File, Environment};

let settings = Config::builder()
    .add_source(File::with_name("config/default"))
    .add_source(File::with_name(&format!("config/{}", env)).required(false))
    .add_source(Environment::with_prefix("VOICE_AGENT"))
    .build()?;

let stt_provider: String = settings.get("stt.provider")?;
```

### Serialization

```toml
[dependencies]
serde = { version = "1", features = ["derive"] }
serde_json = "1"
toml = "0.8"
serde_yaml = "0.9"
```

---

## Observability

### Tracing

```toml
[dependencies]
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
tracing-opentelemetry = "0.22"
opentelemetry = "0.21"
opentelemetry-otlp = "0.14"
```

**Setup:**
```rust
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

tracing_subscriber::registry()
    .with(tracing_subscriber::fmt::layer())
    .with(tracing_opentelemetry::layer())
    .init();
```

**Usage:**
```rust
#[tracing::instrument(skip(audio))]
async fn transcribe(&self, audio: &AudioFrame) -> Result<TranscriptFrame> {
    let span = tracing::info_span!("stt_transcribe", language = ?self.language);
    let _guard = span.enter();

    let start = Instant::now();
    let result = self.model.transcribe(audio).await?;

    tracing::info!(
        duration_ms = start.elapsed().as_millis(),
        confidence = result.confidence,
        "Transcription complete"
    );

    Ok(result)
}
```

### Metrics

```toml
[dependencies]
metrics = "0.22"
metrics-exporter-prometheus = "0.13"
```

**Setup:**
```rust
use metrics_exporter_prometheus::PrometheusBuilder;

PrometheusBuilder::new()
    .with_http_listener(([0, 0, 0, 0], 9090))
    .install()?;
```

**Recording:**
```rust
use metrics::{counter, histogram, gauge};

counter!("conversations_started").increment(1);
histogram!("stt_latency_ms").record(latency_ms as f64);
gauge!("active_conversations").set(count as f64);
```

---

## Risk Matrix

| Component | Risk | Likelihood | Impact | Mitigation |
|-----------|------|------------|--------|------------|
| **IndicConformer ONNX** | Export fails | Medium | High | Whisper fallback |
| **IndicF5 ONNX** | Export fails | Medium | High | Piper fallback |
| **IndicTrans2 ONNX** | Fairseq complexity | High | High | Python gRPC |
| **sherpa-rs** | Incomplete bindings | Low | Medium | Contribute fixes |
| **Kalosm** | Model compatibility | Medium | Medium | Ollama fallback |
| **rust-bert** | Indian NER accuracy | Medium | Low | Regex + LLM |
| **Performance** | Latency > 800ms | Low | High | Profile early |
| **Build** | Dependency conflicts | Low | Low | Lock versions |

---

## Fallback Strategies

### Speech Fallback Chain

```
IndicConformer ONNX
    ↓ (if fails)
Whisper via sherpa-onnx
    ↓ (if fails)
Whisper via Python gRPC
    ↓ (if fails)
External STT API (Sarvam/Deepgram)
```

### Translation Fallback Chain

```
IndicTrans2 ONNX (Rust-native)
    ↓ (if ONNX export fails)
IndicTrans2 via Python gRPC sidecar
    ↓ (if sidecar unavailable)
Google Translate API (degraded accuracy)
```

### LLM Fallback Chain

```
Kalosm (local, Rust-native)
    ↓ (if model issues)
Ollama API (local server)
    ↓ (if local unavailable)
Claude API (cloud)
    ↓ (if API unavailable)
OpenAI API (cloud)
```

### Fallback Implementation Pattern

```rust
pub struct FallbackChain<T> {
    providers: Vec<Box<dyn Provider<Output = T>>>,
}

impl<T> FallbackChain<T> {
    pub async fn execute(&self, input: &Input) -> Result<T, Error> {
        let mut last_error = None;

        for (i, provider) in self.providers.iter().enumerate() {
            match provider.process(input).await {
                Ok(result) => {
                    if i > 0 {
                        tracing::warn!(
                            provider_index = i,
                            "Used fallback provider"
                        );
                        metrics::counter!("fallback_used").increment(1);
                    }
                    return Ok(result);
                }
                Err(e) => {
                    tracing::warn!(
                        provider_index = i,
                        error = %e,
                        "Provider failed, trying next"
                    );
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or(Error::NoProvidersAvailable))
    }
}
```

---

## Cargo.toml Template

```toml
[workspace]
members = [
    "crates/core",
    "crates/config",
    "crates/pipeline",
    "crates/speech",
    "crates/text_processing",
    "crates/rag",
    "crates/agent",
    "crates/personalization",
    "crates/llm",
    "crates/experiments",
    "crates/server",
]
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2021"
rust-version = "1.75"
license = "Proprietary"

[workspace.dependencies]
# Async
tokio = { version = "1", features = ["full"] }
tokio-stream = "0.1"
async-stream = "0.3"
futures = "0.3"
async-trait = "0.1"

# Web
axum = { version = "0.7", features = ["ws"] }
tower = "0.4"
tower-http = { version = "0.5", features = ["cors", "trace"] }
hyper = { version = "1", features = ["full"] }

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"
toml = "0.8"
serde_yaml = "0.9"

# Config
config = "0.14"

# Speech
sherpa-rs = { version = "0.6", features = ["tts", "download-binaries"] }

# ML/AI
kalosm = { version = "0.4", features = ["language"] }
ort = { version = "2", features = ["cuda"] }
rust-bert = "0.22"
tokenizers = "0.15"

# Vector/Search
qdrant-client = "1"
tantivy = "0.22"

# Observability
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
tracing-opentelemetry = "0.22"
opentelemetry = "0.21"
opentelemetry-otlp = "0.14"
metrics = "0.22"
metrics-exporter-prometheus = "0.13"

# Error handling
thiserror = "1"
anyhow = "1"

# Utilities
regex = "1"
chrono = { version = "0.4", features = ["serde"] }
uuid = { version = "1", features = ["v4"] }

# gRPC (for fallbacks)
tonic = "0.10"
prost = "0.12"

[profile.release]
lto = "thin"
codegen-units = 1
opt-level = 3
```

---

## Next Steps

1. **Validate High-Risk Items:**
   - [ ] Export IndicConformer to ONNX
   - [ ] Export IndicF5 to ONNX
   - [ ] Test sherpa-rs with exported models
   - [ ] Export IndicTrans2 to ONNX (or setup gRPC)

2. **Prototype Pipeline:**
   - [ ] Basic STT → LLM → TTS flow
   - [ ] Measure latency
   - [ ] Test streaming

3. **Setup Fallbacks:**
   - [ ] Whisper model via sherpa-rs
   - [ ] Piper TTS via sherpa-rs
   - [ ] Python gRPC sidecar for translation

---

## References

- [sherpa-rs GitHub](https://github.com/thewh1teagle/sherpa-rs)
- [sherpa-onnx Documentation](https://k2-fsa.github.io/sherpa/onnx/index.html)
- [Kalosm Documentation](https://docs.rs/kalosm/latest/kalosm/)
- [ort (ONNX Runtime) Documentation](https://docs.rs/ort/latest/ort/)
- [rust-bert Documentation](https://docs.rs/rust-bert/latest/rust_bert/)
- [Qdrant Rust Client](https://docs.rs/qdrant-client/latest/qdrant_client/)
- [Tantivy Documentation](https://docs.rs/tantivy/latest/tantivy/)
- [AI4Bharat IndicConformer](https://github.com/AI4Bharat/IndicConformerASR)
- [AI4Bharat IndicTrans2](https://github.com/AI4Bharat/IndicTrans2)
