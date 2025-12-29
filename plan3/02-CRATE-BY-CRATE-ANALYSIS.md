# Plan 3: Crate-by-Crate Deep Dive Analysis

---

## 1. Core Crate (`crates/core`)

**Status:** PRODUCTION READY (95%)
**Grade:** A-

### What's Implemented

| Module | Status | Notes |
|--------|--------|-------|
| `traits/speech.rs` | ✅ Complete | SpeechToText, TextToSpeech with streaming |
| `traits/llm.rs` | ✅ Complete | LanguageModel with tool support |
| `traits/retriever.rs` | ✅ Complete | Agentic retrieval, metadata filtering |
| `traits/text_processing.rs` | ✅ Complete | Grammar, Translation, PII, Compliance |
| `traits/pipeline.rs` | ✅ Complete | 16 Frame variants, FrameProcessor |
| `language.rs` | ✅ Complete | 23 languages (22 Indian + English) |
| `pii.rs` | ✅ Complete | 18 PII types, India-specific |
| `compliance.rs` | ✅ Complete | Violations, categories, rewrites |
| `personalization/` | ✅ Complete | Persona, Signals, Adaptation |

### Strengths
1. **22 Languages:** All Eighth Schedule languages with correct ISO codes
2. **RTL Support:** Urdu, Kashmiri, Sindhi properly marked RTL
3. **Sentence Terminators:** Indic danda (।, ॥) properly configured
4. **Personalization Engine:** 6 customer segments with tailored personas
5. **Signal Detection:** 12 behavior signals with Hindi pattern support
6. **Objection Handling:** 10 objection types with segment-specific responses

### Issues (Minor)
1. **DateOfBirth missing from PIISeverity match** (line 93-107)
2. **Objection pattern precedence unclear** (adaptation.rs:116-122)
3. **RedactionStrategy::Hash uses text length** (not cryptographic)

### Recommendations
```rust
// Fix 1: Add DateOfBirth
Self::DateOfBirth => PIISeverity::Medium,

// Fix 2: Clarify precedence
if (lower.contains("safe") && lower.contains("gold"))
    || (lower.contains("security") && lower.contains("gold"))
    || lower.contains("suraksha")
```

---

## 2. Pipeline Crate (`crates/pipeline`)

**Status:** MAJOR REWORK NEEDED (45%)
**Grade:** B-

### What's Implemented

| Module | Status | Notes |
|--------|--------|-------|
| `vad/silero.rs` | ⚠️ Race condition | LSTM-based, but thread-unsafe |
| `vad/magicnet.rs` | ✅ Working | GRU-based, state machine correct |
| `stt/mod.rs` | ⚠️ Not streaming | Batched mel computation |
| `tts/mod.rs` | ✅ Working | Word-level chunking |
| `tts/chunker.rs` | ✅ Working | P2 fix applied (O(n)) |
| `turn_detection/hybrid.rs` | ✅ Working | VAD + semantic |
| `processors/sentence_detector.rs` | ✅ Excellent | Indic support |
| `processors/interrupt_handler.rs` | ✅ Working | 3 modes |
| `orchestrator.rs` | ⚠️ Monolithic | Not frame-based |

### Critical Issues

#### 1. Silero VAD Race Condition
```rust
// Lines 163-189: Lock released mid-processing
drop(state);  // ← RACE WINDOW
let speech_prob = self.compute_probability(&chunk)?;
let mut state = self.mutable.lock();  // ← Re-acquire
```

#### 2. STT Not True Streaming
```rust
// Recomputes FULL mel spectrogram each chunk
// Manual DFT O(n²) instead of FFT O(n log n)
```

#### 3. Architecture Mismatch
- Documented: Frame-based with FrameProcessor chain
- Actual: Monolithic orchestrator with sequential processing
- Only 30% aligned with ARCHITECTURE_v2.md

### Strengths
1. **Sentence Detector:** Excellent Indic terminator support
2. **Turn Detection:** Hybrid VAD + semantic works well
3. **MagicNet VAD:** P0 fix applied (consolidated mutex)
4. **TTS Chunker:** P2 fix applied (boundary-first approach)

### Recommendations
1. Fix Silero race condition (keep lock during compute)
2. Implement sliding-window FFT for true streaming
3. Refactor to frame-based architecture (long-term)
4. Wire sentence detector to TTS output

---

## 3. Text Processing Crate (`crates/text_processing`)

**Status:** MOSTLY COMPLETE (72%)
**Grade:** C+

### What's Implemented

| Module | Status | Notes |
|--------|--------|-------|
| `grammar/llm_corrector.rs` | ✅ Working | Domain-aware |
| `grammar/noop.rs` | ✅ Working | Pass-through |
| `translation/mod.rs` | ❌ STUBBED | No ONNX implementation |
| `translation/detect.rs` | ✅ Excellent | 11 scripts, code-switching |
| `pii/patterns.rs` | ✅ Working | 12 patterns, India-specific |
| `pii/detector.rs` | ✅ Working | Hybrid (regex, NER stub) |
| `compliance/checker.rs` | ✅ Working | Rule-based |
| `compliance/rules.rs` | ✅ Working | TOML-configurable |
| `pipeline.rs` | ✅ Working | Orchestration layer |

### Critical Issues

#### 1. Translation Completely Stubbed
```rust
TranslationProvider::Grpc => {
    tracing::warn!("gRPC translator not yet implemented");
    Box::new(NoopTranslator::new())  // ← DOES NOTHING
}
// NO IndicTrans2 ONNX implementation
```

#### 2. NER Detection Stubbed
```rust
// pii/detector.rs:51-55
async fn detect_ner(&self, _text: &str) -> Vec<PIIEntity> {
    Vec::new()  // ← TODO: Implement NER
}
```

### Strengths
1. **Script Detection:** All 11 Indic scripts with correct Unicode ranges
2. **PII Patterns:** Aadhaar (first digit validation), PAN (holder type), IFSC
3. **Compliance:** 11 forbidden phrases, rate validation (7-24%)
4. **Pipeline:** Translate-Think-Translate pattern ready (once translation works)

### Pattern Quality

| Pattern | Regex | Confidence | Issues |
|---------|-------|------------|--------|
| Aadhaar | `[2-9]\d{3}\s?\d{4}\s?\d{4}` | 0.95 | No Verhoeff checksum |
| PAN | `[A-Z]{3}[ABCFGHLJPT][A-Z][0-9]{4}[A-Z]` | 0.98 | Excellent |
| BankAccount | `\d{9,18}` | 0.60 | Too loose |
| UPI ID | `[a-zA-Z0-9._-]+@[a-zA-Z]+` | 0.70 | Matches emails |

### Recommendations
1. **Priority:** Implement IndicTrans2 ONNX (40 hours)
2. Add Verhoeff checksum for Aadhaar
3. Tighten BankAccount pattern
4. Implement rust-bert NER for names/addresses

---

## 4. LLM Crate (`crates/llm`)

**Status:** FUNCTIONAL (70%)
**Grade:** B

### What's Implemented

| Module | Status | Notes |
|--------|--------|-------|
| `backend.rs` | ✅ Working | Ollama integration |
| `speculative.rs` | ⚠️ Latency issues | SlmFirst, RaceParallel |
| `prompt.rs` | ✅ Working | Persona, context limiting |
| `streaming.rs` | ✅ Working | Token streaming |

### Critical Issues

#### 1. SlmFirst Latency Cliffs
```rust
// 200ms timeout + LLM fallback = 505ms+
let slm_timeout = Duration::from_millis(200);
```

#### 2. Tool Calling Skeleton Only
```rust
// Role::Tool added but no actual tool binding
impl OllamaBackend {
    // generate_with_tools() not properly wired
}
```

### Strengths
1. **KV Cache:** Session context properly maintained
2. **Streaming:** Token-by-token delivery works
3. **Welford Algorithm:** Numerically stable running stats
4. **Retry Logic:** Exponential backoff implemented

### Recommendations
1. Reduce SlmFirst timeout to 100ms
2. OR implement parallel execution (select! on both)
3. Complete tool calling for Ollama

---

## 5. RAG Crate (`crates/rag`)

**Status:** SOLID FOUNDATION (60%)
**Grade:** B

### What's Implemented

| Module | Status | Notes |
|--------|--------|-------|
| `retriever.rs` | ✅ Working | Hybrid (dense + sparse + RRF) |
| `agentic.rs` | ✅ Working | 3-step iterative loop |
| `reranker.rs` | ✅ Working | Cascaded, not layer-level |
| `sparse_search.rs` | ✅ Working | Tantivy BM25 |
| `vector_store.rs` | ✅ Working | Qdrant placeholder |
| `embeddings.rs` | ⚠️ Config needed | Output name varies by model |
| `query_expansion.rs` | ❌ Not wired | Implemented but unused |
| `domain_boost.rs` | ❌ Not wired | Implemented but unused |
| `cross_lingual.rs` | ❌ Not wired | Implemented but unused |

### Critical Issues

#### 1. Early-Exit Reranker Mislabeled
```rust
/// # P0 FIX: Early-Exit Limitation with ONNX Runtime
/// Layer-by-layer early exit is **NOT currently functional**
/// ONNX models don't expose intermediate layer outputs
```
- Actually: Cascaded reranking (pre-filter + full model)
- Works as 2-stage pipeline, just not layer-level

#### 2. Query Expansion Not Integrated
```rust
// query_expansion.rs exists with full implementation
// BUT never called during retrieval!
```

#### 3. Prefetch Not Wired to VAD
```rust
pub async fn prefetch(&self, hint: &str) -> Result<()> {
    // Method exists but never called from VAD pipeline
}
```

### Strengths
1. **RRF Fusion:** Correct implementation `1/(k + rank + 1)`
2. **Agentic Loop:** Sufficiency checking, query rewriting
3. **SimpleScorer:** TF-IDF-like with position weighting
4. **Stopwords:** English and Hindi supported

### Recommendations
1. Wire query expansion to retrieval pipeline
2. Call `prefetch()` from VAD on speech detection
3. Implement stage-aware context budgeting

---

## 6. Agent Crate (`crates/agent`)

**Status:** NEEDS SIGNIFICANT WORK (45%)
**Grade:** C

### What's Implemented

| Module | Status | Notes |
|--------|--------|-------|
| `conversation.rs` | ⚠️ Incomplete | 4/11 intent transitions |
| `memory.rs` | ✅ Working | Hierarchical (Working/Episodic/Semantic) |
| `intent.rs` | ⚠️ ASCII-only | Hindi slot extraction broken |
| `stage.rs` | ✅ Working | 7 stages, valid_transitions() |
| `voice_session.rs` | ✅ Working | STT/TTS/VAD integration |
| `agent.rs` | ⚠️ Fire-and-forget | Memory summarization |

### Critical Issues

#### 1. Hindi Slot Extraction
```rust
// Only English patterns defined
("loan_amount", vec![
    SlotPattern::regex(r"(\d+)...", 0.95),
    // NO Devanagari patterns
]),
```

#### 2. Missing Intent Transitions
```rust
match intent.intent.as_str() {
    "farewell" | "objection" | "schedule_visit" | "affirmative" => ...,
    _ => None,  // negative, interest_rate, eligibility_check, loan_inquiry, complaint IGNORED
}
```

#### 3. Memory Fire-and-Forget
```rust
tokio::spawn(async move {
    if let Err(e) = memory.summarize_pending_async().await {
        tracing::debug!(...);  // Error just logged!
    }
});
```

### Strengths
1. **Memory:** Episodic summarization, VecDeque for O(1) removal
2. **Intent:** 11 gold-loan intents defined
3. **Voice Session:** Barge-in detection, silence timeout

### Recommendations
1. Add Devanagari slot patterns (priority)
2. Add all intent transitions
3. Add watermark-based memory management
4. Add personalization (CustomerSegment not used)

---

## 7. Tools Crate (`crates/tools`)

**Status:** MOSTLY COMPLETE (85%)
**Grade:** B+

### What's Implemented

| Module | Status | Notes |
|--------|--------|-------|
| `mcp.rs` | ✅ Complete | MCP protocol, validation |
| `gold_loan.rs` | ⚠️ Stubs | SMS false positive |
| `registry.rs` | ✅ Working | Discovery, timeouts |
| `integrations.rs` | ✅ Stubs | CRM, Calendar stubs |

### Critical Issues

#### 1. SMS Confirmation False Positive
```rust
"confirmation_sent": true,  // But NO SMS actually sent!
```

#### 2. Tiered Rates Not Used
```rust
"interest_rate_percent": self.config.kotak_interest_rate,  // Always 10.5%
// Config has tiers but not applied
```

### Strengths
1. **MCP Validation:** Type checking, enum values, numeric ranges
2. **Audio Content Block:** P2 fix added audio support
3. **Branch Locator:** Haversine distance, 1600+ branches

### Recommendations
1. Fix SMS to return `confirmation_pending: true`
2. Apply tiered rates in eligibility calculation
3. Complete CRM/Calendar integrations

---

## 8. Config Crate (`crates/config`)

**Status:** PRODUCTION READY (95%)
**Grade:** A

### What's Implemented

| Module | Status | Notes |
|--------|--------|-------|
| `settings.rs` | ✅ Complete | Hot-reload, validation |
| `domain.rs` | ✅ Complete | YAML/JSON loading |
| `branch.rs` | ✅ Complete | 1600+ branches |
| `product.rs` | ✅ Complete | 4 variants |
| `competitor.rs` | ✅ Complete | 6 competitors |
| `prompts.rs` | ✅ Complete | Stage-specific |
| `agent.rs` | ✅ Complete | Consolidated config |

### Strengths
1. **Hot Reload:** DomainConfigManager with `reload()`
2. **Competitor Modeling:** Rates, weaknesses, market share
3. **Product Variants:** Shakti Gold (women), Bullet, Overdraft
4. **Savings Calculation:** MonthlySavings struct

### Issues (Minor)
1. **Gold Price Hardcoded:** `default_gold_price() = 7500.0`
2. **Auth Disabled by Default**
3. **Model paths warn-only** (should error for required models)

### Recommendations
1. Add gold price API integration
2. Add startup warning for auth disabled
3. Make STT/TTS/VAD model paths required

---

## 9. Server Crate (`crates/server`)

**Status:** MOSTLY COMPLETE (85%)
**Grade:** B+

### What's Implemented

| Module | Status | Notes |
|--------|--------|-------|
| `http.rs` | ✅ Complete | P0 CORS fix applied |
| `websocket.rs` | ✅ Complete | Audio/text duplex |
| `session.rs` | ⚠️ Redis stubbed | In-memory only |
| `auth.rs` | ✅ Complete | Timing-safe |
| `metrics.rs` | ✅ Complete | Prometheus |

### Critical Issues

#### 1. Redis Session Store Stubbed
```rust
pub struct RedisSessionStore {
    // TODO: Add Redis connection pool
}
// All methods return stubs
```

### Strengths
1. **CORS:** Dynamic from config, P0 fix applied
2. **Auth:** Constant-time comparison
3. **Metrics:** STT/LLM/TTS latency histograms
4. **WebSocket:** Rate limiting, pipeline integration

### Recommendations
1. Implement Redis session store (priority)
2. Add session affinity for distributed scaling
3. Make CORS require explicit config in production

---

## 10. Transport Crate (`crates/transport`)

**Status:** SELECTIVE IMPLEMENTATION (65%)
**Grade:** B-

### What's Implemented

| Module | Status | Notes |
|--------|--------|-------|
| `traits.rs` | ✅ Complete | Transport, AudioSink, AudioSource |
| `webrtc.rs` | ⚠️ ICE incomplete | Opus, SDP working |
| `websocket.rs` | ⚠️ Stub | Delegated to server |
| `codec.rs` | ✅ Production | Opus encoder/decoder |
| `session.rs` | ✅ Working | Failover, reconnection |

### Critical Issues

#### 1. ICE Handling Incomplete
```rust
// TODO: Add timeout and proper ICE candidate handling
```

#### 2. WebRTC Audio Decoding
```rust
// Decoding RTP as raw i16 instead of Opus
let samples: Vec<f32> = payload.chunks(2).map(|chunk| {
    i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0
});
// Should use OpusDecoder from codec.rs
```

### Strengths
1. **Opus Codec:** Packet loss concealment, resampling
2. **Transport Traits:** Clean abstraction
3. **Failover:** WebRTC → WebSocket fallback

### Recommendations
1. Implement trickle ICE with timeout
2. Use OpusDecoder for RTP audio
3. Fix audio source cloning (use Arc)

---

## Summary Table

| Crate | Files | LOC | Status | Priority |
|-------|-------|-----|--------|----------|
| core | 15 | ~3500 | 95% | Low |
| pipeline | 18 | ~4500 | 45% | CRITICAL |
| text_processing | 12 | ~2800 | 72% | HIGH |
| llm | 5 | ~1500 | 70% | HIGH |
| rag | 10 | ~2800 | 60% | MEDIUM |
| agent | 6 | ~2200 | 45% | CRITICAL |
| tools | 5 | ~1800 | 85% | MEDIUM |
| config | 8 | ~3500 | 95% | Low |
| server | 6 | ~2000 | 85% | HIGH |
| transport | 6 | ~1800 | 65% | MEDIUM |
