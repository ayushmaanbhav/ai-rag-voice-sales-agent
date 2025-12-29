# Plan 3: Critical Findings (P0/P1 Issues)

> **Priority Legend:**
> - P0 = Ship Blocker (must fix before any production use)
> - P1 = Pre-Launch (fix before public launch)
> - P2 = Post-Launch (fix in next sprint)

---

## P0 Critical Issues (Ship Blockers)

### P0-1: Redis Session Persistence Completely Stubbed

**Location:** `crates/server/src/session.rs:153-226`

**Problem:**
```rust
pub struct RedisSessionStore {
    config: RedisSessionConfig,
    // TODO: Add Redis connection pool when implementing
}

impl SessionStore for RedisSessionStore {
    async fn store_metadata(&self, _: &str, _: &SessionMetadata) -> Result<(), SessionError> {
        tracing::debug!("Redis session storage not yet implemented");
        Ok(())  // ← SILENTLY SUCCEEDS WITHOUT STORING
    }

    async fn get_metadata(&self, _: &str) -> Result<Option<SessionMetadata>, SessionError> {
        Ok(None)  // ← ALWAYS RETURNS NONE
    }
}
```

**Impact:**
- Sessions stored only in memory (HashMap)
- Server restart = all sessions lost
- Cannot scale to multiple instances
- No distributed session coordination

**Fix:**
1. Add `redis` crate dependency
2. Create connection pool using `deadpool-redis`
3. Serialize SessionMetadata to JSON
4. Use Redis TTL for session expiration

**Effort:** 16 hours

---

### P0-2: Silero VAD Race Condition

**Location:** `crates/pipeline/src/vad/silero.rs:163-189`

**Problem:**
```rust
let mut state = self.mutable.lock();
state.audio_buffer.extend_from_slice(&frame.samples);

if state.audio_buffer.len() >= self.config.chunk_size {
    let chunk: Vec<f32> = state.audio_buffer.drain(..).collect();
    drop(state);  // ← LOCK RELEASED

    let speech_prob = self.compute_probability(&chunk)?;  // ← RACE WINDOW

    let mut state = self.mutable.lock();  // ← RE-ACQUIRED
}
```

**Impact:**
- Between `drop(state)` and re-acquire, another thread could modify buffer
- Data corruption in audio stream
- Panic if buffer modified during drain

**Fix:**
```rust
let mut state = self.mutable.lock();
state.audio_buffer.extend_from_slice(&frame.samples);

if state.audio_buffer.len() >= self.config.chunk_size {
    let chunk: Vec<f32> = state.audio_buffer.drain(..).collect();
    // Keep lock held, compute within lock
    let speech_prob = self.compute_probability_inner(&mut state, &chunk)?;
    self.update_state_inner(&mut state, is_speech, speech_prob)?;
}
// Lock released here at end of scope
```

**Effort:** 4 hours

---

### P0-3: Hindi Slot Extraction ASCII-Only

**Location:** `crates/agent/src/intent.rs:231-346`

**Problem:**
```rust
fn compile_slot_patterns() -> HashMap<String, Vec<SlotPattern>> {
    // Only English patterns defined
    ("loan_amount", vec![
        SlotPattern::regex(r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(lakh|lac|lakhs|crore|thousand|k)?", 0.95),
    ]),
    // NO Devanagari patterns!
}
```

**Impact:**
- User says: "पांच लाख रुपये का लोन चाहिए" → slot extraction returns None
- Core business logic completely fails for Hindi users
- ~40% of target users cannot use the system

**Fix:**
```rust
// Add Devanagari patterns
("loan_amount", vec![
    // Existing English patterns
    SlotPattern::regex(r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(lakh|lac|lakhs|crore|thousand|k)?", 0.95),
    // Hindi/Devanagari patterns
    SlotPattern::regex(r"([\u0966-\u096F]+)\s*(लाख|करोड़|हज़ार)", 0.95),
    SlotPattern::regex(r"(पांच|दस|बीस|पचास|सौ)\s*(लाख|करोड़|हज़ार)", 0.90),
]),
```

**Effort:** 16 hours (includes testing for all 11 Indic scripts)

---

### P0-4: SMS Confirmation False Positive

**Location:** `crates/tools/src/gold_loan.rs:493-496`

**Problem:**
```rust
let result = json!({
    "status": "confirmed",
    "confirmation_sent": true,  // ← CLAIMS SMS SENT
    "appointment": { ... }
});
// But NO actual SMS provider integration!
```

**Impact:**
- User thinks appointment is confirmed via SMS
- No actual confirmation sent
- User may miss appointment
- Trust erosion

**Fix (Stub):**
```rust
let result = json!({
    "status": "pending_confirmation",
    "confirmation_sent": false,
    "confirmation_method": "agent_will_call",  // ← Honest
    "appointment": { ... }
});
```

**Fix (Real):**
```rust
// Integrate MSG91 or Twilio
let sms_result = sms_provider.send(
    customer_phone,
    format!("Kotak appointment confirmed: {} at {}", date, branch)
).await?;

let result = json!({
    "status": "confirmed",
    "confirmation_sent": sms_result.is_success(),
    "sms_id": sms_result.message_id,
});
```

**Effort:** 2h (stub) / 8h (real SMS)

---

### P0-5: Gold Price Hardcoded

**Location:** `crates/config/src/gold_loan.rs:96`

**Problem:**
```rust
fn default_gold_price() -> f64 {
    7500.0  // INR per gram - STALE WITHIN DAYS
}
```

**Impact:**
- Gold price fluctuates daily (±2-5%)
- All eligibility calculations wrong
- Savings comparisons vs competitors inaccurate
- Legal risk if customer disputes quoted amount

**Fix:**
```rust
// Add to config
pub struct GoldPriceConfig {
    pub source: GoldPriceSource,  // Api, Manual, Fallback
    pub api_url: Option<String>,  // e.g., goldapi.io
    pub api_key: Option<String>,
    pub cache_ttl_seconds: u64,   // 3600 (1 hour)
    pub fallback_price: f64,      // 7500.0
}

// Runtime price fetching
async fn get_current_gold_price(&self) -> f64 {
    if let Some(cached) = self.cache.get() {
        if cached.age < self.config.cache_ttl {
            return cached.price;
        }
    }

    match self.fetch_from_api().await {
        Ok(price) => {
            self.cache.set(price);
            price
        }
        Err(_) => self.config.fallback_price
    }
}
```

**Effort:** 4 hours

---

### P0-6: Translation Completely Stubbed

**Location:** `crates/text_processing/src/translation/mod.rs:61-63`

**Problem:**
```rust
TranslationProvider::Grpc => {
    tracing::warn!("gRPC translator not yet implemented, using no-op");
    Box::new(NoopTranslator::new())
}
// NO ONNX implementation at all
// IndicTrans2 mentioned in architecture but not implemented
```

**Impact:**
- Cannot implement Translate-Think-Translate pattern
- LLM reasons directly in Indic languages (lower quality)
- No translation between 22 Indian languages
- Architecture promise broken

**Fix:**
```rust
// Implement IndicTrans2 ONNX
pub struct IndicTranslator {
    encoder: ort::Session,
    decoder: ort::Session,
    tokenizer: IndicTransTokenizer,
}

impl IndicTranslator {
    pub fn new(model_path: &str) -> Result<Self> {
        let encoder = ort::Session::builder()?
            .with_model_from_file(format!("{}/encoder.onnx", model_path))?;
        let decoder = ort::Session::builder()?
            .with_model_from_file(format!("{}/decoder.onnx", model_path))?;
        // ...
    }
}
```

**Effort:** 40 hours (includes tokenizer, testing all language pairs)

---

### P0-7: SlmFirst Latency Cliffs

**Location:** `crates/llm/src/speculative.rs:175-176`

**Problem:**
```rust
let slm_timeout = Duration::from_millis(self.config.slm_timeout_ms);  // 200ms
match timeout(slm_timeout, self.slm.generate(messages)).await {
    Ok(result) => { /* SLM succeeded */ }
    Err(_) => { /* Timeout, fall back to LLM */ }
}
```

**Timeline:**
- Complexity check: 5ms
- SLM attempt: 200ms (timeout)
- LLM fallback: 300ms
- **Total: 505ms+** (exceeds 500ms budget!)

**Impact:**
- No latency improvement over direct LLM call
- Marketing claim of "latency optimization" fails
- User experience degrades

**Fix:**
```rust
// Option A: Reduce timeout
let slm_timeout = Duration::from_millis(100);  // 100ms, not 200ms

// Option B: Parallel execution
let slm_fut = self.slm.generate(messages);
let llm_fut = self.llm.generate(messages);

tokio::select! {
    slm_result = slm_fut => {
        if is_acceptable(&slm_result?) { return slm_result; }
        llm_fut.await
    }
    llm_result = llm_fut => llm_result
}
```

**Effort:** 8 hours

---

### P0-8: STT Not Truly Streaming

**Location:** `crates/pipeline/src/stt/indicconformer.rs:564-614`

**Problem:**
```rust
// "Streaming" actually recomputes full mel spectrogram each time
// Manual DFT computation - O(n²) complexity instead of FFT O(n log n)
fn compute_mel_spectrogram(&self, audio: &[f32]) -> Array2<f32> {
    // Processes ALL audio from start, not incremental
}
```

**Impact:**
- 10-20x slower than proper streaming FFT
- Latency grows with conversation length
- Likely exceeds 500ms budget on longer utterances

**Fix:**
```rust
// Use rustfft for proper FFT
use rustfft::{FftPlanner, num_complex::Complex};

// Implement sliding window with overlap-add
struct StreamingMelComputer {
    fft: Arc<dyn Fft<f32>>,
    mel_bank: MelFilterbank,
    buffer: VecDeque<f32>,
    hop_size: usize,
}

impl StreamingMelComputer {
    fn process_frame(&mut self, samples: &[f32]) -> Option<Vec<f32>> {
        self.buffer.extend(samples);
        if self.buffer.len() >= self.frame_size {
            let frame = self.buffer.drain(..self.hop_size).collect();
            Some(self.compute_single_frame(&frame))
        } else {
            None
        }
    }
}
```

**Effort:** 16 hours

---

## P1 High Priority Issues

### P1-1: Authentication Disabled by Default

**Location:** `crates/config/src/settings.rs:177-184`

```rust
impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            enabled: false,  // ← SECURITY RISK
            api_key: None,
        }
    }
}
```

**Fix:** Add startup warning, require explicit enable in production
**Effort:** 1 hour

---

### P1-2: FSM Missing Intent Transitions

**Location:** `crates/agent/src/conversation.rs:262-296`

```rust
match intent.intent.as_str() {
    "farewell" => Some(Farewell),
    "objection" => Some(ObjectionHandling),
    "schedule_visit" if current == Presentation => Some(Closing),
    "affirmative" if current == Closing => Some(Farewell),
    _ => None,  // ← ALL OTHER INTENTS IGNORED
}
```

**Missing:** negative, interest_rate, eligibility_check, loan_inquiry, complaint
**Effort:** 4 hours

---

### P1-3: Tiered Rates Not Applied

**Location:** `crates/tools/src/gold_loan.rs:177-214`

```rust
let result = json!({
    "interest_rate_percent": self.config.kotak_interest_rate,  // Always 10.5%!
});
// Config has tiers: <1L: 11.5%, 1-5L: 10.5%, >5L: 9.5%
// BUT NOT USED
```

**Fix:** Call `config.get_tiered_rate(loan_amount)`
**Effort:** 2 hours

---

### P1-4: Memory Fire-and-Forget

**Location:** `crates/agent/src/agent.rs:277-281`

```rust
tokio::spawn(async move {
    if let Err(e) = memory.summarize_pending_async().await {
        tracing::debug!("Memory summarization skipped: {}", e);
        // Error just logged, not handled!
    }
});
```

**Fix:** Add watermark check, emergency truncate after 3 failures
**Effort:** 4 hours

---

### P1-5: CORS Empty by Default

**Location:** `crates/config/src/settings.rs:260`

```rust
cors_origins: Vec::new(),  // Defaults to localhost:3000
```

**Fix:** Require explicit configuration in production
**Effort:** 1 hour

---

### P1-6: Sparse Search Blocks Async

**Location:** `crates/rag/src/retriever.rs:186-192`

```rust
let sparse_future = async {
    self.search_sparse(query)  // ← CPU-intensive BM25 on async thread
};
```

**Fix:** Use `tokio::task::spawn_blocking`
**Effort:** 1 hour

---

### P1-7: WebRTC ICE Handling Incomplete

**Location:** `crates/transport/src/webrtc.rs:542-543`

```rust
// Wait for ICE gathering to complete
// TODO: Add timeout and proper ICE candidate handling
```

**Fix:** Implement trickle ICE, add gathering timeout
**Effort:** 8 hours

---

### P1-8: Model Path Validation Warnings-Only

**Location:** `crates/config/src/settings.rs:64-109`

```rust
if !path_obj.exists() {
    tracing::warn!("Model not found: {} = {}", field, path);  // ← Just warns
}
```

**Fix:** Make VAD/TTS/STT model paths required errors
**Effort:** 2 hours

---

## P2 Medium Priority Issues

| # | Issue | Location | Effort |
|---|-------|----------|--------|
| P2-1 | DateOfBirth missing from PIISeverity | core/pii.rs | 0.5h |
| P2-2 | Objection pattern precedence unclear | core/adaptation.rs | 0.5h |
| P2-3 | BankAccount pattern too loose | text_processing/pii/patterns.rs | 2h |
| P2-4 | Aadhaar Verhoeff checksum not implemented | text_processing/pii/patterns.rs | 4h |
| P2-5 | Query expansion not integrated | rag/query_expansion.rs | 8h |
| P2-6 | Indic numerals only Devanagari tested | agent/intent.rs | 8h |
| P2-7 | Linear resampling instead of Rubato | core/audio.rs | 2h |
| P2-8 | WebRTC audio decoding raw instead of Opus | transport/webrtc.rs | 4h |
| P2-9 | Rate limiting hard-coded | server/rate_limit.rs | 2h |
| P2-10 | Cascaded pre-filter threshold too low | rag/reranker.rs | 1h |

---

## Summary

| Priority | Count | Total Effort |
|----------|-------|--------------|
| **P0** | 8 | 106 hours |
| **P1** | 8 | 23 hours |
| **P2** | 10 | 32 hours |
| **Total** | 26 | 161 hours |

**Timeline:**
- P0: 2.5 dev weeks
- P0+P1: 3.5 dev weeks
- All: 4.5 dev weeks
