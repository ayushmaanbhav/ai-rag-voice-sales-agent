# Plan 3: Prioritized Recommendations

---

## Implementation Roadmap

### Phase 0: Ship Blockers (Week 1-2)

**Goal:** Fix critical bugs that prevent any production use

| # | Issue | Crate | Effort | Owner |
|---|-------|-------|--------|-------|
| 0.1 | Fix Silero VAD race condition | pipeline | 4h | Audio |
| 0.2 | Fix Hindi slot extraction (Devanagari) | agent | 16h | NLP |
| 0.3 | Fix SMS confirmation false positive | tools | 2h | Backend |
| 0.4 | Integrate gold price API | config | 4h | Backend |
| 0.5 | Apply tiered interest rates | tools | 2h | Backend |
| 0.6 | Add missing intent transitions | agent | 4h | Agent |
| 0.7 | Fix SlmFirst latency (100ms timeout) | llm | 4h | LLM |
| 0.8 | Add auth disabled warning | server | 1h | Security |

**Total Phase 0:** 37 hours (1 week)

---

### Phase 1: Core Infrastructure (Week 3-4)

**Goal:** Complete critical infrastructure for production

| # | Issue | Crate | Effort | Owner |
|---|-------|-------|--------|-------|
| 1.1 | Implement Redis session store | server | 16h | Backend |
| 1.2 | Implement gRPC translator (fallback) | text_processing | 16h | NLP |
| 1.3 | Wire sentence detector to TTS | pipeline | 8h | Audio |
| 1.4 | Add memory watermark management | agent | 4h | Agent |
| 1.5 | Make model paths required errors | config | 2h | Backend |
| 1.6 | Require explicit CORS config in prod | server | 2h | Security |
| 1.7 | Fix sparse search async blocking | rag | 1h | RAG |

**Total Phase 1:** 49 hours (1.5 weeks)

---

### Phase 2: Streaming & Latency (Week 5-6)

**Goal:** Achieve <500ms latency budget

| # | Issue | Crate | Effort | Owner |
|---|-------|-------|--------|-------|
| 2.1 | Implement sliding-window FFT for STT | pipeline | 16h | Audio |
| 2.2 | Wire VAD → RAG prefetch | rag | 8h | RAG |
| 2.3 | Implement stage-aware context budget | agent/rag | 8h | Agent |
| 2.4 | Complete WebRTC ICE handling | transport | 8h | Transport |
| 2.5 | Use Opus decoder for WebRTC audio | transport | 4h | Transport |
| 2.6 | Add parallel SlmFirst execution | llm | 8h | LLM |

**Total Phase 2:** 52 hours (1.5 weeks)

---

### Phase 3: Full Multilingual (Week 7-8)

**Goal:** Support all 22 Indian languages properly

| # | Issue | Crate | Effort | Owner |
|---|-------|-------|--------|-------|
| 3.1 | IndicTrans2 ONNX implementation | text_processing | 40h | NLP |
| 3.2 | All 11 Indic script numerals | agent | 8h | NLP |
| 3.3 | Multiplier words all languages | agent | 8h | NLP |
| 3.4 | NER for names/addresses | text_processing | 16h | NLP |
| 3.5 | Aadhaar Verhoeff checksum | text_processing | 4h | NLP |

**Total Phase 3:** 76 hours (2 weeks)

---

### Phase 4: Architecture Alignment (Week 9-10)

**Goal:** Align with documented architecture

| # | Issue | Crate | Effort | Owner |
|---|-------|-------|--------|-------|
| 4.1 | Domain YAML/TOML loading | config | 16h | Backend |
| 4.2 | Wire query expansion to retrieval | rag | 8h | RAG |
| 4.3 | Integrate personalization in agent | agent | 8h | Agent |
| 4.4 | Complete tool calling for Ollama | llm | 8h | LLM |
| 4.5 | RAG timing strategy selection | rag/agent | 8h | RAG |
| 4.6 | Complete CRM/Calendar integrations | tools | 16h | Backend |

**Total Phase 4:** 64 hours (1.5 weeks)

---

### Phase 5: Polish & Optimization (Week 11-12)

**Goal:** Production hardening

| # | Issue | Crate | Effort | Owner |
|---|-------|-------|--------|-------|
| 5.1 | Cascaded pre-filter threshold tuning | rag | 4h | RAG |
| 5.2 | Rate limiting configuration | server | 4h | Backend |
| 5.3 | Linear resampling → Rubato | core | 4h | Audio |
| 5.4 | Tool timeout per-tool config | tools | 4h | Backend |
| 5.5 | Distributed tracing integration | all | 8h | Infra |
| 5.6 | Integration test suite | all | 16h | QA |
| 5.7 | Performance benchmarks | all | 8h | QA |

**Total Phase 5:** 48 hours (1.5 weeks)

---

## Total Effort Summary

| Phase | Duration | Hours | Focus |
|-------|----------|-------|-------|
| Phase 0 | Week 1-2 | 37h | Ship Blockers |
| Phase 1 | Week 3-4 | 49h | Core Infrastructure |
| Phase 2 | Week 5-6 | 52h | Streaming & Latency |
| Phase 3 | Week 7-8 | 76h | Full Multilingual |
| Phase 4 | Week 9-10 | 64h | Architecture Alignment |
| Phase 5 | Week 11-12 | 48h | Polish |
| **Total** | **12 weeks** | **326h** | |

**Assuming 2 developers:** 6 weeks to production-ready (Phase 0-2)

---

## Quick Wins (< 4 hours each)

| Issue | Effort | Impact |
|-------|--------|--------|
| Fix SMS false positive | 2h | User trust |
| Apply tiered rates | 2h | Quote accuracy |
| Add auth warning | 1h | Security |
| Require CORS config | 2h | Security |
| Fix sparse search blocking | 1h | Performance |
| DateOfBirth in PIISeverity | 0.5h | Completeness |
| Objection pattern precedence | 0.5h | Code quality |

**Total Quick Wins:** 9 hours

---

## Code Snippets for Key Fixes

### Fix 0.1: Silero VAD Race Condition
```rust
// crates/pipeline/src/vad/silero.rs
pub async fn process_frame(&self, frame: &AudioFrame) -> Result<VadResult> {
    let mut state = self.mutable.lock();
    state.audio_buffer.extend_from_slice(&frame.samples);

    if state.audio_buffer.len() >= self.config.chunk_size {
        let chunk: Vec<f32> = state.audio_buffer.drain(..self.config.chunk_size).collect();

        // Keep lock during computation
        let speech_prob = self.compute_probability_locked(&mut state, &chunk)?;
        let is_speech = speech_prob >= self.config.threshold;
        self.update_state_locked(&mut state, is_speech, speech_prob)?;

        return Ok(state.last_result.clone());
    }

    Ok(VadResult::NoChange)
}
```

### Fix 0.2: Hindi Slot Extraction
```rust
// crates/agent/src/intent.rs
fn compile_slot_patterns() -> HashMap<String, Vec<SlotPattern>> {
    hashmap! {
        "loan_amount" => vec![
            // English patterns
            SlotPattern::regex(r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(lakh|lac|crore|thousand|k)?", 0.95),
            // Devanagari numerals
            SlotPattern::regex(r"([\u0966-\u096F]+)\s*(लाख|करोड़|हज़ार)", 0.95),
            // Hindi number words
            SlotPattern::regex(r"(एक|दो|तीन|चार|पांच|छह|सात|आठ|नौ|दस)\s*(लाख|करोड़|हज़ार)", 0.90),
            // Tamil numerals
            SlotPattern::regex(r"([\u0BE6-\u0BEF]+)\s*(லட்சம்|கோடி)", 0.95),
        ],
        // ... other slots
    }
}
```

### Fix 0.3: SMS Confirmation
```rust
// crates/tools/src/gold_loan.rs
let result = json!({
    "status": "pending_confirmation",
    "confirmation_method": "agent_will_call_to_confirm",
    "confirmation_sent": false,
    "next_action": "Agent will call customer to confirm appointment",
    "appointment": {
        "date": date,
        "time": slot,
        "branch": branch.name,
        "address": branch.address,
    }
});
```

### Fix 0.7: SlmFirst Latency
```rust
// crates/llm/src/speculative.rs
pub async fn execute(&self, messages: &[Message]) -> Result<String> {
    // Reduce timeout from 200ms to 100ms
    let slm_timeout = Duration::from_millis(100);

    // OR: Parallel execution
    let slm_fut = self.slm.generate(messages);
    let llm_fut = self.llm.generate(messages);

    tokio::select! {
        biased;  // Prefer SLM if ready

        slm_result = timeout(slm_timeout, slm_fut) => {
            if let Ok(Ok(response)) = slm_result {
                if self.is_acceptable(&response) {
                    return Ok(response);
                }
            }
            // Fall through to LLM
            llm_fut.await
        }

        llm_result = llm_fut => llm_result
    }
}
```

### Fix 1.1: Redis Session Store
```rust
// crates/server/src/session.rs
use deadpool_redis::{Pool, redis::AsyncCommands};

pub struct RedisSessionStore {
    pool: Pool,
    ttl_seconds: u64,
}

impl SessionStore for RedisSessionStore {
    async fn store_metadata(&self, id: &str, meta: &SessionMetadata) -> Result<()> {
        let mut conn = self.pool.get().await?;
        let json = serde_json::to_string(meta)?;
        conn.set_ex(format!("session:{}", id), json, self.ttl_seconds).await?;
        Ok(())
    }

    async fn get_metadata(&self, id: &str) -> Result<Option<SessionMetadata>> {
        let mut conn = self.pool.get().await?;
        let json: Option<String> = conn.get(format!("session:{}", id)).await?;
        Ok(json.map(|s| serde_json::from_str(&s)).transpose()?)
    }
}
```

---

## Test Plan

### Unit Tests Required

| Fix | Test Cases |
|-----|------------|
| VAD race | Concurrent frame processing, buffer integrity |
| Hindi slots | "पांच लाख", "दस करोड़", mixed numerals |
| SMS stub | Returns pending, not confirmed |
| Tiered rates | <1L, 1-5L, >5L buckets |
| Intent transitions | All 11 intents → stage changes |

### Integration Tests Required

| Scenario | Components |
|----------|------------|
| Full conversation | VAD → STT → Intent → Agent → TTS |
| Hindi user flow | Hindi input → correct slots → Hindi output |
| Barge-in handling | Interrupt during TTS |
| Session persistence | Create → Restart → Resume |
| RAG timing | Prefetch on VAD → Ready for LLM |

### Performance Tests Required

| Metric | Target | Test |
|--------|--------|------|
| E2E latency | <500ms | Time from audio end to TTS start |
| STT streaming | <150ms per chunk | Mel spectrogram computation |
| LLM response | <300ms | First token time |
| Memory usage | <512MB | Long conversation (100+ turns) |

---

## Risk Mitigation

| Risk | Probability | Mitigation |
|------|-------------|------------|
| IndicTrans2 ONNX complex | HIGH | Start with gRPC fallback |
| Redis integration delays | MEDIUM | Keep in-memory as fallback |
| Streaming STT breaks | MEDIUM | Feature flag, batch fallback |
| Gold API rate limits | LOW | Cache aggressively, fallback price |
| Multilingual edge cases | HIGH | Extensive test data, gradual rollout |

---

## Success Criteria

### Phase 0 Complete When:
- [ ] All P0 bugs fixed
- [ ] Hindi slot extraction working (5 test cases)
- [ ] No false positives in user-facing claims
- [ ] E2E latency <600ms (relaxed target)

### Phase 1 Complete When:
- [ ] Redis sessions persist across restart
- [ ] Translation works (gRPC)
- [ ] Sentence-level TTS streaming
- [ ] No memory leaks in 1hr test

### Phase 2 Complete When:
- [ ] E2E latency <500ms
- [ ] WebRTC audio quality acceptable
- [ ] RAG prefetch reduces latency by 50ms+
- [ ] Stage-aware context saves 30%+ tokens

### Production Ready When:
- [ ] All Phase 0-2 complete
- [ ] 22 Indian languages tested
- [ ] 48hr stability test passed
- [ ] Security review passed
