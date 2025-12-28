# LLM/Speculative Execution Plan

## Component Overview

The LLM crate handles language model inference with speculative execution:
- Ollama backend
- Speculative execution (SLM-first, Race, Hybrid, Draft-Verify)
- Prompt building with persona
- Streaming generation

**Location**: `voice-agent-rust/crates/llm/src/`

---

## Current Status Summary

| Module | Status | Grade |
|--------|--------|-------|
| OllamaBackend | Functional, missing KV cache | C |
| Speculative SlmFirst | Works correctly | B |
| Speculative RaceParallel | Resource leak - doesn't cancel loser | D |
| Speculative DraftVerify | Fundamentally wrong implementation | F |
| PromptBuilder | Good persona support | B+ |
| Streaming | Basic functionality | B |

---

## P0 - Critical Issues (Must Fix)

| Task | File:Line | Description |
|------|-----------|-------------|
| **No KV Cache Management** | `backend.rs` (missing) | Every turn reprocesses entire context - 2-5x latency |
| **RaceParallel Resource Waste** | `speculative.rs:288-341` | Both models run to completion, wasting GPU |
| **DraftVerify is Fundamentally Wrong** | `speculative.rs:423-449` | NOT EAGLE-style - actually doubles latency |
| No keep_alive for Ollama | `backend.rs:130-139` | Model unloads between calls |

---

## P1 - Important Issues

| Task | File:Line | Description |
|------|-----------|-------------|
| Panic on Client Creation | `backend.rs:113-114` | `OllamaBackend::new()` uses `expect()` - should return Result |
| No Retry Logic | `backend.rs:141-145` | Network failures cause immediate failure |
| Hybrid Streaming Discards Output | `speculative.rs:394-404` | When switching to LLM, SLM output is lost |
| Missing Context Window Management | `prompt.rs:232-234` | Prompts can exceed model limits |
| Quality Estimation Penalizes Valid | `speculative.rs:514-519` | "sorry", "cannot" penalize legitimate responses |
| Token count hardcoded | `backend.rs:96-99` | Uses len/4 estimate instead of actual tokenizer |
| SLM Timeout Too High | `speculative.rs:40` | 2000ms default vs 500ms total budget |

---

## P2 - Nice to Have

| Task | File:Line | Description |
|------|-----------|-------------|
| Missing Claude/OpenAI Backends | `backend.rs:1-4` | Doc claims support but not implemented |
| No Clone for OllamaBackend | `backend.rs:103` | Limits composability |
| Statistics precision | `speculative.rs:533-534` | Use Welford's algorithm |
| Unicode word boundaries | `streaming.rs:134` | TokenBuffer doesn't handle properly |
| Missing Tool role | `prompt.rs:9-15` | No function calling support |

---

## Fix DraftVerify or Remove

Current implementation:
```
1. SLM generates full response
2. LLM generates additional response
3. Concatenate both
```

This DOUBLES latency. Real EAGLE-style:
```
1. SLM generates draft tokens speculatively
2. LLM verifies draft in single forward pass
3. Accept correct prefix, regenerate from first error
```

**Recommendation**: Remove DraftVerify mode or rename to "SlmThenLlm" with clear documentation that it's NOT speculative decoding.

---

## KV Cache Implementation Plan

```rust
// Add to OllamaChatRequest
struct OllamaChatRequest {
    // existing fields...
    keep_alive: Option<String>,  // e.g., "5m" or "-1" for indefinite
    context: Option<Vec<i64>>,   // Previous context for continuation
}

// Add to OllamaBackend
impl OllamaBackend {
    /// Store context after generation for reuse
    pub async fn generate_with_cache(
        &self,
        messages: &[Message],
        context: Option<&[i64]>,
    ) -> Result<(GenerationResult, Vec<i64>), LlmError>;
}
```

---

## Test Coverage Gaps

| File | Tests | Coverage Quality |
|------|-------|------------------|
| backend.rs | 2 | Inadequate - no API tests |
| speculative.rs | 2 | Inadequate - no mock backend tests |
| streaming.rs | 3 | Moderate |
| prompt.rs | 4 | Moderate |

**Missing:**
- Mock backend tests for speculative strategies
- Integration test with actual Ollama
- Cancellation behavior tests
- Context overflow handling tests

---

## Implementation Priorities

### Week 1: Critical Fixes
1. Add KV cache to Ollama backend
2. Fix RaceParallel to abort losing model
3. Remove or rename DraftVerify

### Week 2: Reliability
1. Add retry logic with exponential backoff
2. Fix OllamaBackend::new() to return Result
3. Reduce SLM timeout to 200ms

### Week 3: Quality
1. Add mock backend for testing
2. Improve quality estimation
3. Add context window management

---

*Last Updated: 2024-12-27*
