# Pipeline Component Plan

## Component Overview

The pipeline crate handles the real-time audio processing chain:
- Voice Activity Detection (VAD)
- Speech-to-Text (STT)
- Turn Detection
- Text-to-Speech (TTS)
- Orchestration

**Location**: `voice-agent-rust/crates/pipeline/src/`

---

## Current Status Summary

| Module | Status | Grade |
|--------|--------|-------|
| VAD (MagicNet) | Feature-gated, functional | B |
| Turn Detection | Hybrid semantic+VAD working | B+ |
| STT | Placeholder, no real model | D |
| TTS | Placeholder, UNSAFE code | F |
| Orchestrator | Event-driven, barge-in support | B |

---

## P0 - Critical Issues (Must Fix)

| Task | File:Line | Description |
|------|-----------|-------------|
| **UNSAFE mem::zeroed()** | `tts/streaming.rs:147` | Creates invalid Session struct - UNDEFINED BEHAVIOR! |
| No IndicConformer integration | `stt/streaming.rs:141-143` | Fake vocabulary, no actual model loading |
| No IndicF5 integration | `tts/streaming.rs:107-130` | No phoneme conversion, wrong input schema |
| SmolLM2 missing | `turn_detection/semantic.rs` | Plan requires SmolLM2, uses BERT-style classifier |
| Semantic detector always simple | `turn_detection/hybrid.rs:102-106` | ONNX path never used even when enabled |

---

## P1 - Important Issues

| Task | File:Line | Description |
|------|-----------|-------------|
| VadEngine trait not implemented | `vad/mod.rs:17` vs `magicnet.rs:162` | Signature mismatch (&mut self vs &self) |
| **Mutex contention (4 locks)** | `vad/magicnet.rs:99-103` | **FIX**: Consolidate to 1 lock. See [07-deep-dives.md](./07-deep-dives.md#q9-mutex-contention-in-vad-hot-path---solution) |
| VadResult computed but unused | `vad/magicnet.rs:245` | `_result` variable never used |
| Instant::now() inside lock | `turn_detection/hybrid.rs:149-150` | Clock syscall while holding mutex |
| Mutex blocks async runtime | `orchestrator.rs:128` | parking_lot Mutex in async context - use tokio::sync::Mutex |
| Race condition state checks | `orchestrator.rs:179,185` | State checked without holding lock |
| Hardcoded ONNX input names | `stt/streaming.rs:189` | Different models have different names |
| Text-to-phoneme missing | `tts/streaming.rs:212-214` | TTS expects phonemes, gets codepoints |
| Beam search allocations | `stt/decoder.rs:142` | O(beam * top_k) string clones per frame |

---

## P2 - Nice to Have

| Task | File:Line | Description |
|------|-----------|-------------|
| Vec::remove(0) O(n) | `semantic.rs:282`, `decoder.rs:193` | Should use VecDeque |
| Fake FFT in mel filterbank | `vad/magicnet.rs:400-416` | Band averaging, not real FFT |
| Error type lost in conversion | `lib.rs:69` | All errors become Vad variant |
| No parallel STT + Turn Detection | `orchestrator.rs:195-216` | Sequential when could be parallel |
| parse_words() O(n^2) | `tts/chunker.rs:91-115` | String allocations per word |

---

## Test Coverage Gaps

| Module | Unit Tests | ONNX Tests | Integration | Benchmarks |
|--------|------------|------------|-------------|------------|
| vad/magicnet | 2 | None | None | None |
| turn_detection | 7 | None | None | None |
| stt/streaming | 3 | None | None | None |
| tts | 7 | None | None | None |
| orchestrator | 3 | None | None | None |

**Critical Gaps:**
- Zero ONNX code path tests
- No latency benchmarks (plan requires <500ms E2E)
- No Hindi/Hinglish language handling tests

---

## Implementation Priorities

### Week 1: Fix Critical Safety Issues
1. Remove `unsafe { std::mem::zeroed() }` in TTS
2. Create proper model loader abstraction
3. Add fallback for missing ONNX models

### Week 2: Real Model Integration
1. Integrate IndicConformer STT with proper tokenizer
2. Integrate IndicF5 TTS with phoneme conversion
3. Wire up semantic turn detector with ONNX model

### Week 3: Performance & Testing
1. Replace parking_lot Mutex with tokio::sync::Mutex
2. Add latency benchmarks
3. Add integration tests

---

*Last Updated: 2024-12-27*
