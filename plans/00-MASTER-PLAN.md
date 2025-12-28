# Voice Agent Rust - Master Implementation Plan

## Executive Summary

This document tracks the implementation status and next steps for the Gold Loan Voice Agent built in Rust. A comprehensive review of 6 components was completed on 2024-12-27.

**Target**: Production-ready voice agent with <500ms E2E latency for Kotak Mahindra Bank gold loan acquisition.

---

## Component Status Summary

| Component | Grade | P0 Issues | P1 Issues | Plan File |
|-----------|-------|-----------|-----------|-----------|
| Pipeline (VAD, STT, TTS) | C | 5 | 9 | [01-pipeline-plan.md](./01-pipeline-plan.md) |
| LLM/Speculative | C- | 4 | 7 | [02-llm-plan.md](./02-llm-plan.md) |
| RAG (Retriever, Reranker) | C | 3 | 7 | [03-rag-plan.md](./03-rag-plan.md) |
| Agent (Conversation, Intent) | C | 4 | 8 | [04-agent-plan.md](./04-agent-plan.md) |
| Tools (MCP, Gold Loan) | D+ | 4 | 7 | [05-tools-plan.md](./05-tools-plan.md) |
| Core/Infrastructure | B- | 4 | 9 | [06-core-plan.md](./06-core-plan.md) |
| **Deep Dives** | - | - | - | [07-deep-dives.md](./07-deep-dives.md) |

**Total: 24 P0 (Critical) + 47 P1 (Important) issues identified**

---

## Critical Issues Summary (P0)

### SAFETY HAZARD
| Issue | Location | Action |
|-------|----------|--------|
| `unsafe { mem::zeroed() }` | `tts/streaming.rs:147` | **IMMEDIATE FIX** - Undefined behavior |

### Architecture Gaps
| Issue | Location | Action |
|-------|----------|--------|
| No IndicConformer STT | `stt/streaming.rs` | Fake vocab, no real model |
| No IndicF5 TTS | `tts/streaming.rs` | Wrong input schema |
| DraftVerify is wrong | `speculative.rs:423-449` | Doubles latency, not EAGLE-style |
| No KV cache | `backend.rs` | 2-5x latency penalty |
| Reranker never used | `retriever.rs:234-255` | Dead code, SimpleScorer instead |
| Early-exit never called | `reranker.rs:229-255` | should_exit() is dead code |
| No WebRTC transport | N/A | Critical for mobile |
| No Observability | `Cargo.toml:61-67` | Dependencies unused |

### Business Logic
| Issue | Location | Action |
|-------|----------|--------|
| Hardcoded gold price | `gold_loan.rs:66-70` | Need real-time API |
| No CRM integration | `gold_loan.rs:235-267` | Leads not persisted |
| No calendar integration | `gold_loan.rs:316-361` | Appointments not scheduled |
| Mock branch data | `gold_loan.rs:441-501` | Only 5 hardcoded branches |

### Security
| Issue | Location | Action |
|-------|----------|--------|
| No rate limiting | `websocket.rs:56` | DoS vulnerability |
| Insecure CORS default | `settings.rs:128` | Wildcard "*" |

---

## Phase 1: Critical Fixes (Week 1)

### Day 1-2: Safety & Security
- [ ] Remove `unsafe { mem::zeroed() }` from TTS
- [ ] Add rate limiting to WebSocket
- [ ] Fix CORS default configuration

### Day 3-4: Speculative Execution
- [ ] Fix RaceParallel to abort losing model
- [ ] Remove or rename DraftVerify mode
- [ ] Add KV cache to Ollama backend
- [ ] Reduce SLM timeout from 2000ms to 200ms

### Day 5-7: Core Integration
- [ ] Wire up EarlyExitReranker in retriever
- [ ] Integrate semantic turn detector with ONNX model
- [ ] Initialize observability stack

---

## Phase 2: Model Integration (Week 2)

### STT Integration
- [ ] Create proper IndicConformer loader
- [ ] Add real vocabulary/tokenizer
- [ ] Wire up streaming inference

### TTS Integration
- [ ] Add phoneme conversion for IndicF5
- [ ] Fix ONNX input schema
- [ ] Implement word-level streaming

### RAG Enhancements
- [ ] Parallelize dense + sparse search
- [ ] Implement agentic RAG multi-step flow
- [ ] Add prefetch caching

---

## Phase 3: Business Integration (Week 3)

### External APIs
- [ ] Gold price API (MCX/GoldAPI.io)
- [ ] CRM integration (Salesforce/HubSpot)
- [ ] Calendar API (Google/Outlook)
- [ ] Branch database/API

### Agent Improvements
- [ ] Fix slot extraction using regex patterns
- [ ] Implement actual LLM memory summarization
- [ ] Add Devanagari script support
- [ ] Add missing FSM transitions

---

## Phase 4: Production Hardening (Week 4)

### Transport
- [ ] Create WebRTC transport crate
- [ ] Add session persistence (Redis)
- [ ] Implement graceful shutdown

### Reliability
- [ ] Add retry logic with backoff
- [ ] Add authentication middleware
- [ ] Complete health check dependencies
- [ ] Add comprehensive integration tests

---

## Latency Budget Analysis

Target: **<500ms E2E**

| Component | Budget | Current Estimate | Issue |
|-----------|--------|------------------|-------|
| VAD | 10ms | 10ms | OK |
| STT | 100ms | Unknown | No real model |
| Turn Detection | 20ms | 20ms | OK |
| RAG Prefetch | 50ms | 50ms | OK |
| LLM Generation | 200ms | 2000ms | SLM timeout 10x budget |
| TTS First Chunk | 100ms | Unknown | No real model |
| **Total** | **480ms** | **2000ms+** | **4x over budget** |

### Critical Path to 500ms
1. Reduce SLM timeout to 200ms
2. Add KV cache for multi-turn (saves ~100ms)
3. Integrate real STT/TTS models
4. Measure and optimize

---

## Test Coverage Summary

| Component | Unit | Integration | ONNX | Benchmarks |
|-----------|------|-------------|------|------------|
| Pipeline | 25 | 0 | 0 | 0 |
| LLM | 11 | 0 | 0 | 0 |
| RAG | 12 | 0 | 0 | 0 |
| Agent | 18 | 0 | 0 | 0 |
| Tools | 13 | 0 | 0 | 0 |
| Core | 10 | 0 | 0 | 0 |

**Missing:**
- Zero ONNX code path tests
- Zero integration tests
- Zero latency benchmarks
- Zero Hindi/Hinglish tests

---

## Resolved Questions

See **[07-deep-dives.md](./07-deep-dives.md)** for detailed solutions.

| Question | Resolution |
|----------|------------|
| Latency 450-550ms achievable? | **YES** - reduce SLM timeout to 200ms |
| Model deployment strategy | Need download script + NeMo export guide |
| IndicConformer vs Whisper | IndicConformer primary (ONNX), Whisper fallback |
| Translation layer | Pluggable trait design, IndicTrans2 via gRPC/ONNX |
| WebRTC priority | **Yes, planned** - critical for 500ms target |
| Gold price API | Static for MVP, API integration future phase |
| Competitor rates | Static config for now, database later |
| CRM/Calendar | Future phase, not MVP blocker |
| Mutex contention in VAD | **FIXED**: Consolidate 4 locks → 1 lock |
| Integration tests | Add after implementation complete |
| Error recovery | Retry + circuit breaker + fallback chain design |

### Key Architecture Decisions

1. **Pluggable Model Interface**: STT/TTS/Translation via traits for swappable backends
2. **Configurable Agentic RAG**: Enable/disable multi-step retrieval via config
3. **Error Recovery**: Graceful degradation with fallback responses
4. **Language Support**: Hindi+English MVP, pluggable for 22 languages

---

## Review Completion Status

- [x] Pipeline Review - **Complete**
- [x] LLM Review - **Complete**
- [x] RAG Review - **Complete**
- [x] Agent Review - **Complete**
- [x] Tools Review - **Complete**
- [x] Core Review - **Complete**

---

*Last Updated: 2024-12-27*
*Review Agents: 6 parallel reviews completed*
