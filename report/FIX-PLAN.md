# Voice Agent Architecture Review & Fix Plan

## Executive Summary

A comprehensive analysis of 11 Rust crates was conducted by specialized agents to compare the actual implementation against documented architecture. The system is **~75% production-ready** with well-implemented core components, but has critical architectural gaps, unused services, and missing integrations.

### Overall Assessment by Crate

| Crate | Completion | Status | Critical Issues |
|-------|------------|--------|-----------------|
| core | 100% | Production-Ready | None |
| config | 90% | Production-Ready | Duplicate configs |
| llm | 100% | Production-Ready | None |
| server | 98% | Production-Ready | None |
| transport | 95% | Production-Ready | Minor stubs |
| rag | 90% | Production-Ready | Missing ContextCompressor |
| agent | 85% | Functional | No Agent trait, missing grammar correction |
| tools | 70% | Functional | Missing MCP JSON-RPC server |
| pipeline | 55% | **Needs Work** | Text/RAG/LLM not integrated |
| text_processing | 60% | **Needs Work** | Intent in wrong crate, no sentiment |
| persistence | 70% | Functional | SMS/GoldPrice services unused |

---

## Priority 0 (P0) - Critical Fixes

### P0-1: Pipeline Architecture Mismatch

**Problem:** The documented frame-based pipeline architecture is NOT implemented. The pipeline only handles audio I/O (VAD → STT → TTS). Text processing, RAG, and LLM are handled in the agent crate instead of being orchestrated by the pipeline.

**Evidence:**
- `pipeline/Cargo.toml` does NOT import `voice-agent-text-processing`, `voice-agent-rag`, or `voice-agent-llm`
- Pipeline only defines audio frames, missing: `Frame::GrammarCorrected`, `Frame::Translated`, `Frame::LLMChunk`
- Agent crate handles these stages outside the frame pipeline

**Current Flow:**
```
Audio → Pipeline.VAD → Pipeline.STT → (EXIT PIPELINE) → Agent.process() → Pipeline.TTS → Audio
```

**Documented Flow:**
```
Audio → VAD → STT → Grammar → Translation → RAG → LLM → TTS → Audio (all in pipeline)
```

**Fix Options:**
1. **Option A - Refactor Pipeline** (HIGH EFFORT): Add processors for text_processing, rag, llm stages; define new Frame variants
2. **Option B - Document Reality** (LOW EFFORT): Update ARCHITECTURE_v2.md to reflect hybrid architecture with agent handling language stages

**Recommendation:** Option B initially, Option A for v2

**Files to Update:**
- `/home/vscode/goldloan-study/docs/ARCHITECTURE_v2.md` - Update pipeline section
- `/home/vscode/goldloan-study/voice-agent/backend/crates/pipeline/src/lib.rs` - Add frame types if choosing Option A

---

### P0-2: speak_streaming() Not Wired

**Problem:** The `speak_streaming()` method in pipeline is fully implemented but NOT called by voice_session.

**Evidence:**
- `orchestrator.rs:408-475` implements streaming TTS for LLM chunks
- `voice_session.rs` uses `speak()` instead of `speak_streaming()`
- Streaming capability exists but is unused

**Impact:** Higher latency - waiting for full LLM response before TTS instead of streaming

**Fix:**
```rust
// In voice_session.rs, change:
self.pipeline.speak(&response_text).await?;
// To:
let rx = self.pipeline.speak_streaming(llm_chunk_receiver).await?;
```

**Files to Update:**
- `/home/vscode/goldloan-study/voice-agent/backend/crates/agent/src/voice_session.rs`

---

## Priority 1 (P1) - High Priority Fixes

### P1-1: Missing Agent Trait Abstraction

**Problem:** No `Agent` trait exists - `GoldLoanAgent` is a monolithic struct. This prevents easy testing and alternative agent implementations.

**Evidence:**
- Architecture docs show `Agent` trait with `process()`, `handle_tool_result()`, `greeting()`, `farewell()`
- Actual code has `GoldLoanAgent` struct with concrete methods, no trait

**Fix:**
Create `/home/vscode/goldloan-study/voice-agent/backend/crates/agent/src/traits.rs`:
```rust
#[async_trait]
pub trait Agent: Send + Sync {
    async fn process(&self, input: &str, state: &mut AgentState) -> Result<AgentResponse, AgentError>;
    async fn handle_tool_result(&self, result: &ToolOutput) -> Result<String, AgentError>;
    fn greeting(&self) -> String;
    fn farewell(&self) -> String;
}

impl Agent for GoldLoanAgent { ... }
```

**Files to Update:**
- `/home/vscode/goldloan-study/voice-agent/backend/crates/agent/src/traits.rs` (NEW)
- `/home/vscode/goldloan-study/voice-agent/backend/crates/agent/src/lib.rs`
- `/home/vscode/goldloan-study/voice-agent/backend/crates/agent/src/agent.rs`

---

### P1-2: Intent Detection in Wrong Crate

**Problem:** Intent detection is in `agent/intent.rs` instead of `text_processing` crate as documented.

**Evidence:**
- `agent/src/intent.rs` contains IntentDetector with 12 intents
- `text_processing` crate has no intent module
- Violates documented separation of concerns

**Fix:**
1. Create `/home/vscode/goldloan-study/voice-agent/backend/crates/text_processing/src/intent/mod.rs`
2. Move IntentDetector logic to text_processing
3. Update agent to import from text_processing

**Files to Update:**
- `/home/vscode/goldloan-study/voice-agent/backend/crates/text_processing/src/intent/mod.rs` (NEW)
- `/home/vscode/goldloan-study/voice-agent/backend/crates/text_processing/src/lib.rs`
- `/home/vscode/goldloan-study/voice-agent/backend/crates/agent/src/agent.rs`

---

### P1-3: Missing MCP JSON-RPC Transport

**Problem:** MCP protocol types are defined but there's no JSON-RPC 2.0 server. Tools are accessed via REST API instead of MCP protocol.

**Evidence:**
- `tools/src/mcp.rs` defines JsonRpcRequest, JsonRpcResponse, all MCP methods
- No McpServer struct or JSON-RPC handler exists
- Server uses REST endpoints `/api/tools` instead

**Impact:** Not MCP-compliant; can't connect to MCP clients

**Fix:**
Create `/home/vscode/goldloan-study/voice-agent/backend/crates/tools/src/server.rs`:
```rust
pub struct McpServer {
    registry: Arc<ToolRegistry>,
}

impl McpServer {
    pub async fn handle_request(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        match req.method.as_str() {
            "tools/list" => self.list_tools(),
            "tools/call" => self.call_tool(req.params),
            "resources/list" => self.list_resources(),
            _ => JsonRpcResponse::error(-32601, "Method not found"),
        }
    }
}
```

**Files to Update:**
- `/home/vscode/goldloan-study/voice-agent/backend/crates/tools/src/server.rs` (NEW)
- `/home/vscode/goldloan-study/voice-agent/backend/crates/server/src/http.rs` - Add MCP endpoint

---

### P1-4: Unused Persistence Services

**Problem:** SMS and GoldPrice services are fully implemented but never called.

**Evidence:**
- `persistence/src/sms.rs` - 283 lines, complete SimulatedSmsService
- `persistence/src/gold_price.rs` - 329 lines, complete SimulatedGoldPriceService
- Neither are called from agent or tools

**Fix:**
Wire services into tools:
```rust
// In tools/src/gold_loan.rs
pub fn new_with_services(
    sms: Arc<dyn SmsService>,
    gold_price: Arc<dyn GoldPriceService>,
) -> Self { ... }
```

**Files to Update:**
- `/home/vscode/goldloan-study/voice-agent/backend/crates/tools/src/gold_loan.rs`
- `/home/vscode/goldloan-study/voice-agent/backend/crates/server/src/main.rs` - Initialize and inject services

---

## Priority 2 (P2) - Medium Priority Fixes

### P2-1: Missing Sentiment Analysis

**Problem:** No sentiment detection capability exists anywhere in the codebase.

**Evidence:**
- Architecture docs mention sentiment analysis for customer emotion
- Grep for "sentiment" returns no implementations
- Agent has no way to detect frustration/satisfaction

**Fix:**
Create `/home/vscode/goldloan-study/voice-agent/backend/crates/text_processing/src/sentiment/mod.rs`:
```rust
pub enum Sentiment { Positive, Negative, Neutral, Frustrated, Satisfied }

pub struct SentimentAnalyzer { ... }

impl SentimentAnalyzer {
    pub fn analyze(&self, text: &str) -> Sentiment { ... }
}
```

**Files to Update:**
- `/home/vscode/goldloan-study/voice-agent/backend/crates/text_processing/src/sentiment/mod.rs` (NEW)
- `/home/vscode/goldloan-study/voice-agent/backend/crates/text_processing/src/lib.rs`
- `/home/vscode/goldloan-study/voice-agent/backend/crates/text_processing/src/pipeline.rs`

---

### P2-2: Missing RAG ContextCompressor

**Problem:** No ContextCompressor for conversation history management as documented.

**Evidence:**
- Architecture docs specify ContextCompressor for history compression
- `rag/src/context.rs` only handles token budgets, not compression
- No summarization of old conversation turns

**Fix:**
Add to `/home/vscode/goldloan-study/voice-agent/backend/crates/rag/src/compressor.rs`:
```rust
pub struct ContextCompressor {
    llm: Arc<dyn LanguageModel>,
    max_history_turns: usize,
}

impl ContextCompressor {
    pub async fn compress(&self, turns: &[Turn]) -> CompressedContext { ... }
}
```

---

### P2-3: Duplicate Config Definitions

**Problem:** Multiple conflicting config definitions exist.

**Evidence:**
- `config/src/agent.rs` defines `MemoryConfig`, `RagConfig`
- `config/src/settings.rs` also defines `RagConfig`
- `agent/src/conversation.rs` defines another `MemoryConfig`
- Unclear which takes precedence

**Fix:**
1. Consolidate all RagConfig into one location
2. Remove duplicate MemoryConfig
3. Document precedence

**Files to Update:**
- `/home/vscode/goldloan-study/voice-agent/backend/crates/config/src/agent.rs`
- `/home/vscode/goldloan-study/voice-agent/backend/crates/config/src/settings.rs`

---

### P2-4: Placeholder Echo Cancellation / Noise Suppression

**Problem:** AEC/NS flags exist but are documented as "placeholder".

**Evidence:**
- `pipeline/src/config.rs` lines 415-427: "P2 FIX: Currently a placeholder flag"
- Flags can be set but have no effect
- Requires signal processing library

**Fix Options:**
1. Implement with WebRTC audio processing
2. Add warning when enabled but not implemented
3. Remove flags if not planned

**Files to Update:**
- `/home/vscode/goldloan-study/voice-agent/backend/crates/pipeline/src/config.rs`

---

### P2-5: Missing Loan Entity Extraction

**Problem:** Entity extraction exists for PII only, not for loan-specific entities.

**Evidence:**
- `text_processing/src/pii/ner.rs` detects names and addresses
- No extraction for: loan amounts, gold weight, interest rates, tenures

**Fix:**
Create `/home/vscode/goldloan-study/voice-agent/backend/crates/text_processing/src/entities/loan_entities.rs`:
```rust
pub struct LoanEntityExtractor { ... }

impl LoanEntityExtractor {
    pub fn extract_amount(&self, text: &str) -> Option<Currency> { ... }
    pub fn extract_gold_weight(&self, text: &str) -> Option<Weight> { ... }
    pub fn extract_rate(&self, text: &str) -> Option<Percentage> { ... }
}
```

---

## Priority 3 (P3) - Low Priority / Cleanup

### P3-1: Remove Deprecated Redis Session Store

**Problem:** Redis stub implementation still exists with deprecated warnings.

**Evidence:**
- `server/src/session.rs:172-231` - RedisSessionStore with empty implementations
- Marked as deprecated, logs warnings

**Fix:**
Remove or hide behind feature flag.

**Files to Update:**
- `/home/vscode/goldloan-study/voice-agent/backend/crates/server/src/session.rs`

---

### P3-2: Remove Unused LLM Provider Kalosm

**Problem:** Kalosm provider enum variant appears unused.

**Evidence:**
- `config/src/agent.rs` line 210 defines `LlmProvider::Kalosm`
- No implementation found

**Fix:**
Remove variant or add `#[allow(dead_code)]` with comment.

---

### P3-3: Unused ResourceProvider Trait

**Problem:** MCP ResourceProvider trait defined but never implemented.

**Evidence:**
- `tools/src/mcp.rs` defines `ResourceProvider` trait
- No implementations exist
- `resources/list`, `resources/read` are constants only

**Fix:**
Either implement or remove.

---

### P3-4: Conflicting Interest Rate Configs

**Problem:** Two different interest rates in config.

**Evidence:**
- `gold_loan.rs`: `kotak_interest_rate = 10.5%`
- `gold_loan.rs`: `tiered_rates.tier1_rate = 11.5%`
- `get_tiered_rate()` uses tiered_rates, ignoring kotak_interest_rate

**Fix:**
Remove unused `kotak_interest_rate` or document relationship.

---

## Summary: Implementation Effort

| Priority | Count | Estimated Effort |
|----------|-------|------------------|
| P0 | 2 | 2-3 days |
| P1 | 4 | 4-5 days |
| P2 | 5 | 3-4 days |
| P3 | 4 | 1-2 days |
| **Total** | **15** | **~10-14 days** |

---

## Architecture Diagram (Current Reality)

```
                    +------------------+
                    |     SERVER       |
                    | (WebSocket/REST) |
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
      +-------v-------+            +--------v--------+
      |   PIPELINE    |            |     AGENT       |
      | (Audio Only)  |            | (Language+Tools)|
      +-------+-------+            +--------+--------+
              |                             |
    +---------+---------+         +---------+---------+---------+
    |         |         |         |         |         |         |
+---v---+ +---v---+ +---v---+ +---v---+ +---v---+ +---v---+ +---v---+
|  VAD  | |  STT  | |  TTS  | |  LLM  | |  RAG  | | TOOLS | | TEXT  |
+-------+ +-------+ +-------+ +-------+ +-------+ +-------+ +-------+
```

---

## Recommended Fix Order

1. **P0-2**: Wire speak_streaming() (quick win, immediate latency improvement)
2. **P1-4**: Wire SMS/GoldPrice services (unused code should work)
3. **P1-1**: Create Agent trait (improves testability)
4. **P2-1**: Add sentiment analysis (new feature)
5. **P1-2**: Move intent to text_processing (refactor)
6. **P0-1**: Document/update architecture (alignment)
7. **P1-3**: Add MCP JSON-RPC server (new feature)
8. **P2-2**: Add ContextCompressor (enhancement)
9. **Remaining P2/P3**: Cleanup tasks

---

*Report generated: 2025-12-30*
*Analyzed by: 11 specialized code analysis agents*
