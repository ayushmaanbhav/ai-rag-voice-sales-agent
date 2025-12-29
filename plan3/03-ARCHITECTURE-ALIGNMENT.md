# Plan 3: Architecture Alignment Analysis

> Comparison of ARCHITECTURE_v2.md specification vs actual implementation

---

## Overall Alignment: 30%

The implementation has **diverged significantly** from the documented architecture. Key architectural decisions have been implemented differently or not at all.

---

## Section-by-Section Analysis

### Section 5: Voice Pipeline Architecture

#### 5.1 Frame-Based Processing

**Documented:**
```
AudioFrame → FrameProcessor1 → Frame → FrameProcessor2 → Frame → Output
             (tokio spawn)     (channel)   (tokio spawn)    (channel)
```

**Actual:**
```
AudioFrame → VoicePipeline.process_audio()
             ├─ VAD.process_frame() [sequential, no spawn]
             ├─ TurnDetector.process() [sequential]
             ├─ STT.process() [sequential]
             └─ emit PipelineEvent
```

| Feature | Documented | Implemented | Gap |
|---------|------------|-------------|-----|
| FrameProcessor trait | Yes | No | CRITICAL |
| Channel-based communication | Yes | No | CRITICAL |
| Parallel processor spawn | Yes | No | CRITICAL |
| 16 Frame variants | Yes | 3 variants | HIGH |
| Processor chaining | Yes | Monolithic | CRITICAL |

**Alignment:** 10%

---

#### 5.2 Sentence Streaming

**Documented (lines 676-792):**
- LLMChunk frames emitted by LLM
- SentenceDetector accumulates chunks
- Sentence boundaries trigger TTS
- LLMToTTSStreamer for streaming

**Actual:**
- SentenceDetector EXISTS in `processors/sentence_detector.rs`
- Proper Indic terminator support (।, ॥)
- BUT: Not receiving LLMChunk frames
- NOT connected to TTS pipeline

| Feature | Documented | Implemented | Gap |
|---------|------------|-------------|-----|
| SentenceDetector | Yes | Yes | ✅ |
| Indic terminators | Yes | Yes | ✅ |
| LLMChunk frames | Yes | No | HIGH |
| LLMToTTSStreamer | Yes | No | CRITICAL |
| Sentence-to-TTS | Yes | Not wired | HIGH |

**Alignment:** 40%

---

#### 5.3 Interrupt Handling

**Documented:**
```rust
enum InterruptMode {
    Immediate,
    SentenceBoundary,
    WordBoundary,
}
```

**Actual:**
- InterruptHandler EXISTS with 3 modes
- Barge-in detection works
- BUT: Only Immediate mode used in practice

**Alignment:** 70%

---

### Section 7: Text Processing Pipeline

#### 7.1 Grammar Correction

**Documented:**
```rust
pub struct LLMGrammarCorrector {
    llm: Arc<dyn LanguageModel>,
    domain_context: DomainContext,
}
```

**Actual:**
- LLMGrammarCorrector EXISTS in `text_processing/grammar/`
- Domain context with gold_loan vocabulary
- NOT wired into voice pipeline

**Alignment:** 80%

---

#### 7.2 Translation (Translate-Think-Translate)

**Documented:**
```
User (Hindi) → Translate → LLM (English) → Translate → User (Hindi)
```

**Actual:**
```rust
TranslationProvider::Grpc => {
    tracing::warn!("gRPC translator not yet implemented");
    Box::new(NoopTranslator::new())  // ← DOES NOTHING
}
```

| Feature | Documented | Implemented | Gap |
|---------|------------|-------------|-----|
| IndicTrans2 ONNX | Yes | No | CRITICAL |
| Encoder/Decoder sessions | Yes | No | CRITICAL |
| Tokenizer | Yes | No | CRITICAL |
| gRPC fallback | Yes | Stubbed | HIGH |
| ScriptDetector | Yes | Yes | ✅ |

**Alignment:** 15%

---

#### 7.3 PII Detection

**Documented:**
```rust
pub struct HybridPIIDetector {
    regex_patterns: Vec<PIIPattern>,
    ner_model: Option<IndicNER>,
}
```

**Actual:**
- HybridPIIDetector EXISTS
- 12 regex patterns (India-specific)
- NER stub only (returns empty vec)

| Feature | Documented | Implemented | Gap |
|---------|------------|-------------|-----|
| Regex patterns | Yes | Yes (12) | ✅ |
| Aadhaar detection | Yes | Yes | ✅ |
| PAN detection | Yes | Yes | ✅ |
| IFSC detection | Yes | Yes | ✅ |
| NER model | Yes | Stub | MEDIUM |
| Aadhaar Verhoeff | Implied | No | LOW |

**Alignment:** 75%

---

#### 7.4 Compliance Checking

**Documented:**
```rust
pub struct RuleBasedComplianceChecker {
    forbidden_phrases: Vec<ForbiddenPhrase>,
    claim_disclaimers: Vec<ClaimRule>,
    rate_rules: RateRules,
}
```

**Actual:**
- RuleBasedComplianceChecker EXISTS
- 11 forbidden phrases
- Rate validation (7-24%)
- Competitor disparagement detection
- TOML-configurable rules

**Alignment:** 90%

---

### Section 8: RAG Architecture

#### 8.1 Hybrid Retrieval

**Documented:**
```rust
pub struct HybridRetriever {
    dense: DenseSearcher,
    sparse: SparseSearcher,
    reranker: CascadedReranker,
    config: RetrieverConfig,
}
```

**Actual:**
- HybridRetriever EXISTS
- Dense + Sparse + RRF fusion
- Cascaded reranking (not layer-level)
- Parallel execution for dense/sparse

**Alignment:** 85%

---

#### 8.2 Agentic RAG

**Documented:**
```rust
async fn retrieve_agentic(&self, query: &str, context: &ConversationContext) -> Vec<Document> {
    // 1. Initial retrieval
    // 2. Check sufficiency
    // 3. Rewrite query if needed
    // 4. Iterate up to 3 times
}
```

**Actual:**
- Agentic loop EXISTS in `rag/agentic.rs`
- SufficiencyChecker scores results
- QueryRewriter uses LLM
- Max 3 iterations configurable

**Alignment:** 90%

---

#### 8.3 RAG Timing Strategies

**Documented:**
```rust
enum RagTiming {
    Sequential,       // Retrieve before LLM
    PrefetchAsync,    // Start on speech detection
    ParallelInject,   // Retrieve during generation
}
```

**Actual:**
- Sequential works (default)
- `prefetch()` method EXISTS but never called
- ParallelInject skeleton only
- No stage-aware timing selection

| Feature | Documented | Implemented | Gap |
|---------|------------|-------------|-----|
| Sequential | Yes | Yes | ✅ |
| PrefetchAsync | Yes | Method exists | Not wired |
| ParallelInject | Yes | Skeleton | Not wired |
| Stage-aware timing | Yes | No | HIGH |

**Alignment:** 35%

---

#### 8.4 Context Window Management

**Documented:**
```rust
struct ContextBudget {
    greeting: 200,
    discovery: 800,
    qualification: 1500,
    presentation: 2000,
    objection_handling: 1800,
    closing: 500,
    farewell: 200,
}
```

**Actual:**
- Flat 4096 token limit
- No stage-aware budgeting
- `context_window_tokens` config exists but not used per-stage

**Alignment:** 20%

---

### Section 9: Personalization

#### 9.1 Customer Segments

**Documented:**
```rust
enum CustomerSegment {
    MsmeOwner,       // P1
    FirstTimeUser,   // P2
    ShaktiWomen,     // P3
    HighValue,       // P4
    PriceSensitive,  // P5
    TrustSeeker,     // P6
}
```

**Actual:**
- CustomerSegment enum EXISTS in core
- Persona definitions exist
- Signal detection works
- BUT: Not used in agent (no segment detection logic)

**Alignment:** 60%

---

#### 9.2 Signal Detection

**Documented:**
```rust
enum BehaviorSignal {
    Hesitation, Interest, StrongInterest, Urgency,
    Frustration, Confusion, Comparison, Skepticism,
    // ...
}
```

**Actual:**
- BehaviorSignal enum EXISTS with 12 signals
- Pattern detection with Hindi support
- Timing-aware detection
- Trend analysis

**Alignment:** 95%

---

### Section 10: Domain Configuration

**Documented:**
```
domains/
├── gold_loan/
│   ├── knowledge/
│   ├── prompts/
│   ├── compliance.toml
│   └── tools.toml
```

**Actual:**
- All config is CODE-BASED
- No domains/ directory structure
- No YAML/TOML file loading for domains
- Cannot swap domains without recompile

**Alignment:** 20%

---

### Section 11: Multilingual Support

**Documented:**
- 22 Indian languages (Eighth Schedule)
- All 11 Indic scripts
- Numeral normalization
- Multiplier words

**Actual:**
- Language enum has all 22 + English ✅
- Script detection for all 11 scripts ✅
- Numeral normalization: Devanagari only
- Multiplier words: 3 languages only (Hindi, English, partial)

**Alignment:** 65%

---

## Gap Summary Matrix

| Architecture Section | Alignment | Priority |
|---------------------|-----------|----------|
| 5.1 Frame-Based Pipeline | 10% | CRITICAL |
| 5.2 Sentence Streaming | 40% | HIGH |
| 5.3 Interrupt Handling | 70% | LOW |
| 7.1 Grammar Correction | 80% | MEDIUM |
| 7.2 Translation | 15% | CRITICAL |
| 7.3 PII Detection | 75% | LOW |
| 7.4 Compliance | 90% | LOW |
| 8.1 Hybrid Retrieval | 85% | LOW |
| 8.2 Agentic RAG | 90% | LOW |
| 8.3 RAG Timing | 35% | HIGH |
| 8.4 Context Budget | 20% | HIGH |
| 9.1 Customer Segments | 60% | MEDIUM |
| 9.2 Signal Detection | 95% | LOW |
| 10 Domain Config | 20% | MEDIUM |
| 11 Multilingual | 65% | MEDIUM |

---

## Recommended Architecture Updates

### Option A: Update Code to Match Architecture
1. Implement FrameProcessor trait
2. Refactor to channel-based pipeline
3. Complete translation (IndicTrans2)
4. Wire all timing strategies
5. Add domain YAML loading

**Effort:** 8-10 weeks

### Option B: Update Architecture to Match Code
1. Document monolithic orchestrator pattern
2. Remove frame-based claims
3. Simplify timing strategies
4. Accept code-based domain config

**Effort:** 1 week documentation

### Recommendation: Hybrid Approach
1. Keep core architecture vision
2. Add "Current Implementation" section
3. Mark planned vs implemented features
4. Create migration roadmap

---

## Key Architectural Decisions Needed

1. **Frame-Based vs Monolithic Pipeline**
   - Frame-based: More flexible, better for middleware
   - Monolithic: Simpler, already working
   - **Recommendation:** Keep monolithic for MVP, plan frame-based for v2

2. **Translation Strategy**
   - ONNX: Lower latency, offline capable
   - gRPC: Easier to implement, requires Python sidecar
   - **Recommendation:** Implement gRPC first, ONNX for v2

3. **Domain Configuration**
   - Code-based: Type-safe, IDE support
   - YAML/TOML: Easier to modify, non-dev friendly
   - **Recommendation:** Keep code-based for core, YAML for prompts/rules

4. **Stage-Aware Context**
   - Implement now: Better LLM utilization
   - Defer: Simpler implementation
   - **Recommendation:** Implement for v1, significant cost savings
