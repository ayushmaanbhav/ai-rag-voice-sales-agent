# Voice Agent Codebase Analysis Summary

## Analysis Scope

Analyzed 11 Rust crates in `/home/vscode/goldloan-study/voice-agent/backend/crates/`:

1. **core** - Traits and types
2. **config** - Configuration management
3. **pipeline** - Audio processing
4. **agent** - Conversation framework
5. **rag** - Retrieval-Augmented Generation
6. **llm** - Language model integration
7. **tools** - MCP tool interface
8. **server** - HTTP/WebSocket server
9. **transport** - WebRTC/WebSocket transport
10. **text_processing** - NLP pipeline
11. **persistence** - Database storage

---

## Key Findings by Crate

### Core (100% Complete)
- 13 well-defined traits for pluggable backends
- 40+ common types for gold loan domain
- Zero dead code
- Production-ready foundation

### Config (90% Complete)
- 10-module configuration system
- Hot-reload support via DomainConfigManager
- **Issue**: Duplicate MemoryConfig and RagConfig definitions
- **Issue**: Unused Kalosm LLM provider enum

### LLM (100% Complete)
- 3 providers: Claude, OpenAI, Ollama
- Full streaming support
- Prompt templating with stage guidance
- Native tool calling support
- Speculative execution modes

### Server/Transport (98% Complete)
- Full WebSocket with audio streaming
- WebRTC with Trickle ICE (P2 fix)
- 15+ REST endpoints
- Session management with ScyllaDB persistence
- Rate limiting, auth, metrics
- **Minor**: RedisSessionStore is deprecated stub

### RAG (90% Complete)
- HybridRetriever: Qdrant + Tantivy + RRF
- SufficiencyChecker (heuristic + LLM)
- Query rewriting
- CrossEncoderReranker (cascaded approach)
- Stage-aware context sizing
- **Missing**: ContextCompressor for history

### Agent (85% Complete)
- 7 conversation stages implemented
- StageManager with transitions
- ConversationMemory (3-tier hierarchy)
- PersuasionEngine with bilingual support
- P0-P5 fixes integrated
- **Missing**: Agent trait abstraction
- **Missing**: Grammar correction integration

### Tools (70% Complete)
- 8 domain tools (Eligibility, Savings, Branch, etc.)
- ToolRegistry with hot-reload
- Optional CRM/Calendar integration
- **Missing**: MCP JSON-RPC transport server
- **Dead**: ResourceProvider trait unused

### Pipeline (55% Complete)
- Audio pipeline works: VAD → STT → TTS
- Processor chain architecture
- SentenceDetector, TtsProcessor, InterruptHandler
- **Critical Gap**: Text processing, RAG, LLM NOT integrated
- **Unused**: speak_streaming() implemented but not called
- Architecture mismatch vs documentation

### Text Processing (60% Complete)
- PII detection (14 types + NER)
- Language detection (23 languages)
- Grammar correction (LLM-based)
- Translation (IndicTrans2)
- Compliance checking
- **Wrong Location**: Intent detection in agent crate
- **Missing**: Sentiment analysis
- **Missing**: Loan entity extraction

### Persistence (70% Complete)
- ScyllaDB backend fully implemented
- Session persistence working
- Audit logging with Merkle chain
- **Unused**: SimulatedSmsService (never called)
- **Unused**: SimulatedGoldPriceService (never called)
- **Missing**: Conversation history storage

---

## Critical Architecture Gaps

### 1. Pipeline vs Agent Separation
The documented "frame-based pipeline" architecture is NOT implemented. Instead:
- Pipeline: Only handles audio (VAD, STT, TTS)
- Agent: Handles language processing (text, RAG, LLM, tools)

This is a hybrid architecture that works but differs from documentation.

### 2. Missing Streaming Integration
`speak_streaming()` in pipeline is fully implemented but voice_session uses blocking `speak()` instead.

### 3. Unused Services
SMS and GoldPrice persistence services are complete but never wired into the tool chain.

---

## Dead Code Summary

| Location | Code | Status |
|----------|------|--------|
| server/session.rs | RedisSessionStore | Deprecated stub |
| config/agent.rs | LlmProvider::Kalosm | Unused enum |
| tools/mcp.rs | ResourceProvider trait | Never implemented |
| persistence/sms.rs | SimulatedSmsService | Never called |
| persistence/gold_price.rs | SimulatedGoldPriceService | Never called |
| pipeline/config.rs | echo_cancellation/noise_suppression | Placeholder flags |

---

## What's Working End-to-End

1. **Audio Pipeline**: VAD → STT → (agent) → TTS flows correctly
2. **Conversation Flow**: 7 stages with proper transitions
3. **RAG Retrieval**: Hybrid search with reranking works
4. **LLM Integration**: Multiple providers with streaming
5. **Tool Execution**: 8 tools callable from agent
6. **Session Persistence**: ScyllaDB storage with recovery
7. **WebSocket Server**: Real-time audio streaming
8. **WebRTC Support**: Low-latency transport option

---

## Recommended Next Steps

1. Wire `speak_streaming()` for lower latency
2. Connect SMS/GoldPrice services to tools
3. Create Agent trait for testability
4. Add sentiment analysis
5. Document actual architecture vs planned

See [FIX-PLAN.md](./FIX-PLAN.md) for detailed remediation steps.
