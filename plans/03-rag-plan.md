# RAG Component Plan

## Component Overview

The RAG crate handles retrieval-augmented generation:
- Hybrid retrieval (dense + sparse)
- RRF fusion
- Early-exit reranking
- Vector store abstraction

**Location**: `voice-agent-rust/crates/rag/src/`

---

## Current Status Summary

| Module | Status | Grade |
|--------|--------|-------|
| HybridRetriever | RRF fusion works, single-shot only | B |
| EarlyExitReranker | Logic exists but NEVER CALLED | F |
| SimpleEmbedder | Placeholder hash-based | D |
| VectorStore (Qdrant) | Functional, missing auth | B- |
| SparseSearch (BM25) | Works, no Hindi support | B |

---

## P0 - Critical Issues (Must Fix)

| Task | File:Line | Description |
|------|-----------|-------------|
| **Reranker not integrated** | `retriever.rs:234-255` | Uses SimpleScorer instead of EarlyExitReranker |
| **should_exit() never called** | `reranker.rs:229-255` | Early-exit logic exists but is dead code |
| **No per-layer inference** | `reranker.rs:236-254` | PABEE requires layer-by-layer execution |

---

## P1 - Important Issues

| Task | File:Line | Description |
|------|-----------|-------------|
| No parallel dense+sparse | `retriever.rs:154-162` | Sequential, should use tokio::join! |
| No agentic RAG flow | N/A | Missing intent→retrieve→sufficiency→rewrite loop |
| Prefetch not cached | `retriever.rs:258` | Results computed but not reused |
| Embedding blocks async | `embeddings.rs:87-97` | ONNX inference is synchronous |
| API key not used | `vector_store.rs:30, 103` | Config has api_key but never passed to Qdrant |
| Stemming not enabled | `sparse_search.rs:26, 72-80` | Config flag exists but not implemented |
| No Hindi analyzer | `sparse_search.rs:28` | Default English only |

---

## P2 - Nice to Have

| Task | File:Line | Description |
|------|-----------|-------------|
| Hardcoded prefetch params | `retriever.rs:264-284` | Confidence/top_k should be configurable |
| SimpleScorer too naive | `reranker.rs:366-388` | Jaccard similarity, not semantic |
| SimpleEmbedder is hash-based | `embeddings.rs:225-231` | Only for testing |
| Stats not updated | `reranker.rs:251-253` | exits_per_layer never populated |
| Hardcoded output name | `embeddings.rs:169-171` | Assumes last_hidden_state |

---

## Agentic RAG Implementation Plan

**DESIGN COMPLETE** - See [07-deep-dives.md](./07-deep-dives.md#q5-configurable-agentic-rag-architecture) for full implementation.

The architecture doc specifies but is NOT implemented:

```
┌─────────────────────────────────────────────────────────┐
│                    Agentic RAG Flow                     │
├─────────────────────────────────────────────────────────┤
│  1. Intent Classification (FAQ, Product, Complaint)     │
│  2. Initial Retrieval (Hybrid: Dense + Sparse)          │
│  3. Sufficiency Check (Cross-encoder relevance score)   │
│  4. If insufficient:                                     │
│     a. Query Rewriting (LLM-based expansion)            │
│     b. Re-retrieve with expanded query                  │
│     c. Repeat up to max_iterations                      │
│  5. Return context or escalate to human                 │
└─────────────────────────────────────────────────────────┘
```

**TODO**: Create new `agentic_retriever.rs` with:
- `AgenticRetriever` struct
- `SufficiencyChecker` using cross-encoder
- `QueryRewriter` using LLM
- Max iteration limit (default: 3)

---

## Early-Exit Reranker Fix Plan

Current state:
```rust
// reranker.rs:229-255
fn run_with_early_exit(...) -> Result<(f32, Option<usize>), RagError> {
    // Runs full model
    // NEVER calls should_exit()
    // Always returns exit_layer: None
}
```

Fix requires:
1. Export model with intermediate layer outputs (ONNX modification)
2. Process layer-by-layer with exit checks
3. Actually call `should_exit()` between layers

**Alternative**: If per-layer export not feasible, remove early-exit claims and use standard reranking.

---

## Test Coverage

| File | Tests | Quality |
|------|-------|---------|
| retriever.rs | 3 | No async tests, no reranking tests |
| reranker.rs | 3 | No ONNX tests, no early-exit tests |
| embeddings.rs | 2 | No batch tests |
| vector_store.rs | 2 | No Qdrant integration tests |
| sparse_search.rs | 2 | Good basic coverage |

---

## Implementation Priorities

### Week 1: Fix Core Issues
1. Integrate EarlyExitReranker into retriever
2. Parallelize dense + sparse search
3. Wrap embedding in spawn_blocking

### Week 2: Agentic RAG
1. Create AgenticRetriever with multi-step flow
2. Add SufficiencyChecker
3. Add QueryRewriter

### Week 3: Production Hardening
1. Add Qdrant API key support
2. Add Hindi analyzer for BM25
3. Add prefetch caching

---

*Last Updated: 2024-12-27*
