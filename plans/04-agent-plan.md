# Agent Component Plan

## Component Overview

The agent crate handles conversation logic:
- Conversation state machine
- Intent detection and slot filling
- Stage-based flow
- Memory management

**Location**: `voice-agent-rust/crates/agent/src/`

---

## Current Status Summary

| Module | Status | Grade |
|--------|--------|-------|
| GoldLoanAgent | Event-driven, mock fallback | B |
| Conversation | Stage FSM works | B |
| Intent Detection | Keyword-based, regex unused | C |
| Memory | Summarization is fake | D |
| Stage Transitions | Missing some valid paths | B- |

---

## P0 - Critical Issues (Must Fix)

| Task | File:Line | Description |
|------|-----------|-------------|
| **Slot patterns never used** | `intent.rs:217-239` | Regex patterns defined but extract_slot_value uses hardcoded matching |
| **Memory summarization is fake** | `memory.rs:137-141` | Just trims entries, no LLM summarization |
| **No Devanagari support** | `intent.rs` | Hindi in Devanagari script will fail intent detection |
| **Duplicate PersonaTraits** | `agent.rs:45-63` | Defined in agent, config, AND llm crates |

---

## P1 - Important Issues

| Task | File:Line | Description |
|------|-----------|-------------|
| Missing FSM transitions | `stage.rs:108-112, 123-127` | Discovery→ObjectionHandling, ObjectionHandling→Discovery |
| required_intents not checked | `stage.rs:267-290` | Collected but never validated in stage_completed() |
| Incomplete stage mappings | `conversation.rs:263-296` | Many intent→stage transitions missing |
| Hardcoded tool defaults | `agent.rs:229-249` | City "Mumbai", purity "22K" should come from profile |
| SlotType always Text | `intent.rs:312-318` | Ignores actual slot type definition |
| Hardcoded slot confidence | `intent.rs:317` | Always 0.8, should vary |
| No RAG integration | `agent.rs` | rag_enabled flag exists but unused |
| Stage guidance mismatch | `agent.rs:312-314` | Uses display_name but expects lowercase |

---

## P2 - Nice to Have

| Task | File:Line | Description |
|------|-----------|-------------|
| No Hindi word tokenization | `intent.rs:274-303` | split_whitespace doesn't handle Hindi |
| Fragile amount extraction | `intent.rs:329-339` | Only handles "lakh" suffix |
| Poor episodic summaries | `memory.rs:159-168` | Truncates at 50 chars mid-word |
| Mock responses ignore language | `agent.rs:367-405` | Always Hinglish even if config.language is "en" |
| No multi-turn slot filling | N/A | Can't ask follow-up for missing slots |

---

## Slot Extraction Fix Plan

Current (broken):
```rust
// intent.rs:325-366 - Hardcoded keyword matching
fn extract_slot_value(&self, slot_name: &str, text: &str) -> Option<String> {
    match slot_name {
        "loan_amount" => { /* looks for "lakh" */ }
        "gold_weight" => { /* looks for "gram" */ }
        // ...
    }
}
```

Fix - use the defined patterns:
```rust
// intent.rs:217-239 - These patterns exist but aren't used!
slot_patterns.insert("loan_amount", vec![
    (r"(\d+)\s*(?:lakh|lac)", "$1"),
    (r"(?:Rs\.?|₹)\s*(\d+)", "$1"),
]);

// New implementation
fn extract_slot_value(&self, slot_name: &str, text: &str) -> Option<String> {
    if let Some(patterns) = self.slot_patterns.get(slot_name) {
        for (pattern, replacement) in patterns {
            let re = Regex::new(pattern).ok()?;
            if let Some(caps) = re.captures(text) {
                // Use captures to build value
            }
        }
    }
    None
}
```

---

## Memory Summarization Fix

Current (fake):
```rust
// memory.rs:137-141
fn summarize_if_needed(&self) {
    // Just removes old entries, no summarization!
    if turns.len() > MAX_WORKING_MEMORY {
        turns.drain(0..turns.len() - MAX_WORKING_MEMORY);
    }
}
```

Fix - actual LLM summarization:
```rust
async fn summarize_if_needed(&self, llm: &dyn LlmBackend) {
    if turns.len() > MAX_WORKING_MEMORY {
        let to_summarize: Vec<_> = turns.drain(0..turns.len() - MAX_WORKING_MEMORY/2).collect();
        let summary = llm.generate(&[
            Message::system("Summarize this conversation segment in 2-3 sentences, preserving key facts about the customer."),
            Message::user(&format_turns(&to_summarize)),
        ]).await?;
        self.add_episodic_memory("conversation_summary", &summary.text);
    }
}
```

---

## Test Coverage

| File | Tests | Coverage |
|------|-------|----------|
| stage.rs | 4 | ~40% - missing transition validation |
| intent.rs | 4 | ~30% - no Hindi, no slot edge cases |
| memory.rs | 3 | ~30% - no summarization tests |
| conversation.rs | 4 | ~25% - no concurrent access |
| agent.rs | 3 | ~20% - no tool calling, no LLM |

---

## Implementation Priorities

### Week 1: Fix Slot Extraction
1. Compile regex patterns from slot_patterns
2. Use patterns in extract_slot_value
3. Add Hindi numeral conversion

### Week 2: Memory & FSM
1. Implement actual LLM summarization
2. Add missing stage transitions
3. Validate required_intents in stage_completed

### Week 3: Hindi Support
1. Add Devanagari script handling
2. Add transliteration support
3. Test with Hindi inputs

---

*Last Updated: 2024-12-27*
