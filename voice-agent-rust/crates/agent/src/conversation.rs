//! Conversation Management
//!
//! Manages the overall conversation flow and state.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::broadcast;
use parking_lot::Mutex;
use chrono::Utc;

use voice_agent_core::{Turn, TurnRole};
use crate::stage::{StageManager, ConversationStage, TransitionReason};
use crate::memory::{ConversationMemory, MemoryConfig, MemoryEntry};
use crate::intent::{IntentDetector, DetectedIntent};
use crate::AgentError;

/// Conversation configuration
#[derive(Debug, Clone)]
pub struct ConversationConfig {
    /// Maximum duration in seconds
    pub max_duration_seconds: u32,
    /// Session timeout in seconds
    pub session_timeout_seconds: u32,
    /// Memory config
    pub memory: MemoryConfig,
    /// Enable intent detection
    pub intent_detection: bool,
    /// Default language
    pub language: String,
}

impl Default for ConversationConfig {
    fn default() -> Self {
        Self {
            max_duration_seconds: 600, // 10 minutes
            session_timeout_seconds: 60,
            memory: MemoryConfig::default(),
            intent_detection: true,
            language: "hi".to_string(),
        }
    }
}

/// Conversation event
#[derive(Debug, Clone)]
pub enum ConversationEvent {
    /// Conversation started
    Started { session_id: String },
    /// Turn added
    TurnAdded { role: TurnRole, content: String },
    /// Intent detected
    IntentDetected(DetectedIntent),
    /// Stage changed
    StageChanged { from: ConversationStage, to: ConversationStage },
    /// Fact learned
    FactLearned { key: String, value: String },
    /// Tool called
    ToolCalled { name: String, success: bool },
    /// Conversation ended
    Ended { reason: EndReason },
    /// Error occurred
    Error(String),
}

/// Reason for conversation end
#[derive(Debug, Clone)]
pub enum EndReason {
    UserEnded,
    AgentEnded,
    Timeout,
    MaxDuration,
    Error(String),
}

/// Conversation state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversationState {
    Active,
    Paused,
    Ended,
}

/// Conversation manager
pub struct Conversation {
    /// Session ID
    session_id: String,
    /// Configuration
    config: ConversationConfig,
    /// Start time
    start_time: Instant,
    /// Last activity time
    last_activity: Mutex<Instant>,
    /// Current state
    state: Mutex<ConversationState>,
    /// Stage manager
    stage_manager: Arc<StageManager>,
    /// Memory
    memory: Arc<ConversationMemory>,
    /// Intent detector
    intent_detector: Arc<IntentDetector>,
    /// Event sender
    event_tx: broadcast::Sender<ConversationEvent>,
    /// Turn counter
    turn_count: Mutex<usize>,
}

impl Conversation {
    /// Create a new conversation
    pub fn new(session_id: impl Into<String>, config: ConversationConfig) -> Self {
        let (event_tx, _) = broadcast::channel(100);

        Self {
            session_id: session_id.into(),
            config: config.clone(),
            start_time: Instant::now(),
            last_activity: Mutex::new(Instant::now()),
            state: Mutex::new(ConversationState::Active),
            stage_manager: Arc::new(StageManager::new()),
            memory: Arc::new(ConversationMemory::new(config.memory)),
            intent_detector: Arc::new(IntentDetector::new()),
            event_tx,
            turn_count: Mutex::new(0),
        }
    }

    /// Subscribe to conversation events
    pub fn subscribe(&self) -> broadcast::Receiver<ConversationEvent> {
        self.event_tx.subscribe()
    }

    /// Get session ID
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Get current state
    pub fn state(&self) -> ConversationState {
        *self.state.lock()
    }

    /// Get current stage
    pub fn stage(&self) -> ConversationStage {
        self.stage_manager.current()
    }

    /// Get duration
    pub fn duration(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Get turn count
    pub fn turn_count(&self) -> usize {
        *self.turn_count.lock()
    }

    /// Check if conversation is active
    pub fn is_active(&self) -> bool {
        *self.state.lock() == ConversationState::Active
    }

    /// Add user turn
    pub fn add_user_turn(&self, content: &str) -> Result<DetectedIntent, AgentError> {
        self.check_active()?;
        self.update_activity();

        // Create turn
        let turn = Turn {
            role: TurnRole::User,
            content: content.to_string(),
            timestamp: Utc::now(),
            metadata: None,
        };

        // Add to memory
        let mut entry = MemoryEntry::from(&turn);
        entry.stage = Some(self.stage().display_name().to_string());

        // Detect intent
        let detected = if self.config.intent_detection {
            self.intent_detector.detect(content)
        } else {
            DetectedIntent {
                intent: "unknown".to_string(),
                confidence: 0.0,
                slots: std::collections::HashMap::new(),
                alternatives: vec![],
            }
        };

        entry.intents = vec![detected.intent.clone()];

        // Extract and store entities
        for (key, slot) in &detected.slots {
            if let Some(ref value) = slot.value {
                entry.entities.insert(key.clone(), value.clone());
                self.memory.add_fact(key, value, slot.confidence);

                let _ = self.event_tx.send(ConversationEvent::FactLearned {
                    key: key.clone(),
                    value: value.clone(),
                });
            }
        }

        self.memory.add(entry);
        self.stage_manager.record_turn();
        *self.turn_count.lock() += 1;

        // Emit events
        let _ = self.event_tx.send(ConversationEvent::TurnAdded {
            role: TurnRole::User,
            content: content.to_string(),
        });

        let _ = self.event_tx.send(ConversationEvent::IntentDetected(detected.clone()));

        // Check for stage transition triggers
        self.check_stage_transitions(&detected);

        Ok(detected)
    }

    /// Add assistant turn
    pub fn add_assistant_turn(&self, content: &str) -> Result<(), AgentError> {
        self.check_active()?;
        self.update_activity();

        let turn = Turn {
            role: TurnRole::Assistant,
            content: content.to_string(),
            timestamp: Utc::now(),
            metadata: None,
        };

        let mut entry = MemoryEntry::from(&turn);
        entry.stage = Some(self.stage().display_name().to_string());

        self.memory.add(entry);
        *self.turn_count.lock() += 1;

        let _ = self.event_tx.send(ConversationEvent::TurnAdded {
            role: TurnRole::Assistant,
            content: content.to_string(),
        });

        Ok(())
    }

    /// Transition to a new stage
    pub fn transition_stage(&self, to: ConversationStage) -> Result<(), AgentError> {
        let from = self.stage();

        match self.stage_manager.transition(to, TransitionReason::NaturalFlow) {
            Ok(_) => {
                let _ = self.event_tx.send(ConversationEvent::StageChanged { from, to });
                Ok(())
            }
            Err(e) => Err(AgentError::Stage(e)),
        }
    }

    /// Check and perform stage transitions based on intent
    ///
    /// P0 FIX: Added comprehensive intent→stage mappings for gold loan sales flow.
    /// Covers all major intents: greeting, loan_inquiry, eligibility, interest_rate,
    /// branch_locator, schedule_visit, objection, affirmative, negative, farewell, thank_you.
    fn check_stage_transitions(&self, intent: &DetectedIntent) {
        let current = self.stage();

        // Record intent for stage requirement tracking
        self.stage_manager.record_intent(&intent.intent);

        // P0 FIX: Comprehensive intent-based transitions for gold loan sales flow
        let new_stage = match intent.intent.as_str() {
            // Greeting -> Discovery: After initial greeting, move to discovery
            "greeting" if current == ConversationStage::Greeting => {
                // Only transition if we've had at least 1 turn (rapport built)
                if self.stage_manager.current_stage_turns() >= 1 {
                    Some(ConversationStage::Discovery)
                } else {
                    None
                }
            }

            // Loan inquiry / eligibility query: Move to relevant stage
            "loan_inquiry" | "eligibility_query" => {
                match current {
                    ConversationStage::Greeting => Some(ConversationStage::Discovery),
                    ConversationStage::Discovery => Some(ConversationStage::Qualification),
                    _ => None,
                }
            }

            // Interest rate query: Customer interested in rates -> Presentation
            "interest_rate_query" => {
                match current {
                    ConversationStage::Greeting | ConversationStage::Discovery => {
                        Some(ConversationStage::Presentation)
                    }
                    ConversationStage::Qualification => Some(ConversationStage::Presentation),
                    _ => None,
                }
            }

            // Competitor reference: Need to understand their situation
            "competitor_reference" => {
                match current {
                    ConversationStage::Greeting => Some(ConversationStage::Discovery),
                    _ => None, // Stay in current stage but note the competitor info
                }
            }

            // Branch locator: Ready to visit -> Closing
            "branch_locator" => {
                match current {
                    ConversationStage::Presentation => Some(ConversationStage::Closing),
                    ConversationStage::ObjectionHandling => Some(ConversationStage::Closing),
                    _ => None,
                }
            }

            // Schedule visit: Ready to book appointment -> Closing
            "schedule_visit" | "schedule_appointment" | "book_appointment" => {
                match current {
                    ConversationStage::Presentation => Some(ConversationStage::Closing),
                    ConversationStage::ObjectionHandling => Some(ConversationStage::Closing),
                    ConversationStage::Discovery => Some(ConversationStage::Closing), // Fast track
                    _ => None,
                }
            }

            // Objection: Handle objection (from any sales stage)
            "objection" if current != ConversationStage::ObjectionHandling => {
                match current {
                    ConversationStage::Discovery
                    | ConversationStage::Qualification
                    | ConversationStage::Presentation
                    | ConversationStage::Closing => Some(ConversationStage::ObjectionHandling),
                    _ => None,
                }
            }

            // Affirmative: Agreement to proceed to next stage
            "affirmative" => {
                match current {
                    ConversationStage::Greeting => Some(ConversationStage::Discovery),
                    ConversationStage::Discovery => Some(ConversationStage::Qualification),
                    ConversationStage::ObjectionHandling => Some(ConversationStage::Presentation),
                    ConversationStage::Presentation => Some(ConversationStage::Closing),
                    ConversationStage::Closing => Some(ConversationStage::Farewell),
                    _ => None,
                }
            }

            // Negative: Might need objection handling or early exit
            "negative" => {
                match current {
                    ConversationStage::Closing => Some(ConversationStage::ObjectionHandling),
                    ConversationStage::Presentation => Some(ConversationStage::ObjectionHandling),
                    _ => None, // Don't force exit on negative, handle gracefully
                }
            }

            // Thank you: Positive signal, often near end
            "thank_you" => {
                match current {
                    ConversationStage::Closing => Some(ConversationStage::Farewell),
                    ConversationStage::ObjectionHandling => Some(ConversationStage::Presentation),
                    _ => None,
                }
            }

            // Farewell: End conversation (from any stage)
            "farewell" => Some(ConversationStage::Farewell),

            // Confusion: May need to revisit discovery
            "confusion" => {
                match current {
                    ConversationStage::Presentation => Some(ConversationStage::Discovery),
                    _ => None,
                }
            }

            _ => None,
        };

        if let Some(to) = new_stage {
            if current.valid_transitions().contains(&to) {
                let _ = self.stage_manager.transition(to, TransitionReason::IntentDetected(intent.intent.clone()));
                let _ = self.event_tx.send(ConversationEvent::StageChanged {
                    from: current,
                    to,
                });
            }
        }

        // Check if we should suggest next stage
        if let Some(suggested) = self.stage_manager.suggest_next() {
            // Don't auto-transition, just note it
            tracing::debug!("Suggested next stage: {:?}", suggested);
        }
    }

    /// Get memory context
    pub fn get_context(&self) -> String {
        self.memory.get_context()
    }

    /// Get recent messages for LLM
    pub fn get_messages(&self) -> Vec<(String, String)> {
        self.memory.get_recent_messages()
    }

    /// Get stage guidance
    pub fn get_stage_guidance(&self) -> &'static str {
        self.stage().guidance()
    }

    /// Get suggested questions
    pub fn get_suggested_questions(&self) -> Vec<&'static str> {
        self.stage().suggested_questions()
    }

    /// Record a fact
    pub fn record_fact(&self, key: &str, value: &str, confidence: f32) {
        self.memory.add_fact(key, value, confidence);
        self.stage_manager.record_info(key, value);

        let _ = self.event_tx.send(ConversationEvent::FactLearned {
            key: key.to_string(),
            value: value.to_string(),
        });
    }

    /// End the conversation
    pub fn end(&self, reason: EndReason) {
        *self.state.lock() = ConversationState::Ended;
        let _ = self.event_tx.send(ConversationEvent::Ended { reason });
    }

    /// Pause the conversation
    pub fn pause(&self) {
        *self.state.lock() = ConversationState::Paused;
    }

    /// Resume the conversation
    pub fn resume(&self) {
        let mut state = self.state.lock();
        if *state == ConversationState::Paused {
            *state = ConversationState::Active;
            self.update_activity();
        }
    }

    /// Check if active
    fn check_active(&self) -> Result<(), AgentError> {
        let state = *self.state.lock();
        match state {
            ConversationState::Active => {
                // Check timeouts
                if self.duration() > Duration::from_secs(self.config.max_duration_seconds as u64) {
                    self.end(EndReason::MaxDuration);
                    return Err(AgentError::Conversation("Max duration exceeded".to_string()));
                }

                let last = *self.last_activity.lock();
                if last.elapsed() > Duration::from_secs(self.config.session_timeout_seconds as u64) {
                    self.end(EndReason::Timeout);
                    return Err(AgentError::Conversation("Session timeout".to_string()));
                }

                Ok(())
            }
            ConversationState::Paused => {
                Err(AgentError::Conversation("Conversation is paused".to_string()))
            }
            ConversationState::Ended => {
                Err(AgentError::Conversation("Conversation has ended".to_string()))
            }
        }
    }

    /// Update last activity time
    fn update_activity(&self) {
        *self.last_activity.lock() = Instant::now();
    }

    /// Get memory reference
    pub fn memory(&self) -> &ConversationMemory {
        &self.memory
    }

    /// P1 FIX: Get memory Arc for async operations
    pub fn memory_arc(&self) -> Arc<ConversationMemory> {
        Arc::clone(&self.memory)
    }

    /// Get stage manager reference
    pub fn stage_manager(&self) -> &StageManager {
        &self.stage_manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversation_creation() {
        let conv = Conversation::new("test-session", ConversationConfig::default());

        assert_eq!(conv.session_id(), "test-session");
        assert!(conv.is_active());
        assert_eq!(conv.stage(), ConversationStage::Greeting);
    }

    #[test]
    fn test_add_turns() {
        let conv = Conversation::new("test", ConversationConfig::default());

        let intent = conv.add_user_turn("Hello").unwrap();
        assert_eq!(intent.intent, "greeting");

        conv.add_assistant_turn("Hello! How can I help you?").unwrap();

        assert_eq!(conv.turn_count(), 2);
    }

    #[test]
    fn test_stage_transition() {
        let conv = Conversation::new("test", ConversationConfig::default());

        conv.transition_stage(ConversationStage::Discovery).unwrap();
        assert_eq!(conv.stage(), ConversationStage::Discovery);
    }

    #[test]
    fn test_fact_recording() {
        let conv = Conversation::new("test", ConversationConfig::default());

        conv.record_fact("customer_name", "Rajesh", 0.9);

        let fact = conv.memory().get_fact("customer_name");
        assert!(fact.is_some());
        assert_eq!(fact.unwrap().value, "Rajesh");
    }
}
