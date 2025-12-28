//! Stage-Based Dialog Management
//!
//! Manages conversation stages and transitions for gold loan sales flow.

use std::collections::HashMap;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

/// Conversation stage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[derive(Default)]
pub enum ConversationStage {
    /// Initial greeting and rapport building
    #[default]
    Greeting,
    /// Understanding customer needs
    Discovery,
    /// Assessing eligibility and readiness
    Qualification,
    /// Presenting product and benefits
    Presentation,
    /// Handling concerns and objections
    ObjectionHandling,
    /// Moving towards commitment
    Closing,
    /// Wrapping up the conversation
    Farewell,
}

impl ConversationStage {
    /// Get stage display name
    pub fn display_name(&self) -> &'static str {
        match self {
            ConversationStage::Greeting => "Greeting",
            ConversationStage::Discovery => "Discovery",
            ConversationStage::Qualification => "Qualification",
            ConversationStage::Presentation => "Presentation",
            ConversationStage::ObjectionHandling => "Objection Handling",
            ConversationStage::Closing => "Closing",
            ConversationStage::Farewell => "Farewell",
        }
    }

    /// Get guidance for this stage
    pub fn guidance(&self) -> &'static str {
        match self {
            ConversationStage::Greeting =>
                "Warmly greet the customer. Introduce yourself. Build initial rapport before discussing products.",
            ConversationStage::Discovery =>
                "Ask open questions to understand their gold loan needs. Learn about current lender, loan amount, and pain points.",
            ConversationStage::Qualification =>
                "Assess eligibility and readiness to switch. Understand timeline and decision-making process.",
            ConversationStage::Presentation =>
                "Present Kotak's gold loan benefits tailored to their needs. Focus on rate savings and trust.",
            ConversationStage::ObjectionHandling =>
                "Address concerns with empathy. Use social proof and guarantees. Don't be pushy.",
            ConversationStage::Closing =>
                "Summarize benefits and guide to next steps. Schedule appointment or capture lead.",
            ConversationStage::Farewell =>
                "Thank warmly and confirm next steps. Leave door open for future conversations.",
        }
    }

    /// Get suggested questions for this stage
    pub fn suggested_questions(&self) -> Vec<&'static str> {
        match self {
            ConversationStage::Greeting => vec![
                "How are you doing today?",
                "Is this a good time to talk?",
            ],
            ConversationStage::Discovery => vec![
                "Can you tell me about your current gold loan?",
                "What interest rate are you paying currently?",
                "How has your experience been with your current lender?",
                "What would make you consider switching?",
            ],
            ConversationStage::Qualification => vec![
                "How much gold do you have pledged currently?",
                "When does your current loan come up for renewal?",
                "Are you the primary decision maker?",
            ],
            ConversationStage::Presentation => vec![
                "Would you like to know how much you could save?",
                "Have you heard about our Switch & Save program?",
            ],
            ConversationStage::ObjectionHandling => vec![
                "What concerns do you have about switching?",
                "Is there anything holding you back?",
            ],
            ConversationStage::Closing => vec![
                "Would you like me to schedule a branch visit?",
                "Can I have someone call you with more details?",
            ],
            ConversationStage::Farewell => vec![
                "Is there anything else I can help with?",
                "Do you have any other questions?",
            ],
        }
    }

    /// Get all valid transitions from this stage
    pub fn valid_transitions(&self) -> Vec<ConversationStage> {
        match self {
            ConversationStage::Greeting => vec![
                ConversationStage::Discovery,
                ConversationStage::Farewell,
            ],
            ConversationStage::Discovery => vec![
                ConversationStage::Qualification,
                ConversationStage::Presentation,
                ConversationStage::ObjectionHandling, // P1 FIX: Customer may object early
                ConversationStage::Farewell,
            ],
            ConversationStage::Qualification => vec![
                ConversationStage::Presentation,
                ConversationStage::Discovery,
                ConversationStage::Farewell,
            ],
            ConversationStage::Presentation => vec![
                ConversationStage::ObjectionHandling,
                ConversationStage::Closing,
                ConversationStage::Farewell,
            ],
            ConversationStage::ObjectionHandling => vec![
                ConversationStage::Presentation,
                ConversationStage::Discovery, // P1 FIX: May need to revisit needs
                ConversationStage::Closing,
                ConversationStage::Farewell,
            ],
            ConversationStage::Closing => vec![
                ConversationStage::ObjectionHandling,
                ConversationStage::Farewell,
            ],
            ConversationStage::Farewell => vec![],
        }
    }
}


/// Stage transition
#[derive(Debug, Clone)]
pub struct StageTransition {
    /// From stage
    pub from: ConversationStage,
    /// To stage
    pub to: ConversationStage,
    /// Reason for transition
    pub reason: TransitionReason,
    /// Confidence in the transition
    pub confidence: f32,
}

/// Reason for stage transition
#[derive(Debug, Clone)]
pub enum TransitionReason {
    /// Intent detected that triggers transition
    IntentDetected(String),
    /// Minimum requirements met for current stage
    StageCompleted,
    /// Customer explicitly requested
    CustomerRequest,
    /// Natural conversation flow
    NaturalFlow,
    /// Timeout or stall in current stage
    Timeout,
    /// Manual override
    Manual,
}

/// Stage requirements for completion
#[derive(Debug, Clone)]
pub struct StageRequirements {
    /// Minimum turns in this stage
    pub min_turns: usize,
    /// Required information collected
    pub required_info: Vec<String>,
    /// Required intents detected
    pub required_intents: Vec<String>,
}

/// Stage manager for tracking and transitioning conversation stages
pub struct StageManager {
    current_stage: Mutex<ConversationStage>,
    stage_history: Mutex<Vec<StageTransition>>,
    stage_turns: Mutex<HashMap<ConversationStage, usize>>,
    collected_info: Mutex<HashMap<String, String>>,
    /// P0 FIX: Track detected intents for stage requirement validation
    detected_intents: Mutex<Vec<String>>,
    requirements: HashMap<ConversationStage, StageRequirements>,
}

impl StageManager {
    /// Create a new stage manager
    pub fn new() -> Self {
        Self {
            current_stage: Mutex::new(ConversationStage::Greeting),
            stage_history: Mutex::new(Vec::new()),
            stage_turns: Mutex::new(HashMap::new()),
            collected_info: Mutex::new(HashMap::new()),
            detected_intents: Mutex::new(Vec::new()),
            requirements: Self::default_requirements(),
        }
    }

    /// Get default stage requirements
    fn default_requirements() -> HashMap<ConversationStage, StageRequirements> {
        let mut req = HashMap::new();

        req.insert(ConversationStage::Greeting, StageRequirements {
            min_turns: 1,
            required_info: vec![],
            required_intents: vec![],
        });

        req.insert(ConversationStage::Discovery, StageRequirements {
            min_turns: 2,
            required_info: vec!["current_lender".into()],
            required_intents: vec![],
        });

        req.insert(ConversationStage::Qualification, StageRequirements {
            min_turns: 1,
            required_info: vec!["gold_weight".into()],
            required_intents: vec![],
        });

        req.insert(ConversationStage::Presentation, StageRequirements {
            min_turns: 1,
            required_info: vec![],
            required_intents: vec![],
        });

        req.insert(ConversationStage::ObjectionHandling, StageRequirements {
            min_turns: 1,
            required_info: vec![],
            required_intents: vec!["objection_raised".into()],
        });

        req.insert(ConversationStage::Closing, StageRequirements {
            min_turns: 1,
            required_info: vec![],
            required_intents: vec![],
        });

        req.insert(ConversationStage::Farewell, StageRequirements {
            min_turns: 1,
            required_info: vec![],
            required_intents: vec![],
        });

        req
    }

    /// Get current stage
    pub fn current(&self) -> ConversationStage {
        *self.current_stage.lock()
    }

    /// Record a turn in the current stage
    pub fn record_turn(&self) {
        let stage = self.current();
        let mut turns = self.stage_turns.lock();
        *turns.entry(stage).or_insert(0) += 1;
    }

    /// Record collected information
    pub fn record_info(&self, key: &str, value: &str) {
        self.collected_info.lock().insert(key.to_string(), value.to_string());
    }

    /// Record a detected intent
    ///
    /// P0 FIX: Tracks intents for stage requirement validation.
    pub fn record_intent(&self, intent: &str) {
        let mut intents = self.detected_intents.lock();
        if !intents.contains(&intent.to_string()) {
            intents.push(intent.to_string());
            tracing::debug!("Recorded intent: {}", intent);
        }
    }

    /// Check if a specific intent has been detected
    pub fn has_intent(&self, intent: &str) -> bool {
        self.detected_intents.lock().contains(&intent.to_string())
    }

    /// Check if current stage requirements are met
    ///
    /// P0 FIX: Now validates required_intents in addition to min_turns and required_info.
    pub fn stage_completed(&self) -> bool {
        let stage = self.current();
        let turns = self.stage_turns.lock();
        let info = self.collected_info.lock();
        let intents = self.detected_intents.lock();

        if let Some(req) = self.requirements.get(&stage) {
            // Check minimum turns
            let stage_turns = turns.get(&stage).copied().unwrap_or(0);
            if stage_turns < req.min_turns {
                return false;
            }

            // Check required info
            for key in &req.required_info {
                if !info.contains_key(key) {
                    return false;
                }
            }

            // P0 FIX: Check required intents
            for intent in &req.required_intents {
                if !intents.contains(intent) {
                    tracing::debug!(
                        "Stage {:?} incomplete: missing required intent '{}'",
                        stage,
                        intent
                    );
                    return false;
                }
            }

            true
        } else {
            true // No requirements, always completed
        }
    }

    /// Transition to a new stage
    pub fn transition(&self, to: ConversationStage, reason: TransitionReason) -> Result<StageTransition, String> {
        let from = self.current();

        // Check if transition is valid
        if !from.valid_transitions().contains(&to) && to != from {
            return Err(format!(
                "Invalid transition from {:?} to {:?}",
                from, to
            ));
        }

        let transition = StageTransition {
            from,
            to,
            reason,
            confidence: 1.0,
        };

        // Update state
        *self.current_stage.lock() = to;
        self.stage_history.lock().push(transition.clone());

        Ok(transition)
    }

    /// Suggest next stage based on current state
    pub fn suggest_next(&self) -> Option<ConversationStage> {
        let current = self.current();

        if self.stage_completed() {
            // Suggest natural next stage
            match current {
                ConversationStage::Greeting => Some(ConversationStage::Discovery),
                ConversationStage::Discovery => Some(ConversationStage::Qualification),
                ConversationStage::Qualification => Some(ConversationStage::Presentation),
                ConversationStage::Presentation => Some(ConversationStage::Closing),
                ConversationStage::ObjectionHandling => Some(ConversationStage::Presentation),
                ConversationStage::Closing => Some(ConversationStage::Farewell),
                ConversationStage::Farewell => None,
            }
        } else {
            None // Stay in current stage
        }
    }

    /// Get stage history
    pub fn history(&self) -> Vec<StageTransition> {
        self.stage_history.lock().clone()
    }

    /// Get turns in current stage
    pub fn current_stage_turns(&self) -> usize {
        let stage = self.current();
        self.stage_turns.lock().get(&stage).copied().unwrap_or(0)
    }

    /// Reset manager
    pub fn reset(&self) {
        *self.current_stage.lock() = ConversationStage::Greeting;
        self.stage_history.lock().clear();
        self.stage_turns.lock().clear();
        self.collected_info.lock().clear();
    }
}

impl Default for StageManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_transitions() {
        let manager = StageManager::new();

        assert_eq!(manager.current(), ConversationStage::Greeting);

        // Valid transition
        let result = manager.transition(ConversationStage::Discovery, TransitionReason::NaturalFlow);
        assert!(result.is_ok());
        assert_eq!(manager.current(), ConversationStage::Discovery);
    }

    #[test]
    fn test_invalid_transition() {
        let manager = StageManager::new();

        // Invalid: can't go from Greeting to Closing
        let result = manager.transition(ConversationStage::Closing, TransitionReason::Manual);
        assert!(result.is_err());
    }

    #[test]
    fn test_stage_completion() {
        let manager = StageManager::new();

        // Not completed initially
        assert!(!manager.stage_completed());

        // Record a turn
        manager.record_turn();

        // Now completed (Greeting only needs 1 turn)
        assert!(manager.stage_completed());
    }

    #[test]
    fn test_suggest_next() {
        let manager = StageManager::new();
        manager.record_turn();

        let next = manager.suggest_next();
        assert_eq!(next, Some(ConversationStage::Discovery));
    }
}
