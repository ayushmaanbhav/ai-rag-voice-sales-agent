//! Intent Detection and Slot Filling
//!
//! Detects user intents and extracts relevant entities.

use std::collections::HashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Intent definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intent {
    /// Intent name
    pub name: String,
    /// Description
    pub description: String,
    /// Required slots
    pub required_slots: Vec<String>,
    /// Optional slots
    pub optional_slots: Vec<String>,
    /// Example utterances
    pub examples: Vec<String>,
}

/// Slot/Entity definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Slot {
    /// Slot name
    pub name: String,
    /// Slot type
    pub slot_type: SlotType,
    /// Extracted value
    pub value: Option<String>,
    /// Confidence
    pub confidence: f32,
}

/// Slot types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SlotType {
    Text,
    Number,
    Currency,
    Phone,
    Date,
    Time,
    Location,
    Enum(Vec<String>),
}

/// Detected intent with slots
#[derive(Debug, Clone)]
pub struct DetectedIntent {
    /// Intent name
    pub intent: String,
    /// Confidence score
    pub confidence: f32,
    /// Extracted slots
    pub slots: HashMap<String, Slot>,
    /// Alternative intents
    pub alternatives: Vec<(String, f32)>,
}

/// Intent detector
pub struct IntentDetector {
    intents: RwLock<Vec<Intent>>,
    slot_patterns: HashMap<String, Vec<(String, String)>>, // slot_name -> (pattern, regex)
}

impl IntentDetector {
    /// Create a new intent detector with gold loan intents
    pub fn new() -> Self {
        let mut detector = Self {
            intents: RwLock::new(Vec::new()),
            slot_patterns: HashMap::new(),
        };

        detector.register_gold_loan_intents();
        detector.register_slot_patterns();

        detector
    }

    /// Register gold loan specific intents
    fn register_gold_loan_intents(&self) {
        let intents = vec![
            Intent {
                name: "loan_inquiry".to_string(),
                description: "User wants to know about gold loan".to_string(),
                required_slots: vec![],
                optional_slots: vec!["loan_amount".to_string(), "gold_weight".to_string()],
                examples: vec![
                    "I want a gold loan".to_string(),
                    "Tell me about gold loan".to_string(),
                    "Gold loan kaise milega".to_string(),
                ],
            },
            Intent {
                name: "interest_rate".to_string(),
                description: "User asking about interest rates".to_string(),
                required_slots: vec![],
                optional_slots: vec!["loan_amount".to_string()],
                examples: vec![
                    "What is the interest rate".to_string(),
                    "Interest rate kitna hai".to_string(),
                    "Rate of interest".to_string(),
                ],
            },
            Intent {
                name: "eligibility_check".to_string(),
                description: "User wants to check eligibility".to_string(),
                required_slots: vec!["gold_weight".to_string()],
                optional_slots: vec!["gold_purity".to_string()],
                examples: vec![
                    "Am I eligible".to_string(),
                    "Can I get a loan".to_string(),
                    "Kitna loan milega".to_string(),
                ],
            },
            Intent {
                name: "switch_lender".to_string(),
                description: "User wants to switch from current lender".to_string(),
                required_slots: vec!["current_lender".to_string()],
                optional_slots: vec!["current_rate".to_string(), "loan_amount".to_string()],
                examples: vec![
                    "I want to switch from Muthoot".to_string(),
                    "Transfer my loan".to_string(),
                    "Can I move my gold loan".to_string(),
                ],
            },
            Intent {
                name: "objection".to_string(),
                description: "User has concerns or objections".to_string(),
                required_slots: vec![],
                optional_slots: vec!["objection_type".to_string()],
                examples: vec![
                    "I'm not sure".to_string(),
                    "What if something goes wrong".to_string(),
                    "Is it safe".to_string(),
                    "Mujhe dar lagta hai".to_string(),
                ],
            },
            Intent {
                name: "schedule_visit".to_string(),
                description: "User wants to visit branch".to_string(),
                required_slots: vec![],
                optional_slots: vec!["location".to_string(), "date".to_string(), "time".to_string()],
                examples: vec![
                    "I want to visit".to_string(),
                    "Schedule appointment".to_string(),
                    "Kab aa sakte hain".to_string(),
                ],
            },
            Intent {
                name: "documentation".to_string(),
                description: "User asking about required documents".to_string(),
                required_slots: vec![],
                optional_slots: vec![],
                examples: vec![
                    "What documents needed".to_string(),
                    "Kya documents chahiye".to_string(),
                    "Paper work".to_string(),
                ],
            },
            Intent {
                name: "greeting".to_string(),
                description: "User greeting".to_string(),
                required_slots: vec![],
                optional_slots: vec![],
                examples: vec![
                    "Hello".to_string(),
                    "Hi".to_string(),
                    "Namaste".to_string(),
                ],
            },
            Intent {
                name: "farewell".to_string(),
                description: "User saying goodbye".to_string(),
                required_slots: vec![],
                optional_slots: vec![],
                examples: vec![
                    "Bye".to_string(),
                    "Thank you".to_string(),
                    "Dhanyavaad".to_string(),
                ],
            },
            Intent {
                name: "affirmative".to_string(),
                description: "User agreeing".to_string(),
                required_slots: vec![],
                optional_slots: vec![],
                examples: vec![
                    "Yes".to_string(),
                    "Sure".to_string(),
                    "Haan".to_string(),
                    "Okay".to_string(),
                ],
            },
            Intent {
                name: "negative".to_string(),
                description: "User declining".to_string(),
                required_slots: vec![],
                optional_slots: vec![],
                examples: vec![
                    "No".to_string(),
                    "Not now".to_string(),
                    "Nahi".to_string(),
                ],
            },
        ];

        *self.intents.write() = intents;
    }

    /// Register slot patterns
    fn register_slot_patterns(&mut self) {
        // Loan amount patterns
        self.slot_patterns.insert("loan_amount".to_string(), vec![
            ("rs_amount".to_string(), r"(?:Rs\.?|₹|INR)\s*(\d+(?:,\d+)*(?:\.\d+)?)".to_string()),
            ("lakh".to_string(), r"(\d+(?:\.\d+)?)\s*(?:lakh|lac|L)".to_string()),
            ("thousand".to_string(), r"(\d+(?:\.\d+)?)\s*(?:thousand|k|K)".to_string()),
        ]);

        // Gold weight patterns
        self.slot_patterns.insert("gold_weight".to_string(), vec![
            ("grams".to_string(), r"(\d+(?:\.\d+)?)\s*(?:grams?|gms?|g)".to_string()),
            ("tola".to_string(), r"(\d+(?:\.\d+)?)\s*(?:tola|tole)".to_string()),
        ]);

        // Phone patterns
        self.slot_patterns.insert("phone".to_string(), vec![
            ("indian".to_string(), r"(?:\+91)?[6-9]\d{9}".to_string()),
        ]);

        // Current lender patterns
        self.slot_patterns.insert("current_lender".to_string(), vec![
            ("muthoot".to_string(), r"(?i)muthoot".to_string()),
            ("manappuram".to_string(), r"(?i)manappuram".to_string()),
            ("iifl".to_string(), r"(?i)iifl|ii\s*fl".to_string()),
        ]);
    }

    /// Detect intent from text
    pub fn detect(&self, text: &str) -> DetectedIntent {
        let intents = self.intents.read();
        let text_lower = text.to_lowercase();

        let mut scores: Vec<(String, f32)> = intents
            .iter()
            .map(|intent| {
                let score = self.calculate_intent_score(&text_lower, intent);
                (intent.name.clone(), score)
            })
            .collect();

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let (best_intent, best_score) = scores.first()
            .cloned()
            .unwrap_or(("unknown".to_string(), 0.0));

        // Extract slots
        let slots = self.extract_slots(text);

        DetectedIntent {
            intent: best_intent,
            confidence: best_score,
            slots,
            alternatives: scores.into_iter().skip(1).take(3).collect(),
        }
    }

    /// Calculate intent match score
    fn calculate_intent_score(&self, text: &str, intent: &Intent) -> f32 {
        let mut score: f32 = 0.0;

        // Check examples
        for example in &intent.examples {
            let example_lower = example.to_lowercase();

            // Exact match
            if text == example_lower {
                return 1.0;
            }

            // Contains check
            if text.contains(&example_lower) {
                score = score.max(0.9);
            }

            // Word overlap
            let example_words: std::collections::HashSet<&str> = example_lower.split_whitespace().collect();
            let text_words: std::collections::HashSet<&str> = text.split_whitespace().collect();

            let overlap = example_words.intersection(&text_words).count();
            if overlap > 0 {
                let overlap_score = overlap as f32 / example_words.len().max(1) as f32;
                score = score.max(overlap_score * 0.8);
            }
        }

        score
    }

    /// Extract slots from text
    pub fn extract_slots(&self, text: &str) -> HashMap<String, Slot> {
        let mut slots = HashMap::new();

        // Simple keyword-based extraction (in production, use regex or NER)
        for slot_name in self.slot_patterns.keys() {
            if let Some(value) = self.extract_slot_value(text, slot_name) {
                slots.insert(slot_name.clone(), Slot {
                    name: slot_name.clone(),
                    slot_type: SlotType::Text,
                    value: Some(value),
                    confidence: 0.8,
                });
            }
        }

        slots
    }

    /// Extract slot value using patterns
    fn extract_slot_value(&self, text: &str, slot_name: &str) -> Option<String> {
        let text_lower = text.to_lowercase();

        match slot_name {
            "loan_amount" => {
                // Look for amount patterns
                if let Some(idx) = text_lower.find("lakh") {
                    let before: String = text_lower[..idx].chars().rev()
                        .take_while(|c| c.is_ascii_digit() || *c == '.' || c.is_whitespace())
                        .collect::<String>()
                        .chars().rev().collect();
                    let num = before.trim().parse::<f64>().ok()?;
                    return Some(format!("{}", (num * 100000.0) as i64));
                }
                None
            }
            "gold_weight" => {
                // Look for weight in grams
                for word in text_lower.split_whitespace() {
                    if let Ok(num) = word.parse::<f64>() {
                        // Check if next word is grams
                        if text_lower.contains("gram") || text_lower.contains("gm") {
                            return Some(format!("{}", num));
                        }
                    }
                }
                None
            }
            "current_lender" => {
                if text_lower.contains("muthoot") {
                    Some("Muthoot".to_string())
                } else if text_lower.contains("manappuram") {
                    Some("Manappuram".to_string())
                } else if text_lower.contains("iifl") {
                    Some("IIFL".to_string())
                } else {
                    None
                }
            }
            _ => None
        }
    }

    /// Get intent by name
    pub fn get_intent(&self, name: &str) -> Option<Intent> {
        self.intents.read()
            .iter()
            .find(|i| i.name == name)
            .cloned()
    }

    /// List all intents
    pub fn list_intents(&self) -> Vec<String> {
        self.intents.read()
            .iter()
            .map(|i| i.name.clone())
            .collect()
    }
}

impl Default for IntentDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intent_detection() {
        let detector = IntentDetector::new();

        let result = detector.detect("I want a gold loan");
        assert_eq!(result.intent, "loan_inquiry");
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_interest_rate_intent() {
        let detector = IntentDetector::new();

        let result = detector.detect("What is the interest rate");
        assert_eq!(result.intent, "interest_rate");
    }

    #[test]
    fn test_slot_extraction() {
        let detector = IntentDetector::new();

        let slots = detector.extract_slots("I have a loan from Muthoot");
        assert!(slots.contains_key("current_lender"));
        assert_eq!(slots.get("current_lender").unwrap().value, Some("Muthoot".to_string()));
    }

    #[test]
    fn test_greeting() {
        let detector = IntentDetector::new();

        let result = detector.detect("Hello");
        assert_eq!(result.intent, "greeting");
    }
}
