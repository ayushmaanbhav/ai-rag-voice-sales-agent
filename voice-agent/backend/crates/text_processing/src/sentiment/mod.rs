//! Sentiment Analysis for Voice Agent
//!
//! P2-1 FIX: Provides sentiment detection for customer conversations.
//!
//! Features:
//! - Multi-language support (Hindi, English, Hinglish)
//! - Domain-specific patterns for gold loan context
//! - Confidence scoring
//! - Frustration and satisfaction detection
//!
//! # Example
//!
//! ```
//! use voice_agent_text_processing::sentiment::{SentimentAnalyzer, Sentiment};
//!
//! let analyzer = SentimentAnalyzer::new();
//! let result = analyzer.analyze("This is very helpful, thank you!");
//!
//! assert_eq!(result.sentiment, Sentiment::Positive);
//! assert!(result.confidence > 0.7);
//! ```

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Sentiment categories for customer interactions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Sentiment {
    /// Customer is positive, interested, agreeable
    Positive,
    /// Customer is negative, unhappy, disagreeable
    Negative,
    /// Customer is neutral, informational
    #[default]
    Neutral,
    /// Customer is specifically frustrated (subset of Negative)
    Frustrated,
    /// Customer is specifically satisfied (subset of Positive)
    Satisfied,
}

impl Sentiment {
    /// Check if sentiment is generally positive
    pub fn is_positive(&self) -> bool {
        matches!(self, Sentiment::Positive | Sentiment::Satisfied)
    }

    /// Check if sentiment is generally negative
    pub fn is_negative(&self) -> bool {
        matches!(self, Sentiment::Negative | Sentiment::Frustrated)
    }

    /// Get sentiment polarity score (-1.0 to 1.0)
    pub fn polarity(&self) -> f32 {
        match self {
            Sentiment::Satisfied => 1.0,
            Sentiment::Positive => 0.5,
            Sentiment::Neutral => 0.0,
            Sentiment::Negative => -0.5,
            Sentiment::Frustrated => -1.0,
        }
    }
}

/// Result of sentiment analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentResult {
    /// Detected sentiment
    pub sentiment: Sentiment,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
    /// Matched patterns that contributed to the sentiment
    pub matched_patterns: Vec<String>,
    /// Polarity score (-1.0 to 1.0)
    pub polarity: f32,
}

impl Default for SentimentResult {
    fn default() -> Self {
        Self {
            sentiment: Sentiment::Neutral,
            confidence: 0.5,
            matched_patterns: Vec::new(),
            polarity: 0.0,
        }
    }
}

/// Configuration for sentiment analyzer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentConfig {
    /// Minimum confidence threshold for non-neutral classification
    pub confidence_threshold: f32,
    /// Enable Hindi/Hinglish pattern matching
    pub enable_hindi: bool,
    /// Enable domain-specific (gold loan) patterns
    pub enable_domain_patterns: bool,
}

impl Default for SentimentConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.3,
            enable_hindi: true,
            enable_domain_patterns: true,
        }
    }
}

// ============================================================================
// Pattern definitions
// ============================================================================

// Positive patterns (English)
static POSITIVE_ENGLISH: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        "thank you",
        "thanks",
        "great",
        "good",
        "excellent",
        "perfect",
        "wonderful",
        "amazing",
        "helpful",
        "appreciate",
        "happy",
        "glad",
        "pleased",
        "satisfied",
        "love",
        "like",
        "yes",
        "sure",
        "okay",
        "sounds good",
        "that's great",
        "works for me",
        "i agree",
        "interested",
        "tell me more",
        "go ahead",
        "proceed",
    ]
    .into_iter()
    .collect()
});

// Positive patterns (Hindi/Hinglish)
static POSITIVE_HINDI: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        "धन्यवाद",
        "शुक्रिया",
        "बहुत अच्छा",
        "अच्छा",
        "बढ़िया",
        "शानदार",
        "हाँ",
        "हां",
        "जी हाँ",
        "जी",
        "ठीक है",
        "ठीक",
        "चलो",
        "मुझे पसंद",
        "खुश",
        "संतुष्ट",
        "shukriya",
        "bahut accha",
        "accha",
        "badhiya",
        "haan",
        "ji",
        "theek hai",
        "theek",
        "mujhe pasand",
        "khush",
        "interested hun",
        "batao",
    ]
    .into_iter()
    .collect()
});

// Negative patterns (English)
static NEGATIVE_ENGLISH: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        "no",
        "not interested",
        "don't want",
        "bad",
        "poor",
        "terrible",
        "awful",
        "horrible",
        "disappointed",
        "unhappy",
        "angry",
        "upset",
        "annoyed",
        "frustrated",
        "confused",
        "don't understand",
        "waste of time",
        "not helpful",
        "useless",
        "wrong",
        "incorrect",
        "stop",
        "cancel",
        "end",
        "quit",
        "bye",
        "goodbye",
    ]
    .into_iter()
    .collect()
});

// Negative patterns (Hindi/Hinglish)
static NEGATIVE_HINDI: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        "नहीं",
        "नही",
        "मुझे नहीं चाहिए",
        "बुरा",
        "खराब",
        "गलत",
        "निराश",
        "नाराज़",
        "परेशान",
        "समझ नहीं आया",
        "बंद करो",
        "रुको",
        "छोड़ो",
        "nahi",
        "nahi chahiye",
        "bura",
        "kharab",
        "galat",
        "nirash",
        "naraz",
        "pareshan",
        "samajh nahi aaya",
        "band karo",
        "ruko",
        "chodo",
        "time waste",
    ]
    .into_iter()
    .collect()
});

// Frustration patterns (stronger negative signals)
// These are stored as (pattern, score_weight) pairs
static FRUSTRATION_PATTERNS: Lazy<Vec<(Regex, f32)>> = Lazy::new(|| {
    vec![
        // English frustration (standard weight)
        (
            Regex::new(r"(?i)\b(frustrat|irritat|annoy|angry|furious|fed up)\w*\b").unwrap(),
            0.7,
        ),
        (
            Regex::new(r"(?i)\b(waste\s+(of\s+)?(my\s+)?time)\b").unwrap(),
            0.5,
        ),
        (
            Regex::new(r"(?i)\b(not\s+listening|don'?t\s+understand|how\s+many\s+times)\b")
                .unwrap(),
            0.5,
        ),
        // Escalation patterns - wanting to talk to human (high weight - strong frustration signal)
        (
            Regex::new(r"(?i)(talk|speak|connect).*(human|person|real)").unwrap(),
            0.7,
        ),
        (
            Regex::new(r"(?i)(want|need).*(human|person|agent|manager)").unwrap(),
            0.7,
        ),
        (
            Regex::new(r"(?i)\b(this\s+is\s+(so\s+)?(stupid|useless|ridiculous))\b").unwrap(),
            0.7,
        ),
        // Hindi frustration
        (
            Regex::new(r"(?i)(परेशान|गुस्सा|नाराज़|तंग|थक गया|बोर)").unwrap(),
            0.7,
        ),
        (
            Regex::new(r"(?i)(समझ\s+में\s+नहीं|क्या\s+बकवास|कितनी\s+बार)").unwrap(),
            0.6,
        ),
        (Regex::new(r"(?i)insaan\s*se\s*baat").unwrap(), 0.7),
        (
            Regex::new(r"(?i)(aadmi\s+se\s+connect|manager\s+se)").unwrap(),
            0.7,
        ),
    ]
});

// Satisfaction patterns (stronger positive signals)
static SATISFACTION_PATTERNS: Lazy<Vec<Regex>> = Lazy::new(|| {
    vec![
        // English satisfaction
        Regex::new(r"(?i)\b(very\s+help(ful|ed)|really\s+(good|great|helpful))\b").unwrap(),
        Regex::new(r"(?i)\b(exactly\s+what\s+i\s+(need|want)ed?)\b").unwrap(),
        Regex::new(r"(?i)\b(thank\s+you\s+(so\s+much|very\s+much))\b").unwrap(),
        Regex::new(r"(?i)\b(excellent|perfect|wonderful|amazing)\s+(service|help|information)\b")
            .unwrap(),
        Regex::new(r"(?i)\b(i('m|\s+am)\s+(very\s+)?(happy|pleased|satisfied))\b").unwrap(),
        // Hindi satisfaction
        Regex::new(r"(?i)(बहुत\s+(अच्छा|बढ़िया|शुक्रिया|धन्यवाद))").unwrap(),
        Regex::new(r"(?i)(bahut\s+(accha|badhiya|shukriya|helpful))").unwrap(),
        Regex::new(r"(?i)(yahi\s+chahiye\s+tha|exactly|perfect)").unwrap(),
    ]
});

// Domain-specific positive patterns (gold loan context)
static DOMAIN_POSITIVE: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        "good rate",
        "better rate",
        "lower interest",
        "quick process",
        "easy documentation",
        "nearby branch",
        "same day",
        "achcha rate",
        "kam interest",
        "jaldi",
        "aasan",
    ]
    .into_iter()
    .collect()
});

// Domain-specific negative patterns (gold loan context)
static DOMAIN_NEGATIVE: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        "high interest",
        "too expensive",
        "too far",
        "complicated",
        "too much documentation",
        "hidden charges",
        "not safe",
        "zyada interest",
        "bahut door",
        "mushkil",
        "safe nahi",
    ]
    .into_iter()
    .collect()
});

/// Sentiment Analyzer for customer conversations
pub struct SentimentAnalyzer {
    config: SentimentConfig,
}

impl SentimentAnalyzer {
    /// Create a new sentiment analyzer with default configuration
    pub fn new() -> Self {
        Self {
            config: SentimentConfig::default(),
        }
    }

    /// Create a sentiment analyzer with custom configuration
    pub fn with_config(config: SentimentConfig) -> Self {
        Self { config }
    }

    /// Analyze text and return sentiment result
    pub fn analyze(&self, text: &str) -> SentimentResult {
        let text_lower = text.to_lowercase();
        let mut matched_patterns = Vec::new();

        // Check for strong signals first (frustration/satisfaction)
        let frustration_score = self.check_frustration(&text_lower, &mut matched_patterns);
        if frustration_score > 0.6 {
            return SentimentResult {
                sentiment: Sentiment::Frustrated,
                confidence: frustration_score,
                matched_patterns,
                polarity: Sentiment::Frustrated.polarity(),
            };
        }

        let satisfaction_score = self.check_satisfaction(&text_lower, &mut matched_patterns);
        if satisfaction_score > 0.6 {
            return SentimentResult {
                sentiment: Sentiment::Satisfied,
                confidence: satisfaction_score,
                matched_patterns,
                polarity: Sentiment::Satisfied.polarity(),
            };
        }

        // Check general positive/negative patterns
        let (positive_score, positive_matches) = self.count_matches(&text_lower, true);
        let (negative_score, negative_matches) = self.count_matches(&text_lower, false);

        matched_patterns.extend(positive_matches);
        matched_patterns.extend(negative_matches);

        // Determine sentiment based on scores
        let total = positive_score + negative_score;
        if total < self.config.confidence_threshold {
            return SentimentResult {
                sentiment: Sentiment::Neutral,
                confidence: 0.5,
                matched_patterns,
                polarity: 0.0,
            };
        }

        let positive_ratio = positive_score / total.max(0.001);

        if positive_ratio > 0.6 {
            let confidence = (positive_ratio * 0.7 + 0.3).min(0.95);
            SentimentResult {
                sentiment: Sentiment::Positive,
                confidence,
                matched_patterns,
                polarity: Sentiment::Positive.polarity(),
            }
        } else if positive_ratio < 0.4 {
            let confidence = ((1.0 - positive_ratio) * 0.7 + 0.3).min(0.95);
            SentimentResult {
                sentiment: Sentiment::Negative,
                confidence,
                matched_patterns,
                polarity: Sentiment::Negative.polarity(),
            }
        } else {
            SentimentResult {
                sentiment: Sentiment::Neutral,
                confidence: 0.5,
                matched_patterns,
                polarity: 0.0,
            }
        }
    }

    /// Check for frustration patterns
    fn check_frustration(&self, text: &str, matched: &mut Vec<String>) -> f32 {
        let mut score: f32 = 0.0;

        for (pattern, weight) in FRUSTRATION_PATTERNS.iter() {
            if let Some(m) = pattern.find(text) {
                matched.push(format!("frustration:{}", m.as_str()));
                score += weight;
            }
        }

        score.min(1.0)
    }

    /// Check for satisfaction patterns
    fn check_satisfaction(&self, text: &str, matched: &mut Vec<String>) -> f32 {
        let mut score: f32 = 0.0;

        for pattern in SATISFACTION_PATTERNS.iter() {
            if let Some(m) = pattern.find(text) {
                matched.push(format!("satisfaction:{}", m.as_str()));
                score += 0.4;
            }
        }

        score.min(1.0)
    }

    /// Count pattern matches and return score
    fn count_matches(&self, text: &str, positive: bool) -> (f32, Vec<String>) {
        let mut score = 0.0;
        let mut matched = Vec::new();

        // English patterns
        let english_set = if positive {
            &*POSITIVE_ENGLISH
        } else {
            &*NEGATIVE_ENGLISH
        };
        for pattern in english_set.iter() {
            if text.contains(pattern) {
                matched.push(format!(
                    "{}:{}",
                    if positive { "pos" } else { "neg" },
                    pattern
                ));
                score += 0.2;
            }
        }

        // Hindi patterns
        if self.config.enable_hindi {
            let hindi_set = if positive {
                &*POSITIVE_HINDI
            } else {
                &*NEGATIVE_HINDI
            };
            for pattern in hindi_set.iter() {
                if text.contains(pattern) {
                    matched.push(format!(
                        "{}:{}",
                        if positive { "pos_hi" } else { "neg_hi" },
                        pattern
                    ));
                    score += 0.2;
                }
            }
        }

        // Domain patterns
        if self.config.enable_domain_patterns {
            let domain_set = if positive {
                &*DOMAIN_POSITIVE
            } else {
                &*DOMAIN_NEGATIVE
            };
            for pattern in domain_set.iter() {
                if text.contains(pattern) {
                    matched.push(format!(
                        "{}:{}",
                        if positive { "domain_pos" } else { "domain_neg" },
                        pattern
                    ));
                    score += 0.25;
                }
            }
        }

        (score, matched)
    }

    /// Analyze a batch of texts and return aggregate sentiment
    pub fn analyze_conversation(&self, texts: &[&str]) -> SentimentResult {
        if texts.is_empty() {
            return SentimentResult::default();
        }

        let results: Vec<_> = texts.iter().map(|t| self.analyze(t)).collect();

        // Weight recent messages more heavily
        let mut weighted_polarity = 0.0;
        let mut total_weight = 0.0;
        let mut all_patterns = Vec::new();

        for (i, result) in results.iter().enumerate() {
            let weight = 1.0 + (i as f32 * 0.5); // More recent = higher weight
            weighted_polarity += result.polarity * weight;
            total_weight += weight;
            all_patterns.extend(result.matched_patterns.clone());
        }

        let avg_polarity = weighted_polarity / total_weight;

        let sentiment = if avg_polarity > 0.5 {
            Sentiment::Satisfied
        } else if avg_polarity > 0.2 {
            Sentiment::Positive
        } else if avg_polarity < -0.5 {
            Sentiment::Frustrated
        } else if avg_polarity < -0.2 {
            Sentiment::Negative
        } else {
            Sentiment::Neutral
        };

        let confidence = (avg_polarity.abs() * 0.7 + 0.3).min(0.95);

        SentimentResult {
            sentiment,
            confidence,
            matched_patterns: all_patterns,
            polarity: avg_polarity,
        }
    }
}

impl Default for SentimentAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positive_english() {
        let analyzer = SentimentAnalyzer::new();
        let result = analyzer.analyze("Thank you, this is very helpful!");
        assert!(result.sentiment.is_positive());
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_negative_english() {
        let analyzer = SentimentAnalyzer::new();
        let result = analyzer.analyze("I'm not interested, this is useless");
        assert!(result.sentiment.is_negative());
    }

    #[test]
    fn test_neutral() {
        let analyzer = SentimentAnalyzer::new();
        let result = analyzer.analyze("What is the interest rate?");
        assert_eq!(result.sentiment, Sentiment::Neutral);
    }

    #[test]
    fn test_frustrated() {
        let analyzer = SentimentAnalyzer::new();
        let result = analyzer.analyze("This is so frustrating! I want to talk to a human!");
        assert_eq!(result.sentiment, Sentiment::Frustrated);
        assert!(result.confidence > 0.6);
    }

    #[test]
    fn test_satisfied() {
        let analyzer = SentimentAnalyzer::new();
        let result = analyzer.analyze("Thank you so much! This was exactly what I needed!");
        assert_eq!(result.sentiment, Sentiment::Satisfied);
    }

    #[test]
    fn test_hindi_positive() {
        let analyzer = SentimentAnalyzer::new();
        let result = analyzer.analyze("बहुत अच्छा, धन्यवाद!");
        assert!(result.sentiment.is_positive());
    }

    #[test]
    fn test_hindi_negative() {
        let analyzer = SentimentAnalyzer::new();
        let result = analyzer.analyze("नहीं चाहिए, यह गलत है");
        assert!(result.sentiment.is_negative());
    }

    #[test]
    fn test_hinglish() {
        let analyzer = SentimentAnalyzer::new();
        let result = analyzer.analyze("bahut accha hai, interested hun");
        assert!(result.sentiment.is_positive());
    }

    #[test]
    fn test_domain_positive() {
        let analyzer = SentimentAnalyzer::new();
        let result = analyzer.analyze("good rate and quick process");
        assert!(result.sentiment.is_positive());
    }

    #[test]
    fn test_domain_negative() {
        let analyzer = SentimentAnalyzer::new();
        let result = analyzer.analyze("too expensive and hidden charges");
        assert!(result.sentiment.is_negative());
    }

    #[test]
    fn test_polarity() {
        assert_eq!(Sentiment::Satisfied.polarity(), 1.0);
        assert_eq!(Sentiment::Positive.polarity(), 0.5);
        assert_eq!(Sentiment::Neutral.polarity(), 0.0);
        assert_eq!(Sentiment::Negative.polarity(), -0.5);
        assert_eq!(Sentiment::Frustrated.polarity(), -1.0);
    }

    #[test]
    fn test_conversation_analysis() {
        let analyzer = SentimentAnalyzer::new();
        let texts = vec![
            "What is the interest rate?",   // Neutral
            "That seems high",              // Slight negative
            "Actually, that's a good rate", // Positive
            "Thank you, I'm interested!",   // Positive
        ];
        let result = analyzer.analyze_conversation(&texts.iter().map(|s| *s).collect::<Vec<_>>());
        assert!(result.sentiment.is_positive());
    }

    #[test]
    fn test_escalation_detection() {
        let analyzer = SentimentAnalyzer::new();

        // English escalation
        let result = analyzer.analyze("I want to talk to a human agent");
        assert_eq!(result.sentiment, Sentiment::Frustrated);

        // Hindi escalation
        let result = analyzer.analyze("insaan se baat karo");
        assert_eq!(result.sentiment, Sentiment::Frustrated);
    }
}
