//! Hybrid PII detector (regex + optional NER)

use async_trait::async_trait;
use voice_agent_core::{PIIRedactor, PIIEntity, PIIType, RedactionStrategy, DetectionMethod, Result};
use super::patterns::INDIAN_PII_PATTERNS;
use std::collections::HashSet;

/// Hybrid PII detector using regex and optional NER
pub struct HybridPIIDetector {
    enabled_types: HashSet<PIIType>,
    use_ner: bool,
}

impl HybridPIIDetector {
    /// Create with specified entity types
    pub fn new(entity_names: &[String], use_ner: bool) -> Self {
        Self {
            enabled_types: parse_entity_names(entity_names),
            use_ner,
        }
    }

    /// Create with regex only
    pub fn regex_only(entity_names: &[String]) -> Self {
        Self::new(entity_names, false)
    }

    /// Detect using regex patterns
    fn detect_regex(&self, text: &str) -> Vec<PIIEntity> {
        let mut entities = Vec::new();

        for pii_type in &self.enabled_types {
            if let Some(pattern) = INDIAN_PII_PATTERNS.get(pii_type) {
                for capture in pattern.regex.find_iter(text) {
                    entities.push(PIIEntity {
                        pii_type: *pii_type,
                        text: capture.as_str().to_string(),
                        start: capture.start(),
                        end: capture.end(),
                        confidence: pattern.confidence,
                        method: DetectionMethod::Regex,
                    });
                }
            }
        }

        entities
    }

    /// Detect using NER (for names, addresses)
    async fn detect_ner(&self, _text: &str) -> Vec<PIIEntity> {
        // TODO: Implement NER-based detection
        // This would use a model like rust-bert for named entity recognition
        vec![]
    }

    /// Merge and deduplicate detections
    fn merge_detections(&self, mut entities: Vec<PIIEntity>) -> Vec<PIIEntity> {
        // Sort by start position
        entities.sort_by_key(|e| e.start);

        // Remove overlaps, keeping higher confidence
        let mut result: Vec<PIIEntity> = Vec::new();
        for entity in entities {
            if let Some(last) = result.last_mut() {
                if entity.start < last.end {
                    // Overlap - keep higher confidence
                    if entity.confidence > last.confidence {
                        *last = entity;
                    }
                    continue;
                }
            }
            result.push(entity);
        }

        result
    }

    /// Apply redaction to a single entity
    fn apply_redaction(&self, text: &str, entity: &PIIEntity, strategy: &RedactionStrategy) -> String {
        strategy.apply(text, entity.pii_type)
    }
}

#[async_trait]
impl PIIRedactor for HybridPIIDetector {
    async fn detect(&self, text: &str) -> Result<Vec<PIIEntity>> {
        let mut entities = self.detect_regex(text);

        if self.use_ner {
            entities.extend(self.detect_ner(text).await);
        }

        Ok(self.merge_detections(entities))
    }

    async fn redact(&self, text: &str, strategy: &RedactionStrategy) -> Result<String> {
        let entities = self.detect(text).await?;

        if entities.is_empty() {
            return Ok(text.to_string());
        }

        let mut result = text.to_string();

        // Apply redactions in reverse order to preserve indices
        for entity in entities.into_iter().rev() {
            let replacement = self.apply_redaction(&entity.text, &entity, strategy);
            result.replace_range(entity.start..entity.end, &replacement);
        }

        Ok(result)
    }

    fn supported_types(&self) -> &[PIIType] {
        static TYPES: &[PIIType] = &[
            PIIType::Aadhaar,
            PIIType::PAN,
            PIIType::PhoneNumber,
            PIIType::Email,
            PIIType::IFSC,
            PIIType::BankAccount,
            PIIType::VoterId,
            PIIType::DrivingLicense,
            PIIType::Passport,
            PIIType::UpiId,
            PIIType::CardNumber,
            PIIType::GSTIN,
        ];
        TYPES
    }
}

/// Parse entity names to PIIType
fn parse_entity_names(names: &[String]) -> HashSet<PIIType> {
    names
        .iter()
        .filter_map(|name| match name.to_lowercase().as_str() {
            "aadhaar" | "aadhar" => Some(PIIType::Aadhaar),
            "pan" => Some(PIIType::PAN),
            "phone" | "phonenumber" | "mobile" => Some(PIIType::PhoneNumber),
            "email" => Some(PIIType::Email),
            "ifsc" => Some(PIIType::IFSC),
            "bankaccount" | "account" => Some(PIIType::BankAccount),
            "voterid" | "epic" => Some(PIIType::VoterId),
            "drivinglicense" | "dl" => Some(PIIType::DrivingLicense),
            "passport" => Some(PIIType::Passport),
            "upi" | "upiid" => Some(PIIType::UpiId),
            "card" | "cardnumber" => Some(PIIType::CardNumber),
            "gstin" | "gst" => Some(PIIType::GSTIN),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_detect_aadhaar() {
        let detector = HybridPIIDetector::new(&["Aadhaar".to_string()], false);
        let text = "My Aadhaar is 2345 6789 0123";

        let entities = detector.detect(text).await.unwrap();
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].pii_type, PIIType::Aadhaar);
    }

    #[tokio::test]
    async fn test_detect_multiple() {
        let detector = HybridPIIDetector::new(
            &["Aadhaar".to_string(), "PAN".to_string(), "PhoneNumber".to_string()],
            false,
        );
        let text = "Aadhaar: 2345 6789 0123, PAN: ABCPD1234E, Phone: 9876543210";

        let entities = detector.detect(text).await.unwrap();
        assert_eq!(entities.len(), 3);
    }

    #[tokio::test]
    async fn test_redact_partial_mask() {
        let detector = HybridPIIDetector::new(&["PhoneNumber".to_string()], false);
        let text = "Call me at 9876543210";

        let strategy = RedactionStrategy::PartialMask {
            visible_start: 2,
            visible_end: 2,
        };
        let redacted = detector.redact(text, &strategy).await.unwrap();
        assert!(redacted.contains("98******10"));
    }

    #[tokio::test]
    async fn test_redact_type_mask() {
        let detector = HybridPIIDetector::new(&["Email".to_string()], false);
        let text = "Email: user@example.com";

        let strategy = RedactionStrategy::TypeMask;
        let redacted = detector.redact(text, &strategy).await.unwrap();
        assert!(redacted.contains("[EMAIL]"));
    }

    #[tokio::test]
    async fn test_no_pii() {
        let detector = HybridPIIDetector::new(&["Aadhaar".to_string()], false);
        let text = "Hello, how are you?";

        let entities = detector.detect(text).await.unwrap();
        assert!(entities.is_empty());
    }

    #[test]
    fn test_parse_entity_names() {
        let names = vec!["Aadhaar".to_string(), "pan".to_string(), "PHONE".to_string()];
        let types = parse_entity_names(&names);
        assert!(types.contains(&PIIType::Aadhaar));
        assert!(types.contains(&PIIType::PAN));
        assert!(types.contains(&PIIType::PhoneNumber));
    }
}
