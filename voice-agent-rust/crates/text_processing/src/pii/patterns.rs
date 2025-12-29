//! India-specific PII regex patterns

use regex::Regex;
use voice_agent_core::PIIType;
use std::collections::HashMap;
use once_cell::sync::Lazy;

/// Compiled PII patterns for India
pub static INDIAN_PII_PATTERNS: Lazy<HashMap<PIIType, CompiledPattern>> = Lazy::new(|| {
    let mut patterns = HashMap::new();

    // Aadhaar: 12 digits, often with spaces (XXXX XXXX XXXX)
    // First digit cannot be 0 or 1
    patterns.insert(
        PIIType::Aadhaar,
        CompiledPattern {
            regex: Regex::new(r"\b[2-9]\d{3}\s?\d{4}\s?\d{4}\b").unwrap(),
            description: "Aadhaar number (12 digits)",
            confidence: 0.95,
        },
    );

    // PAN: 5 letters, 4 digits, 1 letter (ABCDE1234F)
    // 4th char indicates holder type: P=Person, C=Company, etc.
    patterns.insert(
        PIIType::PAN,
        CompiledPattern {
            regex: Regex::new(r"\b[A-Z]{3}[ABCFGHLJPT][A-Z][0-9]{4}[A-Z]\b").unwrap(),
            description: "PAN card number",
            confidence: 0.98,
        },
    );

    // Indian phone: +91 or 0 followed by 10 digits starting with 6-9
    patterns.insert(
        PIIType::PhoneNumber,
        CompiledPattern {
            regex: Regex::new(r"(?:\+91[\-\s]?)?(?:0)?[6-9]\d{9}\b").unwrap(),
            description: "Indian phone number",
            confidence: 0.90,
        },
    );

    // Email
    patterns.insert(
        PIIType::Email,
        CompiledPattern {
            regex: Regex::new(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b").unwrap(),
            description: "Email address",
            confidence: 0.95,
        },
    );

    // IFSC: 4 letters, 0, 6 alphanumeric (SBIN0001234)
    patterns.insert(
        PIIType::IFSC,
        CompiledPattern {
            regex: Regex::new(r"\b[A-Z]{4}0[A-Z0-9]{6}\b").unwrap(),
            description: "IFSC code",
            confidence: 0.98,
        },
    );

    // Bank Account: 9-18 digits (varies by bank)
    // More restrictive to avoid false positives
    patterns.insert(
        PIIType::BankAccount,
        CompiledPattern {
            regex: Regex::new(r"\b\d{9,18}\b").unwrap(),
            description: "Bank account number",
            confidence: 0.60, // Lower confidence due to false positives
        },
    );

    // Voter ID (EPIC): 3 letters followed by 7 digits
    patterns.insert(
        PIIType::VoterId,
        CompiledPattern {
            regex: Regex::new(r"\b[A-Z]{3}\d{7}\b").unwrap(),
            description: "Voter ID (EPIC)",
            confidence: 0.85,
        },
    );

    // Driving License: State code (2 letters) + 13-15 alphanumeric
    // Format varies by state
    patterns.insert(
        PIIType::DrivingLicense,
        CompiledPattern {
            regex: Regex::new(r"\b[A-Z]{2}[\-\s]?\d{2}[\-\s]?\d{4}[\-\s]?\d{7}\b").unwrap(),
            description: "Driving license number",
            confidence: 0.80,
        },
    );

    // Passport: Letter + 7 digits (Indian passport)
    patterns.insert(
        PIIType::Passport,
        CompiledPattern {
            regex: Regex::new(r"\b[A-Z][0-9]{7}\b").unwrap(),
            description: "Indian passport number",
            confidence: 0.85,
        },
    );

    // UPI ID: name@upihandle
    patterns.insert(
        PIIType::UpiId,
        CompiledPattern {
            regex: Regex::new(r"\b[a-zA-Z0-9._-]+@[a-zA-Z]+\b").unwrap(),
            description: "UPI ID",
            confidence: 0.70, // Can have false positives with emails
        },
    );

    // Credit/Debit Card: 16 digits, often with spaces/dashes
    patterns.insert(
        PIIType::CardNumber,
        CompiledPattern {
            regex: Regex::new(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b").unwrap(),
            description: "Credit/Debit card number",
            confidence: 0.90,
        },
    );

    // GSTIN: 15 character alphanumeric
    patterns.insert(
        PIIType::GSTIN,
        CompiledPattern {
            regex: Regex::new(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z\d]Z[A-Z\d]\b").unwrap(),
            description: "GSTIN number",
            confidence: 0.95,
        },
    );

    patterns
});

/// Compiled regex pattern with metadata
#[derive(Debug, Clone)]
pub struct CompiledPattern {
    pub regex: Regex,
    pub description: &'static str,
    pub confidence: f32,
}

/// Indian PII patterns helper
#[derive(Debug, Clone, Default)]
pub struct IndianPIIPatterns;

impl IndianPIIPatterns {
    /// Get pattern for a PII type
    pub fn get(pii_type: PIIType) -> Option<&'static CompiledPattern> {
        INDIAN_PII_PATTERNS.get(&pii_type)
    }

    /// Find all matches of a PII type in text
    pub fn find_matches(text: &str, pii_type: PIIType) -> Vec<PatternMatch> {
        if let Some(pattern) = INDIAN_PII_PATTERNS.get(&pii_type) {
            pattern.regex
                .find_iter(text)
                .map(|m| PatternMatch {
                    pii_type,
                    text: m.as_str().to_string(),
                    start: m.start(),
                    end: m.end(),
                    confidence: pattern.confidence,
                })
                .collect()
        } else {
            vec![]
        }
    }

    /// Find all PII in text
    pub fn find_all(text: &str, types: &[PIIType]) -> Vec<PatternMatch> {
        let mut matches = Vec::new();

        for pii_type in types {
            matches.extend(Self::find_matches(text, *pii_type));
        }

        // Sort by start position
        matches.sort_by_key(|m| m.start);
        matches
    }

    /// Validate Aadhaar checksum (Verhoeff algorithm)
    pub fn validate_aadhaar(number: &str) -> bool {
        let digits: Vec<u32> = number
            .chars()
            .filter(|c| c.is_ascii_digit())
            .filter_map(|c| c.to_digit(10))
            .collect();

        if digits.len() != 12 {
            return false;
        }

        // First digit cannot be 0 or 1
        if digits[0] < 2 {
            return false;
        }

        // TODO: Implement Verhoeff checksum validation
        true
    }

    /// Validate PAN format
    pub fn validate_pan(pan: &str) -> bool {
        let pan = pan.to_uppercase();
        if pan.len() != 10 {
            return false;
        }

        // First 3 characters: letters
        // 4th character: holder type (P, C, H, A, B, G, J, L, F, T)
        // 5th character: first letter of last name
        // 6-9: digits
        // 10: check letter

        let chars: Vec<char> = pan.chars().collect();

        chars[0..3].iter().all(|c| c.is_ascii_uppercase())
            && ['P', 'C', 'H', 'A', 'B', 'G', 'J', 'L', 'F', 'T'].contains(&chars[3])
            && chars[4].is_ascii_uppercase()
            && chars[5..9].iter().all(|c| c.is_ascii_digit())
            && chars[9].is_ascii_uppercase()
    }
}

/// Pattern match result
#[derive(Debug, Clone)]
pub struct PatternMatch {
    pub pii_type: PIIType,
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aadhaar_pattern() {
        let text = "My Aadhaar is 2345 6789 0123";
        let matches = IndianPIIPatterns::find_matches(text, PIIType::Aadhaar);
        assert_eq!(matches.len(), 1);
        assert!(matches[0].text.contains("2345"));
    }

    #[test]
    fn test_pan_pattern() {
        let text = "My PAN is ABCPD1234E";
        let matches = IndianPIIPatterns::find_matches(text, PIIType::PAN);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].text, "ABCPD1234E");
    }

    #[test]
    fn test_pan_validation() {
        assert!(IndianPIIPatterns::validate_pan("ABCPD1234E"));
        assert!(!IndianPIIPatterns::validate_pan("12345")); // Too short
        assert!(!IndianPIIPatterns::validate_pan("ABCXD1234E")); // Invalid 4th char
    }

    #[test]
    fn test_phone_pattern() {
        let text = "Call me at +91 9876543210 or 9123456789";
        let matches = IndianPIIPatterns::find_matches(text, PIIType::PhoneNumber);
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_email_pattern() {
        let text = "Email: user@example.com";
        let matches = IndianPIIPatterns::find_matches(text, PIIType::Email);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].text, "user@example.com");
    }

    #[test]
    fn test_ifsc_pattern() {
        let text = "IFSC: SBIN0001234";
        let matches = IndianPIIPatterns::find_matches(text, PIIType::IFSC);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].text, "SBIN0001234");
    }

    #[test]
    fn test_find_all() {
        let text = "PAN: ABCPD1234E, Phone: 9876543210, Email: test@example.com";
        let matches = IndianPIIPatterns::find_all(
            text,
            &[PIIType::PAN, PIIType::PhoneNumber, PIIType::Email],
        );
        assert_eq!(matches.len(), 3);
    }
}
