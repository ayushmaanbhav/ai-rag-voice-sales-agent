//! P2-5 FIX: Loan Entity Extraction
//!
//! Extracts loan-specific entities from text:
//! - Loan amounts (with lakh/crore support)
//! - Gold weight (grams, tola)
//! - Interest rates (percentage)
//! - Tenures (months, years)
//! - Customer names
//!
//! # Example
//!
//! ```ignore
//! use voice_agent_text_processing::entities::LoanEntityExtractor;
//!
//! let extractor = LoanEntityExtractor::new();
//! let entities = extractor.extract("I want 5 lakh loan for 12 months at 10% interest");
//!
//! assert_eq!(entities.amount, Some(Currency { value: 500000, unit: "INR" }));
//! assert_eq!(entities.tenure, Some(Duration { value: 12, unit: "months" }));
//! assert_eq!(entities.rate, Some(Percentage { value: 10.0 }));
//! ```

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};

/// Currency value extracted from text
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Currency {
    /// Amount in base units (paise for INR)
    pub value: i64,
    /// Currency code (default: INR)
    pub unit: String,
    /// Original text span
    pub text: String,
}

impl Currency {
    /// Format as rupees string
    pub fn as_rupees(&self) -> String {
        let rupees = self.value / 100;
        format!("₹{}", rupees)
    }

    /// Get value in rupees
    pub fn rupees(&self) -> f64 {
        self.value as f64 / 100.0
    }
}

/// Weight value extracted from text
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Weight {
    /// Weight in milligrams
    pub value_mg: i64,
    /// Original unit (grams, tola, etc.)
    pub unit: String,
    /// Original text span
    pub text: String,
}

impl Weight {
    /// Get weight in grams
    pub fn grams(&self) -> f64 {
        self.value_mg as f64 / 1000.0
    }

    /// Get weight in tola
    pub fn tola(&self) -> f64 {
        self.grams() / 11.66
    }
}

/// Percentage value extracted from text
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Percentage {
    /// Percentage value (e.g., 10.5 for 10.5%)
    pub value: f64,
    /// Original text span
    pub text: String,
}

/// Duration value extracted from text
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Duration {
    /// Duration in days
    pub days: i32,
    /// Original unit (months, years, days)
    pub unit: String,
    /// Original text span
    pub text: String,
}

impl Duration {
    /// Get duration in months
    pub fn months(&self) -> f64 {
        self.days as f64 / 30.0
    }

    /// Get duration in years
    pub fn years(&self) -> f64 {
        self.days as f64 / 365.0
    }
}

/// All entities extracted from text
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LoanEntities {
    /// Loan amount
    pub amount: Option<Currency>,
    /// Gold weight
    pub gold_weight: Option<Weight>,
    /// Interest rate
    pub interest_rate: Option<Percentage>,
    /// Loan tenure
    pub tenure: Option<Duration>,
    /// Customer name (if mentioned)
    pub customer_name: Option<String>,
    /// Gold purity (karat)
    pub gold_purity: Option<u8>,
    /// Current lender (for balance transfer)
    pub current_lender: Option<String>,
}

impl LoanEntities {
    /// Check if any entities were extracted
    pub fn is_empty(&self) -> bool {
        self.amount.is_none()
            && self.gold_weight.is_none()
            && self.interest_rate.is_none()
            && self.tenure.is_none()
            && self.customer_name.is_none()
            && self.gold_purity.is_none()
            && self.current_lender.is_none()
    }

    /// Merge with another LoanEntities, preferring non-None values from other
    pub fn merge(&mut self, other: &LoanEntities) {
        if other.amount.is_some() {
            self.amount = other.amount.clone();
        }
        if other.gold_weight.is_some() {
            self.gold_weight = other.gold_weight.clone();
        }
        if other.interest_rate.is_some() {
            self.interest_rate = other.interest_rate.clone();
        }
        if other.tenure.is_some() {
            self.tenure = other.tenure.clone();
        }
        if other.customer_name.is_some() {
            self.customer_name = other.customer_name.clone();
        }
        if other.gold_purity.is_some() {
            self.gold_purity = other.gold_purity;
        }
        if other.current_lender.is_some() {
            self.current_lender = other.current_lender.clone();
        }
    }
}

// Compiled regex patterns
static AMOUNT_PATTERN: Lazy<Regex> = Lazy::new(|| {
    // P2-5 FIX: Use word boundaries to avoid matching "l" in "loan" as "lakh"
    Regex::new(r"(?i)(?:rs\.?|rupees?|₹|inr)?\s*(\d+(?:\.\d+)?)\s*\b(lakh|lac|lakhs?|crore|crores?|hazar|hazaar|thousand|k\b|l\b|cr\b)?\b?(?:\s*(?:rs\.?|rupees?|₹|inr))?").unwrap()
});

static HINDI_AMOUNT_PATTERN: Lazy<Regex> = Lazy::new(|| {
    // Hindi number words
    Regex::new(r"(?i)(एक|दो|तीन|चार|पांच|पाँच|छह|छः|सात|आठ|नौ|दस|बीस|तीस|चालीस|पचास|साठ|सत्तर|अस्सी|नब्बे|सौ)\s*(लाख|करोड़|हज़ार|हजार)?").unwrap()
});

static WEIGHT_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(\d+(?:\.\d+)?)\s*(gram|grams?|gm|g|tola|tolas?|kg|kilogram)s?").unwrap()
});

static RATE_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)(\d+(?:\.\d+)?)\s*(?:%|percent|प्रतिशत|prतिshat)").unwrap());

static TENURE_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(\d+)\s*(month|months?|year|years?|yr|yrs?|day|days?|mahine?|saal)s?").unwrap()
});

static PURITY_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)(\d{1,2})\s*(?:k|karat|carat|kt)").unwrap());

static NAME_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(?:my\s+name\s+is|i\s+am|mera\s+naam|मेरा\s+नाम)\s+([A-Za-z\u0900-\u097F]+(?:\s+[A-Za-z\u0900-\u097F]+)?)").unwrap()
});

static LENDER_PATTERNS: Lazy<Vec<(&'static str, Regex)>> = Lazy::new(|| {
    vec![
        ("Muthoot", Regex::new(r"(?i)muthoot").unwrap()),
        ("Manappuram", Regex::new(r"(?i)manappuram").unwrap()),
        ("IIFL", Regex::new(r"(?i)iifl").unwrap()),
        ("HDFC", Regex::new(r"(?i)hdfc").unwrap()),
        ("SBI", Regex::new(r"(?i)\bsbi\b").unwrap()),
        ("ICICI", Regex::new(r"(?i)icici").unwrap()),
        ("Axis", Regex::new(r"(?i)axis").unwrap()),
    ]
});

/// Loan entity extractor
pub struct LoanEntityExtractor {
    /// Whether to extract Hindi/Devanagari numbers
    pub support_hindi: bool,
}

impl Default for LoanEntityExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl LoanEntityExtractor {
    /// Create a new extractor with default settings
    pub fn new() -> Self {
        Self {
            support_hindi: true,
        }
    }

    /// Extract all loan entities from text
    pub fn extract(&self, text: &str) -> LoanEntities {
        LoanEntities {
            amount: self.extract_amount(text),
            gold_weight: self.extract_weight(text),
            interest_rate: self.extract_rate(text),
            tenure: self.extract_tenure(text),
            customer_name: self.extract_name(text),
            gold_purity: self.extract_purity(text),
            current_lender: self.extract_lender(text),
        }
    }

    /// Extract loan amount
    pub fn extract_amount(&self, text: &str) -> Option<Currency> {
        // Try English pattern first
        if let Some(caps) = AMOUNT_PATTERN.captures(text) {
            let num_str = caps.get(1)?.as_str();
            let multiplier_str = caps.get(2).map(|m| m.as_str().to_lowercase());

            let base: f64 = num_str.parse().ok()?;
            let multiplier = match multiplier_str.as_deref() {
                Some("lakh") | Some("lac") | Some("lakhs") | Some("l") => 100_000.0,
                Some("crore") | Some("crores") | Some("cr") => 10_000_000.0,
                Some("hazar") | Some("hazaar") | Some("thousand") | Some("k") => 1_000.0,
                _ => 1.0,
            };

            let value = (base * multiplier * 100.0) as i64; // Store in paise
            return Some(Currency {
                value,
                unit: "INR".to_string(),
                text: caps.get(0)?.as_str().to_string(),
            });
        }

        // Try Hindi pattern
        if self.support_hindi {
            if let Some(caps) = HINDI_AMOUNT_PATTERN.captures(text) {
                let hindi_num = caps.get(1)?.as_str();
                let multiplier_str = caps.get(2).map(|m| m.as_str());

                let base = self.hindi_to_number(hindi_num)?;
                let multiplier = match multiplier_str {
                    Some("लाख") => 100_000.0,
                    Some("करोड़") => 10_000_000.0,
                    Some("हज़ार") | Some("हजार") => 1_000.0,
                    _ => 1.0,
                };

                let value = (base * multiplier * 100.0) as i64;
                return Some(Currency {
                    value,
                    unit: "INR".to_string(),
                    text: caps.get(0)?.as_str().to_string(),
                });
            }
        }

        None
    }

    /// Extract gold weight
    pub fn extract_weight(&self, text: &str) -> Option<Weight> {
        let caps = WEIGHT_PATTERN.captures(text)?;
        let num_str = caps.get(1)?.as_str();
        let unit_str = caps.get(2)?.as_str().to_lowercase();

        let base: f64 = num_str.parse().ok()?;

        // Convert to milligrams
        let (value_mg, unit) = match unit_str.as_str() {
            "gram" | "grams" | "gm" | "g" => ((base * 1000.0) as i64, "grams"),
            "tola" | "tolas" => ((base * 11660.0) as i64, "tola"), // 1 tola = 11.66 grams
            "kg" | "kilogram" => ((base * 1_000_000.0) as i64, "kg"),
            _ => return None,
        };

        Some(Weight {
            value_mg,
            unit: unit.to_string(),
            text: caps.get(0)?.as_str().to_string(),
        })
    }

    /// Extract interest rate
    pub fn extract_rate(&self, text: &str) -> Option<Percentage> {
        let caps = RATE_PATTERN.captures(text)?;
        let value: f64 = caps.get(1)?.as_str().parse().ok()?;

        Some(Percentage {
            value,
            text: caps.get(0)?.as_str().to_string(),
        })
    }

    /// Extract loan tenure
    pub fn extract_tenure(&self, text: &str) -> Option<Duration> {
        let caps = TENURE_PATTERN.captures(text)?;
        let num: i32 = caps.get(1)?.as_str().parse().ok()?;
        let unit_str = caps.get(2)?.as_str().to_lowercase();

        let (days, unit) = match unit_str.as_str() {
            "month" | "months" | "mahine" => (num * 30, "months"),
            "year" | "years" | "yr" | "yrs" | "saal" => (num * 365, "years"),
            "day" | "days" => (num, "days"),
            _ => return None,
        };

        Some(Duration {
            days,
            unit: unit.to_string(),
            text: caps.get(0)?.as_str().to_string(),
        })
    }

    /// Extract customer name
    pub fn extract_name(&self, text: &str) -> Option<String> {
        let caps = NAME_PATTERN.captures(text)?;
        Some(caps.get(1)?.as_str().trim().to_string())
    }

    /// Extract gold purity (karat)
    pub fn extract_purity(&self, text: &str) -> Option<u8> {
        let caps = PURITY_PATTERN.captures(text)?;
        let karat: u8 = caps.get(1)?.as_str().parse().ok()?;

        // Validate karat value (typically 18, 20, 22, 24)
        if (10..=24).contains(&karat) {
            Some(karat)
        } else {
            None
        }
    }

    /// Extract current lender name
    pub fn extract_lender(&self, text: &str) -> Option<String> {
        for (name, pattern) in LENDER_PATTERNS.iter() {
            if pattern.is_match(text) {
                return Some(name.to_string());
            }
        }
        None
    }

    /// Convert Hindi number word to f64
    fn hindi_to_number(&self, hindi: &str) -> Option<f64> {
        match hindi {
            "एक" => Some(1.0),
            "दो" => Some(2.0),
            "तीन" => Some(3.0),
            "चार" => Some(4.0),
            "पांच" | "पाँच" => Some(5.0),
            "छह" | "छः" => Some(6.0),
            "सात" => Some(7.0),
            "आठ" => Some(8.0),
            "नौ" => Some(9.0),
            "दस" => Some(10.0),
            "बीस" => Some(20.0),
            "तीस" => Some(30.0),
            "चालीस" => Some(40.0),
            "पचास" => Some(50.0),
            "साठ" => Some(60.0),
            "सत्तर" => Some(70.0),
            "अस्सी" => Some(80.0),
            "नब्बे" => Some(90.0),
            "सौ" => Some(100.0),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_amount_lakh() {
        let extractor = LoanEntityExtractor::new();

        let result = extractor.extract_amount("I want 5 lakh loan");
        assert!(result.is_some());
        let amount = result.unwrap();
        assert_eq!(amount.rupees(), 500000.0);
    }

    #[test]
    fn test_extract_amount_crore() {
        let extractor = LoanEntityExtractor::new();

        let result = extractor.extract_amount("Need 1 crore for business");
        assert!(result.is_some());
        let amount = result.unwrap();
        assert_eq!(amount.rupees(), 10_000_000.0);
    }

    #[test]
    fn test_extract_amount_with_currency_symbol() {
        let extractor = LoanEntityExtractor::new();

        let result = extractor.extract_amount("Rs. 50000 loan needed");
        assert!(result.is_some());
        let amount = result.unwrap();
        assert_eq!(amount.rupees(), 50000.0);
    }

    #[test]
    fn test_extract_weight_grams() {
        let extractor = LoanEntityExtractor::new();

        let result = extractor.extract_weight("I have 50 grams of gold");
        assert!(result.is_some());
        let weight = result.unwrap();
        assert_eq!(weight.grams(), 50.0);
    }

    #[test]
    fn test_extract_weight_tola() {
        let extractor = LoanEntityExtractor::new();

        let result = extractor.extract_weight("Gold weighing 10 tola");
        assert!(result.is_some());
        let weight = result.unwrap();
        // 10 tola = 116.6 grams
        assert!((weight.grams() - 116.6).abs() < 0.1);
    }

    #[test]
    fn test_extract_rate() {
        let extractor = LoanEntityExtractor::new();

        let result = extractor.extract_rate("Interest rate is 10.5%");
        assert!(result.is_some());
        let rate = result.unwrap();
        assert_eq!(rate.value, 10.5);
    }

    #[test]
    fn test_extract_tenure_months() {
        let extractor = LoanEntityExtractor::new();

        let result = extractor.extract_tenure("12 month loan");
        assert!(result.is_some());
        let tenure = result.unwrap();
        assert_eq!(tenure.months(), 12.0);
    }

    #[test]
    fn test_extract_tenure_years() {
        let extractor = LoanEntityExtractor::new();

        let result = extractor.extract_tenure("2 year tenure");
        assert!(result.is_some());
        let tenure = result.unwrap();
        assert_eq!(tenure.years(), 2.0);
    }

    #[test]
    fn test_extract_name() {
        let extractor = LoanEntityExtractor::new();

        let result = extractor.extract_name("My name is Rajesh Kumar");
        assert_eq!(result, Some("Rajesh Kumar".to_string()));
    }

    #[test]
    fn test_extract_purity() {
        let extractor = LoanEntityExtractor::new();

        let result = extractor.extract_purity("22k gold");
        assert_eq!(result, Some(22));

        let result = extractor.extract_purity("18 karat gold");
        assert_eq!(result, Some(18));
    }

    #[test]
    fn test_extract_lender() {
        let extractor = LoanEntityExtractor::new();

        let result = extractor.extract_lender("I have loan from Muthoot Finance");
        assert_eq!(result, Some("Muthoot".to_string()));

        let result = extractor.extract_lender("Currently with IIFL");
        assert_eq!(result, Some("IIFL".to_string()));
    }

    #[test]
    fn test_extract_all_entities() {
        let extractor = LoanEntityExtractor::new();

        let text = "My name is Rahul. I want 5 lakh loan for 12 months at 10% interest. I have 50 grams of 22k gold. Currently with Muthoot.";
        let entities = extractor.extract(text);

        assert!(entities.amount.is_some());
        assert_eq!(entities.amount.as_ref().unwrap().rupees(), 500000.0);

        assert!(entities.tenure.is_some());
        assert_eq!(entities.tenure.as_ref().unwrap().months(), 12.0);

        assert!(entities.interest_rate.is_some());
        assert_eq!(entities.interest_rate.as_ref().unwrap().value, 10.0);

        assert!(entities.gold_weight.is_some());
        assert_eq!(entities.gold_weight.as_ref().unwrap().grams(), 50.0);

        assert_eq!(entities.gold_purity, Some(22));
        assert_eq!(entities.customer_name, Some("Rahul".to_string()));
        assert_eq!(entities.current_lender, Some("Muthoot".to_string()));
    }

    #[test]
    fn test_hindi_amount() {
        let extractor = LoanEntityExtractor::new();

        let result = extractor.extract_amount("पांच लाख");
        assert!(result.is_some());
        let amount = result.unwrap();
        assert_eq!(amount.rupees(), 500000.0);
    }

    #[test]
    fn test_merge_entities() {
        let mut entities1 = LoanEntities::default();
        entities1.amount = Some(Currency {
            value: 50000000, // 5 lakh in paise
            unit: "INR".to_string(),
            text: "5 lakh".to_string(),
        });

        let mut entities2 = LoanEntities::default();
        entities2.tenure = Some(Duration {
            days: 360,
            unit: "months".to_string(),
            text: "12 months".to_string(),
        });

        entities1.merge(&entities2);

        assert!(entities1.amount.is_some());
        assert!(entities1.tenure.is_some());
    }

    #[test]
    fn test_empty_text() {
        let extractor = LoanEntityExtractor::new();
        let entities = extractor.extract("");
        assert!(entities.is_empty());
    }
}
