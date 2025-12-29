//! Gold Loan Domain Tools
//!
//! Specific tools for the gold loan voice agent.

use async_trait::async_trait;
use chrono::{NaiveDate, Utc};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::Path;
use std::sync::{Arc, RwLock};
use voice_agent_config::GoldLoanConfig;
use crate::integrations::{CrmIntegration, CalendarIntegration, CrmLead, LeadSource, InterestLevel, LeadStatus, Appointment, AppointmentPurpose, AppointmentStatus};

use crate::mcp::{Tool, ToolSchema, ToolOutput, ToolError, InputSchema, PropertySchema};

/// P0 FIX: Branch data structure for JSON loading
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchData {
    pub branch_id: String,
    pub name: String,
    pub city: String,
    pub area: String,
    pub address: String,
    #[serde(default)]
    pub pincode: String,
    pub phone: String,
    pub gold_loan_available: bool,
    pub timing: String,
    #[serde(default)]
    pub facilities: Vec<String>,
}

/// Branch data file structure
#[derive(Debug, Deserialize)]
struct BranchDataFile {
    branches: Vec<BranchData>,
}

/// P0 FIX: Global branch data loaded from JSON
static BRANCH_DATA: Lazy<RwLock<Vec<BranchData>>> = Lazy::new(|| {
    // Try to load from default paths
    let default_paths = [
        "data/branches.json",
        "../data/branches.json",
        "../../data/branches.json",
        "./branches.json",
    ];

    for path in &default_paths {
        if let Ok(data) = load_branches_from_file(path) {
            tracing::info!("Loaded {} branches from {}", data.len(), path);
            return RwLock::new(data);
        }
    }

    // Fall back to embedded default data
    tracing::warn!("Could not load branches from file, using embedded defaults");
    RwLock::new(get_default_branches())
});

/// Load branches from a JSON file
pub fn load_branches_from_file<P: AsRef<Path>>(path: P) -> Result<Vec<BranchData>, std::io::Error> {
    let content = std::fs::read_to_string(path)?;
    let file: BranchDataFile = serde_json::from_str(&content)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok(file.branches)
}

/// Reload branches from a file (for runtime updates)
pub fn reload_branches<P: AsRef<Path>>(path: P) -> Result<usize, std::io::Error> {
    let branches = load_branches_from_file(path)?;
    let count = branches.len();
    *BRANCH_DATA.write().unwrap() = branches;
    Ok(count)
}

/// Get all loaded branches
pub fn get_branches() -> Vec<BranchData> {
    BRANCH_DATA.read().unwrap().clone()
}

/// Get default embedded branches (fallback)
fn get_default_branches() -> Vec<BranchData> {
    vec![
        BranchData {
            branch_id: "KMBL001".to_string(),
            name: "Kotak Mahindra Bank - Andheri West".to_string(),
            city: "Mumbai".to_string(),
            area: "Andheri West".to_string(),
            address: "Ground Floor, Kora Kendra, S.V. Road, Andheri West, Mumbai - 400058".to_string(),
            pincode: "400058".to_string(),
            phone: "022-66006060".to_string(),
            gold_loan_available: true,
            timing: "10:00 AM - 5:00 PM (Mon-Sat)".to_string(),
            facilities: vec!["Gold Valuation".to_string(), "Same Day Disbursement".to_string()],
        },
        BranchData {
            branch_id: "KMBL101".to_string(),
            name: "Kotak Mahindra Bank - Connaught Place".to_string(),
            city: "Delhi".to_string(),
            area: "Connaught Place".to_string(),
            address: "M-Block, Connaught Place, New Delhi - 110001".to_string(),
            pincode: "110001".to_string(),
            phone: "011-66006060".to_string(),
            gold_loan_available: true,
            timing: "10:00 AM - 5:00 PM (Mon-Sat)".to_string(),
            facilities: vec!["Gold Valuation".to_string(), "Same Day Disbursement".to_string()],
        },
        BranchData {
            branch_id: "KMBL201".to_string(),
            name: "Kotak Mahindra Bank - MG Road".to_string(),
            city: "Bangalore".to_string(),
            area: "MG Road".to_string(),
            address: "Church Street, MG Road, Bangalore - 560001".to_string(),
            pincode: "560001".to_string(),
            phone: "080-66006060".to_string(),
            gold_loan_available: true,
            timing: "10:00 AM - 5:00 PM (Mon-Sat)".to_string(),
            facilities: vec!["Gold Valuation".to_string(), "Same Day Disbursement".to_string()],
        },
        BranchData {
            branch_id: "KMBL301".to_string(),
            name: "Kotak Mahindra Bank - T Nagar".to_string(),
            city: "Chennai".to_string(),
            area: "T Nagar".to_string(),
            address: "Usman Road, T Nagar, Chennai - 600017".to_string(),
            pincode: "600017".to_string(),
            phone: "044-66006060".to_string(),
            gold_loan_available: true,
            timing: "10:00 AM - 5:00 PM (Mon-Sat)".to_string(),
            facilities: vec!["Gold Valuation".to_string(), "Same Day Disbursement".to_string()],
        },
    ]
}

/// Check eligibility tool
pub struct EligibilityCheckTool {
    config: GoldLoanConfig,
}

impl EligibilityCheckTool {
    pub fn new() -> Self {
        Self {
            config: GoldLoanConfig::default(),
        }
    }

    pub fn with_config(config: GoldLoanConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl Tool for EligibilityCheckTool {
    fn name(&self) -> &str {
        "check_eligibility"
    }

    fn description(&self) -> &str {
        "Check customer eligibility for gold loan based on gold weight and purity"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: InputSchema::object()
                .property("gold_weight_grams", PropertySchema::number("Gold weight in grams"), true)
                .property("gold_purity", PropertySchema::enum_type(
                    "Gold purity (22K, 18K, etc.)",
                    vec!["24K".into(), "22K".into(), "18K".into(), "14K".into()]
                ).with_default(json!("22K")), false)
                .property("existing_loan_amount", PropertySchema::number("Existing loan amount if any"), false),
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let weight: f64 = input.get("gold_weight_grams")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| ToolError::invalid_params("gold_weight_grams is required"))?;

        let purity = input.get("gold_purity")
            .and_then(|v| v.as_str())
            .unwrap_or("22K");

        let existing_loan = input.get("existing_loan_amount")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        // Calculate eligibility using config values
        let gold_value = self.config.calculate_gold_value(weight, purity);
        let max_loan = self.config.calculate_max_loan(gold_value);
        let available_loan = max_loan - existing_loan;

        // P2 FIX: Use tiered interest rates based on loan amount
        // Higher loan amounts get better rates
        let interest_rate = self.config.get_tiered_rate(available_loan.max(0.0));

        let result = json!({
            "eligible": available_loan >= self.config.min_loan_amount,
            "gold_value_inr": gold_value.round(),
            "max_loan_amount_inr": max_loan.round(),
            "existing_loan_inr": existing_loan,
            "available_loan_inr": available_loan.max(0.0).round(),
            "ltv_percent": self.config.ltv_percent,
            "interest_rate_percent": interest_rate,
            "processing_fee_percent": self.config.processing_fee_percent,
            // P2 FIX: Include rate tier info for transparency
            "rate_tier": if available_loan <= 100000.0 {
                "Standard"
            } else if available_loan <= 500000.0 {
                "Premium"
            } else {
                "Elite"
            },
            "message": if available_loan >= self.config.min_loan_amount {
                format!(
                    "You are eligible for a gold loan up to ₹{:.0} at {}% interest!",
                    available_loan, interest_rate
                )
            } else if available_loan > 0.0 {
                format!("You can get an additional ₹{:.0} on your gold.", available_loan)
            } else {
                "Based on your existing loan, no additional loan is available at this time.".to_string()
            }
        });

        Ok(ToolOutput::json(result))
    }
}

impl Default for EligibilityCheckTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Savings calculator tool
pub struct SavingsCalculatorTool {
    config: GoldLoanConfig,
}

impl SavingsCalculatorTool {
    pub fn new() -> Self {
        Self {
            config: GoldLoanConfig::default(),
        }
    }

    pub fn with_config(config: GoldLoanConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl Tool for SavingsCalculatorTool {
    fn name(&self) -> &str {
        "calculate_savings"
    }

    fn description(&self) -> &str {
        "Calculate potential savings when switching from NBFC to Kotak gold loan"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: InputSchema::object()
                .property("current_loan_amount", PropertySchema::number("Current loan amount in INR"), true)
                .property("current_interest_rate", PropertySchema::number("Current interest rate (%)").with_range(10.0, 30.0), true)
                .property("remaining_tenure_months", PropertySchema::integer("Remaining tenure in months"), true)
                .property("current_lender", PropertySchema::enum_type(
                    "Current lender",
                    vec!["Muthoot".into(), "Manappuram".into(), "IIFL".into(), "Other NBFC".into()]
                ), false),
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let loan_amount: f64 = input.get("current_loan_amount")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| ToolError::invalid_params("current_loan_amount is required"))?;

        // Get current rate - either from input or infer from lender
        let current_lender = input.get("current_lender")
            .and_then(|v| v.as_str())
            .unwrap_or("Other NBFC");

        let current_rate: f64 = input.get("current_interest_rate")
            .and_then(|v| v.as_f64())
            .unwrap_or_else(|| self.config.get_competitor_rate(current_lender));

        let tenure_months: i64 = input.get("remaining_tenure_months")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| ToolError::invalid_params("remaining_tenure_months is required"))?;

        // P2 FIX: Use tiered rate based on loan amount
        // Higher loan amounts qualify for better rates
        let kotak_rate = self.config.get_tiered_rate(loan_amount);

        // Calculate monthly savings using tiered rates
        let monthly_savings = self.config.calculate_monthly_savings_tiered(loan_amount, current_rate);

        // Calculate monthly payments
        let current_monthly = loan_amount * (current_rate / 100.0 / 12.0);
        let kotak_monthly = loan_amount * (kotak_rate / 100.0 / 12.0);

        // Total interest over remaining tenure
        let total_savings = monthly_savings * tenure_months as f64;

        // P2 FIX: Determine rate tier for customer communication
        let rate_tier = if loan_amount <= 100000.0 {
            "Standard"
        } else if loan_amount <= 500000.0 {
            "Premium"
        } else {
            "Elite"
        };

        let result = json!({
            "current_lender": current_lender,
            "current_interest_rate_percent": current_rate,
            "kotak_interest_rate_percent": kotak_rate,
            "rate_reduction_percent": current_rate - kotak_rate,
            "current_monthly_interest_inr": current_monthly.round(),
            "kotak_monthly_interest_inr": kotak_monthly.round(),
            "monthly_savings_inr": monthly_savings.round(),
            "total_savings_inr": total_savings.round(),
            "tenure_months": tenure_months,
            // P2 FIX: Include tier info for transparency
            "rate_tier": rate_tier,
            "message": format!(
                "By switching to Kotak at our {} rate of {}%, you can save ₹{:.0} per month and ₹{:.0} over the remaining {} months!",
                rate_tier, kotak_rate, monthly_savings, total_savings, tenure_months
            )
        });

        Ok(ToolOutput::json(result))
    }
}

impl Default for SavingsCalculatorTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Lead capture tool
/// P4 FIX: Now wired to CrmIntegration for actual lead creation
pub struct LeadCaptureTool {
    /// CRM integration for creating leads
    crm: Option<Arc<dyn CrmIntegration>>,
}

impl LeadCaptureTool {
    pub fn new() -> Self {
        Self { crm: None }
    }

    /// Create with CRM integration
    pub fn with_crm(crm: Arc<dyn CrmIntegration>) -> Self {
        Self { crm: Some(crm) }
    }
}

#[async_trait]
impl Tool for LeadCaptureTool {
    fn name(&self) -> &str {
        "capture_lead"
    }

    fn description(&self) -> &str {
        "Capture customer lead information for follow-up"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: InputSchema::object()
                .property("customer_name", PropertySchema::string("Customer's full name"), true)
                .property("phone_number", PropertySchema::string("10-digit mobile number"), true)
                .property("city", PropertySchema::string("Customer's city"), false)
                .property("preferred_branch", PropertySchema::string("Preferred branch location"), false)
                .property("estimated_gold_weight", PropertySchema::number("Estimated gold weight in grams"), false)
                .property("interest_level", PropertySchema::enum_type(
                    "Customer's interest level",
                    vec!["High".into(), "Medium".into(), "Low".into()]
                ), false)
                .property("notes", PropertySchema::string("Additional notes from conversation"), false),
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let name = input.get("customer_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("customer_name is required"))?;

        let phone = input.get("phone_number")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("phone_number is required"))?;

        // Validate phone number
        if phone.len() != 10 || !phone.chars().all(|c| c.is_ascii_digit()) {
            return Err(ToolError::invalid_params("phone_number must be 10 digits"));
        }

        // Extract optional fields
        let city = input.get("city").and_then(|v| v.as_str()).map(String::from);
        let estimated_gold = input.get("estimated_gold_weight").and_then(|v| v.as_f64());
        let notes = input.get("notes").and_then(|v| v.as_str()).map(String::from);
        let interest_str = input.get("interest_level").and_then(|v| v.as_str()).unwrap_or("Medium");

        // Parse interest level
        let interest_level = match interest_str.to_lowercase().as_str() {
            "high" => InterestLevel::High,
            "low" => InterestLevel::Low,
            _ => InterestLevel::Medium,
        };

        // P4 FIX: Use CRM integration if available
        if let Some(ref crm) = self.crm {
            let lead = CrmLead {
                id: None,
                name: name.to_string(),
                phone: phone.to_string(),
                email: None,
                city,
                source: LeadSource::VoiceAgent,
                interest_level,
                estimated_gold_grams: estimated_gold,
                current_lender: None,
                notes,
                assigned_to: None,
                status: LeadStatus::New,
            };

            match crm.create_lead(lead).await {
                Ok(lead_id) => {
                    let result = json!({
                        "success": true,
                        "lead_id": lead_id,
                        "customer_name": name,
                        "phone_number": phone,
                        "city": input.get("city").and_then(|v| v.as_str()),
                        "interest_level": interest_str,
                        "estimated_gold_weight": estimated_gold,
                        "created_at": Utc::now().to_rfc3339(),
                        "crm_integrated": true,
                        "message": format!("Lead captured successfully! A representative will contact {} shortly.", name)
                    });
                    return Ok(ToolOutput::json(result));
                }
                Err(e) => {
                    tracing::warn!("CRM integration failed, falling back to local: {}", e);
                    // Fall through to local generation
                }
            }
        }

        // Fallback: Generate lead ID locally (no CRM integration)
        let lead_id = format!("GL{}", uuid::Uuid::new_v4().to_string()[..8].to_uppercase());

        let result = json!({
            "success": true,
            "lead_id": lead_id,
            "customer_name": name,
            "phone_number": phone,
            "city": input.get("city").and_then(|v| v.as_str()),
            "preferred_branch": input.get("preferred_branch").and_then(|v| v.as_str()),
            "estimated_gold_weight": estimated_gold,
            "interest_level": interest_str,
            "notes": input.get("notes").and_then(|v| v.as_str()),
            "created_at": Utc::now().to_rfc3339(),
            "crm_integrated": false,
            "message": format!("Lead captured successfully! A representative will contact {} shortly.", name)
        });

        Ok(ToolOutput::json(result))
    }

    /// P5 FIX: CRM integrations may need more time
    fn timeout_secs(&self) -> u64 {
        45 // 45 seconds for CRM operations
    }
}

impl Default for LeadCaptureTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Appointment scheduler tool
/// P4 FIX: Now wired to CalendarIntegration for actual scheduling
pub struct AppointmentSchedulerTool {
    /// Calendar integration for scheduling appointments
    calendar: Option<Arc<dyn CalendarIntegration>>,
}

impl AppointmentSchedulerTool {
    pub fn new() -> Self {
        Self { calendar: None }
    }

    /// Create with calendar integration
    pub fn with_calendar(calendar: Arc<dyn CalendarIntegration>) -> Self {
        Self { calendar: Some(calendar) }
    }
}

#[async_trait]
impl Tool for AppointmentSchedulerTool {
    fn name(&self) -> &str {
        "schedule_appointment"
    }

    fn description(&self) -> &str {
        "Schedule a branch visit appointment for gold valuation"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: InputSchema::object()
                .property("customer_name", PropertySchema::string("Customer's name"), true)
                .property("phone_number", PropertySchema::string("Contact number"), true)
                .property("branch_id", PropertySchema::string("Branch ID or location"), true)
                .property("preferred_date", PropertySchema::string("Preferred date (YYYY-MM-DD)"), true)
                .property("preferred_time", PropertySchema::enum_type(
                    "Preferred time slot",
                    vec!["10:00 AM".into(), "11:00 AM".into(), "12:00 PM".into(),
                         "2:00 PM".into(), "3:00 PM".into(), "4:00 PM".into(), "5:00 PM".into()]
                ), true)
                .property("purpose", PropertySchema::enum_type(
                    "Purpose of visit",
                    vec!["New Gold Loan".into(), "Gold Loan Transfer".into(), "Top-up".into(), "Closure".into()]
                ), false),
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let name = input.get("customer_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("customer_name is required"))?;

        let phone = input.get("phone_number")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("phone_number is required"))?;

        let branch = input.get("branch_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("branch_id is required"))?;

        let date_str = input.get("preferred_date")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("preferred_date is required"))?;

        // P1 FIX: Validate date format and ensure it's not in the past
        let parsed_date = NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
            .or_else(|_| NaiveDate::parse_from_str(date_str, "%d-%m-%Y"))
            .or_else(|_| NaiveDate::parse_from_str(date_str, "%d/%m/%Y"))
            .map_err(|_| ToolError::invalid_params(
                "preferred_date must be in format YYYY-MM-DD, DD-MM-YYYY, or DD/MM/YYYY"
            ))?;

        let today = Utc::now().date_naive();
        if parsed_date < today {
            return Err(ToolError::invalid_params("preferred_date cannot be in the past"));
        }

        // Use standardized format
        let date = parsed_date.format("%Y-%m-%d").to_string();

        let time = input.get("preferred_time")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("preferred_time is required"))?;

        let purpose_str = input.get("purpose")
            .and_then(|v| v.as_str())
            .unwrap_or("New Gold Loan");

        // Parse purpose enum
        let purpose_enum = match purpose_str {
            "Gold Loan Transfer" => AppointmentPurpose::GoldLoanTransfer,
            "Top-up" => AppointmentPurpose::TopUp,
            "Closure" => AppointmentPurpose::Closure,
            "Consultation" => AppointmentPurpose::Consultation,
            _ => AppointmentPurpose::NewGoldLoan,
        };

        // P4 FIX: Use calendar integration if available
        if let Some(ref calendar) = self.calendar {
            let appointment = Appointment {
                id: None,
                customer_name: name.to_string(),
                customer_phone: phone.to_string(),
                branch_id: branch.to_string(),
                date: date.clone(),
                time_slot: time.to_string(),
                purpose: purpose_enum,
                notes: None,
                status: AppointmentStatus::Scheduled,
                confirmation_sent: false,
            };

            match calendar.schedule_appointment(appointment).await {
                Ok(appointment_id) => {
                    // P0 FIX: Don't claim confirmation sent until actually sent
                    // Try to send confirmation, but don't fail if it doesn't work
                    let confirmation_sent = calendar.send_confirmation(&appointment_id).await.is_ok();

                    let result = json!({
                        "success": true,
                        "appointment_id": appointment_id,
                        "customer_name": name,
                        "phone_number": phone,
                        "branch_id": branch,
                        "date": date,
                        "time": time,
                        "purpose": purpose_str,
                        "confirmation_sent": confirmation_sent,
                        "calendar_integrated": true,
                        "status": "pending_confirmation",
                        "confirmation_method": "agent_will_call_to_confirm",
                        "next_action": "Agent will call customer to confirm appointment",
                        "message": if confirmation_sent {
                            format!(
                                "Appointment scheduled for {} on {} at {}. Confirmation sent to {}.",
                                name, date, time, phone
                            )
                        } else {
                            format!(
                                "Appointment scheduled for {} on {} at {}. Our team will call to confirm.",
                                name, date, time
                            )
                        }
                    });
                    return Ok(ToolOutput::json(result));
                }
                Err(e) => {
                    tracing::warn!("Calendar integration failed, falling back to local: {}", e);
                    // Fall through to local generation
                }
            }
        }

        // Fallback: Generate appointment ID locally (no calendar integration)
        // P0 FIX: Don't claim SMS confirmation sent when we have no integration
        let appointment_id = format!("APT{}", uuid::Uuid::new_v4().to_string()[..8].to_uppercase());

        let result = json!({
            "success": true,
            "appointment_id": appointment_id,
            "customer_name": name,
            "phone_number": phone,
            "branch_id": branch,
            "date": date,
            "time": time,
            "purpose": purpose_str,
            "confirmation_sent": false,
            "calendar_integrated": false,
            "status": "pending_confirmation",
            "confirmation_method": "agent_will_call_to_confirm",
            "next_action": "Agent will call customer to confirm appointment",
            "message": format!(
                "Appointment scheduled for {} on {} at {}. Our team will call to confirm.",
                name, date, time
            )
        });

        Ok(ToolOutput::json(result))
    }

    /// P5 FIX: Calendar integrations may be slower, allow more time
    fn timeout_secs(&self) -> u64 {
        60 // 60 seconds for appointment scheduling with external calendar
    }
}

impl Default for AppointmentSchedulerTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Branch locator tool
pub struct BranchLocatorTool;

impl BranchLocatorTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for BranchLocatorTool {
    fn name(&self) -> &str {
        "find_branches"
    }

    fn description(&self) -> &str {
        "Find nearby Kotak Mahindra Bank branches offering gold loan services"
    }

    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name().to_string(),
            description: self.description().to_string(),
            input_schema: InputSchema::object()
                .property("city", PropertySchema::string("City name"), true)
                .property("area", PropertySchema::string("Area or locality"), false)
                .property("pincode", PropertySchema::string("6-digit PIN code"), false)
                .property("max_results", PropertySchema::integer("Maximum results to return").with_default(json!(5)), false),
        }
    }

    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> {
        let city = input.get("city")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ToolError::invalid_params("city is required"))?;

        let area = input.get("area").and_then(|v| v.as_str());
        let pincode = input.get("pincode").and_then(|v| v.as_str());
        let max_results = input.get("max_results")
            .and_then(|v| v.as_i64())
            .unwrap_or(5) as usize;

        // Mock branch data (in production, this would query a database or API)
        let branches = get_mock_branches(city, area, pincode, max_results);

        let result = json!({
            "city": city,
            "area": area,
            "branches_found": branches.len(),
            "branches": branches,
            "message": if branches.is_empty() {
                format!("No branches found in {}. Please try a nearby city.", city)
            } else {
                format!("Found {} branches in {}.", branches.len(), city)
            }
        });

        Ok(ToolOutput::json(result))
    }
}

impl Default for BranchLocatorTool {
    fn default() -> Self {
        Self::new()
    }
}

/// P0 FIX: Get branches from JSON data instead of hardcoded mock
fn get_mock_branches(city: &str, area: Option<&str>, pincode: Option<&str>, max: usize) -> Vec<Value> {
    let city_lower = city.to_lowercase();
    let branches = get_branches();

    // Filter by city
    let mut filtered: Vec<BranchData> = branches
        .into_iter()
        .filter(|b| {
            b.city.to_lowercase().contains(&city_lower) ||
            city_lower.contains(&b.city.to_lowercase())
        })
        .collect();

    // Filter by pincode if provided (exact match)
    if let Some(pin) = pincode {
        let pin_matches: Vec<BranchData> = filtered.iter()
            .filter(|b| b.pincode == pin)
            .cloned()
            .collect();
        if !pin_matches.is_empty() {
            filtered = pin_matches;
        }
    }

    // Filter by area if provided
    if let Some(area_str) = area {
        let area_lower = area_str.to_lowercase();
        let area_matches: Vec<BranchData> = filtered.iter()
            .filter(|b| b.area.to_lowercase().contains(&area_lower))
            .cloned()
            .collect();
        if !area_matches.is_empty() {
            filtered = area_matches;
        }
    }

    // Convert to JSON Value and truncate
    filtered.truncate(max);
    filtered.into_iter()
        .map(|b| json!({
            "branch_id": b.branch_id,
            "name": b.name,
            "city": b.city,
            "area": b.area,
            "address": b.address,
            "pincode": b.pincode,
            "phone": b.phone,
            "gold_loan_available": b.gold_loan_available,
            "timing": b.timing,
            "facilities": b.facilities
        }))
        .collect()
}


#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_eligibility_check() {
        let tool = EligibilityCheckTool::new();
        let input = json!({
            "gold_weight_grams": 50.0,
            "gold_purity": "22K"
        });

        let output = tool.execute(input).await.unwrap();
        assert!(!output.is_error);
    }

    #[tokio::test]
    async fn test_savings_calculator() {
        let tool = SavingsCalculatorTool::new();
        let input = json!({
            "current_loan_amount": 100000.0,
            "current_interest_rate": 18.0,
            "remaining_tenure_months": 12,
            "current_lender": "Muthoot"
        });

        let output = tool.execute(input).await.unwrap();
        assert!(!output.is_error);
    }

    #[tokio::test]
    async fn test_lead_capture() {
        let tool = LeadCaptureTool::new();
        let input = json!({
            "customer_name": "Rajesh Kumar",
            "phone_number": "9876543210",
            "city": "Mumbai"
        });

        let output = tool.execute(input).await.unwrap();
        assert!(!output.is_error);
    }

    #[tokio::test]
    async fn test_branch_locator() {
        let tool = BranchLocatorTool::new();
        let input = json!({
            "city": "Mumbai"
        });

        let output = tool.execute(input).await.unwrap();
        assert!(!output.is_error);
    }

    // P4 FIX: Tests for CRM/Calendar integration wiring

    #[tokio::test]
    async fn test_lead_capture_with_crm() {
        use crate::integrations::StubCrmIntegration;

        let crm = Arc::new(StubCrmIntegration::new());
        let tool = LeadCaptureTool::with_crm(crm);
        let input = json!({
            "customer_name": "Rajesh Kumar",
            "phone_number": "9876543210",
            "city": "Mumbai",
            "interest_level": "High"
        });

        let output = tool.execute(input).await.unwrap();
        assert!(!output.is_error);

        // Parse output to verify CRM integration flag
        let text = output.content.iter()
            .filter_map(|c| match c {
                crate::mcp::ContentBlock::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");

        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(json["crm_integrated"], true);
        assert!(json["lead_id"].as_str().unwrap().starts_with("LEAD-"));
    }

    #[tokio::test]
    async fn test_lead_capture_without_crm() {
        let tool = LeadCaptureTool::new();
        let input = json!({
            "customer_name": "Rajesh Kumar",
            "phone_number": "9876543210"
        });

        let output = tool.execute(input).await.unwrap();

        let text = output.content.iter()
            .filter_map(|c| match c {
                crate::mcp::ContentBlock::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");

        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(json["crm_integrated"], false);
        assert!(json["lead_id"].as_str().unwrap().starts_with("GL"));
    }

    #[tokio::test]
    async fn test_appointment_scheduler_with_calendar() {
        use crate::integrations::StubCalendarIntegration;

        let calendar = Arc::new(StubCalendarIntegration::new());
        let tool = AppointmentSchedulerTool::with_calendar(calendar);

        // Use a future date
        let future_date = (chrono::Utc::now() + chrono::Duration::days(7))
            .format("%Y-%m-%d")
            .to_string();

        let input = json!({
            "customer_name": "Rajesh Kumar",
            "phone_number": "9876543210",
            "branch_id": "KMBL001",
            "preferred_date": future_date,
            "preferred_time": "10:00 AM",
            "purpose": "New Gold Loan"
        });

        let output = tool.execute(input).await.unwrap();
        assert!(!output.is_error);

        let text = output.content.iter()
            .filter_map(|c| match c {
                crate::mcp::ContentBlock::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");

        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(json["calendar_integrated"], true);
        assert!(json["appointment_id"].as_str().unwrap().starts_with("APT-"));
        // P0 FIX: Verify we don't falsely claim SMS confirmation
        assert_eq!(json["status"], "pending_confirmation");
    }

    #[tokio::test]
    async fn test_appointment_scheduler_without_calendar() {
        let tool = AppointmentSchedulerTool::new();

        let future_date = (chrono::Utc::now() + chrono::Duration::days(7))
            .format("%Y-%m-%d")
            .to_string();

        let input = json!({
            "customer_name": "Rajesh Kumar",
            "phone_number": "9876543210",
            "branch_id": "KMBL001",
            "preferred_date": future_date,
            "preferred_time": "10:00 AM"
        });

        let output = tool.execute(input).await.unwrap();

        let text = output.content.iter()
            .filter_map(|c| match c {
                crate::mcp::ContentBlock::Text { text } => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");

        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(json["calendar_integrated"], false);
        assert!(json["appointment_id"].as_str().unwrap().starts_with("APT"));
        // P0 FIX: Verify we don't falsely claim SMS confirmation
        assert_eq!(json["confirmation_sent"], false);
        assert_eq!(json["status"], "pending_confirmation");
    }
}
