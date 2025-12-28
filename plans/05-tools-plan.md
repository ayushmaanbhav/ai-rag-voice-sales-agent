# Tools Component Plan

## Component Overview

The tools crate provides MCP-compatible tool implementations:
- MCP protocol interface
- Gold loan domain tools
- Tool registry

**Location**: `voice-agent-rust/crates/tools/src/`

---

## Current Status Summary

| Module | Status | Grade |
|--------|--------|-------|
| MCP Interface | JSON-RPC compliant | B |
| EligibilityCheck | Uses hardcoded gold price | C |
| SavingsCalculator | Static competitor rates | C |
| LeadCapture | No CRM integration | D |
| AppointmentScheduler | No calendar integration | D |
| BranchLocator | Only 5 mock branches | F |

---

## P0 - Critical Issues (Must Fix)

| Task | File:Line | Description |
|------|-----------|-------------|
| **Hardcoded gold price** | `gold_loan.rs:66-70`, `config/gold_loan.rs:70` | 7500 INR/gram is stale - need real-time API |
| **No CRM integration** | `gold_loan.rs:235-267` | Leads are returned but not persisted |
| **No calendar integration** | `gold_loan.rs:316-361` | Appointments not actually scheduled |
| **Mock branch data** | `gold_loan.rs:441-501` | Only 5 branches in 4 cities |

---

## P1 - Important Issues

| Task | File:Line | Description |
|------|-----------|-------------|
| No execution timeout | `registry.rs:89-98` | Tool can block indefinitely |
| Static competitor rates | `gold_loan.rs:147-153` | Should fetch from pricing API |
| SMS not sent | `gold_loan.rs:353` | Claims confirmation sent but doesn't |
| Date not validated | `gold_loan.rs:329-331` | Accepts any string as date |
| Phone validation India-only | `gold_loan.rs:245-247` | No country code support |
| Pincode unused | `gold_loan.rs:438` | Parameter exists but ignored |
| MCP missing timeout | `mcp.rs:274-311` | No timeout/cancellation in Tool trait |

---

## P2 - Nice to Have

| Task | File:Line | Description |
|------|-----------|-------------|
| Missing Audio ContentBlock | `mcp.rs:141-147` | For voice response support |
| Basic schema validation | `mcp.rs:290-310` | Only checks required, not types |
| O(n) history removal | `registry.rs:141-146` | Should use VecDeque |
| Error type unused | `lib.rs:20-42` | ToolsError defined but mcp.rs uses ToolError |
| Tiered interest rates | `config/gold_loan.rs:74-75` | Single rate, should tier by amount |

---

## External Integration Plan

### Gold Price API

Options:
1. **MCX API** - Official exchange rates
2. **GoldAPI.io** - Real-time spot prices
3. **Metals-API** - Historical + live prices

Implementation:
```rust
// config/gold_loan.rs
pub struct GoldPriceConfig {
    pub provider: GoldPriceProvider,
    pub api_key: Option<String>,
    pub cache_ttl_secs: u64,  // e.g., 300 for 5 min cache
}

// New file: tools/src/gold_price.rs
pub struct GoldPriceService {
    client: reqwest::Client,
    cache: RwLock<Option<(f64, Instant)>>,
}

impl GoldPriceService {
    pub async fn get_price_per_gram(&self) -> Result<f64, ToolError>;
}
```

### CRM Integration

Options:
1. **Salesforce** - Enterprise standard
2. **HubSpot** - Free tier available
3. **Zoho CRM** - Popular in India

Implementation:
```rust
// New file: tools/src/crm.rs
#[async_trait]
pub trait CrmClient: Send + Sync {
    async fn create_lead(&self, lead: Lead) -> Result<String, ToolError>;
    async fn update_lead(&self, id: &str, lead: Lead) -> Result<(), ToolError>;
}

pub struct SalesforceCrm { /* ... */ }
pub struct HubSpotCrm { /* ... */ }
```

### Calendar Integration

Options:
1. **Google Calendar API**
2. **Microsoft Graph** (Outlook)
3. **Internal booking system**

Implementation:
```rust
// New file: tools/src/calendar.rs
#[async_trait]
pub trait CalendarClient: Send + Sync {
    async fn get_available_slots(&self, branch: &str, date: NaiveDate) -> Result<Vec<TimeSlot>, ToolError>;
    async fn book_appointment(&self, apt: Appointment) -> Result<String, ToolError>;
}
```

### Branch Database

Options:
1. **Kotak internal API**
2. **PostgreSQL database**
3. **Static JSON file** (better than hardcoded)

Minimum data per branch:
```json
{
  "branch_id": "KMBL001",
  "name": "Kotak - Andheri West",
  "city": "Mumbai",
  "area": "Andheri West",
  "pincode": "400058",
  "lat": 19.1364,
  "lon": 72.8296,
  "gold_loan_enabled": true,
  "timings": "10:00-17:00",
  "phone": "022-66006060"
}
```

---

## Test Coverage

| File | Tests | Quality |
|------|-------|---------|
| mcp.rs | 3 | Schema building, output, errors |
| registry.rs | 3 | CRUD, listing, tracking |
| gold_loan.rs | 4 | Happy path only |

**Missing:**
- Invalid input tests
- Boundary value tests
- Concurrent access tests
- Timeout tests

---

## Implementation Priorities

### Week 1: Core Integrations
1. Add gold price API client with caching
2. Add execution timeout wrapper
3. Add date validation with chrono

### Week 2: CRM & Calendar
1. Add CRM client abstraction
2. Add Salesforce/HubSpot implementation
3. Add calendar availability check

### Week 3: Branch & SMS
1. Replace mock branches with database/API
2. Add SMS gateway integration (MSG91/Twilio)
3. Add geolocation-based branch search

---

*Last Updated: 2024-12-27*
