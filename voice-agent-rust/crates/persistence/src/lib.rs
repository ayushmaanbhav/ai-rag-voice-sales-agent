//! ScyllaDB persistence layer for voice-agent-rust
//!
//! Provides persistent storage for:
//! - Sessions (replaces Redis stub)
//! - SMS messages (simulated, persisted for audit)
//! - Gold prices (simulated with realistic fluctuation)
//! - Appointments

pub mod client;
pub mod error;
pub mod schema;
pub mod sessions;
pub mod sms;
pub mod gold_price;
pub mod appointments;

pub use client::{ScyllaClient, ScyllaConfig};
pub use error::PersistenceError;
pub use sessions::{SessionStore, ScyllaSessionStore, SessionData};
pub use sms::{SmsService, SimulatedSmsService, SmsMessage, SmsStatus, SmsType};
pub use gold_price::{GoldPriceService, SimulatedGoldPriceService, GoldPrice};
pub use appointments::{AppointmentStore, ScyllaAppointmentStore, Appointment, AppointmentStatus};

/// Initialize the persistence layer with ScyllaDB
pub async fn init(config: ScyllaConfig) -> Result<PersistenceLayer, PersistenceError> {
    let client = ScyllaClient::connect(config).await?;
    client.ensure_schema().await?;

    Ok(PersistenceLayer {
        sessions: ScyllaSessionStore::new(client.clone()),
        sms: SimulatedSmsService::new(client.clone()),
        gold_price: SimulatedGoldPriceService::new(client.clone(), 7500.0),
        appointments: ScyllaAppointmentStore::new(client),
    })
}

/// Combined persistence layer with all services
pub struct PersistenceLayer {
    pub sessions: ScyllaSessionStore,
    pub sms: SimulatedSmsService,
    pub gold_price: SimulatedGoldPriceService,
    pub appointments: ScyllaAppointmentStore,
}
