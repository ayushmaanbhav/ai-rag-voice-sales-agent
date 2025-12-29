# Plan 3: ScyllaDB Persistence Strategy

> **Decision:** Use ScyllaDB instead of Redis for all persistence needs

---

## Why ScyllaDB?

| Feature | ScyllaDB | Redis |
|---------|----------|-------|
| Data model | Wide-column (Cassandra-compatible) | Key-value |
| Persistence | Native disk persistence | AOF/RDB snapshots |
| Scalability | Linear horizontal scaling | Clustering complex |
| Query flexibility | CQL (SQL-like) | Limited |
| Time-series | Excellent (TTL, time-based partitioning) | Manual |

---

## Schema Design

### 1. Sessions Table

```cql
CREATE KEYSPACE IF NOT EXISTS voice_agent
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

CREATE TABLE voice_agent.sessions (
    session_id TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    expires_at TIMESTAMP,
    customer_phone TEXT,
    customer_segment TEXT,
    language TEXT,
    conversation_stage TEXT,
    memory_json TEXT,
    metadata_json TEXT,
    PRIMARY KEY (session_id)
) WITH default_time_to_live = 86400;  -- 24 hour TTL
```

### 2. SMS Messages Table (Simulated)

```cql
CREATE TABLE voice_agent.sms_messages (
    message_id TIMEUUID,
    session_id TEXT,
    phone_number TEXT,
    message_text TEXT,
    message_type TEXT,  -- 'appointment_confirmation', 'follow_up', etc.
    status TEXT,        -- 'queued', 'simulated_sent', 'delivered'
    created_at TIMESTAMP,
    sent_at TIMESTAMP,
    metadata_json TEXT,
    PRIMARY KEY ((phone_number), message_id)
) WITH CLUSTERING ORDER BY (message_id DESC);
```

### 3. Gold Prices Table (Simulated)

```cql
CREATE TABLE voice_agent.gold_prices (
    date DATE,
    time_bucket INT,  -- Hour of day (0-23) for granularity
    price_per_gram DECIMAL,
    price_24k DECIMAL,
    price_22k DECIMAL,
    price_18k DECIMAL,
    source TEXT,      -- 'simulated', 'api', 'manual'
    created_at TIMESTAMP,
    PRIMARY KEY ((date), time_bucket)
) WITH CLUSTERING ORDER BY (time_bucket DESC);

-- Latest price view
CREATE TABLE voice_agent.gold_price_latest (
    singleton INT,  -- Always 1, single row
    price_per_gram DECIMAL,
    price_24k DECIMAL,
    price_22k DECIMAL,
    price_18k DECIMAL,
    updated_at TIMESTAMP,
    source TEXT,
    PRIMARY KEY (singleton)
);
```

### 4. Appointments Table

```cql
CREATE TABLE voice_agent.appointments (
    appointment_id TIMEUUID,
    session_id TEXT,
    customer_phone TEXT,
    customer_name TEXT,
    branch_id TEXT,
    branch_name TEXT,
    appointment_date DATE,
    appointment_time TEXT,
    status TEXT,  -- 'scheduled', 'confirmed', 'cancelled', 'completed'
    created_at TIMESTAMP,
    confirmation_sms_id TIMEUUID,
    notes TEXT,
    PRIMARY KEY ((customer_phone), appointment_id)
) WITH CLUSTERING ORDER BY (appointment_id DESC);
```

---

## Implementation Plan

### New Crate: `crates/persistence`

```
crates/persistence/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── client.rs      -- ScyllaDB connection pool
│   ├── sessions.rs    -- Session CRUD
│   ├── sms.rs         -- SMS simulation & persistence
│   ├── gold_price.rs  -- Gold price simulation & caching
│   ├── appointments.rs -- Appointment persistence
│   └── schema.rs      -- Schema creation utilities
```

### Dependencies

```toml
[dependencies]
scylla = "0.13"
tokio = { version = "1", features = ["full"] }
uuid = { version = "1", features = ["v4", "v1"] }
chrono = { version = "0.4", features = ["serde"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tracing = "0.1"
rand = "0.8"  # For simulated price fluctuations
```

---

## Simulated Services

### SMS Simulation

```rust
pub struct SimulatedSmsService {
    db: Arc<ScyllaClient>,
}

impl SimulatedSmsService {
    /// "Sends" an SMS by persisting to ScyllaDB
    /// Returns immediately with simulated success
    pub async fn send_sms(&self, phone: &str, message: &str, msg_type: SmsType) -> Result<SmsResult> {
        let message_id = Uuid::now_v1(&[1, 2, 3, 4, 5, 6]);

        // Persist to ScyllaDB
        self.db.execute(
            "INSERT INTO voice_agent.sms_messages
             (message_id, phone_number, message_text, message_type, status, created_at, sent_at)
             VALUES (?, ?, ?, ?, 'simulated_sent', toTimestamp(now()), toTimestamp(now()))",
            (message_id, phone, message, msg_type.as_str())
        ).await?;

        // Log for debugging
        tracing::info!(
            phone = %phone,
            message_id = %message_id,
            "SMS simulated and persisted to ScyllaDB"
        );

        Ok(SmsResult {
            message_id,
            status: SmsStatus::SimulatedSent,
            sent_at: Utc::now(),
        })
    }
}
```

### Gold Price Simulation

```rust
pub struct SimulatedGoldPriceService {
    db: Arc<ScyllaClient>,
    base_price: f64,  // Base price for simulation
}

impl SimulatedGoldPriceService {
    /// Gets current gold price with realistic daily fluctuation
    pub async fn get_current_price(&self) -> Result<GoldPrice> {
        // Check cache first
        if let Some(cached) = self.get_cached_price().await? {
            if cached.updated_at > Utc::now() - Duration::minutes(5) {
                return Ok(cached);
            }
        }

        // Generate simulated price with ±2% daily fluctuation
        let fluctuation = (rand::random::<f64>() - 0.5) * 0.04;  // -2% to +2%
        let price_24k = self.base_price * (1.0 + fluctuation);
        let price_22k = price_24k * 0.916;  // 22k = 91.6% pure
        let price_18k = price_24k * 0.75;   // 18k = 75% pure

        let price = GoldPrice {
            price_per_gram: price_22k,  // Default to 22k for jewelry
            price_24k,
            price_22k,
            price_18k,
            source: "simulated".to_string(),
            updated_at: Utc::now(),
        };

        // Persist to ScyllaDB
        self.update_latest_price(&price).await?;
        self.record_price_history(&price).await?;

        Ok(price)
    }

    async fn update_latest_price(&self, price: &GoldPrice) -> Result<()> {
        self.db.execute(
            "INSERT INTO voice_agent.gold_price_latest
             (singleton, price_per_gram, price_24k, price_22k, price_18k, updated_at, source)
             VALUES (1, ?, ?, ?, ?, ?, ?)",
            (price.price_per_gram, price.price_24k, price.price_22k,
             price.price_18k, price.updated_at, &price.source)
        ).await
    }
}
```

---

## Migration from Redis Stubs

### Before (Redis Stub)
```rust
// server/session.rs
impl SessionStore for RedisSessionStore {
    async fn store_metadata(&self, _: &str, _: &SessionMetadata) -> Result<()> {
        tracing::debug!("Redis not implemented");
        Ok(())  // Silent no-op
    }
}
```

### After (ScyllaDB)
```rust
// persistence/sessions.rs
impl SessionStore for ScyllaSessionStore {
    async fn store_metadata(&self, id: &str, meta: &SessionMetadata) -> Result<()> {
        self.db.execute(
            "INSERT INTO voice_agent.sessions
             (session_id, created_at, updated_at, expires_at, customer_phone,
              customer_segment, language, conversation_stage, metadata_json)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (id, meta.created_at, Utc::now(), meta.expires_at,
             &meta.customer_phone, &meta.segment, &meta.language,
             &meta.stage, serde_json::to_string(meta)?)
        ).await?;
        Ok(())
    }
}
```

---

## Configuration

```toml
# config/voice_agent.toml
[persistence]
provider = "scylladb"  # or "memory" for testing

[persistence.scylladb]
hosts = ["127.0.0.1:9042"]
keyspace = "voice_agent"
replication_factor = 1
connection_pool_size = 10
request_timeout_ms = 5000

[persistence.gold_price]
base_price = 7500.0  # INR per gram (22k)
fluctuation_percent = 2.0
cache_ttl_seconds = 300

[persistence.sms]
simulation_mode = true  # Always true for now
log_messages = true
```

---

## Testing Strategy

### Unit Tests
```rust
#[tokio::test]
async fn test_session_persistence() {
    let db = setup_test_scylla().await;
    let store = ScyllaSessionStore::new(db);

    let meta = SessionMetadata::new("test-session");
    store.store_metadata("test-session", &meta).await.unwrap();

    let retrieved = store.get_metadata("test-session").await.unwrap();
    assert!(retrieved.is_some());
}

#[tokio::test]
async fn test_sms_simulation() {
    let sms = SimulatedSmsService::new(db);
    let result = sms.send_sms("+919876543210", "Test message", SmsType::Confirmation).await;

    assert!(result.is_ok());
    assert_eq!(result.unwrap().status, SmsStatus::SimulatedSent);

    // Verify persisted
    let messages = sms.get_messages_for_phone("+919876543210").await.unwrap();
    assert_eq!(messages.len(), 1);
}
```

### Integration Tests
- Session survives restart
- SMS messages queryable by phone
- Gold price fluctuates within bounds
- Appointments linked to sessions

---

## Rollout Plan

1. **Create persistence crate** with ScyllaDB client
2. **Implement session store** (replace Redis stub)
3. **Implement SMS simulation** (fix false positive)
4. **Implement gold price simulation** (fix hardcoded value)
5. **Wire into server/agent** crates
6. **Add Docker Compose** with ScyllaDB container
7. **Test with real conversations**
