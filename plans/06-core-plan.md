# Core/Infrastructure Plan

## Component Overview

Core infrastructure crates:
- Core types (AudioFrame, TranscriptResult)
- Configuration management
- Server/Transport (WebSocket)

**Locations**:
- `voice-agent-rust/crates/core/src/`
- `voice-agent-rust/crates/config/src/`
- `voice-agent-rust/crates/server/src/`

---

## Current Status Summary

| Module | Status | Grade |
|--------|--------|-------|
| Core Types | Well-designed | A- |
| Configuration | Layered loading | B+ |
| WebSocket Transport | Functional | B |
| HTTP API | REST endpoints | B |
| Session Management | Basic | C |
| Observability | Dependencies only, not used | F |

---

## P0 - Critical Issues (Must Fix)

| Task | File:Line | Description |
|------|-----------|-------------|
| **No WebRTC transport** | N/A | Critical for mobile apps and low latency |
| **No Metrics/Observability** | Cargo.toml:61-67 | Dependencies declared but never initialized |
| **No Rate Limiting** | `websocket.rs:56` | DoS vulnerability |
| **Insecure CORS default** | `settings.rs:128` | Wildcard "*" in production |

---

## P1 - Important Issues

| Task | File:Line | Description |
|------|-----------|-------------|
| Linear resampling | `audio.rs:216` | Aliasing - use rubato crate |
| No config hot-reload | `settings.rs:298` | Requires restart for config changes |
| No auth middleware | `http.rs:22` | No JWT/API key support |
| No session persistence | `session.rs:63` | Lost on restart |
| WebSocket lock contention | `websocket.rs:73` | Arc<Mutex> for sender |
| No graceful shutdown | `main.rs:38` | Connections dropped abruptly |
| Health check incomplete | `http.rs:196` | Doesn't verify models/dependencies |
| ort pre-release | `Cargo.toml:45` | 2.0.0-rc.9 may break |
| API key plain text | `agent.rs:134` | No secret management |

---

## P2 - Nice to Have

| Task | File:Line | Description |
|------|-----------|-------------|
| No Opus codec | `audio.rs:63` | For WebRTC compatibility |
| No speaker diarization | `transcript.rs:28` | Add speaker_id field |
| Hard-coded gold price | `customer.rs:226` | Should use config |
| String error variants | `error.rs:32` | Use structured types |
| Static transitions | `conversation.rs:30` | Use lazy_static |
| Model path validation | `settings.rs:51` | Only warns, doesn't fail |
| AEC flags unused | `pipeline.rs:408` | Implement or remove |
| Passive session cleanup | `session.rs:123` | No background task |
| Outdated opentelemetry | `Cargo.toml:64` | 0.21 vs current 0.27 |
| No cargo-deny | Cargo.toml | Security auditing |

---

## WebRTC Implementation Plan

**Status**: NOT IMPLEMENTED - Critical Gap

### Recommended Approach

1. Create new transport crate:
```
voice-agent-rust/crates/transport/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── trait.rs        # Transport trait
│   ├── websocket.rs    # Existing, move here
│   └── webrtc.rs       # New
```

2. Use `webrtc-rs` crate for Rust-native WebRTC

3. Key components:
```rust
// trait.rs
#[async_trait]
pub trait Transport: Send + Sync {
    async fn send_audio(&self, frame: AudioFrame) -> Result<(), TransportError>;
    async fn recv_audio(&self) -> Result<AudioFrame, TransportError>;
    async fn send_text(&self, msg: &str) -> Result<(), TransportError>;
    async fn recv_text(&self) -> Result<String, TransportError>;
}

// webrtc.rs
pub struct WebRTCTransport {
    peer_connection: RTCPeerConnection,
    audio_track: Arc<TrackLocalStaticSample>,
    data_channel: Arc<RTCDataChannel>,
}
```

4. ICE/STUN/TURN configuration:
```toml
[transport.webrtc]
stun_servers = ["stun:stun.l.google.com:19302"]
turn_servers = []
ice_candidate_pool_size = 10
```

---

## Observability Implementation Plan

**Status**: Dependencies exist but NOT USED

### Step 1: Initialize in main.rs

```rust
// main.rs
use opentelemetry::global;
use opentelemetry_otlp::WithExportConfig;
use tracing_subscriber::prelude::*;

async fn init_telemetry(config: &Settings) -> Result<(), Box<dyn std::error::Error>> {
    // Tracer
    let tracer = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_exporter(opentelemetry_otlp::new_exporter().tonic())
        .install_batch(opentelemetry::runtime::Tokio)?;

    // Subscriber
    tracing_subscriber::registry()
        .with(tracing_opentelemetry::layer().with_tracer(tracer))
        .with(tracing_subscriber::fmt::layer())
        .init();

    Ok(())
}
```

### Step 2: Define Key Metrics

```rust
// metrics.rs
use metrics::{counter, histogram, gauge};

pub fn record_e2e_latency(duration_ms: f64) {
    histogram!("voice_agent.e2e_latency_ms").record(duration_ms);
}

pub fn record_stt_latency(duration_ms: f64) {
    histogram!("voice_agent.stt_latency_ms").record(duration_ms);
}

pub fn increment_session_count() {
    gauge!("voice_agent.active_sessions").increment(1.0);
}
```

### Step 3: Expose /metrics Endpoint

```rust
// http.rs
async fn metrics_handler() -> impl IntoResponse {
    let metrics = prometheus::TextEncoder::new()
        .encode_to_string(&prometheus::gather())
        .unwrap_or_default();
    ([(header::CONTENT_TYPE, "text/plain")], metrics)
}
```

---

## Graceful Shutdown Plan

```rust
// main.rs
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c().await.expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("Shutdown signal received, draining connections...");
}

async fn main() {
    // ...
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    // Cleanup
    global::shutdown_tracer_provider();
}
```

---

## Test Coverage

| File | Tests | Quality |
|------|-------|---------|
| audio.rs | 4 | Good |
| transcript.rs | 3 | Good |
| customer.rs | 3 | Good |
| websocket.rs | 0 | None |
| http.rs | 0 | None |
| session.rs | 0 | None |

**Missing:**
- WebSocket message handling tests
- HTTP endpoint integration tests
- Session lifecycle tests

---

## Implementation Priorities

### Week 1: Critical Security & Observability
1. Add rate limiting to WebSocket
2. Initialize observability stack
3. Remove wildcard CORS default

### Week 2: Transport & Sessions
1. Start WebRTC crate
2. Add session persistence (Redis)
3. Add graceful shutdown

### Week 3: Reliability
1. Add health check dependencies
2. Upgrade ort to stable
3. Add authentication middleware

---

*Last Updated: 2024-12-27*
