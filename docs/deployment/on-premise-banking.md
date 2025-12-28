# On-Premise Banking Deployment Guide

## Overview

This document details the deployment architecture for the voice agent in an on-premise banking environment. Banks have stringent requirements around data residency, security, compliance, and audit trails that fundamentally shape deployment architecture.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ON-PREMISE BANKING ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                         DMZ (Demilitarized Zone)                      │  │
│   │   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │  │
│   │   │   WAF/DDoS   │───▶│   Load       │───▶│   API Gateway        │  │  │
│   │   │   Protection │    │   Balancer   │    │   (Rate Limiting)    │  │  │
│   │   └──────────────┘    └──────────────┘    └──────────────────────┘  │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                        │                                    │
│                                        ▼                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    Application Zone (Private)                        │  │
│   │   ┌──────────────────────────────────────────────────────────────┐  │  │
│   │   │                 Voice Agent Cluster                           │  │  │
│   │   │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────┐ │  │  │
│   │   │  │ Audio   │  │  STT    │  │  TTS    │  │   Orchestrator  │ │  │  │
│   │   │  │ Gateway │  │ Workers │  │ Workers │  │   (Stateless)   │ │  │  │
│   │   │  └─────────┘  └─────────┘  └─────────┘  └─────────────────┘ │  │  │
│   │   └──────────────────────────────────────────────────────────────┘  │  │
│   │                                                                      │  │
│   │   ┌──────────────────────────────────────────────────────────────┐  │  │
│   │   │                    AI Services Layer                          │  │  │
│   │   │  ┌───────────────┐  ┌───────────────┐  ┌──────────────────┐ │  │  │
│   │   │  │  LLM Inference │  │  RAG Engine   │  │  Translation     │ │  │  │
│   │   │  │  (GPU Cluster) │  │  (Vector DB)  │  │  Service         │ │  │  │
│   │   │  └───────────────┘  └───────────────┘  └──────────────────┘ │  │  │
│   │   └──────────────────────────────────────────────────────────────┘  │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                        │                                    │
│                                        ▼                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      Data Zone (Restricted)                          │  │
│   │   ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐   │  │
│   │   │   Customer   │  │   Vector     │  │   Audit Logs           │   │  │
│   │   │   Database   │  │   Store      │  │   (Immutable)          │   │  │
│   │   └──────────────┘  └──────────────┘  └────────────────────────┘   │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Security Architecture

### Defense in Depth

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SECURITY LAYERS                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Layer 1: Network Security                                                  │
│   ─────────────────────────────────────────────────────────────────────────│
│   • Network segmentation (VLANs)                                            │
│   • Micro-segmentation between services                                     │
│   • Zero-trust network architecture                                         │
│   • East-west traffic encryption (mTLS)                                     │
│                                                                              │
│   Layer 2: Application Security                                              │
│   ─────────────────────────────────────────────────────────────────────────│
│   • Input validation at every boundary                                      │
│   • Content Security Policy (CSP)                                           │
│   • API authentication (mTLS + JWT)                                         │
│   • Rate limiting and throttling                                            │
│                                                                              │
│   Layer 3: Data Security                                                     │
│   ─────────────────────────────────────────────────────────────────────────│
│   • Encryption at rest (AES-256)                                            │
│   • Encryption in transit (TLS 1.3)                                         │
│   • Field-level encryption for PII                                          │
│   • Hardware Security Modules (HSM) for key management                      │
│                                                                              │
│   Layer 4: Access Control                                                    │
│   ─────────────────────────────────────────────────────────────────────────│
│   • Role-based access control (RBAC)                                        │
│   • Just-in-time access provisioning                                        │
│   • Multi-factor authentication                                             │
│   • Privileged Access Management (PAM)                                      │
│                                                                              │
│   Layer 5: Monitoring & Audit                                                │
│   ─────────────────────────────────────────────────────────────────────────│
│   • Real-time security monitoring (SIEM)                                    │
│   • Immutable audit logs                                                    │
│   • Anomaly detection                                                       │
│   • Incident response automation                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Network Segmentation

```rust
/// Network zone definitions
pub enum NetworkZone {
    /// Internet-facing, heavily firewalled
    Dmz,
    /// Application servers, no direct internet access
    Application,
    /// AI workloads (GPU servers)
    AiCompute,
    /// Database and storage
    Data,
    /// Management and monitoring
    Management,
}

/// Allowed network flows
pub struct NetworkPolicy {
    pub from_zone: NetworkZone,
    pub to_zone: NetworkZone,
    pub protocol: Protocol,
    pub ports: Vec<u16>,
    pub direction: Direction,
}

/// Define allowed inter-zone communication
pub fn get_network_policies() -> Vec<NetworkPolicy> {
    vec![
        // DMZ can reach Application zone only
        NetworkPolicy {
            from_zone: NetworkZone::Dmz,
            to_zone: NetworkZone::Application,
            protocol: Protocol::Https,
            ports: vec![443],
            direction: Direction::Inbound,
        },
        // Application can reach AI Compute
        NetworkPolicy {
            from_zone: NetworkZone::Application,
            to_zone: NetworkZone::AiCompute,
            protocol: Protocol::Grpc,
            ports: vec![50051, 50052],
            direction: Direction::Outbound,
        },
        // Application can reach Data zone
        NetworkPolicy {
            from_zone: NetworkZone::Application,
            to_zone: NetworkZone::Data,
            protocol: Protocol::Custom("postgres".into()),
            ports: vec![5432],
            direction: Direction::Outbound,
        },
        // AI Compute can reach Data zone (vector DB)
        NetworkPolicy {
            from_zone: NetworkZone::AiCompute,
            to_zone: NetworkZone::Data,
            protocol: Protocol::Grpc,
            ports: vec![6333, 6334], // Qdrant ports
            direction: Direction::Outbound,
        },
        // Management can reach all zones (monitoring)
        NetworkPolicy {
            from_zone: NetworkZone::Management,
            to_zone: NetworkZone::Application,
            protocol: Protocol::Https,
            ports: vec![9090, 9091], // Prometheus/Grafana
            direction: Direction::Outbound,
        },
    ]
}
```

### mTLS Configuration

```rust
use rustls::{Certificate, PrivateKey, ServerConfig, ClientConfig};
use std::sync::Arc;

/// mTLS configuration for service-to-service communication
pub struct MtlsConfig {
    /// Server certificate chain
    pub server_certs: Vec<Certificate>,
    /// Server private key
    pub server_key: PrivateKey,
    /// CA certificates for client verification
    pub ca_certs: Vec<Certificate>,
    /// Client certificate for outbound connections
    pub client_cert: Certificate,
    /// Client private key
    pub client_key: PrivateKey,
}

impl MtlsConfig {
    /// Load certificates from HSM or secure storage
    pub async fn from_hsm(hsm_config: &HsmConfig) -> Result<Self, SecurityError> {
        let hsm = HsmClient::connect(hsm_config).await?;

        Ok(Self {
            server_certs: hsm.get_certificate_chain("voice-agent-server").await?,
            server_key: hsm.get_private_key("voice-agent-server").await?,
            ca_certs: hsm.get_ca_certificates().await?,
            client_cert: hsm.get_certificate("voice-agent-client").await?,
            client_key: hsm.get_private_key("voice-agent-client").await?,
        })
    }

    /// Build server TLS configuration
    pub fn server_config(&self) -> Result<Arc<ServerConfig>, SecurityError> {
        let client_cert_verifier = rustls::server::AllowAnyAuthenticatedClient::new(
            self.build_root_cert_store()?
        );

        let config = ServerConfig::builder()
            .with_safe_defaults()
            .with_client_cert_verifier(Arc::new(client_cert_verifier))
            .with_single_cert(
                self.server_certs.clone(),
                self.server_key.clone(),
            )?;

        Ok(Arc::new(config))
    }

    /// Build client TLS configuration
    pub fn client_config(&self) -> Result<Arc<ClientConfig>, SecurityError> {
        let config = ClientConfig::builder()
            .with_safe_defaults()
            .with_root_certificates(self.build_root_cert_store()?)
            .with_client_auth_cert(
                vec![self.client_cert.clone()],
                self.client_key.clone(),
            )?;

        Ok(Arc::new(config))
    }

    fn build_root_cert_store(&self) -> Result<rustls::RootCertStore, SecurityError> {
        let mut store = rustls::RootCertStore::empty();
        for cert in &self.ca_certs {
            store.add(cert)?;
        }
        Ok(store)
    }
}
```

## Compliance Requirements

### RBI Guidelines for Voice AI in Banking

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RBI COMPLIANCE REQUIREMENTS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   1. Data Localization (RBI/2017-18/153)                                    │
│   ─────────────────────────────────────────────────────────────────────────│
│   • All customer data must be stored within India                           │
│   • Processing must occur within Indian data centers                        │
│   • No cross-border data transfer without explicit consent                  │
│   • Implementation: On-premise deployment mandatory                         │
│                                                                              │
│   2. Customer Consent & Disclosure                                           │
│   ─────────────────────────────────────────────────────────────────────────│
│   • Clear disclosure that customer is interacting with AI                   │
│   • Opt-out mechanism to speak with human agent                             │
│   • Recording consent for voice conversations                               │
│   • Implementation: Pre-call IVR consent + in-call disclosure              │
│                                                                              │
│   3. Data Retention & Privacy                                                │
│   ─────────────────────────────────────────────────────────────────────────│
│   • Voice recordings: Maximum 7 years retention                             │
│   • Transaction logs: 10 years retention                                    │
│   • Right to erasure (with exceptions for regulatory records)               │
│   • Implementation: Automated retention policies                            │
│                                                                              │
│   4. Audit Trail Requirements                                                │
│   ─────────────────────────────────────────────────────────────────────────│
│   • Complete audit trail of all AI decisions                                │
│   • Explainability for loan recommendations                                 │
│   • Tamper-proof logging                                                    │
│   • Implementation: Immutable audit log with merkle tree verification       │
│                                                                              │
│   5. Risk Management                                                         │
│   ─────────────────────────────────────────────────────────────────────────│
│   • AI model risk assessment before deployment                              │
│   • Regular model validation and backtesting                                │
│   • Human oversight for high-value decisions                                │
│   • Implementation: Model governance framework                              │
│                                                                              │
│   6. Business Continuity                                                     │
│   ─────────────────────────────────────────────────────────────────────────│
│   • Disaster recovery within 4 hours (RTO)                                  │
│   • Data loss tolerance: Near-zero (RPO)                                    │
│   • Annual DR testing mandatory                                             │
│   • Implementation: Multi-DC active-passive with async replication          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Compliance Implementation

```rust
/// Compliance configuration
#[derive(Debug, Clone)]
pub struct ComplianceConfig {
    /// Enable AI disclosure at conversation start
    pub ai_disclosure_enabled: bool,
    /// Disclosure message
    pub ai_disclosure_message: String,

    /// Enable human handoff option
    pub human_handoff_enabled: bool,
    /// Phrases that trigger human handoff offer
    pub human_handoff_triggers: Vec<String>,

    /// Recording consent required
    pub recording_consent_required: bool,

    /// Maximum data retention periods
    pub retention_policy: RetentionPolicy,

    /// PII handling rules
    pub pii_policy: PiiPolicy,
}

#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Voice recordings retention (days)
    pub voice_recording_days: u32,
    /// Transaction logs retention (days)
    pub transaction_log_days: u32,
    /// Chat transcripts retention (days)
    pub transcript_days: u32,
    /// PII data retention (days)
    pub pii_data_days: u32,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            voice_recording_days: 2555, // 7 years
            transaction_log_days: 3650, // 10 years
            transcript_days: 2555,      // 7 years
            pii_data_days: 2555,        // 7 years
        }
    }
}

#[derive(Debug, Clone)]
pub struct PiiPolicy {
    /// Mask PII in logs
    pub mask_in_logs: bool,
    /// Encrypt PII at rest (field-level)
    pub encrypt_at_rest: bool,
    /// Tokenize PII for analytics
    pub tokenize_for_analytics: bool,
    /// PII access requires justification
    pub access_requires_justification: bool,
}

/// Compliance service
pub struct ComplianceService {
    config: ComplianceConfig,
    audit_log: Arc<dyn AuditLog>,
    pii_handler: Arc<dyn PiiHandler>,
}

impl ComplianceService {
    /// Check if AI disclosure is needed
    pub fn needs_disclosure(&self, conversation: &Conversation) -> bool {
        self.config.ai_disclosure_enabled && !conversation.has_received_disclosure
    }

    /// Get disclosure message
    pub fn get_disclosure_message(&self, language: &Language) -> String {
        // Localized disclosure
        match language.code.as_str() {
            "hi" => "यह एक AI सहायक है। आप किसी भी समय मानव एजेंट से बात कर सकते हैं।".into(),
            "ta" => "இது AI உதவியாளர். நீங்கள் எந்த நேரத்திலும் மனித முகவருடன் பேசலாம்.".into(),
            _ => "This is an AI assistant. You can speak with a human agent at any time.".into(),
        }
    }

    /// Check for human handoff triggers
    pub fn check_handoff_triggers(&self, text: &str) -> bool {
        if !self.config.human_handoff_enabled {
            return false;
        }

        let text_lower = text.to_lowercase();
        self.config.human_handoff_triggers.iter().any(|trigger| {
            text_lower.contains(&trigger.to_lowercase())
        })
    }

    /// Log compliance event
    pub async fn log_event(&self, event: ComplianceEvent) -> Result<(), ComplianceError> {
        self.audit_log.log(AuditEntry {
            timestamp: chrono::Utc::now(),
            event_type: event.event_type,
            conversation_id: event.conversation_id,
            details: event.details,
            actor: event.actor,
            compliance_tags: event.compliance_tags,
        }).await
    }
}

#[derive(Debug)]
pub struct ComplianceEvent {
    pub event_type: ComplianceEventType,
    pub conversation_id: String,
    pub details: serde_json::Value,
    pub actor: Actor,
    pub compliance_tags: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
pub enum ComplianceEventType {
    AiDisclosureGiven,
    RecordingConsentObtained,
    HumanHandoffRequested,
    PiiAccessed,
    PiiRedacted,
    LoanRecommendationMade,
    DataRetentionPolicyApplied,
}
```

## Hardware Configurations

### Tier-Based Hardware Profiles

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HARDWARE TIER CONFIGURATIONS                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   TIER 1: PILOT / POC (50-100 concurrent sessions)                          │
│   ─────────────────────────────────────────────────────────────────────────│
│   Application Servers (3x HA):                                               │
│   • CPU: 16 cores (Intel Xeon Gold 6326)                                    │
│   • RAM: 64 GB                                                              │
│   • Storage: 500 GB NVMe SSD                                                │
│   • Network: 10 Gbps                                                        │
│                                                                              │
│   GPU Servers (2x):                                                          │
│   • GPU: NVIDIA A10 (24 GB VRAM each)                                       │
│   • CPU: 32 cores                                                           │
│   • RAM: 128 GB                                                             │
│   • Storage: 1 TB NVMe                                                      │
│                                                                              │
│   Database (3-node cluster):                                                 │
│   • CPU: 8 cores                                                            │
│   • RAM: 64 GB                                                              │
│   • Storage: 2 TB NVMe (RAID 10)                                            │
│                                                                              │
│   ─────────────────────────────────────────────────────────────────────────│
│                                                                              │
│   TIER 2: PRODUCTION (500-1000 concurrent sessions)                          │
│   ─────────────────────────────────────────────────────────────────────────│
│   Application Servers (6x HA):                                               │
│   • CPU: 32 cores (Intel Xeon Platinum 8358)                                │
│   • RAM: 128 GB                                                             │
│   • Storage: 1 TB NVMe SSD                                                  │
│   • Network: 25 Gbps                                                        │
│                                                                              │
│   GPU Servers (4x):                                                          │
│   • GPU: NVIDIA A100 (40 GB VRAM each)                                      │
│   • CPU: 64 cores                                                           │
│   • RAM: 256 GB                                                             │
│   • Storage: 2 TB NVMe                                                      │
│   • NVLink interconnect                                                     │
│                                                                              │
│   Database (5-node cluster):                                                 │
│   • CPU: 16 cores                                                           │
│   • RAM: 128 GB                                                             │
│   • Storage: 4 TB NVMe (RAID 10)                                            │
│                                                                              │
│   Vector Database (3-node):                                                  │
│   • CPU: 16 cores                                                           │
│   • RAM: 256 GB (for in-memory indexes)                                     │
│   • Storage: 2 TB NVMe                                                      │
│                                                                              │
│   ─────────────────────────────────────────────────────────────────────────│
│                                                                              │
│   TIER 3: ENTERPRISE (5000+ concurrent sessions)                             │
│   ─────────────────────────────────────────────────────────────────────────│
│   Application Servers (12x HA):                                              │
│   • CPU: 64 cores (AMD EPYC 7763)                                           │
│   • RAM: 256 GB                                                             │
│   • Storage: 2 TB NVMe SSD                                                  │
│   • Network: 100 Gbps                                                       │
│                                                                              │
│   GPU Cluster (8x):                                                          │
│   • GPU: NVIDIA A100 (80 GB VRAM each)                                      │
│   • Multi-instance GPU (MIG) enabled                                        │
│   • CPU: 128 cores                                                          │
│   • RAM: 512 GB                                                             │
│   • InfiniBand interconnect (200 Gbps)                                      │
│                                                                              │
│   Database (7-node cluster + read replicas):                                 │
│   • CPU: 32 cores                                                           │
│   • RAM: 256 GB                                                             │
│   • Storage: 8 TB NVMe (RAID 10)                                            │
│   • Separate read replicas for analytics                                    │
│                                                                              │
│   Vector Database (5-node + sharding):                                       │
│   • CPU: 32 cores                                                           │
│   • RAM: 512 GB                                                             │
│   • Storage: 4 TB NVMe per node                                             │
│   • Horizontal sharding for scale                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Hardware Configuration Code

```rust
/// Hardware tier enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareTier {
    /// 50-100 concurrent sessions
    Pilot,
    /// 500-1000 concurrent sessions
    Production,
    /// 5000+ concurrent sessions
    Enterprise,
}

/// Hardware profile for deployment
#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub tier: HardwareTier,
    pub app_servers: ServerSpec,
    pub gpu_servers: GpuServerSpec,
    pub database: DatabaseSpec,
    pub vector_db: VectorDbSpec,
}

#[derive(Debug, Clone)]
pub struct ServerSpec {
    pub count: u32,
    pub cpu_cores: u32,
    pub ram_gb: u32,
    pub storage_gb: u32,
    pub network_gbps: u32,
}

#[derive(Debug, Clone)]
pub struct GpuServerSpec {
    pub count: u32,
    pub gpu_model: String,
    pub gpu_vram_gb: u32,
    pub cpu_cores: u32,
    pub ram_gb: u32,
    pub storage_gb: u32,
}

#[derive(Debug, Clone)]
pub struct DatabaseSpec {
    pub cluster_nodes: u32,
    pub cpu_cores: u32,
    pub ram_gb: u32,
    pub storage_gb: u32,
    pub read_replicas: u32,
}

#[derive(Debug, Clone)]
pub struct VectorDbSpec {
    pub cluster_nodes: u32,
    pub cpu_cores: u32,
    pub ram_gb: u32,
    pub storage_gb: u32,
    pub shards: u32,
}

impl HardwareProfile {
    pub fn for_tier(tier: HardwareTier) -> Self {
        match tier {
            HardwareTier::Pilot => Self::pilot(),
            HardwareTier::Production => Self::production(),
            HardwareTier::Enterprise => Self::enterprise(),
        }
    }

    fn pilot() -> Self {
        Self {
            tier: HardwareTier::Pilot,
            app_servers: ServerSpec {
                count: 3,
                cpu_cores: 16,
                ram_gb: 64,
                storage_gb: 500,
                network_gbps: 10,
            },
            gpu_servers: GpuServerSpec {
                count: 2,
                gpu_model: "NVIDIA A10".into(),
                gpu_vram_gb: 24,
                cpu_cores: 32,
                ram_gb: 128,
                storage_gb: 1000,
            },
            database: DatabaseSpec {
                cluster_nodes: 3,
                cpu_cores: 8,
                ram_gb: 64,
                storage_gb: 2000,
                read_replicas: 0,
            },
            vector_db: VectorDbSpec {
                cluster_nodes: 1,
                cpu_cores: 8,
                ram_gb: 64,
                storage_gb: 500,
                shards: 1,
            },
        }
    }

    fn production() -> Self {
        Self {
            tier: HardwareTier::Production,
            app_servers: ServerSpec {
                count: 6,
                cpu_cores: 32,
                ram_gb: 128,
                storage_gb: 1000,
                network_gbps: 25,
            },
            gpu_servers: GpuServerSpec {
                count: 4,
                gpu_model: "NVIDIA A100 40GB".into(),
                gpu_vram_gb: 40,
                cpu_cores: 64,
                ram_gb: 256,
                storage_gb: 2000,
            },
            database: DatabaseSpec {
                cluster_nodes: 5,
                cpu_cores: 16,
                ram_gb: 128,
                storage_gb: 4000,
                read_replicas: 2,
            },
            vector_db: VectorDbSpec {
                cluster_nodes: 3,
                cpu_cores: 16,
                ram_gb: 256,
                storage_gb: 2000,
                shards: 3,
            },
        }
    }

    fn enterprise() -> Self {
        Self {
            tier: HardwareTier::Enterprise,
            app_servers: ServerSpec {
                count: 12,
                cpu_cores: 64,
                ram_gb: 256,
                storage_gb: 2000,
                network_gbps: 100,
            },
            gpu_servers: GpuServerSpec {
                count: 8,
                gpu_model: "NVIDIA A100 80GB".into(),
                gpu_vram_gb: 80,
                cpu_cores: 128,
                ram_gb: 512,
                storage_gb: 4000,
            },
            database: DatabaseSpec {
                cluster_nodes: 7,
                cpu_cores: 32,
                ram_gb: 256,
                storage_gb: 8000,
                read_replicas: 4,
            },
            vector_db: VectorDbSpec {
                cluster_nodes: 5,
                cpu_cores: 32,
                ram_gb: 512,
                storage_gb: 4000,
                shards: 10,
            },
        }
    }

    /// Calculate estimated concurrent session capacity
    pub fn estimated_capacity(&self) -> u32 {
        // Based on profiling: ~50 sessions per GPU (VRAM limited)
        // and ~20 sessions per app server core
        let gpu_capacity = self.gpu_servers.count * 50;
        let cpu_capacity = self.app_servers.count * self.app_servers.cpu_cores * 2;

        gpu_capacity.min(cpu_capacity)
    }

    /// Calculate estimated cost per month (approximate)
    pub fn estimated_monthly_cost_usd(&self) -> u32 {
        let app_cost = self.app_servers.count * 800; // ~$800/server/month
        let gpu_cost = self.gpu_servers.count * 5000; // ~$5000/GPU server/month
        let db_cost = self.database.cluster_nodes * 500;
        let vector_cost = self.vector_db.cluster_nodes * 400;

        app_cost + gpu_cost + db_cost + vector_cost
    }
}
```

## Audit Logging

### Immutable Audit Trail

```rust
use sha2::{Sha256, Digest};
use chrono::{DateTime, Utc};

/// Audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Unique entry ID
    pub id: String,

    /// Timestamp (UTC)
    pub timestamp: DateTime<Utc>,

    /// Event type
    pub event_type: String,

    /// Actor (user, system, or service)
    pub actor: Actor,

    /// Target resource
    pub resource: Resource,

    /// Action performed
    pub action: String,

    /// Outcome (success/failure)
    pub outcome: Outcome,

    /// Additional details (JSON)
    pub details: serde_json::Value,

    /// Previous entry hash (merkle chain)
    pub previous_hash: String,

    /// This entry's hash
    pub hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Actor {
    pub actor_type: ActorType,
    pub id: String,
    pub name: Option<String>,
    pub ip_address: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ActorType {
    User,
    System,
    Service,
    Customer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    pub resource_type: String,
    pub id: String,
    pub attributes: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Outcome {
    Success,
    Failure,
    Partial,
}

impl AuditEntry {
    /// Create new audit entry with hash chain
    pub fn new(
        event_type: impl Into<String>,
        actor: Actor,
        resource: Resource,
        action: impl Into<String>,
        outcome: Outcome,
        details: serde_json::Value,
        previous_hash: String,
    ) -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        let timestamp = Utc::now();
        let event_type = event_type.into();
        let action = action.into();

        // Calculate hash
        let hash = Self::calculate_hash(
            &id,
            &timestamp,
            &event_type,
            &actor,
            &action,
            &outcome,
            &details,
            &previous_hash,
        );

        Self {
            id,
            timestamp,
            event_type,
            actor,
            resource,
            action,
            outcome,
            details,
            previous_hash,
            hash,
        }
    }

    fn calculate_hash(
        id: &str,
        timestamp: &DateTime<Utc>,
        event_type: &str,
        actor: &Actor,
        action: &str,
        outcome: &Outcome,
        details: &serde_json::Value,
        previous_hash: &str,
    ) -> String {
        let mut hasher = Sha256::new();

        hasher.update(id.as_bytes());
        hasher.update(timestamp.to_rfc3339().as_bytes());
        hasher.update(event_type.as_bytes());
        hasher.update(serde_json::to_string(actor).unwrap_or_default().as_bytes());
        hasher.update(action.as_bytes());
        hasher.update(format!("{:?}", outcome).as_bytes());
        hasher.update(details.to_string().as_bytes());
        hasher.update(previous_hash.as_bytes());

        format!("{:x}", hasher.finalize())
    }

    /// Verify hash chain integrity
    pub fn verify(&self, expected_previous_hash: &str) -> bool {
        if self.previous_hash != expected_previous_hash {
            return false;
        }

        let calculated = Self::calculate_hash(
            &self.id,
            &self.timestamp,
            &self.event_type,
            &self.actor,
            &self.action,
            &self.outcome,
            &self.details,
            &self.previous_hash,
        );

        calculated == self.hash
    }
}

/// Audit log service
#[async_trait::async_trait]
pub trait AuditLog: Send + Sync {
    /// Log an audit entry
    async fn log(&self, entry: AuditEntry) -> Result<(), AuditError>;

    /// Query audit entries
    async fn query(&self, query: AuditQuery) -> Result<Vec<AuditEntry>, AuditError>;

    /// Verify chain integrity
    async fn verify_integrity(&self, from: DateTime<Utc>, to: DateTime<Utc>)
        -> Result<IntegrityReport, AuditError>;

    /// Export audit log for regulatory submission
    async fn export(&self, query: AuditQuery, format: ExportFormat)
        -> Result<Vec<u8>, AuditError>;
}

#[derive(Debug)]
pub struct AuditQuery {
    pub from: Option<DateTime<Utc>>,
    pub to: Option<DateTime<Utc>>,
    pub event_types: Option<Vec<String>>,
    pub actor_id: Option<String>,
    pub resource_id: Option<String>,
    pub limit: Option<u32>,
    pub offset: Option<u32>,
}

#[derive(Debug)]
pub struct IntegrityReport {
    pub verified_entries: u64,
    pub invalid_entries: Vec<String>,
    pub chain_breaks: Vec<ChainBreak>,
    pub is_valid: bool,
}

#[derive(Debug)]
pub struct ChainBreak {
    pub entry_id: String,
    pub expected_previous: String,
    pub actual_previous: String,
}

#[derive(Debug, Clone, Copy)]
pub enum ExportFormat {
    Json,
    Csv,
    Pdf,
}
```

## Data Encryption

### Field-Level Encryption for PII

```rust
use aes_gcm::{Aes256Gcm, Key, Nonce, aead::{Aead, NewAead}};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};

/// Field-level encryption for PII
pub struct FieldEncryption {
    cipher: Aes256Gcm,
    key_id: String,
}

impl FieldEncryption {
    /// Create from HSM-managed key
    pub async fn from_hsm(hsm: &HsmClient, key_id: &str) -> Result<Self, SecurityError> {
        let key_bytes = hsm.get_data_encryption_key(key_id).await?;
        let key = Key::<Aes256Gcm>::from_slice(&key_bytes);
        let cipher = Aes256Gcm::new(key);

        Ok(Self {
            cipher,
            key_id: key_id.to_string(),
        })
    }

    /// Encrypt a field value
    pub fn encrypt(&self, plaintext: &str) -> Result<EncryptedField, SecurityError> {
        // Generate random nonce
        let nonce_bytes: [u8; 12] = rand::random();
        let nonce = Nonce::from_slice(&nonce_bytes);

        // Encrypt
        let ciphertext = self.cipher.encrypt(nonce, plaintext.as_bytes())
            .map_err(|_| SecurityError::EncryptionFailed)?;

        Ok(EncryptedField {
            key_id: self.key_id.clone(),
            nonce: BASE64.encode(nonce_bytes),
            ciphertext: BASE64.encode(ciphertext),
        })
    }

    /// Decrypt a field value
    pub fn decrypt(&self, encrypted: &EncryptedField) -> Result<String, SecurityError> {
        // Verify key ID matches
        if encrypted.key_id != self.key_id {
            return Err(SecurityError::KeyMismatch);
        }

        // Decode
        let nonce_bytes = BASE64.decode(&encrypted.nonce)
            .map_err(|_| SecurityError::InvalidCiphertext)?;
        let ciphertext = BASE64.decode(&encrypted.ciphertext)
            .map_err(|_| SecurityError::InvalidCiphertext)?;

        let nonce = Nonce::from_slice(&nonce_bytes);

        // Decrypt
        let plaintext = self.cipher.decrypt(nonce, ciphertext.as_slice())
            .map_err(|_| SecurityError::DecryptionFailed)?;

        String::from_utf8(plaintext)
            .map_err(|_| SecurityError::InvalidPlaintext)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedField {
    pub key_id: String,
    pub nonce: String,      // Base64-encoded
    pub ciphertext: String, // Base64-encoded
}

/// Customer record with encrypted PII
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedCustomer {
    pub id: String,
    pub account_number: String, // Not PII, plain text

    // Encrypted PII fields
    pub name: EncryptedField,
    pub phone: EncryptedField,
    pub email: Option<EncryptedField>,
    pub aadhaar: Option<EncryptedField>,
    pub pan: Option<EncryptedField>,
    pub address: Option<EncryptedField>,

    // Non-PII metadata (plain)
    pub segment: String,
    pub created_at: DateTime<Utc>,
    pub last_interaction: DateTime<Utc>,
}
```

## Disaster Recovery

### Multi-Datacenter Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DISASTER RECOVERY ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   PRIMARY DC (Mumbai)                    SECONDARY DC (Chennai)              │
│   ┌─────────────────────────┐           ┌─────────────────────────┐        │
│   │                         │           │                         │        │
│   │   ┌───────────────┐    │  Async    │   ┌───────────────┐    │        │
│   │   │ Voice Agent   │────┼──Repl.───▶│   │ Voice Agent   │    │        │
│   │   │ Cluster       │    │           │   │ Cluster       │    │        │
│   │   │ (ACTIVE)      │    │           │   │ (STANDBY)     │    │        │
│   │   └───────────────┘    │           │   └───────────────┘    │        │
│   │                         │           │                         │        │
│   │   ┌───────────────┐    │  Sync     │   ┌───────────────┐    │        │
│   │   │ Database      │────┼──Repl.───▶│   │ Database      │    │        │
│   │   │ (Primary)     │    │           │   │ (Replica)     │    │        │
│   │   └───────────────┘    │           │   └───────────────┘    │        │
│   │                         │           │                         │        │
│   │   ┌───────────────┐    │  Async    │   ┌───────────────┐    │        │
│   │   │ Vector DB     │────┼──Repl.───▶│   │ Vector DB     │    │        │
│   │   │               │    │           │   │               │    │        │
│   │   └───────────────┘    │           │   └───────────────┘    │        │
│   │                         │           │                         │        │
│   │   ┌───────────────┐    │  Async    │   ┌───────────────┐    │        │
│   │   │ Model Store   │────┼──Sync────▶│   │ Model Store   │    │        │
│   │   │               │    │           │   │ (Full Copy)   │    │        │
│   │   └───────────────┘    │           │   └───────────────┘    │        │
│   │                         │           │                         │        │
│   └─────────────────────────┘           └─────────────────────────┘        │
│                                                                              │
│   RTO: 4 hours                          RPO: Near-zero (sync repl.)         │
│   Failover: Automatic (DNS + Health)    Failback: Manual                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### DR Configuration

```rust
/// Disaster recovery configuration
#[derive(Debug, Clone)]
pub struct DisasterRecoveryConfig {
    /// Recovery Time Objective (minutes)
    pub rto_minutes: u32,

    /// Recovery Point Objective (minutes)
    pub rpo_minutes: u32,

    /// Primary datacenter
    pub primary_dc: DatacenterConfig,

    /// Secondary datacenter
    pub secondary_dc: DatacenterConfig,

    /// Replication configuration
    pub replication: ReplicationConfig,

    /// Failover configuration
    pub failover: FailoverConfig,
}

#[derive(Debug, Clone)]
pub struct DatacenterConfig {
    pub name: String,
    pub region: String,
    pub endpoints: DatacenterEndpoints,
    pub is_active: bool,
}

#[derive(Debug, Clone)]
pub struct DatacenterEndpoints {
    pub api_gateway: String,
    pub database: String,
    pub vector_db: String,
    pub model_store: String,
}

#[derive(Debug, Clone)]
pub struct ReplicationConfig {
    /// Database replication mode
    pub database_mode: ReplicationMode,

    /// Vector DB replication mode
    pub vector_db_mode: ReplicationMode,

    /// Model sync mode
    pub model_sync_mode: ReplicationMode,

    /// Replication lag threshold for alerts (seconds)
    pub lag_alert_threshold_seconds: u32,
}

#[derive(Debug, Clone, Copy)]
pub enum ReplicationMode {
    /// Synchronous replication (zero data loss)
    Synchronous,
    /// Asynchronous replication (potential data loss)
    Asynchronous,
    /// Periodic sync (batch)
    Periodic { interval_minutes: u32 },
}

#[derive(Debug, Clone)]
pub struct FailoverConfig {
    /// Automatic failover enabled
    pub automatic: bool,

    /// Health check interval (seconds)
    pub health_check_interval_seconds: u32,

    /// Failures before failover
    pub failure_threshold: u32,

    /// Cooldown after failover (minutes)
    pub cooldown_minutes: u32,

    /// Notification endpoints
    pub notification_endpoints: Vec<String>,
}

/// DR orchestrator
pub struct DrOrchestrator {
    config: DisasterRecoveryConfig,
    health_checker: HealthChecker,
    dns_manager: DnsManager,
}

impl DrOrchestrator {
    /// Initiate failover to secondary DC
    pub async fn failover(&self) -> Result<FailoverResult, DrError> {
        tracing::warn!("Initiating failover to secondary datacenter");

        // 1. Verify secondary is healthy
        if !self.health_checker.check_dc(&self.config.secondary_dc).await? {
            return Err(DrError::SecondaryUnhealthy);
        }

        // 2. Stop writes to primary (if accessible)
        self.stop_primary_writes().await.ok(); // Best effort

        // 3. Wait for replication to catch up (with timeout)
        let sync_result = tokio::time::timeout(
            std::time::Duration::from_secs(60),
            self.wait_for_replication_sync(),
        ).await;

        if sync_result.is_err() {
            tracing::warn!("Replication sync timed out, proceeding with potential data loss");
        }

        // 4. Promote secondary
        self.promote_secondary().await?;

        // 5. Update DNS
        self.dns_manager.update_to_secondary().await?;

        // 6. Send notifications
        self.send_failover_notifications().await?;

        Ok(FailoverResult {
            success: true,
            new_active_dc: self.config.secondary_dc.name.clone(),
            data_loss_estimate_seconds: sync_result.err().map(|_| 60).unwrap_or(0),
        })
    }

    /// Check if failover is needed
    pub async fn check_failover_needed(&self) -> bool {
        let primary_health = self.health_checker.check_dc(&self.config.primary_dc).await;

        match primary_health {
            Ok(false) | Err(_) => {
                // Primary unhealthy - check if threshold met
                self.health_checker.consecutive_failures() >= self.config.failover.failure_threshold
            }
            Ok(true) => false,
        }
    }

    async fn stop_primary_writes(&self) -> Result<(), DrError> {
        // Implementation: set database to read-only
        Ok(())
    }

    async fn wait_for_replication_sync(&self) -> Result<(), DrError> {
        // Implementation: poll replication lag until zero
        Ok(())
    }

    async fn promote_secondary(&self) -> Result<(), DrError> {
        // Implementation: promote replica to primary
        Ok(())
    }

    async fn send_failover_notifications(&self) -> Result<(), DrError> {
        // Implementation: send alerts to configured endpoints
        Ok(())
    }
}

#[derive(Debug)]
pub struct FailoverResult {
    pub success: bool,
    pub new_active_dc: String,
    pub data_loss_estimate_seconds: u32,
}
```

## Model Lifecycle Management

### Air-Gapped Model Updates

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MODEL UPDATE WORKFLOW (AIR-GAPPED)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   EXTERNAL (Internet-Connected)         INTERNAL (Air-Gapped)               │
│   ┌─────────────────────────┐           ┌─────────────────────────┐        │
│   │                         │           │                         │        │
│   │   ┌───────────────┐    │           │   ┌───────────────┐    │        │
│   │   │ Model         │    │   USB/    │   │ Staging       │    │        │
│   │   │ Registry      │────┼──Secure──▶│   │ Environment   │    │        │
│   │   │ (Hugging Face)│    │  Transfer │   │               │    │        │
│   │   └───────────────┘    │           │   └───────┬───────┘    │        │
│   │                         │           │           │             │        │
│   │   ┌───────────────┐    │           │           ▼             │        │
│   │   │ Validation    │    │           │   ┌───────────────┐    │        │
│   │   │ Pipeline      │    │           │   │ Security      │    │        │
│   │   │ (CI/CD)       │    │           │   │ Scanning      │    │        │
│   │   └───────────────┘    │           │   └───────┬───────┘    │        │
│   │                         │           │           │             │        │
│   │   ┌───────────────┐    │           │           ▼             │        │
│   │   │ Model Card +  │    │           │   ┌───────────────┐    │        │
│   │   │ Checksum      │    │           │   │ Validation    │    │        │
│   │   │ Generation    │    │           │   │ Testing       │    │        │
│   │   └───────────────┘    │           │   └───────┬───────┘    │        │
│   │                         │           │           │             │        │
│   └─────────────────────────┘           │           ▼             │        │
│                                         │   ┌───────────────┐    │        │
│                                         │   │ Blue/Green    │    │        │
│                                         │   │ Deployment    │    │        │
│                                         │   └───────────────┘    │        │
│                                         │                         │        │
│                                         └─────────────────────────┘        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Model Management Implementation

```rust
/// Model artifact definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArtifact {
    pub id: String,
    pub name: String,
    pub version: String,
    pub model_type: ModelType,

    /// SHA256 checksum of model files
    pub checksum: String,

    /// Model card with performance metrics
    pub model_card: ModelCard,

    /// File paths within artifact
    pub files: Vec<ModelFile>,

    /// Deployment status
    pub status: ModelStatus,

    /// Created timestamp
    pub created_at: DateTime<Utc>,

    /// Validated timestamp
    pub validated_at: Option<DateTime<Utc>>,

    /// Deployed timestamp
    pub deployed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ModelType {
    Stt,
    Tts,
    Llm,
    Translation,
    Embedding,
    Vad,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ModelStatus {
    Received,
    SecurityScanning,
    Validating,
    Validated,
    Deploying,
    Active,
    Deprecated,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCard {
    pub description: String,
    pub training_data: String,
    pub languages: Vec<String>,
    pub performance_metrics: std::collections::HashMap<String, f64>,
    pub limitations: Vec<String>,
    pub ethical_considerations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFile {
    pub path: String,
    pub size_bytes: u64,
    pub checksum: String,
}

/// Model lifecycle manager
pub struct ModelLifecycleManager {
    artifact_store: Arc<dyn ArtifactStore>,
    security_scanner: Arc<dyn SecurityScanner>,
    validator: Arc<dyn ModelValidator>,
    deployer: Arc<dyn ModelDeployer>,
    audit_log: Arc<dyn AuditLog>,
}

impl ModelLifecycleManager {
    /// Receive new model artifact
    pub async fn receive_artifact(
        &self,
        artifact_data: &[u8],
        metadata: ModelMetadata,
        actor: &Actor,
    ) -> Result<ModelArtifact, ModelError> {
        // 1. Verify checksum
        let calculated_checksum = self.calculate_checksum(artifact_data);
        if calculated_checksum != metadata.expected_checksum {
            return Err(ModelError::ChecksumMismatch);
        }

        // 2. Create artifact record
        let artifact = ModelArtifact {
            id: uuid::Uuid::new_v4().to_string(),
            name: metadata.name,
            version: metadata.version,
            model_type: metadata.model_type,
            checksum: calculated_checksum,
            model_card: metadata.model_card,
            files: vec![],
            status: ModelStatus::Received,
            created_at: Utc::now(),
            validated_at: None,
            deployed_at: None,
        };

        // 3. Store artifact
        self.artifact_store.store(&artifact, artifact_data).await?;

        // 4. Log receipt
        self.audit_log.log(AuditEntry::new(
            "model_received",
            actor.clone(),
            Resource {
                resource_type: "model_artifact".into(),
                id: artifact.id.clone(),
                attributes: std::collections::HashMap::new(),
            },
            "receive",
            Outcome::Success,
            serde_json::json!({
                "name": artifact.name,
                "version": artifact.version,
                "checksum": artifact.checksum,
            }),
            "".into(),
        )).await?;

        Ok(artifact)
    }

    /// Run security scan on artifact
    pub async fn security_scan(&self, artifact_id: &str) -> Result<SecurityScanResult, ModelError> {
        let artifact = self.artifact_store.get(artifact_id).await?;
        let artifact_data = self.artifact_store.get_data(artifact_id).await?;

        // Update status
        self.update_status(artifact_id, ModelStatus::SecurityScanning).await?;

        // Run scans
        let result = self.security_scanner.scan(&artifact_data).await?;

        if result.passed {
            self.update_status(artifact_id, ModelStatus::Validating).await?;
        } else {
            self.update_status(artifact_id, ModelStatus::Rejected).await?;
        }

        Ok(result)
    }

    /// Validate model performance
    pub async fn validate(&self, artifact_id: &str) -> Result<ValidationResult, ModelError> {
        let artifact = self.artifact_store.get(artifact_id).await?;

        // Load model into staging
        let staged_model = self.validator.stage_model(&artifact).await?;

        // Run validation tests
        let result = self.validator.run_validation(&staged_model, &artifact.model_type).await?;

        if result.passed {
            self.update_status(artifact_id, ModelStatus::Validated).await?;

            // Update validated timestamp
            self.artifact_store.update_validated_at(artifact_id, Utc::now()).await?;
        } else {
            self.update_status(artifact_id, ModelStatus::Rejected).await?;
        }

        Ok(result)
    }

    /// Deploy model using blue/green strategy
    pub async fn deploy(&self, artifact_id: &str, actor: &Actor) -> Result<DeploymentResult, ModelError> {
        let artifact = self.artifact_store.get(artifact_id).await?;

        if artifact.status != ModelStatus::Validated {
            return Err(ModelError::NotValidated);
        }

        self.update_status(artifact_id, ModelStatus::Deploying).await?;

        // Blue/green deployment
        let result = self.deployer.deploy_blue_green(&artifact).await?;

        if result.success {
            self.update_status(artifact_id, ModelStatus::Active).await?;
            self.artifact_store.update_deployed_at(artifact_id, Utc::now()).await?;

            // Deprecate previous version
            if let Some(prev_id) = result.previous_artifact_id.as_ref() {
                self.update_status(prev_id, ModelStatus::Deprecated).await?;
            }
        } else {
            self.update_status(artifact_id, ModelStatus::Validated).await?;
        }

        // Log deployment
        self.audit_log.log(AuditEntry::new(
            "model_deployed",
            actor.clone(),
            Resource {
                resource_type: "model_artifact".into(),
                id: artifact_id.to_string(),
                attributes: std::collections::HashMap::new(),
            },
            "deploy",
            if result.success { Outcome::Success } else { Outcome::Failure },
            serde_json::json!({
                "name": artifact.name,
                "version": artifact.version,
                "deployment_id": result.deployment_id,
            }),
            "".into(),
        )).await?;

        Ok(result)
    }

    fn calculate_checksum(&self, data: &[u8]) -> String {
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(data);
        format!("{:x}", hasher.finalize())
    }

    async fn update_status(&self, artifact_id: &str, status: ModelStatus) -> Result<(), ModelError> {
        self.artifact_store.update_status(artifact_id, status).await
    }
}

#[derive(Debug)]
pub struct SecurityScanResult {
    pub passed: bool,
    pub findings: Vec<SecurityFinding>,
    pub scanned_at: DateTime<Utc>,
}

#[derive(Debug)]
pub struct SecurityFinding {
    pub severity: Severity,
    pub description: String,
    pub file: Option<String>,
}

#[derive(Debug)]
pub struct ValidationResult {
    pub passed: bool,
    pub metrics: std::collections::HashMap<String, f64>,
    pub failures: Vec<String>,
}

#[derive(Debug)]
pub struct DeploymentResult {
    pub success: bool,
    pub deployment_id: String,
    pub previous_artifact_id: Option<String>,
}
```

## Monitoring and Alerting

### Banking-Grade Monitoring

```rust
/// Critical metrics that require immediate alerting
pub struct CriticalMetrics {
    /// Service availability
    pub availability_percent: f64,

    /// Error rate
    pub error_rate_percent: f64,

    /// Latency P99
    pub latency_p99_ms: f64,

    /// Security events
    pub security_events_count: u64,

    /// Compliance violations
    pub compliance_violations_count: u64,

    /// Data replication lag
    pub replication_lag_seconds: f64,
}

/// Alert thresholds
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// Availability below this triggers P1
    pub availability_critical: f64,      // 99.5%
    pub availability_warning: f64,       // 99.9%

    /// Error rate above this triggers P1
    pub error_rate_critical: f64,        // 5%
    pub error_rate_warning: f64,         // 1%

    /// Latency above this triggers alert
    pub latency_critical_ms: f64,        // 2000ms
    pub latency_warning_ms: f64,         // 1000ms

    /// Any security event is P1
    pub security_events_critical: u64,   // 1

    /// Any compliance violation is P1
    pub compliance_violations_critical: u64, // 1

    /// Replication lag above this triggers P1
    pub replication_lag_critical_seconds: f64, // 60
    pub replication_lag_warning_seconds: f64,  // 10
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            availability_critical: 99.5,
            availability_warning: 99.9,
            error_rate_critical: 5.0,
            error_rate_warning: 1.0,
            latency_critical_ms: 2000.0,
            latency_warning_ms: 1000.0,
            security_events_critical: 1,
            compliance_violations_critical: 1,
            replication_lag_critical_seconds: 60.0,
            replication_lag_warning_seconds: 10.0,
        }
    }
}

/// Alert severity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertSeverity {
    /// Critical - immediate response required
    P1,
    /// High - response within 1 hour
    P2,
    /// Medium - response within 4 hours
    P3,
    /// Low - response within 24 hours
    P4,
}

/// Alerting service
pub struct AlertingService {
    thresholds: AlertThresholds,
    notification_channels: Vec<NotificationChannel>,
}

impl AlertingService {
    pub fn evaluate(&self, metrics: &CriticalMetrics) -> Vec<Alert> {
        let mut alerts = Vec::new();

        // Availability
        if metrics.availability_percent < self.thresholds.availability_critical {
            alerts.push(Alert {
                severity: AlertSeverity::P1,
                title: "Critical availability drop".into(),
                description: format!(
                    "Service availability at {:.2}% (threshold: {:.2}%)",
                    metrics.availability_percent,
                    self.thresholds.availability_critical
                ),
                metric_name: "availability".into(),
                metric_value: metrics.availability_percent,
            });
        } else if metrics.availability_percent < self.thresholds.availability_warning {
            alerts.push(Alert {
                severity: AlertSeverity::P2,
                title: "Availability degradation".into(),
                description: format!(
                    "Service availability at {:.2}%",
                    metrics.availability_percent
                ),
                metric_name: "availability".into(),
                metric_value: metrics.availability_percent,
            });
        }

        // Error rate
        if metrics.error_rate_percent > self.thresholds.error_rate_critical {
            alerts.push(Alert {
                severity: AlertSeverity::P1,
                title: "Critical error rate".into(),
                description: format!(
                    "Error rate at {:.2}% (threshold: {:.2}%)",
                    metrics.error_rate_percent,
                    self.thresholds.error_rate_critical
                ),
                metric_name: "error_rate".into(),
                metric_value: metrics.error_rate_percent,
            });
        }

        // Security events
        if metrics.security_events_count >= self.thresholds.security_events_critical {
            alerts.push(Alert {
                severity: AlertSeverity::P1,
                title: "Security event detected".into(),
                description: format!(
                    "{} security events detected",
                    metrics.security_events_count
                ),
                metric_name: "security_events".into(),
                metric_value: metrics.security_events_count as f64,
            });
        }

        // Compliance violations
        if metrics.compliance_violations_count >= self.thresholds.compliance_violations_critical {
            alerts.push(Alert {
                severity: AlertSeverity::P1,
                title: "Compliance violation detected".into(),
                description: format!(
                    "{} compliance violations detected",
                    metrics.compliance_violations_count
                ),
                metric_name: "compliance_violations".into(),
                metric_value: metrics.compliance_violations_count as f64,
            });
        }

        // Replication lag
        if metrics.replication_lag_seconds > self.thresholds.replication_lag_critical_seconds {
            alerts.push(Alert {
                severity: AlertSeverity::P1,
                title: "Critical replication lag".into(),
                description: format!(
                    "Replication lag at {:.0}s (threshold: {:.0}s)",
                    metrics.replication_lag_seconds,
                    self.thresholds.replication_lag_critical_seconds
                ),
                metric_name: "replication_lag".into(),
                metric_value: metrics.replication_lag_seconds,
            });
        }

        alerts
    }

    pub async fn send_alerts(&self, alerts: Vec<Alert>) -> Result<(), AlertError> {
        for alert in alerts {
            for channel in &self.notification_channels {
                channel.send(&alert).await?;
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Alert {
    pub severity: AlertSeverity,
    pub title: String,
    pub description: String,
    pub metric_name: String,
    pub metric_value: f64,
}

#[async_trait::async_trait]
pub trait NotificationChannel: Send + Sync {
    async fn send(&self, alert: &Alert) -> Result<(), AlertError>;
}
```

## Summary

This deployment guide covers:

1. **Network Architecture**: Multi-zone with DMZ, application, AI compute, and data layers
2. **Security**: Defense in depth with mTLS, field-level encryption, HSM key management
3. **Compliance**: RBI guidelines implementation including data localization, consent, audit trails
4. **Hardware Tiers**: Pilot (50-100), Production (500-1000), Enterprise (5000+) configurations
5. **Audit Logging**: Immutable merkle-chain audit trail for regulatory compliance
6. **DR/BCP**: Multi-DC active-passive with 4-hour RTO, near-zero RPO
7. **Model Management**: Air-gapped model updates with security scanning and validation
8. **Monitoring**: Banking-grade alerting with P1-P4 severity levels

Key compliance features:
- Data never leaves Indian borders
- Complete audit trail of all AI decisions
- Human handoff always available
- Clear AI disclosure to customers
