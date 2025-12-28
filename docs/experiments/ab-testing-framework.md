# A/B Testing and Experimentation Framework

## Overview

This document details the experimentation framework for testing different configurations, models, and strategies in the voice agent. The framework supports controlled experiments with statistical rigor while protecting customer experience through guardrails.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EXPERIMENTATION ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      Experiment Definition                           │  │
│   │   ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐   │  │
│   │   │  Hypothesis  │  │   Variants   │  │   Success Metrics      │   │  │
│   │   │  Definition  │  │   (A/B/n)    │  │   (Primary/Secondary)  │   │  │
│   │   └──────────────┘  └──────────────┘  └────────────────────────┘   │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                        │                                    │
│                                        ▼                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      Traffic Allocation                              │  │
│   │   ┌──────────────────────────────────────────────────────────────┐  │  │
│   │   │  Incoming Session                                             │  │  │
│   │   │       │                                                       │  │  │
│   │   │       ▼                                                       │  │  │
│   │   │  ┌─────────────┐     ┌─────────────────────────────────────┐ │  │  │
│   │   │  │ Eligibility │────▶│  Hash-based Bucket Assignment       │ │  │  │
│   │   │  │ Check       │     │  (Consistent across sessions)       │ │  │  │
│   │   │  └─────────────┘     └─────────────────────────────────────┘ │  │  │
│   │   │                              │                                │  │  │
│   │   │              ┌───────────────┼───────────────┐               │  │  │
│   │   │              ▼               ▼               ▼               │  │  │
│   │   │       ┌──────────┐    ┌──────────┐    ┌──────────┐          │  │  │
│   │   │       │ Control  │    │ Variant  │    │ Variant  │          │  │  │
│   │   │       │   (A)    │    │   (B)    │    │   (C)    │          │  │  │
│   │   │       └──────────┘    └──────────┘    └──────────┘          │  │  │
│   │   └──────────────────────────────────────────────────────────────┘  │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                        │                                    │
│                                        ▼                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      Metrics Collection                              │  │
│   │   ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐   │  │
│   │   │   Funnel     │  │  Sentiment   │  │   Quality Metrics      │   │  │
│   │   │   Progress   │  │  Summary     │  │   (Latency, Errors)    │   │  │
│   │   └──────────────┘  └──────────────┘  └────────────────────────┘   │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                        │                                    │
│                                        ▼                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      Analysis & Decision                             │  │
│   │   ┌──────────────────────────────────────────────────────────────┐  │  │
│   │   │  Statistical Significance  │  Guardrail Checks  │  Decision  │  │  │
│   │   │  (Bayesian + Frequentist)  │  (Harm Prevention) │  (Ship/No) │  │  │
│   │   └──────────────────────────────────────────────────────────────┘  │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Experiment Definition

### Core Data Structures

```rust
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Experiment definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    /// Unique experiment ID
    pub id: String,

    /// Human-readable name
    pub name: String,

    /// Detailed description
    pub description: String,

    /// Hypothesis being tested
    pub hypothesis: Hypothesis,

    /// Experiment variants (including control)
    pub variants: Vec<Variant>,

    /// Primary success metric
    pub primary_metric: MetricDefinition,

    /// Secondary metrics
    pub secondary_metrics: Vec<MetricDefinition>,

    /// Guardrail metrics (must not degrade)
    pub guardrail_metrics: Vec<GuardrailMetric>,

    /// Traffic allocation
    pub allocation: TrafficAllocation,

    /// Targeting rules
    pub targeting: TargetingRules,

    /// Experiment status
    pub status: ExperimentStatus,

    /// Timeline
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub ended_at: Option<DateTime<Utc>>,

    /// Required sample size for significance
    pub required_sample_size: u64,

    /// Owner
    pub owner: String,
}

/// Hypothesis structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hypothesis {
    /// What we believe will happen
    pub statement: String,

    /// Expected effect size (percentage improvement)
    pub expected_effect_size: f64,

    /// Rationale for the hypothesis
    pub rationale: String,
}

/// Experiment variant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variant {
    /// Variant ID
    pub id: String,

    /// Variant name (e.g., "control", "treatment_a")
    pub name: String,

    /// Is this the control group?
    pub is_control: bool,

    /// Traffic percentage (0-100)
    pub traffic_percentage: f64,

    /// Configuration overrides
    pub config_overrides: HashMap<String, serde_json::Value>,

    /// Description of what's different
    pub description: String,
}

/// Metric definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDefinition {
    /// Metric name
    pub name: String,

    /// Metric type
    pub metric_type: MetricType,

    /// How to aggregate
    pub aggregation: AggregationType,

    /// Higher is better?
    pub higher_is_better: bool,

    /// Minimum detectable effect (for power analysis)
    pub minimum_detectable_effect: f64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MetricType {
    /// Binary outcome (conversion, success/failure)
    Binary,
    /// Continuous value (latency, rating)
    Continuous,
    /// Count (messages, errors)
    Count,
    /// Ratio (rate, percentage)
    Ratio,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AggregationType {
    Mean,
    Median,
    Sum,
    P50,
    P75,
    P90,
    P95,
    P99,
    Rate,
}

/// Guardrail metric (must not degrade significantly)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailMetric {
    pub metric: MetricDefinition,

    /// Maximum acceptable degradation (percentage)
    pub max_degradation_percent: f64,

    /// Confidence level for guardrail check
    pub confidence_level: f64,
}

/// Traffic allocation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficAllocation {
    /// Allocation strategy
    pub strategy: AllocationStrategy,

    /// Ramp-up schedule
    pub ramp_schedule: Option<RampSchedule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationStrategy {
    /// Fixed percentage allocation
    Fixed,
    /// Multi-armed bandit (adaptive)
    Bandit { exploration_rate: f64 },
    /// Sequential testing (adaptive stopping)
    Sequential,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RampSchedule {
    pub stages: Vec<RampStage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RampStage {
    /// Hours after start
    pub hours_after_start: u32,
    /// Traffic percentage at this stage
    pub traffic_percentage: f64,
}

/// Targeting rules for experiment eligibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetingRules {
    /// Customer segments to include
    pub segments: Option<Vec<String>>,

    /// Languages to include
    pub languages: Option<Vec<String>>,

    /// Regions to include
    pub regions: Option<Vec<String>>,

    /// Exclude customers in other experiments
    pub exclusive: bool,

    /// Custom targeting expression
    pub custom_expression: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExperimentStatus {
    Draft,
    Scheduled,
    Running,
    Paused,
    Stopped,
    Completed,
    Graduated,  // Winner rolled out to 100%
}
```

### Example Experiment Definitions

```rust
/// Example: RAG Timing Strategy Experiment
pub fn create_rag_timing_experiment() -> Experiment {
    Experiment {
        id: "exp_rag_timing_001".into(),
        name: "RAG Timing Strategy Comparison".into(),
        description: "Compare sequential, prefetch async, and parallel inject RAG strategies for latency and quality trade-offs".into(),

        hypothesis: Hypothesis {
            statement: "Prefetch Async RAG will reduce end-to-end latency by 20% without degrading response quality".into(),
            expected_effect_size: 20.0,
            rationale: "By fetching RAG context while user is still speaking, we can hide retrieval latency".into(),
        },

        variants: vec![
            Variant {
                id: "control".into(),
                name: "Sequential RAG".into(),
                is_control: true,
                traffic_percentage: 34.0,
                config_overrides: hashmap! {
                    "rag.timing_strategy".into() => json!("sequential"),
                },
                description: "Wait for complete utterance, then retrieve, then generate".into(),
            },
            Variant {
                id: "treatment_a".into(),
                name: "Prefetch Async RAG".into(),
                is_control: false,
                traffic_percentage: 33.0,
                config_overrides: hashmap! {
                    "rag.timing_strategy".into() => json!("prefetch_async"),
                    "rag.prefetch_confidence_threshold".into() => json!(0.7),
                },
                description: "Start RAG on partial transcript, use results if query is similar".into(),
            },
            Variant {
                id: "treatment_b".into(),
                name: "Parallel Inject RAG".into(),
                is_control: false,
                traffic_percentage: 33.0,
                config_overrides: hashmap! {
                    "rag.timing_strategy".into() => json!("parallel_inject"),
                },
                description: "Always inject top-k context, let LLM decide relevance".into(),
            },
        ],

        primary_metric: MetricDefinition {
            name: "end_to_end_latency_p50".into(),
            metric_type: MetricType::Continuous,
            aggregation: AggregationType::P50,
            higher_is_better: false,
            minimum_detectable_effect: 0.10, // 10% change
        },

        secondary_metrics: vec![
            MetricDefinition {
                name: "response_quality_rating".into(),
                metric_type: MetricType::Continuous,
                aggregation: AggregationType::Mean,
                higher_is_better: true,
                minimum_detectable_effect: 0.05,
            },
            MetricDefinition {
                name: "rag_cache_hit_rate".into(),
                metric_type: MetricType::Ratio,
                aggregation: AggregationType::Rate,
                higher_is_better: true,
                minimum_detectable_effect: 0.10,
            },
        ],

        guardrail_metrics: vec![
            GuardrailMetric {
                metric: MetricDefinition {
                    name: "conversation_completion_rate".into(),
                    metric_type: MetricType::Ratio,
                    aggregation: AggregationType::Rate,
                    higher_is_better: true,
                    minimum_detectable_effect: 0.02,
                },
                max_degradation_percent: 2.0,
                confidence_level: 0.95,
            },
            GuardrailMetric {
                metric: MetricDefinition {
                    name: "factual_accuracy_rate".into(),
                    metric_type: MetricType::Ratio,
                    aggregation: AggregationType::Rate,
                    higher_is_better: true,
                    minimum_detectable_effect: 0.01,
                },
                max_degradation_percent: 1.0,
                confidence_level: 0.95,
            },
        ],

        allocation: TrafficAllocation {
            strategy: AllocationStrategy::Fixed,
            ramp_schedule: Some(RampSchedule {
                stages: vec![
                    RampStage { hours_after_start: 0, traffic_percentage: 10.0 },
                    RampStage { hours_after_start: 24, traffic_percentage: 30.0 },
                    RampStage { hours_after_start: 72, traffic_percentage: 100.0 },
                ],
            }),
        },

        targeting: TargetingRules {
            segments: None, // All segments
            languages: Some(vec!["hi".into(), "en".into()]), // Hindi and English first
            regions: None,
            exclusive: true, // Don't overlap with other experiments
            custom_expression: None,
        },

        status: ExperimentStatus::Draft,
        created_at: Utc::now(),
        started_at: None,
        ended_at: None,
        required_sample_size: 5000,
        owner: "ml-team@bank.com".into(),
    }
}
```

## Traffic Allocation

### Consistent Hashing for Bucketing

```rust
use std::hash::{Hash, Hasher};
use siphasher::sip::SipHasher24;

/// Experiment bucketing service
pub struct ExperimentBucketer {
    /// Active experiments
    experiments: Vec<Experiment>,

    /// Hash seed for consistent bucketing
    hash_seed: u64,
}

impl ExperimentBucketer {
    pub fn new(experiments: Vec<Experiment>, hash_seed: u64) -> Self {
        Self { experiments, hash_seed }
    }

    /// Determine variant assignment for a session
    pub fn get_assignment(
        &self,
        session_id: &str,
        customer_id: &str,
        context: &SessionContext,
    ) -> Vec<ExperimentAssignment> {
        let mut assignments = Vec::new();

        for experiment in &self.experiments {
            if experiment.status != ExperimentStatus::Running {
                continue;
            }

            // Check targeting eligibility
            if !self.is_eligible(&experiment.targeting, context) {
                continue;
            }

            // Check exclusivity (skip if in another exclusive experiment)
            if experiment.targeting.exclusive {
                let in_other_exclusive = assignments.iter().any(|a: &ExperimentAssignment| {
                    self.experiments
                        .iter()
                        .find(|e| e.id == a.experiment_id)
                        .map(|e| e.targeting.exclusive)
                        .unwrap_or(false)
                });

                if in_other_exclusive {
                    continue;
                }
            }

            // Compute bucket
            let bucket = self.compute_bucket(customer_id, &experiment.id);

            // Assign to variant based on bucket
            if let Some(variant) = self.bucket_to_variant(bucket, &experiment.variants) {
                assignments.push(ExperimentAssignment {
                    experiment_id: experiment.id.clone(),
                    experiment_name: experiment.name.clone(),
                    variant_id: variant.id.clone(),
                    variant_name: variant.name.clone(),
                    config_overrides: variant.config_overrides.clone(),
                    assigned_at: Utc::now(),
                });
            }
        }

        assignments
    }

    /// Compute bucket (0-9999) using consistent hashing
    fn compute_bucket(&self, customer_id: &str, experiment_id: &str) -> u32 {
        let mut hasher = SipHasher24::new_with_key(&self.hash_seed.to_le_bytes());

        // Hash customer + experiment for consistent assignment
        customer_id.hash(&mut hasher);
        experiment_id.hash(&mut hasher);

        let hash = hasher.finish();

        // Map to 0-9999 range (for 0.01% granularity)
        (hash % 10000) as u32
    }

    /// Map bucket to variant
    fn bucket_to_variant(&self, bucket: u32, variants: &[Variant]) -> Option<&Variant> {
        let mut cumulative = 0.0;

        for variant in variants {
            cumulative += variant.traffic_percentage;
            let threshold = (cumulative * 100.0) as u32; // Convert to 0-10000 range

            if bucket < threshold {
                return Some(variant);
            }
        }

        None
    }

    /// Check if session is eligible for experiment
    fn is_eligible(&self, targeting: &TargetingRules, context: &SessionContext) -> bool {
        // Check segment
        if let Some(ref segments) = targeting.segments {
            if !segments.contains(&context.customer_segment) {
                return false;
            }
        }

        // Check language
        if let Some(ref languages) = targeting.languages {
            if !languages.contains(&context.language) {
                return false;
            }
        }

        // Check region
        if let Some(ref regions) = targeting.regions {
            if !regions.contains(&context.region) {
                return false;
            }
        }

        true
    }
}

/// Experiment assignment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentAssignment {
    pub experiment_id: String,
    pub experiment_name: String,
    pub variant_id: String,
    pub variant_name: String,
    pub config_overrides: HashMap<String, serde_json::Value>,
    pub assigned_at: DateTime<Utc>,
}

/// Session context for targeting
#[derive(Debug, Clone)]
pub struct SessionContext {
    pub customer_id: String,
    pub customer_segment: String,
    pub language: String,
    pub region: String,
    pub device_type: String,
    pub is_new_customer: bool,
}
```

## Metrics Collection

### Funnel Metrics

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VOICE AGENT CONVERSION FUNNEL                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Stage 1: Session Start                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  100% ████████████████████████████████████████████████████████████  │  │
│   │  Metrics: session_started, consent_given                            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                        │                                    │
│                                        ▼                                    │
│   Stage 2: Engagement                                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  85%  ██████████████████████████████████████████████████████████    │  │
│   │  Metrics: first_response, interaction_count > 2                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                        │                                    │
│                                        ▼                                    │
│   Stage 3: Interest Expressed                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  60%  ██████████████████████████████████████████                    │  │
│   │  Metrics: product_inquiry, savings_calculation_viewed                │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                        │                                    │
│                                        ▼                                    │
│   Stage 4: Objections Handled                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  40%  ██████████████████████████████                                │  │
│   │  Metrics: objection_raised, objection_addressed                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                        │                                    │
│                                        ▼                                    │
│   Stage 5: Intent to Switch                                                  │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  25%  ██████████████████                                            │  │
│   │  Metrics: switch_intent_expressed, documentation_discussed           │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                        │                                    │
│                                        ▼                                    │
│   Stage 6: Appointment Booked                                                │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  15%  ████████████                                                  │  │
│   │  Metrics: appointment_scheduled, branch_visit_booked                 │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                        │                                    │
│                                        ▼                                    │
│   Stage 7: Conversion (Loan Disbursed)                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  8%   ██████                                                        │  │
│   │  Metrics: loan_application_submitted, loan_disbursed                 │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Metrics Collection Implementation

```rust
use std::sync::Arc;
use tokio::sync::RwLock;

/// Funnel stage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FunnelStage {
    SessionStart,
    Engagement,
    InterestExpressed,
    ObjectionsHandled,
    IntentToSwitch,
    AppointmentBooked,
    Conversion,
}

impl FunnelStage {
    pub fn as_index(&self) -> usize {
        match self {
            Self::SessionStart => 0,
            Self::Engagement => 1,
            Self::InterestExpressed => 2,
            Self::ObjectionsHandled => 3,
            Self::IntentToSwitch => 4,
            Self::AppointmentBooked => 5,
            Self::Conversion => 6,
        }
    }

    pub fn from_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::SessionStart),
            1 => Some(Self::Engagement),
            2 => Some(Self::InterestExpressed),
            3 => Some(Self::ObjectionsHandled),
            4 => Some(Self::IntentToSwitch),
            5 => Some(Self::AppointmentBooked),
            6 => Some(Self::Conversion),
            _ => None,
        }
    }
}

/// Session metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetrics {
    pub session_id: String,
    pub customer_id: String,
    pub experiment_assignments: Vec<ExperimentAssignment>,

    /// Funnel progression
    pub funnel_stage: FunnelStage,
    pub stage_timestamps: HashMap<FunnelStage, DateTime<Utc>>,

    /// Sentiment
    pub sentiment_scores: Vec<SentimentScore>,
    pub overall_sentiment: f64, // -1.0 to 1.0

    /// Quality metrics
    pub total_latency_ms: u64,
    pub turn_count: u32,
    pub error_count: u32,
    pub barge_in_count: u32,

    /// Timing
    pub session_duration_seconds: f64,
    pub started_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,

    /// Outcome
    pub outcome: SessionOutcome,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentScore {
    pub turn_number: u32,
    pub score: f64,        // -1.0 to 1.0
    pub confidence: f64,   // 0.0 to 1.0
    pub keywords: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SessionOutcome {
    Completed,
    CustomerHangup,
    AgentHandoff,
    Error,
    Timeout,
}

/// Metrics collector
pub struct MetricsCollector {
    /// Storage backend
    storage: Arc<dyn MetricsStorage>,

    /// Real-time aggregates (per experiment/variant)
    aggregates: Arc<RwLock<HashMap<String, VariantAggregates>>>,
}

#[async_trait::async_trait]
pub trait MetricsStorage: Send + Sync {
    async fn store_session_metrics(&self, metrics: &SessionMetrics) -> Result<(), MetricsError>;
    async fn query_metrics(&self, query: MetricsQuery) -> Result<Vec<SessionMetrics>, MetricsError>;
    async fn get_aggregates(&self, experiment_id: &str) -> Result<HashMap<String, VariantAggregates>, MetricsError>;
}

/// Aggregated metrics for a variant
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VariantAggregates {
    pub variant_id: String,
    pub sample_count: u64,

    /// Funnel conversion rates
    pub funnel_rates: HashMap<FunnelStage, f64>,

    /// Latency percentiles
    pub latency_p50: f64,
    pub latency_p90: f64,
    pub latency_p99: f64,

    /// Sentiment
    pub avg_sentiment: f64,
    pub sentiment_std: f64,

    /// Quality
    pub error_rate: f64,
    pub completion_rate: f64,
    pub avg_turn_count: f64,

    /// Streaming aggregation state
    pub sum_latency: f64,
    pub sum_latency_sq: f64,
    pub sum_sentiment: f64,
    pub sum_sentiment_sq: f64,
}

impl MetricsCollector {
    /// Record session start
    pub async fn record_session_start(
        &self,
        session_id: &str,
        customer_id: &str,
        assignments: Vec<ExperimentAssignment>,
    ) -> Result<(), MetricsError> {
        let metrics = SessionMetrics {
            session_id: session_id.to_string(),
            customer_id: customer_id.to_string(),
            experiment_assignments: assignments,
            funnel_stage: FunnelStage::SessionStart,
            stage_timestamps: hashmap! {
                FunnelStage::SessionStart => Utc::now(),
            },
            sentiment_scores: Vec::new(),
            overall_sentiment: 0.0,
            total_latency_ms: 0,
            turn_count: 0,
            error_count: 0,
            barge_in_count: 0,
            session_duration_seconds: 0.0,
            started_at: Utc::now(),
            ended_at: None,
            outcome: SessionOutcome::Completed,
        };

        self.storage.store_session_metrics(&metrics).await
    }

    /// Update funnel stage
    pub async fn advance_funnel(
        &self,
        session_id: &str,
        new_stage: FunnelStage,
    ) -> Result<(), MetricsError> {
        // In practice, this would update the stored session
        // and update real-time aggregates

        let mut aggregates = self.aggregates.write().await;

        // Update aggregates for each experiment assignment
        // (Implementation would fetch session, get assignments, update each)

        Ok(())
    }

    /// Record sentiment for a turn
    pub async fn record_sentiment(
        &self,
        session_id: &str,
        turn_number: u32,
        score: f64,
        confidence: f64,
        keywords: Vec<String>,
    ) -> Result<(), MetricsError> {
        let sentiment = SentimentScore {
            turn_number,
            score,
            confidence,
            keywords,
        };

        // Update session and recalculate overall sentiment
        Ok(())
    }

    /// Record turn latency
    pub async fn record_latency(
        &self,
        session_id: &str,
        latency_ms: u64,
    ) -> Result<(), MetricsError> {
        // Update session total and streaming aggregates
        Ok(())
    }

    /// Finalize session
    pub async fn end_session(
        &self,
        session_id: &str,
        outcome: SessionOutcome,
    ) -> Result<(), MetricsError> {
        // Finalize metrics and update aggregates
        Ok(())
    }

    /// Generate sentiment summary
    pub fn generate_sentiment_summary(scores: &[SentimentScore]) -> SentimentSummary {
        if scores.is_empty() {
            return SentimentSummary::default();
        }

        let total: f64 = scores.iter().map(|s| s.score).sum();
        let avg = total / scores.len() as f64;

        // Calculate trend
        let trend = if scores.len() >= 3 {
            let first_half: f64 = scores[..scores.len() / 2].iter().map(|s| s.score).sum();
            let second_half: f64 = scores[scores.len() / 2..].iter().map(|s| s.score).sum();
            let first_avg = first_half / (scores.len() / 2) as f64;
            let second_avg = second_half / (scores.len() - scores.len() / 2) as f64;

            if second_avg > first_avg + 0.1 {
                SentimentTrend::Improving
            } else if second_avg < first_avg - 0.1 {
                SentimentTrend::Declining
            } else {
                SentimentTrend::Stable
            }
        } else {
            SentimentTrend::Stable
        };

        // Collect key moments
        let low_points: Vec<_> = scores.iter()
            .filter(|s| s.score < -0.5)
            .map(|s| s.turn_number)
            .collect();

        let high_points: Vec<_> = scores.iter()
            .filter(|s| s.score > 0.5)
            .map(|s| s.turn_number)
            .collect();

        SentimentSummary {
            average: avg,
            trend,
            low_points,
            high_points,
            overall_label: Self::label_sentiment(avg),
        }
    }

    fn label_sentiment(score: f64) -> String {
        match score {
            s if s >= 0.5 => "Very Positive".into(),
            s if s >= 0.2 => "Positive".into(),
            s if s >= -0.2 => "Neutral".into(),
            s if s >= -0.5 => "Negative".into(),
            _ => "Very Negative".into(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SentimentSummary {
    pub average: f64,
    pub trend: SentimentTrend,
    pub low_points: Vec<u32>,
    pub high_points: Vec<u32>,
    pub overall_label: String,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub enum SentimentTrend {
    Improving,
    #[default]
    Stable,
    Declining,
}
```

## Statistical Analysis

### Frequentist and Bayesian Analysis

```rust
use statrs::distribution::{Normal, ContinuousCDF};
use std::f64::consts::PI;

/// Statistical analysis engine
pub struct StatisticalAnalyzer {
    /// Confidence level for frequentist tests (e.g., 0.95)
    confidence_level: f64,

    /// Prior parameters for Bayesian analysis
    bayesian_prior: BayesianPrior,
}

#[derive(Debug, Clone)]
pub struct BayesianPrior {
    /// Beta prior for conversion rates: Beta(alpha, beta)
    pub conversion_alpha: f64,
    pub conversion_beta: f64,

    /// Normal prior for continuous metrics: N(mu, sigma^2)
    pub continuous_mu: f64,
    pub continuous_sigma: f64,
}

impl Default for BayesianPrior {
    fn default() -> Self {
        Self {
            // Weakly informative prior
            conversion_alpha: 1.0,
            conversion_beta: 1.0,
            continuous_mu: 0.0,
            continuous_sigma: 1.0,
        }
    }
}

/// Experiment analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentAnalysis {
    pub experiment_id: String,
    pub analysis_timestamp: DateTime<Utc>,

    /// Per-variant results
    pub variant_results: HashMap<String, VariantAnalysis>,

    /// Comparison results
    pub comparisons: Vec<VariantComparison>,

    /// Recommended action
    pub recommendation: ExperimentRecommendation,

    /// Guardrail check results
    pub guardrail_results: Vec<GuardrailResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantAnalysis {
    pub variant_id: String,
    pub sample_size: u64,

    /// Primary metric
    pub primary_metric_value: f64,
    pub primary_metric_ci: ConfidenceInterval,

    /// Secondary metrics
    pub secondary_metrics: HashMap<String, MetricResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricResult {
    pub value: f64,
    pub confidence_interval: ConfidenceInterval,
    pub standard_error: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
    pub level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantComparison {
    pub control_variant: String,
    pub treatment_variant: String,

    /// Relative lift
    pub relative_lift: f64,
    pub lift_ci: ConfidenceInterval,

    /// Frequentist results
    pub p_value: f64,
    pub is_significant: bool,

    /// Bayesian results
    pub probability_better: f64,
    pub expected_loss: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailResult {
    pub metric_name: String,
    pub control_value: f64,
    pub treatment_value: f64,
    pub relative_change: f64,
    pub passed: bool,
    pub p_value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExperimentRecommendation {
    /// Continue running, not enough data
    Continue {
        estimated_days_remaining: u32,
        current_power: f64,
    },
    /// Ship treatment, significant improvement
    ShipTreatment {
        winning_variant: String,
        confidence: f64,
    },
    /// Ship control, treatment is worse
    ShipControl {
        reason: String,
    },
    /// Stop experiment, guardrails violated
    Stop {
        violated_guardrails: Vec<String>,
    },
    /// Inconclusive, no significant difference
    Inconclusive {
        recommendation: String,
    },
}

impl StatisticalAnalyzer {
    pub fn new(confidence_level: f64) -> Self {
        Self {
            confidence_level,
            bayesian_prior: BayesianPrior::default(),
        }
    }

    /// Analyze experiment results
    pub fn analyze(&self, experiment: &Experiment, data: &ExperimentData) -> ExperimentAnalysis {
        let mut variant_results = HashMap::new();
        let mut comparisons = Vec::new();

        // Analyze each variant
        for variant in &experiment.variants {
            if let Some(variant_data) = data.variants.get(&variant.id) {
                let analysis = self.analyze_variant(
                    variant_data,
                    &experiment.primary_metric,
                    &experiment.secondary_metrics,
                );
                variant_results.insert(variant.id.clone(), analysis);
            }
        }

        // Compare treatments to control
        let control = experiment.variants.iter()
            .find(|v| v.is_control)
            .map(|v| &v.id);

        if let Some(control_id) = control {
            for variant in &experiment.variants {
                if !variant.is_control {
                    if let (Some(control_data), Some(treatment_data)) = (
                        data.variants.get(control_id),
                        data.variants.get(&variant.id),
                    ) {
                        let comparison = self.compare_variants(
                            control_data,
                            treatment_data,
                            &experiment.primary_metric,
                        );
                        comparisons.push(comparison);
                    }
                }
            }
        }

        // Check guardrails
        let guardrail_results = self.check_guardrails(
            experiment,
            data,
            control.unwrap_or(&"".to_string()),
        );

        // Determine recommendation
        let recommendation = self.make_recommendation(
            experiment,
            &comparisons,
            &guardrail_results,
        );

        ExperimentAnalysis {
            experiment_id: experiment.id.clone(),
            analysis_timestamp: Utc::now(),
            variant_results,
            comparisons,
            recommendation,
            guardrail_results,
        }
    }

    /// Two-sample t-test for continuous metrics
    fn t_test(&self, control: &[f64], treatment: &[f64]) -> (f64, f64) {
        let n1 = control.len() as f64;
        let n2 = treatment.len() as f64;

        let mean1: f64 = control.iter().sum::<f64>() / n1;
        let mean2: f64 = treatment.iter().sum::<f64>() / n2;

        let var1: f64 = control.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0);
        let var2: f64 = treatment.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0);

        let se = (var1 / n1 + var2 / n2).sqrt();
        let t_stat = (mean2 - mean1) / se;

        // Welch's degrees of freedom
        let df = ((var1 / n1 + var2 / n2).powi(2))
            / ((var1 / n1).powi(2) / (n1 - 1.0) + (var2 / n2).powi(2) / (n2 - 1.0));

        // Approximate p-value using normal distribution for large samples
        let normal = Normal::new(0.0, 1.0).unwrap();
        let p_value = 2.0 * (1.0 - normal.cdf(t_stat.abs()));

        (t_stat, p_value)
    }

    /// Chi-squared test for proportions
    fn proportion_test(&self, control_successes: u64, control_total: u64,
                       treatment_successes: u64, treatment_total: u64) -> f64 {
        let p1 = control_successes as f64 / control_total as f64;
        let p2 = treatment_successes as f64 / treatment_total as f64;
        let p_pooled = (control_successes + treatment_successes) as f64
            / (control_total + treatment_total) as f64;

        let se = (p_pooled * (1.0 - p_pooled) * (1.0 / control_total as f64 + 1.0 / treatment_total as f64)).sqrt();

        if se == 0.0 {
            return 1.0;
        }

        let z = (p2 - p1) / se;

        let normal = Normal::new(0.0, 1.0).unwrap();
        2.0 * (1.0 - normal.cdf(z.abs()))
    }

    /// Bayesian probability that treatment is better
    fn bayesian_probability_better(
        &self,
        control: &VariantData,
        treatment: &VariantData,
        metric_type: MetricType,
    ) -> f64 {
        match metric_type {
            MetricType::Binary | MetricType::Ratio => {
                // Beta-Binomial model
                self.beta_probability_better(
                    control.conversions, control.sample_size,
                    treatment.conversions, treatment.sample_size,
                )
            }
            MetricType::Continuous => {
                // Normal-Normal model with known variance (simplified)
                self.normal_probability_better(
                    &control.values, &treatment.values,
                )
            }
            MetricType::Count => {
                // Poisson-Gamma model
                self.poisson_probability_better(
                    control.total_count, control.sample_size,
                    treatment.total_count, treatment.sample_size,
                )
            }
        }
    }

    /// Beta probability that B > A using Monte Carlo
    fn beta_probability_better(
        &self,
        a_successes: u64, a_total: u64,
        b_successes: u64, b_total: u64,
    ) -> f64 {
        // Posterior parameters
        let alpha_a = self.bayesian_prior.conversion_alpha + a_successes as f64;
        let beta_a = self.bayesian_prior.conversion_beta + (a_total - a_successes) as f64;
        let alpha_b = self.bayesian_prior.conversion_alpha + b_successes as f64;
        let beta_b = self.bayesian_prior.conversion_beta + (b_total - b_successes) as f64;

        // Monte Carlo simulation
        use rand::distributions::Distribution;
        use statrs::distribution::Beta;

        let dist_a = Beta::new(alpha_a, beta_a).unwrap();
        let dist_b = Beta::new(alpha_b, beta_b).unwrap();

        let mut rng = rand::thread_rng();
        let n_samples = 10000;
        let mut b_wins = 0;

        for _ in 0..n_samples {
            let sample_a = dist_a.sample(&mut rng);
            let sample_b = dist_b.sample(&mut rng);
            if sample_b > sample_a {
                b_wins += 1;
            }
        }

        b_wins as f64 / n_samples as f64
    }

    /// Normal probability that B > A (continuous metrics)
    fn normal_probability_better(&self, a_values: &[f64], b_values: &[f64]) -> f64 {
        if a_values.is_empty() || b_values.is_empty() {
            return 0.5;
        }

        let mean_a: f64 = a_values.iter().sum::<f64>() / a_values.len() as f64;
        let mean_b: f64 = b_values.iter().sum::<f64>() / b_values.len() as f64;

        let var_a: f64 = a_values.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>()
            / a_values.len() as f64;
        let var_b: f64 = b_values.iter().map(|x| (x - mean_b).powi(2)).sum::<f64>()
            / b_values.len() as f64;

        let se_a = (var_a / a_values.len() as f64).sqrt();
        let se_b = (var_b / b_values.len() as f64).sqrt();

        // Difference distribution
        let diff_mean = mean_b - mean_a;
        let diff_se = (se_a.powi(2) + se_b.powi(2)).sqrt();

        let normal = Normal::new(0.0, 1.0).unwrap();
        normal.cdf(diff_mean / diff_se)
    }

    /// Poisson probability that B > A (counts)
    fn poisson_probability_better(
        &self,
        a_count: u64, a_exposure: u64,
        b_count: u64, b_exposure: u64,
    ) -> f64 {
        // Gamma-Poisson model with Monte Carlo
        use rand::distributions::Distribution;
        use statrs::distribution::Gamma;

        // Posterior: Gamma(alpha + count, beta + exposure)
        let alpha_prior = 1.0;
        let beta_prior = 1.0;

        let alpha_a = alpha_prior + a_count as f64;
        let rate_a = beta_prior + a_exposure as f64;
        let alpha_b = alpha_prior + b_count as f64;
        let rate_b = beta_prior + b_exposure as f64;

        let dist_a = Gamma::new(alpha_a, 1.0 / rate_a).unwrap();
        let dist_b = Gamma::new(alpha_b, 1.0 / rate_b).unwrap();

        let mut rng = rand::thread_rng();
        let n_samples = 10000;
        let mut b_wins = 0;

        for _ in 0..n_samples {
            let sample_a = dist_a.sample(&mut rng);
            let sample_b = dist_b.sample(&mut rng);
            if sample_b > sample_a {
                b_wins += 1;
            }
        }

        b_wins as f64 / n_samples as f64
    }

    fn analyze_variant(
        &self,
        data: &VariantData,
        primary: &MetricDefinition,
        secondary: &[MetricDefinition],
    ) -> VariantAnalysis {
        // Calculate primary metric with CI
        let (primary_value, primary_ci) = self.calculate_metric_with_ci(data, primary);

        // Calculate secondary metrics
        let secondary_metrics: HashMap<String, MetricResult> = secondary.iter()
            .map(|m| {
                let (value, ci) = self.calculate_metric_with_ci(data, m);
                let se = (ci.upper - ci.lower) / (2.0 * 1.96); // Approximate
                (m.name.clone(), MetricResult {
                    value,
                    confidence_interval: ci,
                    standard_error: se,
                })
            })
            .collect();

        VariantAnalysis {
            variant_id: data.variant_id.clone(),
            sample_size: data.sample_size,
            primary_metric_value: primary_value,
            primary_metric_ci: primary_ci,
            secondary_metrics,
        }
    }

    fn calculate_metric_with_ci(&self, data: &VariantData, metric: &MetricDefinition) -> (f64, ConfidenceInterval) {
        let z = 1.96; // 95% CI

        match metric.metric_type {
            MetricType::Binary | MetricType::Ratio => {
                let p = data.conversions as f64 / data.sample_size as f64;
                let se = (p * (1.0 - p) / data.sample_size as f64).sqrt();
                (p, ConfidenceInterval {
                    lower: (p - z * se).max(0.0),
                    upper: (p + z * se).min(1.0),
                    level: 0.95,
                })
            }
            MetricType::Continuous => {
                if data.values.is_empty() {
                    return (0.0, ConfidenceInterval { lower: 0.0, upper: 0.0, level: 0.95 });
                }
                let mean: f64 = data.values.iter().sum::<f64>() / data.values.len() as f64;
                let var: f64 = data.values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                    / (data.values.len() - 1) as f64;
                let se = (var / data.values.len() as f64).sqrt();
                (mean, ConfidenceInterval {
                    lower: mean - z * se,
                    upper: mean + z * se,
                    level: 0.95,
                })
            }
            MetricType::Count => {
                let rate = data.total_count as f64 / data.sample_size as f64;
                let se = (rate / data.sample_size as f64).sqrt();
                (rate, ConfidenceInterval {
                    lower: (rate - z * se).max(0.0),
                    upper: rate + z * se,
                    level: 0.95,
                })
            }
        }
    }

    fn compare_variants(
        &self,
        control: &VariantData,
        treatment: &VariantData,
        metric: &MetricDefinition,
    ) -> VariantComparison {
        let control_value = self.calculate_metric_with_ci(control, metric).0;
        let treatment_value = self.calculate_metric_with_ci(treatment, metric).0;

        let relative_lift = if control_value > 0.0 {
            (treatment_value - control_value) / control_value
        } else {
            0.0
        };

        // Frequentist test
        let p_value = match metric.metric_type {
            MetricType::Binary | MetricType::Ratio => {
                self.proportion_test(
                    control.conversions, control.sample_size,
                    treatment.conversions, treatment.sample_size,
                )
            }
            MetricType::Continuous => {
                self.t_test(&control.values, &treatment.values).1
            }
            MetricType::Count => {
                // Rate ratio test
                self.proportion_test(
                    control.total_count, control.sample_size,
                    treatment.total_count, treatment.sample_size,
                )
            }
        };

        let is_significant = p_value < (1.0 - self.confidence_level);

        // Bayesian analysis
        let probability_better = self.bayesian_probability_better(
            control, treatment, metric.metric_type,
        );

        // Expected loss (simplified)
        let expected_loss = if probability_better > 0.5 {
            (1.0 - probability_better) * relative_lift.abs()
        } else {
            probability_better * relative_lift.abs()
        };

        VariantComparison {
            control_variant: control.variant_id.clone(),
            treatment_variant: treatment.variant_id.clone(),
            relative_lift,
            lift_ci: ConfidenceInterval {
                lower: relative_lift - 0.1, // Placeholder
                upper: relative_lift + 0.1,
                level: 0.95,
            },
            p_value,
            is_significant,
            probability_better,
            expected_loss,
        }
    }

    fn check_guardrails(
        &self,
        experiment: &Experiment,
        data: &ExperimentData,
        control_id: &str,
    ) -> Vec<GuardrailResult> {
        let mut results = Vec::new();

        let control = match data.variants.get(control_id) {
            Some(c) => c,
            None => return results,
        };

        for guardrail in &experiment.guardrail_metrics {
            for (variant_id, treatment) in &data.variants {
                if variant_id == control_id {
                    continue;
                }

                let (control_value, _) = self.calculate_metric_with_ci(control, &guardrail.metric);
                let (treatment_value, _) = self.calculate_metric_with_ci(treatment, &guardrail.metric);

                let relative_change = if control_value > 0.0 {
                    (treatment_value - control_value) / control_value * 100.0
                } else {
                    0.0
                };

                // Check if degradation exceeds threshold
                let degradation = if guardrail.metric.higher_is_better {
                    -relative_change
                } else {
                    relative_change
                };

                let p_value = match guardrail.metric.metric_type {
                    MetricType::Binary | MetricType::Ratio => {
                        self.proportion_test(
                            control.conversions, control.sample_size,
                            treatment.conversions, treatment.sample_size,
                        )
                    }
                    _ => {
                        self.t_test(&control.values, &treatment.values).1
                    }
                };

                let passed = degradation <= guardrail.max_degradation_percent
                    || p_value >= (1.0 - guardrail.confidence_level);

                results.push(GuardrailResult {
                    metric_name: format!("{}_{}", guardrail.metric.name, variant_id),
                    control_value,
                    treatment_value,
                    relative_change,
                    passed,
                    p_value,
                });
            }
        }

        results
    }

    fn make_recommendation(
        &self,
        experiment: &Experiment,
        comparisons: &[VariantComparison],
        guardrail_results: &[GuardrailResult],
    ) -> ExperimentRecommendation {
        // Check guardrails first
        let violated: Vec<String> = guardrail_results.iter()
            .filter(|g| !g.passed)
            .map(|g| g.metric_name.clone())
            .collect();

        if !violated.is_empty() {
            return ExperimentRecommendation::Stop { violated_guardrails: violated };
        }

        // Find best treatment
        let best_comparison = comparisons.iter()
            .filter(|c| c.is_significant && c.probability_better > 0.95)
            .max_by(|a, b| a.relative_lift.partial_cmp(&b.relative_lift).unwrap_or(std::cmp::Ordering::Equal));

        if let Some(best) = best_comparison {
            if best.relative_lift > 0.0 && experiment.primary_metric.higher_is_better
                || best.relative_lift < 0.0 && !experiment.primary_metric.higher_is_better
            {
                return ExperimentRecommendation::ShipTreatment {
                    winning_variant: best.treatment_variant.clone(),
                    confidence: best.probability_better,
                };
            }
        }

        // Check if any treatment is significantly worse
        let significantly_worse = comparisons.iter().any(|c| {
            c.is_significant && c.probability_better < 0.05
        });

        if significantly_worse {
            return ExperimentRecommendation::ShipControl {
                reason: "Treatment is significantly worse than control".into(),
            };
        }

        // Check sample size
        let total_samples: u64 = comparisons.iter()
            .map(|_| experiment.required_sample_size) // Placeholder
            .sum();

        if total_samples < experiment.required_sample_size {
            let estimated_days = ((experiment.required_sample_size - total_samples) / 100) as u32;
            return ExperimentRecommendation::Continue {
                estimated_days_remaining: estimated_days,
                current_power: 0.8, // Placeholder
            };
        }

        ExperimentRecommendation::Inconclusive {
            recommendation: "Consider running longer or with larger effect size".into(),
        }
    }
}

/// Raw experiment data
#[derive(Debug, Clone)]
pub struct ExperimentData {
    pub variants: HashMap<String, VariantData>,
}

#[derive(Debug, Clone)]
pub struct VariantData {
    pub variant_id: String,
    pub sample_size: u64,
    pub conversions: u64,
    pub values: Vec<f64>,
    pub total_count: u64,
}
```

## Experiment Dashboard

### Visualization and Reporting

```rust
/// Dashboard data for experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentDashboard {
    pub experiment: Experiment,
    pub analysis: ExperimentAnalysis,

    /// Time series data
    pub daily_metrics: Vec<DailyMetrics>,

    /// Sample size over time
    pub sample_growth: Vec<SamplePoint>,

    /// Statistical power curve
    pub power_analysis: PowerAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DailyMetrics {
    pub date: chrono::NaiveDate,
    pub variant_metrics: HashMap<String, VariantDailyMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantDailyMetrics {
    pub sample_count: u64,
    pub primary_metric: f64,
    pub conversion_rate: f64,
    pub avg_latency_ms: f64,
    pub avg_sentiment: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplePoint {
    pub timestamp: DateTime<Utc>,
    pub variant_id: String,
    pub cumulative_samples: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerAnalysis {
    /// Current statistical power
    pub current_power: f64,

    /// Samples needed for 80% power
    pub samples_for_80_power: u64,

    /// Samples needed for 95% power
    pub samples_for_95_power: u64,

    /// Power curve points
    pub power_curve: Vec<PowerPoint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerPoint {
    pub sample_size: u64,
    pub power: f64,
}

/// Generate experiment report
pub fn generate_report(dashboard: &ExperimentDashboard) -> ExperimentReport {
    ExperimentReport {
        title: format!("Experiment Report: {}", dashboard.experiment.name),
        generated_at: Utc::now(),

        summary: ReportSummary {
            hypothesis: dashboard.experiment.hypothesis.statement.clone(),
            status: format!("{:?}", dashboard.experiment.status),
            duration_days: dashboard.experiment.started_at
                .map(|s| (Utc::now() - s).num_days() as u32)
                .unwrap_or(0),
            total_samples: dashboard.analysis.variant_results.values()
                .map(|v| v.sample_size)
                .sum(),
            recommendation: format!("{:?}", dashboard.analysis.recommendation),
        },

        key_findings: generate_key_findings(&dashboard.analysis),

        variant_performance: dashboard.analysis.variant_results.iter()
            .map(|(id, v)| VariantPerformanceReport {
                variant_id: id.clone(),
                sample_size: v.sample_size,
                primary_metric: v.primary_metric_value,
                ci_lower: v.primary_metric_ci.lower,
                ci_upper: v.primary_metric_ci.upper,
            })
            .collect(),

        guardrail_summary: dashboard.analysis.guardrail_results.iter()
            .map(|g| GuardrailSummary {
                metric: g.metric_name.clone(),
                passed: g.passed,
                change_percent: g.relative_change,
            })
            .collect(),
    }
}

fn generate_key_findings(analysis: &ExperimentAnalysis) -> Vec<String> {
    let mut findings = Vec::new();

    for comparison in &analysis.comparisons {
        if comparison.is_significant {
            findings.push(format!(
                "{} shows {:.1}% {} compared to {} (p={:.4}, P(better)={:.1}%)",
                comparison.treatment_variant,
                comparison.relative_lift.abs() * 100.0,
                if comparison.relative_lift > 0.0 { "improvement" } else { "degradation" },
                comparison.control_variant,
                comparison.p_value,
                comparison.probability_better * 100.0,
            ));
        }
    }

    for guardrail in &analysis.guardrail_results {
        if !guardrail.passed {
            findings.push(format!(
                "GUARDRAIL VIOLATED: {} degraded by {:.1}%",
                guardrail.metric_name,
                guardrail.relative_change,
            ));
        }
    }

    findings
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExperimentReport {
    pub title: String,
    pub generated_at: DateTime<Utc>,
    pub summary: ReportSummary,
    pub key_findings: Vec<String>,
    pub variant_performance: Vec<VariantPerformanceReport>,
    pub guardrail_summary: Vec<GuardrailSummary>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReportSummary {
    pub hypothesis: String,
    pub status: String,
    pub duration_days: u32,
    pub total_samples: u64,
    pub recommendation: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VariantPerformanceReport {
    pub variant_id: String,
    pub sample_size: u64,
    pub primary_metric: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GuardrailSummary {
    pub metric: String,
    pub passed: bool,
    pub change_percent: f64,
}
```

## Common Experiments

### RAG Timing Strategy (Primary Use Case)

```rust
/// RAG timing experiment configurations
pub mod rag_timing {
    use super::*;

    /// Sequential RAG configuration
    pub fn sequential_config() -> HashMap<String, serde_json::Value> {
        hashmap! {
            "rag.timing_strategy".into() => json!("sequential"),
            "rag.max_retrieval_time_ms".into() => json!(500),
        }
    }

    /// Prefetch Async configuration
    pub fn prefetch_async_config() -> HashMap<String, serde_json::Value> {
        hashmap! {
            "rag.timing_strategy".into() => json!("prefetch_async"),
            "rag.prefetch_confidence_threshold".into() => json!(0.7),
            "rag.prefetch_min_words".into() => json!(3),
            "rag.cache_ttl_ms".into() => json!(500),
        }
    }

    /// Parallel Inject configuration
    pub fn parallel_inject_config() -> HashMap<String, serde_json::Value> {
        hashmap! {
            "rag.timing_strategy".into() => json!("parallel_inject"),
            "rag.always_include_top_k".into() => json!(3),
            "rag.context_token_budget".into() => json!(500),
        }
    }

    /// Metrics to track
    pub fn rag_metrics() -> (MetricDefinition, Vec<MetricDefinition>) {
        let primary = MetricDefinition {
            name: "time_to_first_audio_ms".into(),
            metric_type: MetricType::Continuous,
            aggregation: AggregationType::P50,
            higher_is_better: false,
            minimum_detectable_effect: 0.10,
        };

        let secondary = vec![
            MetricDefinition {
                name: "response_relevance_score".into(),
                metric_type: MetricType::Continuous,
                aggregation: AggregationType::Mean,
                higher_is_better: true,
                minimum_detectable_effect: 0.05,
            },
            MetricDefinition {
                name: "rag_utilization_rate".into(),
                metric_type: MetricType::Ratio,
                aggregation: AggregationType::Rate,
                higher_is_better: true,
                minimum_detectable_effect: 0.05,
            },
            MetricDefinition {
                name: "context_token_usage".into(),
                metric_type: MetricType::Continuous,
                aggregation: AggregationType::Mean,
                higher_is_better: false,
                minimum_detectable_effect: 0.10,
            },
        ];

        (primary, secondary)
    }
}
```

### Voice/TTS Experiments

```rust
pub mod voice_experiments {
    use super::*;

    /// TTS voice comparison
    pub fn create_voice_experiment() -> Experiment {
        Experiment {
            id: "exp_voice_001".into(),
            name: "TTS Voice Comparison".into(),
            description: "Compare different TTS voices for customer preference and trust".into(),

            hypothesis: Hypothesis {
                statement: "A warmer, more natural voice will increase customer engagement and trust scores".into(),
                expected_effect_size: 15.0,
                rationale: "Voice quality directly impacts perceived trustworthiness in financial services".into(),
            },

            variants: vec![
                Variant {
                    id: "control".into(),
                    name: "Standard Voice".into(),
                    is_control: true,
                    traffic_percentage: 50.0,
                    config_overrides: hashmap! {
                        "tts.voice_id".into() => json!("indicf5_hindi_default"),
                        "tts.rate".into() => json!(1.0),
                    },
                    description: "Current production voice".into(),
                },
                Variant {
                    id: "treatment".into(),
                    name: "Warm Voice".into(),
                    is_control: false,
                    traffic_percentage: 50.0,
                    config_overrides: hashmap! {
                        "tts.voice_id".into() => json!("indicf5_hindi_warm"),
                        "tts.rate".into() => json!(0.95),
                        "tts.pitch".into() => json!(-0.5),
                    },
                    description: "Warmer, slightly slower voice".into(),
                },
            ],

            primary_metric: MetricDefinition {
                name: "conversation_completion_rate".into(),
                metric_type: MetricType::Ratio,
                aggregation: AggregationType::Rate,
                higher_is_better: true,
                minimum_detectable_effect: 0.05,
            },

            secondary_metrics: vec![
                MetricDefinition {
                    name: "avg_sentiment".into(),
                    metric_type: MetricType::Continuous,
                    aggregation: AggregationType::Mean,
                    higher_is_better: true,
                    minimum_detectable_effect: 0.05,
                },
            ],

            guardrail_metrics: vec![
                GuardrailMetric {
                    metric: MetricDefinition {
                        name: "error_rate".into(),
                        metric_type: MetricType::Ratio,
                        aggregation: AggregationType::Rate,
                        higher_is_better: false,
                        minimum_detectable_effect: 0.01,
                    },
                    max_degradation_percent: 1.0,
                    confidence_level: 0.95,
                },
            ],

            allocation: TrafficAllocation {
                strategy: AllocationStrategy::Fixed,
                ramp_schedule: None,
            },

            targeting: TargetingRules {
                segments: None,
                languages: Some(vec!["hi".into()]),
                regions: None,
                exclusive: false,
                custom_expression: None,
            },

            status: ExperimentStatus::Draft,
            created_at: Utc::now(),
            started_at: None,
            ended_at: None,
            required_sample_size: 3000,
            owner: "voice-team@bank.com".into(),
        }
    }
}
```

## Integration with Voice Agent

### Runtime Integration

```rust
use std::sync::Arc;

/// Experiment-aware configuration provider
pub struct ExperimentalConfig {
    base_config: Config,
    bucketer: Arc<ExperimentBucketer>,
    metrics_collector: Arc<MetricsCollector>,

    /// Current session's assignments
    assignments: Vec<ExperimentAssignment>,
}

impl ExperimentalConfig {
    /// Initialize for a new session
    pub async fn for_session(
        base_config: Config,
        bucketer: Arc<ExperimentBucketer>,
        metrics_collector: Arc<MetricsCollector>,
        session_id: &str,
        customer_id: &str,
        context: &SessionContext,
    ) -> Self {
        let assignments = bucketer.get_assignment(session_id, customer_id, context);

        // Record session start with assignments
        metrics_collector.record_session_start(
            session_id,
            customer_id,
            assignments.clone(),
        ).await.ok();

        Self {
            base_config,
            bucketer,
            metrics_collector,
            assignments,
        }
    }

    /// Get config value with experiment overrides
    pub fn get<T: serde::de::DeserializeOwned>(&self, key: &str) -> Option<T> {
        // Check experiment overrides first
        for assignment in &self.assignments {
            if let Some(value) = assignment.config_overrides.get(key) {
                return serde_json::from_value(value.clone()).ok();
            }
        }

        // Fall back to base config
        self.base_config.get(key)
    }

    /// Get RAG timing strategy
    pub fn rag_timing_strategy(&self) -> RagTimingStrategy {
        self.get("rag.timing_strategy")
            .unwrap_or(RagTimingStrategy::Sequential)
    }

    /// Get TTS voice ID
    pub fn tts_voice_id(&self) -> String {
        self.get("tts.voice_id")
            .unwrap_or_else(|| "default".to_string())
    }

    /// Record metric for experiments
    pub async fn record_metric(&self, metric_name: &str, value: f64) {
        // Implementation would record to metrics collector
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RagTimingStrategy {
    Sequential,
    PrefetchAsync,
    ParallelInject,
}
```

## Summary

The A/B testing framework provides:

1. **Experiment Definition**: Structured hypothesis, variants, metrics, and guardrails
2. **Consistent Bucketing**: Hash-based assignment ensures customers see same variant across sessions
3. **Funnel Metrics**: Track progress through conversation stages with sentiment summaries
4. **Statistical Analysis**: Both frequentist (p-values, CIs) and Bayesian (probability better, expected loss)
5. **Guardrails**: Prevent shipping changes that degrade critical metrics
6. **Recommendations**: Automated analysis with ship/stop/continue decisions

Key experiments planned:
- RAG timing strategy (sequential vs prefetch vs parallel)
- TTS voice comparison
- Conversation flow variations
- Objection handling strategies

All experiments integrate seamlessly with the voice agent runtime through `ExperimentalConfig`.
