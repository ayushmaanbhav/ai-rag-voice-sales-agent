//! Hybrid Turn Detection
//!
//! Combines VAD-based silence detection with semantic completeness analysis.
//! Architecture: Silence detector + Lightweight transformer classifier

mod hybrid;
mod semantic;

pub use hybrid::{HybridTurnDetector, TurnDetectionConfig, TurnDetectionResult, TurnState};
pub use semantic::SemanticTurnDetector;
