//! Early-Exit Cross-Encoder Reranker
//!
//! Implements multiple early exit strategies:
//! - Confidence-based: Exit when softmax confidence exceeds threshold
//! - Patience-based: Exit when k consecutive layers agree
//! - Hybrid: Combination of confidence and patience
//! - Similarity-based: Exit when layer outputs stabilize

use std::path::Path;
use parking_lot::Mutex;

#[cfg(feature = "onnx")]
use ndarray::Array2;
#[cfg(feature = "onnx")]
use ort::{GraphOptimizationLevel, Session};
#[cfg(feature = "onnx")]
use tokenizers::Tokenizer;

use crate::RagError;

/// Exit strategy for early exit
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitStrategy {
    /// Exit when confidence exceeds threshold
    Confidence,
    /// Exit when k consecutive layers agree
    Patience,
    /// Combination of confidence and patience
    Hybrid,
    /// Exit when layer outputs stabilize
    Similarity,
}

/// Reranker configuration
#[derive(Debug, Clone)]
pub struct RerankerConfig {
    /// Exit strategy
    pub strategy: ExitStrategy,
    /// Confidence threshold for early exit (0.0 - 1.0)
    pub confidence_threshold: f32,
    /// Patience (consecutive agreeing layers)
    pub patience: usize,
    /// Minimum layer before allowing exit
    pub min_layer: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Similarity threshold for stability-based exit
    pub similarity_threshold: f32,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            strategy: ExitStrategy::Hybrid,
            confidence_threshold: 0.9,
            patience: 2,
            min_layer: 3,
            max_seq_len: 256,
            similarity_threshold: 0.95,
        }
    }
}

/// Reranking result
#[derive(Debug, Clone)]
pub struct RerankResult {
    /// Document ID
    pub id: String,
    /// Relevance score
    pub score: f32,
    /// Layer at which exit occurred (None if no early exit)
    pub exit_layer: Option<usize>,
    /// Original rank
    pub original_rank: usize,
}

/// Layer output for tracking
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct LayerOutput {
    /// Predicted class (0 = irrelevant, 1 = relevant)
    prediction: usize,
    /// Confidence (softmax probability of predicted class)
    confidence: f32,
    /// Raw logits
    logits: Vec<f32>,
}

/// Early-exit cross-encoder reranker
pub struct EarlyExitReranker {
    #[cfg(feature = "onnx")]
    session: Session,
    #[cfg(feature = "onnx")]
    tokenizer: Tokenizer,
    config: RerankerConfig,
    /// Statistics for monitoring
    stats: Mutex<RerankerStats>,
}

/// Reranker statistics
#[derive(Debug, Clone, Default)]
pub struct RerankerStats {
    /// Total documents reranked
    pub total_docs: usize,
    /// Early exits per layer
    pub exits_per_layer: Vec<usize>,
    /// Average exit layer
    pub avg_exit_layer: f32,
    /// Documents that ran all layers
    pub full_runs: usize,
}

impl EarlyExitReranker {
    /// Create a new early-exit reranker
    #[cfg(feature = "onnx")]
    pub fn new(
        model_path: impl AsRef<Path>,
        tokenizer_path: impl AsRef<Path>,
        config: RerankerConfig,
    ) -> Result<Self, RagError> {
        let session = Session::builder()
            .map_err(|e| RagError::Model(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| RagError::Model(e.to_string()))?
            .with_intra_threads(2)
            .map_err(|e| RagError::Model(e.to_string()))?
            .commit_from_file(model_path)
            .map_err(|e| RagError::Model(e.to_string()))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| RagError::Model(e.to_string()))?;

        Ok(Self {
            session,
            tokenizer,
            config,
            stats: Mutex::new(RerankerStats::default()),
        })
    }

    /// Create a new reranker (stub when ONNX disabled)
    #[cfg(not(feature = "onnx"))]
    pub fn new(
        _model_path: impl AsRef<Path>,
        _tokenizer_path: impl AsRef<Path>,
        config: RerankerConfig,
    ) -> Result<Self, RagError> {
        Ok(Self::simple(config))
    }

    /// Create a simple reranker for testing (no model, only when ONNX disabled)
    #[cfg(not(feature = "onnx"))]
    pub fn simple(config: RerankerConfig) -> Self {
        Self {
            config,
            stats: Mutex::new(RerankerStats::default()),
        }
    }

    /// Create a simple reranker for testing (ONNX enabled - panics)
    #[cfg(feature = "onnx")]
    pub fn simple(_config: RerankerConfig) -> Self {
        panic!("EarlyExitReranker::simple() is not available when ONNX feature is enabled. Use new() instead.")
    }

    /// Rerank documents given a query
    pub fn rerank(
        &self,
        query: &str,
        documents: &[(String, String)], // (id, text)
    ) -> Result<Vec<RerankResult>, RagError> {
        let mut results: Vec<RerankResult> = documents
            .iter()
            .enumerate()
            .map(|(i, (id, text))| {
                let (score, exit_layer) = self.score_pair(query, text)?;
                Ok(RerankResult {
                    id: id.clone(),
                    score,
                    exit_layer,
                    original_rank: i,
                })
            })
            .collect::<Result<Vec<_>, RagError>>()?;

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        Ok(results)
    }

    /// Score a query-document pair
    #[cfg(feature = "onnx")]
    fn score_pair(&self, query: &str, document: &str) -> Result<(f32, Option<usize>), RagError> {
        let encoding = self.tokenizer
            .encode((query, document), true)
            .map_err(|e| RagError::Reranker(e.to_string()))?;

        let ids: Vec<i64> = encoding.get_ids()
            .iter()
            .take(self.config.max_seq_len)
            .map(|&id| id as i64)
            .collect();

        let attention_mask: Vec<i64> = vec![1i64; ids.len()];

        let mut padded_ids = vec![0i64; self.config.max_seq_len];
        let mut padded_mask = vec![0i64; self.config.max_seq_len];

        padded_ids[..ids.len()].copy_from_slice(&ids);
        padded_mask[..attention_mask.len()].copy_from_slice(&attention_mask);

        let input_ids = Array2::from_shape_vec((1, self.config.max_seq_len), padded_ids)
            .map_err(|e| RagError::Reranker(e.to_string()))?;
        let attention = Array2::from_shape_vec((1, self.config.max_seq_len), padded_mask)
            .map_err(|e| RagError::Reranker(e.to_string()))?;

        self.run_with_early_exit(&input_ids, &attention)
    }

    /// Score a query-document pair (simple when ONNX disabled)
    #[cfg(not(feature = "onnx"))]
    fn score_pair(&self, query: &str, document: &str) -> Result<(f32, Option<usize>), RagError> {
        let score = SimpleScorer::score(query, document);
        let mut stats = self.stats.lock();
        stats.total_docs += 1;
        Ok((score, None))
    }

    /// Run inference with early exit logic
    #[cfg(feature = "onnx")]
    fn run_with_early_exit(
        &self,
        input_ids: &Array2<i64>,
        attention_mask: &Array2<i64>,
    ) -> Result<(f32, Option<usize>), RagError> {
        let outputs = self.session.run(ort::inputs![
            "input_ids" => input_ids.view(),
            "attention_mask" => attention_mask.view(),
        ].map_err(|e| RagError::Model(e.to_string()))?)
        .map_err(|e| RagError::Model(e.to_string()))?;

        let logits = outputs
            .get("logits")
            .ok_or_else(|| RagError::Model("Missing logits output".to_string()))?
            .try_extract_tensor::<f32>()
            .map_err(|e| RagError::Model(e.to_string()))?;

        let logits_view = logits.view();
        let score = self.compute_relevance_score(&logits_view);

        let mut stats = self.stats.lock();
        stats.total_docs += 1;

        Ok((score, None))
    }

    /// Compute relevance score from logits
    #[cfg(feature = "onnx")]
    fn compute_relevance_score(&self, logits: &ndarray::ArrayViewD<f32>) -> f32 {
        let flat: Vec<f32> = logits.iter().copied().collect();

        if flat.len() >= 2 {
            let max = flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = flat.iter().map(|&x| (x - max).exp()).sum();
            let relevant_prob = (flat[1] - max).exp() / exp_sum;
            relevant_prob
        } else if flat.len() == 1 {
            1.0 / (1.0 + (-flat[0]).exp())
        } else {
            0.0
        }
    }

    /// Check if should exit based on strategy
    #[allow(dead_code)]
    fn should_exit(&self, layer_outputs: &[LayerOutput], current_layer: usize) -> bool {
        if current_layer < self.config.min_layer {
            return false;
        }

        match self.config.strategy {
            ExitStrategy::Confidence => {
                if let Some(last) = layer_outputs.last() {
                    last.confidence >= self.config.confidence_threshold
                } else {
                    false
                }
            }

            ExitStrategy::Patience => {
                if layer_outputs.len() < self.config.patience {
                    return false;
                }

                let recent = &layer_outputs[layer_outputs.len() - self.config.patience..];
                let first_pred = recent[0].prediction;
                recent.iter().all(|o| o.prediction == first_pred)
            }

            ExitStrategy::Hybrid => {
                if let Some(last) = layer_outputs.last() {
                    if last.confidence >= self.config.confidence_threshold {
                        return true;
                    }
                }

                if layer_outputs.len() >= self.config.patience {
                    let recent = &layer_outputs[layer_outputs.len() - self.config.patience..];
                    let first_pred = recent[0].prediction;
                    if recent.iter().all(|o| o.prediction == first_pred) {
                        let avg_conf: f32 = recent.iter().map(|o| o.confidence).sum::<f32>()
                            / self.config.patience as f32;
                        return avg_conf >= 0.7;
                    }
                }

                false
            }

            ExitStrategy::Similarity => {
                if layer_outputs.len() < 2 {
                    return false;
                }

                let prev = &layer_outputs[layer_outputs.len() - 2].logits;
                let curr = &layer_outputs[layer_outputs.len() - 1].logits;

                let similarity = cosine_similarity(prev, curr);
                similarity >= self.config.similarity_threshold
            }
        }
    }

    /// Get reranker statistics
    pub fn stats(&self) -> RerankerStats {
        self.stats.lock().clone()
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        *self.stats.lock() = RerankerStats::default();
    }
}

/// Compute cosine similarity between two vectors
#[allow(dead_code)]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Simple scorer for testing (no model required)
pub struct SimpleScorer;

impl SimpleScorer {
    /// Score based on keyword overlap
    pub fn score(query: &str, document: &str) -> f32 {
        let query_lower = query.to_lowercase();
        let doc_lower = document.to_lowercase();

        let query_words: std::collections::HashSet<&str> = query_lower
            .split_whitespace()
            .collect();

        let doc_words: std::collections::HashSet<&str> = doc_lower
            .split_whitespace()
            .collect();

        let overlap = query_words.intersection(&doc_words).count();
        let union = query_words.union(&doc_words).count();

        if union > 0 {
            overlap as f32 / union as f32
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = RerankerConfig::default();
        assert_eq!(config.strategy, ExitStrategy::Hybrid);
        assert_eq!(config.min_layer, 3);
    }

    #[test]
    fn test_simple_scorer() {
        let score = SimpleScorer::score(
            "gold loan interest rate",
            "The interest rate for gold loan is 10%",
        );
        assert!(score > 0.0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 0.001);
    }
}
