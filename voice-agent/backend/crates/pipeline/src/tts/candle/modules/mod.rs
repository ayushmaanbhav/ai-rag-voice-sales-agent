//! Neural Network Modules for IndicF5 TTS
//!
//! This module provides all the building blocks needed for the F5-TTS architecture:
//!
//! - **norm**: Layer normalization variants (LayerNorm, RMSNorm, AdaLayerNorm)
//! - **attention**: Self-attention with RoPE position encoding
//! - **feedforward**: MLP with GELU/SwiGLU activation
//! - **conv**: ConvNeXt V2 blocks and convolutional position embedding
//! - **embedding**: Text, mel, and time embeddings

pub mod attention;
pub mod conv;
pub mod embedding;
pub mod feedforward;
pub mod norm;

// Re-export commonly used types
pub use attention::{QKVProjection, RotaryEmbedding, SelfAttention};
pub use conv::{CausalConv1d, ConvNeXtV2Block, ConvPositionEmbedding, GRN};
pub use embedding::{
    DurationEmbedding, InputEmbedding, SinusoidalPositionalEmbedding, TextEmbedding, TimeEmbedding,
};
pub use feedforward::{Dropout, FeedForward, GatedFeedForward};
pub use norm::{AdaLayerNorm, AdaLayerNormOutput, LayerNorm, RMSNorm};

#[cfg(test)]
mod tests {
    #[test]
    fn test_modules_export() {
        // Verify all modules are accessible
    }
}
