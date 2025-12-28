//! Core traits and types for the voice agent
//!
//! This crate provides foundational types used across all other crates:
//! - Audio frame types and processing
//! - Error types
//! - Common traits
//! - Conversation types

pub mod audio;
pub mod error;
pub mod transcript;
pub mod conversation;
pub mod customer;

pub use audio::{AudioFrame, AudioEncoding, Channels, SampleRate};
pub use error::{Error, Result};
pub use transcript::{TranscriptResult, WordTimestamp};
pub use conversation::{Turn, TurnRole, ConversationStage};
pub use customer::{CustomerProfile, CustomerSegment};
