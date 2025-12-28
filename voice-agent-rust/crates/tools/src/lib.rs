//! MCP Tools for Gold Loan Voice Agent
//!
//! Implements MCP (Model Context Protocol) compatible tool interface
//! with domain-specific tools for gold loan operations.

pub mod mcp;
pub mod registry;
pub mod gold_loan;

pub use mcp::{Tool, ToolInput, ToolOutput, ToolSchema, ToolError};
pub use registry::{ToolRegistry, ToolExecutor};
pub use gold_loan::{
    EligibilityCheckTool,
    SavingsCalculatorTool,
    LeadCaptureTool,
    AppointmentSchedulerTool,
    BranchLocatorTool,
};

use thiserror::Error;

/// Tool errors
#[derive(Error, Debug)]
pub enum ToolsError {
    #[error("Tool not found: {0}")]
    NotFound(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Execution error: {0}")]
    Execution(String),

    #[error("API error: {0}")]
    Api(String),

    #[error("Timeout")]
    Timeout,

    #[error("Permission denied: {0}")]
    PermissionDenied(String),
}

impl From<ToolsError> for voice_agent_core::Error {
    fn from(err: ToolsError) -> Self {
        voice_agent_core::Error::Tool(voice_agent_core::error::ToolError::ExecutionFailed(err.to_string()))
    }
}
