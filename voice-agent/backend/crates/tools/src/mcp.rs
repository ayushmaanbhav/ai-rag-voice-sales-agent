//! MCP (Model Context Protocol) Tool Interface
//!
//! Full MCP protocol compliance including:
//! - JSON-RPC 2.0 request/response envelopes
//! - Tool listing and execution
//! - Resource management
//! - Progress reporting
//!
//! P3 FIX: Tool trait and types are now defined in voice-agent-core
//! and re-exported here for backwards compatibility.

use serde::{Deserialize, Serialize};
use serde_json::Value;

// Re-export all tool types from core crate
pub use voice_agent_core::traits::{
    Tool, ToolError, ErrorCode, ToolInput, ToolOutput, ContentBlock,
    ToolSchema, InputSchema, PropertySchema,
};

// ============================================================================
// P3-3 FIX: Full MCP Protocol Compliance
// ============================================================================

/// JSON-RPC 2.0 Request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    /// JSON-RPC version (always "2.0")
    pub jsonrpc: String,
    /// Request ID (optional for notifications)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<RequestId>,
    /// Method name
    pub method: String,
    /// Method parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl JsonRpcRequest {
    /// Create a new request
    pub fn new(method: impl Into<String>, id: impl Into<RequestId>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: Some(id.into()),
            method: method.into(),
            params: None,
        }
    }

    /// Create a request with parameters
    pub fn with_params(method: impl Into<String>, id: impl Into<RequestId>, params: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: Some(id.into()),
            method: method.into(),
            params: Some(params),
        }
    }

    /// Create a notification (no response expected)
    pub fn notification(method: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: None,
            method: method.into(),
            params: None,
        }
    }

    /// Check if this is a notification
    pub fn is_notification(&self) -> bool {
        self.id.is_none()
    }
}

/// JSON-RPC 2.0 Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    /// JSON-RPC version (always "2.0")
    pub jsonrpc: String,
    /// Request ID
    pub id: Option<RequestId>,
    /// Result (present on success)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// Error (present on failure)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

impl JsonRpcResponse {
    /// Create a success response
    pub fn success(id: impl Into<RequestId>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: Some(id.into()),
            result: Some(result),
            error: None,
        }
    }

    /// Create an error response
    pub fn error(id: Option<RequestId>, error: JsonRpcError) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(error),
        }
    }

    /// Create error from ToolError
    pub fn from_tool_error(id: Option<RequestId>, err: ToolError) -> Self {
        Self::error(id, JsonRpcError::from(err))
    }
}

/// JSON-RPC Error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    /// Error code
    pub code: i32,
    /// Error message
    pub message: String,
    /// Additional data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl From<ToolError> for JsonRpcError {
    fn from(err: ToolError) -> Self {
        Self {
            code: err.code.into(),
            message: err.message,
            data: err.data,
        }
    }
}

/// Request ID (string or number)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(untagged)]
pub enum RequestId {
    String(String),
    Number(i64),
}

impl From<String> for RequestId {
    fn from(s: String) -> Self {
        RequestId::String(s)
    }
}

impl From<&str> for RequestId {
    fn from(s: &str) -> Self {
        RequestId::String(s.to_string())
    }
}

impl From<i64> for RequestId {
    fn from(n: i64) -> Self {
        RequestId::Number(n)
    }
}

impl From<i32> for RequestId {
    fn from(n: i32) -> Self {
        RequestId::Number(n as i64)
    }
}

/// MCP Method names
pub mod methods {
    /// List available tools
    pub const TOOLS_LIST: &str = "tools/list";
    /// Call a tool
    pub const TOOLS_CALL: &str = "tools/call";
    /// List resources
    pub const RESOURCES_LIST: &str = "resources/list";
    /// Read a resource
    pub const RESOURCES_READ: &str = "resources/read";
    /// Subscribe to resource updates
    pub const RESOURCES_SUBSCRIBE: &str = "resources/subscribe";
    /// Unsubscribe from resource updates
    pub const RESOURCES_UNSUBSCRIBE: &str = "resources/unsubscribe";
    /// Report progress (notification)
    pub const PROGRESS: &str = "$/progress";
    /// Cancel request (notification)
    pub const CANCEL: &str = "$/cancelRequest";
}

/// Tool call parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallParams {
    /// Tool name
    pub name: String,
    /// Tool arguments
    #[serde(default)]
    pub arguments: Value,
    /// Progress token for progress reporting
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress_token: Option<ProgressToken>,
}

/// Progress token for tracking long operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(untagged)]
pub enum ProgressToken {
    String(String),
    Number(i64),
}

impl From<String> for ProgressToken {
    fn from(s: String) -> Self {
        ProgressToken::String(s)
    }
}

impl From<i64> for ProgressToken {
    fn from(n: i64) -> Self {
        ProgressToken::Number(n)
    }
}

/// Progress notification params
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressParams {
    /// Progress token
    pub token: ProgressToken,
    /// Progress value (0.0 - 1.0)
    pub value: f64,
    /// Progress message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

impl ProgressParams {
    /// Create new progress notification
    pub fn new(token: ProgressToken, value: f64) -> Self {
        Self {
            token,
            value: value.clamp(0.0, 1.0),
            message: None,
        }
    }

    /// Create with message
    pub fn with_message(token: ProgressToken, value: f64, message: impl Into<String>) -> Self {
        Self {
            token,
            value: value.clamp(0.0, 1.0),
            message: Some(message.into()),
        }
    }
}

/// Resource definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Resource {
    /// Resource URI
    pub uri: String,
    /// Resource name
    pub name: String,
    /// Resource description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// MIME type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

impl Resource {
    /// Create a new resource
    pub fn new(uri: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            name: name.into(),
            description: None,
            mime_type: None,
        }
    }

    /// Add description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add MIME type
    pub fn with_mime_type(mut self, mime: impl Into<String>) -> Self {
        self.mime_type = Some(mime.into());
        self
    }
}

/// Resource content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceContent {
    /// Resource URI
    pub uri: String,
    /// MIME type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
    /// Text content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    /// Binary content (base64)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blob: Option<String>,
}

impl ResourceContent {
    /// Create text resource
    pub fn text(uri: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            mime_type: Some("text/plain".to_string()),
            text: Some(content.into()),
            blob: None,
        }
    }

    /// Create JSON resource
    pub fn json(uri: impl Into<String>, value: &impl Serialize) -> Self {
        Self {
            uri: uri.into(),
            mime_type: Some("application/json".to_string()),
            text: serde_json::to_string_pretty(value).ok(),
            blob: None,
        }
    }

    /// Create binary resource
    pub fn binary(uri: impl Into<String>, data: &[u8], mime_type: impl Into<String>) -> Self {
        use base64::Engine;
        Self {
            uri: uri.into(),
            mime_type: Some(mime_type.into()),
            text: None,
            blob: Some(base64::engine::general_purpose::STANDARD.encode(data)),
        }
    }
}

/// Resource provider trait
pub trait ResourceProvider: Send + Sync {
    /// List available resources
    fn list(&self) -> Vec<Resource>;

    /// Read a resource by URI
    fn read(&self, uri: &str) -> Option<ResourceContent>;

    /// Check if provider supports subscriptions
    fn supports_subscriptions(&self) -> bool {
        false
    }
}

/// MCP Server capabilities
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServerCapabilities {
    /// Tool capabilities
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolCapabilities>,
    /// Resource capabilities
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourceCapabilities>,
    /// Experimental features
    #[serde(skip_serializing_if = "Option::is_none")]
    pub experimental: Option<Value>,
}

/// Tool capabilities
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolCapabilities {
    /// Server supports listing tools that have changed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

/// Resource capabilities
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceCapabilities {
    /// Server supports subscriptions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subscribe: Option<bool>,
    /// Server supports listing resources that have changed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub list_changed: Option<bool>,
}

impl ServerCapabilities {
    /// Create capabilities with tools support
    pub fn with_tools() -> Self {
        Self {
            tools: Some(ToolCapabilities::default()),
            ..Default::default()
        }
    }

    /// Add resources support
    pub fn with_resources(mut self, subscribe: bool) -> Self {
        self.resources = Some(ResourceCapabilities {
            subscribe: Some(subscribe),
            list_changed: Some(true),
        });
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_schema() {
        let schema = InputSchema::object()
            .property("name", PropertySchema::string("Customer name"), true)
            .property("amount", PropertySchema::number("Loan amount"), true)
            .property("city", PropertySchema::string("City"), false);

        assert_eq!(schema.properties.len(), 3);
        assert_eq!(schema.required.len(), 2);
    }

    #[test]
    fn test_tool_output() {
        let output = ToolOutput::text("Hello world");
        assert!(!output.is_error);
        assert_eq!(output.content.len(), 1);

        let error = ToolOutput::error("Something went wrong");
        assert!(error.is_error);
    }

    #[test]
    fn test_tool_error() {
        let err = ToolError::invalid_params("Bad input");
        assert_eq!(err.code, ErrorCode::InvalidParams);
    }

    #[test]
    fn test_validate_property_type() {
        let string_schema = PropertySchema::string("Test");

        // Valid string
        assert!(validate_property("name", &serde_json::json!("hello"), &string_schema).is_ok());

        // Invalid: number instead of string
        assert!(validate_property("name", &serde_json::json!(123), &string_schema).is_err());
    }

    #[test]
    fn test_validate_enum() {
        let enum_schema = PropertySchema::enum_type("Status", vec!["active".into(), "inactive".into()]);

        // Valid enum value
        assert!(validate_property("status", &serde_json::json!("active"), &enum_schema).is_ok());

        // Invalid enum value
        let result = validate_property("status", &serde_json::json!("unknown"), &enum_schema);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("must be one of"));
    }

    #[test]
    fn test_validate_range() {
        let range_schema = PropertySchema::number("Amount").with_range(100.0, 1000.0);

        // Valid: within range
        assert!(validate_property("amount", &serde_json::json!(500), &range_schema).is_ok());

        // Invalid: below minimum
        let result = validate_property("amount", &serde_json::json!(50), &range_schema);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains(">="));

        // Invalid: above maximum
        let result = validate_property("amount", &serde_json::json!(2000), &range_schema);
        assert!(result.is_err());
        assert!(result.unwrap_err().message.contains("<="));
    }
}
