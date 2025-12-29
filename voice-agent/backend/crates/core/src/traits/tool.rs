//! Tool trait for MCP-compatible tool implementations
//!
//! This module provides a standardized interface for tools that can be:
//! - Called by agents during conversation
//! - Validated against JSON Schema
//! - Executed with timeout support
//!
//! # Example
//!
//! ```ignore
//! use voice_agent_core::traits::{Tool, ToolSchema, ToolOutput, ToolError};
//! use async_trait::async_trait;
//! use serde_json::Value;
//!
//! struct MyTool;
//!
//! #[async_trait]
//! impl Tool for MyTool {
//!     fn name(&self) -> &str { "my_tool" }
//!     fn description(&self) -> &str { "Does something useful" }
//!     fn schema(&self) -> ToolSchema { /* ... */ }
//!     async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError> { /* ... */ }
//! }
//! ```

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Tool error with MCP error codes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolError {
    /// Error code (MCP compatible)
    pub code: ErrorCode,
    /// Human-readable message
    pub message: String,
    /// Additional error data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl ToolError {
    /// Create an invalid parameters error
    pub fn invalid_params(message: impl Into<String>) -> Self {
        Self {
            code: ErrorCode::InvalidParams,
            message: message.into(),
            data: None,
        }
    }

    /// Create an internal error
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            code: ErrorCode::InternalError,
            message: message.into(),
            data: None,
        }
    }

    /// Create a not found error
    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            code: ErrorCode::MethodNotFound,
            message: message.into(),
            data: None,
        }
    }

    /// Create a timeout error
    pub fn timeout(tool_name: &str, timeout_secs: u64) -> Self {
        Self {
            code: ErrorCode::InternalError,
            message: format!("Tool '{}' timed out after {}s", tool_name, timeout_secs),
            data: None,
        }
    }

    /// Create an error with custom data
    pub fn with_data(mut self, data: Value) -> Self {
        self.data = Some(data);
        self
    }
}

impl std::fmt::Display for ToolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?}] {}", self.code, self.message)
    }
}

impl std::error::Error for ToolError {}

/// MCP Error codes (JSON-RPC 2.0 compatible)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(into = "i32", try_from = "i32")]
pub enum ErrorCode {
    /// Invalid JSON was received
    ParseError,
    /// The JSON sent is not a valid Request object
    InvalidRequest,
    /// The method does not exist / is not available
    MethodNotFound,
    /// Invalid method parameter(s)
    InvalidParams,
    /// Internal JSON-RPC error
    InternalError,
    /// Custom error code
    Custom(i32),
}

impl From<ErrorCode> for i32 {
    fn from(code: ErrorCode) -> Self {
        match code {
            ErrorCode::ParseError => -32700,
            ErrorCode::InvalidRequest => -32600,
            ErrorCode::MethodNotFound => -32601,
            ErrorCode::InvalidParams => -32602,
            ErrorCode::InternalError => -32603,
            ErrorCode::Custom(c) => c,
        }
    }
}

impl TryFrom<i32> for ErrorCode {
    type Error = &'static str;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        Ok(match value {
            -32700 => ErrorCode::ParseError,
            -32600 => ErrorCode::InvalidRequest,
            -32601 => ErrorCode::MethodNotFound,
            -32602 => ErrorCode::InvalidParams,
            -32603 => ErrorCode::InternalError,
            c => ErrorCode::Custom(c),
        })
    }
}

/// Tool input for execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInput {
    /// Tool name
    pub name: String,
    /// Input arguments
    pub arguments: Value,
}

/// Tool output with content blocks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    /// Output content
    pub content: Vec<ContentBlock>,
    /// Is this an error response?
    #[serde(default)]
    pub is_error: bool,
}

impl ToolOutput {
    /// Create a text output
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::Text { text: text.into() }],
            is_error: false,
        }
    }

    /// Create a JSON output (serialized as pretty-printed text)
    pub fn json(value: impl Serialize) -> Self {
        let text = serde_json::to_string_pretty(&value).unwrap_or_default();
        Self {
            content: vec![ContentBlock::Text { text }],
            is_error: false,
        }
    }

    /// Create an error output
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::Text { text: message.into() }],
            is_error: true,
        }
    }

    /// Create an audio output
    pub fn audio(
        data: impl Into<String>,
        mime_type: impl Into<String>,
        sample_rate: Option<u32>,
        duration_ms: Option<u64>,
    ) -> Self {
        Self {
            content: vec![ContentBlock::Audio {
                data: data.into(),
                mime_type: mime_type.into(),
                sample_rate,
                duration_ms,
            }],
            is_error: false,
        }
    }

    /// Create an image output
    pub fn image(data: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::Image {
                data: data.into(),
                mime_type: mime_type.into(),
            }],
            is_error: false,
        }
    }

    /// Create a resource reference output
    pub fn resource(uri: impl Into<String>, mime_type: Option<String>) -> Self {
        Self {
            content: vec![ContentBlock::Resource {
                uri: uri.into(),
                mime_type,
            }],
            is_error: false,
        }
    }
}

/// Content block types for tool output
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Text content
    Text { text: String },
    /// Base64-encoded image
    Image { data: String, mime_type: String },
    /// Resource reference by URI
    Resource { uri: String, mime_type: Option<String> },
    /// Base64-encoded audio
    Audio {
        /// Base64-encoded audio data
        data: String,
        /// MIME type (e.g., "audio/wav", "audio/mp3", "audio/opus")
        mime_type: String,
        /// Sample rate in Hz (e.g., 16000, 22050, 44100)
        #[serde(skip_serializing_if = "Option::is_none")]
        sample_rate: Option<u32>,
        /// Duration in milliseconds
        #[serde(skip_serializing_if = "Option::is_none")]
        duration_ms: Option<u64>,
    },
}

/// Tool schema (JSON Schema format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    /// Tool name
    pub name: String,
    /// Tool description
    pub description: String,
    /// Input schema (JSON Schema)
    pub input_schema: InputSchema,
}

/// Input schema for tool parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputSchema {
    /// Schema type (always "object" for tool inputs)
    #[serde(rename = "type")]
    pub schema_type: String,
    /// Property definitions
    #[serde(default)]
    pub properties: HashMap<String, PropertySchema>,
    /// Required property names
    #[serde(default)]
    pub required: Vec<String>,
}

impl InputSchema {
    /// Create an empty object schema
    pub fn object() -> Self {
        Self {
            schema_type: "object".to_string(),
            properties: HashMap::new(),
            required: Vec::new(),
        }
    }

    /// Add a property to the schema
    pub fn property(mut self, name: &str, schema: PropertySchema, required: bool) -> Self {
        self.properties.insert(name.to_string(), schema);
        if required {
            self.required.push(name.to_string());
        }
        self
    }
}

/// Property schema for input parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertySchema {
    /// Property type (string, number, integer, boolean, array, object)
    #[serde(rename = "type")]
    pub prop_type: String,
    /// Property description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Default value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
    /// Allowed enum values (for string type)
    #[serde(rename = "enum", skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
    /// Minimum value (for number/integer)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum: Option<f64>,
    /// Maximum value (for number/integer)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum: Option<f64>,
}

impl PropertySchema {
    /// Create a string property
    pub fn string(description: impl Into<String>) -> Self {
        Self {
            prop_type: "string".to_string(),
            description: Some(description.into()),
            default: None,
            enum_values: None,
            minimum: None,
            maximum: None,
        }
    }

    /// Create a number property
    pub fn number(description: impl Into<String>) -> Self {
        Self {
            prop_type: "number".to_string(),
            description: Some(description.into()),
            default: None,
            enum_values: None,
            minimum: None,
            maximum: None,
        }
    }

    /// Create an integer property
    pub fn integer(description: impl Into<String>) -> Self {
        Self {
            prop_type: "integer".to_string(),
            description: Some(description.into()),
            default: None,
            enum_values: None,
            minimum: None,
            maximum: None,
        }
    }

    /// Create a boolean property
    pub fn boolean(description: impl Into<String>) -> Self {
        Self {
            prop_type: "boolean".to_string(),
            description: Some(description.into()),
            default: None,
            enum_values: None,
            minimum: None,
            maximum: None,
        }
    }

    /// Create an enum property
    pub fn enum_type(description: impl Into<String>, values: Vec<String>) -> Self {
        Self {
            prop_type: "string".to_string(),
            description: Some(description.into()),
            default: None,
            enum_values: Some(values),
            minimum: None,
            maximum: None,
        }
    }

    /// Add a default value
    pub fn with_default(mut self, default: Value) -> Self {
        self.default = Some(default);
        self
    }

    /// Add a numeric range constraint
    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.minimum = Some(min);
        self.maximum = Some(max);
        self
    }
}

/// Tool trait for MCP-compatible tool implementations
#[async_trait]
pub trait Tool: Send + Sync {
    /// Get tool name (used for invocation)
    fn name(&self) -> &str;

    /// Get tool description (shown to LLM)
    fn description(&self) -> &str;

    /// Get input schema (JSON Schema format)
    fn schema(&self) -> ToolSchema;

    /// Execute the tool with given input
    async fn execute(&self, input: Value) -> Result<ToolOutput, ToolError>;

    /// Validate input against schema (default implementation)
    ///
    /// Checks required fields, types, enum values, and numeric ranges.
    fn validate(&self, input: &Value) -> Result<(), ToolError> {
        let schema = self.schema();

        if let Value::Object(obj) = input {
            // Check required fields
            for required in &schema.input_schema.required {
                if !obj.contains_key(required) {
                    return Err(ToolError::invalid_params(format!(
                        "Missing required field: {}",
                        required
                    )));
                }
            }

            // Validate each property's type and constraints
            for (name, value) in obj {
                if let Some(prop_schema) = schema.input_schema.properties.get(name) {
                    validate_property(name, value, prop_schema)?;
                }
                // Unknown properties are allowed (no additionalProperties: false)
            }

            Ok(())
        } else if schema.input_schema.properties.is_empty() {
            // No properties required
            Ok(())
        } else {
            Err(ToolError::invalid_params("Input must be an object"))
        }
    }

    /// Get per-tool timeout in seconds
    ///
    /// Tools can override this to specify custom timeouts.
    /// Default is 30 seconds.
    fn timeout_secs(&self) -> u64 {
        30
    }
}

/// Validate a property value against its schema
pub fn validate_property(name: &str, value: &Value, schema: &PropertySchema) -> Result<(), ToolError> {
    // Check type
    let type_valid = match schema.prop_type.as_str() {
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.is_i64() || value.is_u64(),
        "boolean" => value.is_boolean(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        "null" => value.is_null(),
        _ => true, // Unknown types pass through
    };

    if !type_valid {
        return Err(ToolError::invalid_params(format!(
            "Field '{}' must be of type '{}', got '{}'",
            name,
            schema.prop_type,
            json_type_name(value)
        )));
    }

    // Check enum values
    if let Some(enum_values) = &schema.enum_values {
        if let Some(s) = value.as_str() {
            if !enum_values.contains(&s.to_string()) {
                return Err(ToolError::invalid_params(format!(
                    "Field '{}' must be one of: [{}], got '{}'",
                    name,
                    enum_values.join(", "),
                    s
                )));
            }
        }
    }

    // Check numeric range
    if let Some(num) = value.as_f64() {
        if let Some(min) = schema.minimum {
            if num < min {
                return Err(ToolError::invalid_params(format!(
                    "Field '{}' must be >= {}, got {}",
                    name, min, num
                )));
            }
        }
        if let Some(max) = schema.maximum {
            if num > max {
                return Err(ToolError::invalid_params(format!(
                    "Field '{}' must be <= {}, got {}",
                    name, max, num
                )));
            }
        }
    }

    Ok(())
}

/// Get a human-readable type name for a JSON value
fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_property_schema_builders() {
        let s = PropertySchema::string("A string property");
        assert_eq!(s.prop_type, "string");

        let n = PropertySchema::number("A number").with_range(0.0, 100.0);
        assert_eq!(n.minimum, Some(0.0));
        assert_eq!(n.maximum, Some(100.0));

        let e = PropertySchema::enum_type("An enum", vec!["a".into(), "b".into()]);
        assert_eq!(e.enum_values, Some(vec!["a".into(), "b".into()]));
    }

    #[test]
    fn test_input_schema_builder() {
        let schema = InputSchema::object()
            .property("name", PropertySchema::string("Name"), true)
            .property("age", PropertySchema::integer("Age"), false);

        assert_eq!(schema.properties.len(), 2);
        assert_eq!(schema.required, vec!["name"]);
    }

    #[test]
    fn test_tool_output_constructors() {
        let text = ToolOutput::text("Hello");
        assert!(!text.is_error);
        assert!(matches!(&text.content[0], ContentBlock::Text { text } if text == "Hello"));

        let err = ToolOutput::error("Something went wrong");
        assert!(err.is_error);
    }

    #[test]
    fn test_validate_property_type() {
        let schema = PropertySchema::string("test");
        assert!(validate_property("field", &json!("valid"), &schema).is_ok());
        assert!(validate_property("field", &json!(123), &schema).is_err());
    }

    #[test]
    fn test_validate_property_enum() {
        let schema = PropertySchema::enum_type("test", vec!["a".into(), "b".into()]);
        assert!(validate_property("field", &json!("a"), &schema).is_ok());
        assert!(validate_property("field", &json!("c"), &schema).is_err());
    }

    #[test]
    fn test_validate_property_range() {
        let schema = PropertySchema::number("test").with_range(0.0, 10.0);
        assert!(validate_property("field", &json!(5.0), &schema).is_ok());
        assert!(validate_property("field", &json!(-1.0), &schema).is_err());
        assert!(validate_property("field", &json!(15.0), &schema).is_err());
    }

    #[test]
    fn test_error_code_serialization() {
        let code: i32 = ErrorCode::InvalidParams.into();
        assert_eq!(code, -32602);

        let parsed = ErrorCode::try_from(-32602).unwrap();
        assert_eq!(parsed, ErrorCode::InvalidParams);
    }
}
