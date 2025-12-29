//! MCP Tools for Gold Loan Voice Agent
//!
//! Implements MCP (Model Context Protocol) compatible tool interface
//! with domain-specific tools for gold loan operations.

pub mod mcp;
pub mod registry;
pub mod gold_loan;
pub mod integrations;

pub use mcp::{
    // Core tool types (from voice_agent_core)
    Tool, ToolInput, ToolOutput, ToolSchema, ToolError, InputSchema, PropertySchema,
    ContentBlock, ErrorCode,
    // P3-3 FIX: Full MCP protocol types
    JsonRpcRequest, JsonRpcResponse, JsonRpcError, RequestId,
    ToolCallParams, ProgressToken, ProgressParams,
    Resource, ResourceContent, ResourceProvider,
    ServerCapabilities, ToolCapabilities, ResourceCapabilities,
    methods,
};
pub use registry::{
    ToolRegistry, ToolExecutor, IntegrationConfig, FullIntegrationConfig,
    create_registry_with_integrations, create_registry_with_persistence,
    // P0-4 FIX: Domain config wiring with hot-reload
    create_registry_with_config, create_registry_with_domain_config,
    ConfigurableToolRegistry,
};
pub use gold_loan::{
    EligibilityCheckTool,
    SavingsCalculatorTool,
    LeadCaptureTool,
    AppointmentSchedulerTool,
    BranchLocatorTool,
    BranchData,
    get_branches,
    reload_branches,
    // P0 FIX: New missing MCP tools
    GetGoldPriceTool,
    EscalateToHumanTool,
    SendSmsTool,
};
pub use integrations::{
    CrmIntegration, StubCrmIntegration, CrmLead, LeadSource, LeadStatus, InterestLevel,
    CalendarIntegration, StubCalendarIntegration, Appointment, AppointmentPurpose, AppointmentStatus, TimeSlot,
    IntegrationError,
};

// P2 FIX: Removed redundant ToolsError enum.
// Use mcp::ToolError for tool execution errors instead.
// This unifies error handling across the tools crate.
//
// P3 FIX: The From<ToolError> impl is now in voice_agent_core::error
// since both types are defined in the core crate.
