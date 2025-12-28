//! Application State
//!
//! Shared state across all handlers.

use std::sync::Arc;

use voice_agent_config::Settings;
use voice_agent_tools::ToolRegistry;

use crate::session::SessionManager;

/// Application state
#[derive(Clone)]
pub struct AppState {
    /// Configuration
    pub config: Arc<Settings>,
    /// Session manager
    pub sessions: Arc<SessionManager>,
    /// Tool registry
    pub tools: Arc<ToolRegistry>,
}

impl AppState {
    /// Create new application state
    pub fn new(config: Settings) -> Self {
        Self {
            config: Arc::new(config),
            sessions: Arc::new(SessionManager::new(100)),
            tools: Arc::new(voice_agent_tools::registry::create_default_registry()),
        }
    }
}
