//! Session Management
//!
//! Manages voice agent sessions.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::RwLock;

use voice_agent_agent::{GoldLoanAgent, AgentConfig};

use crate::ServerError;

/// Session state
pub struct Session {
    /// Session ID
    pub id: String,
    /// Agent instance
    pub agent: Arc<GoldLoanAgent>,
    /// Creation time
    pub created_at: Instant,
    /// Last activity
    pub last_activity: RwLock<Instant>,
    /// Is active
    pub active: RwLock<bool>,
}

impl Session {
    /// Create a new session
    pub fn new(id: impl Into<String>, config: AgentConfig) -> Self {
        let id = id.into();
        Self {
            agent: Arc::new(GoldLoanAgent::new(&id, config)),
            id,
            created_at: Instant::now(),
            last_activity: RwLock::new(Instant::now()),
            active: RwLock::new(true),
        }
    }

    /// Update last activity
    pub fn touch(&self) {
        *self.last_activity.write() = Instant::now();
    }

    /// Check if session is expired
    pub fn is_expired(&self, timeout: Duration) -> bool {
        self.last_activity.read().elapsed() > timeout
    }

    /// Close session
    pub fn close(&self) {
        *self.active.write() = false;
    }

    /// Is session active
    pub fn is_active(&self) -> bool {
        *self.active.read()
    }
}

/// Session manager
pub struct SessionManager {
    sessions: RwLock<HashMap<String, Arc<Session>>>,
    max_sessions: usize,
    session_timeout: Duration,
}

impl SessionManager {
    /// Create a new session manager
    pub fn new(max_sessions: usize) -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
            max_sessions,
            session_timeout: Duration::from_secs(3600), // 1 hour
        }
    }

    /// Create a new session
    pub fn create(&self, config: AgentConfig) -> Result<Arc<Session>, ServerError> {
        let mut sessions = self.sessions.write();

        // Check capacity
        if sessions.len() >= self.max_sessions {
            // Try to clean expired sessions
            self.cleanup_expired_internal(&mut sessions);

            if sessions.len() >= self.max_sessions {
                return Err(ServerError::Session("Max sessions reached".to_string()));
            }
        }

        let id = uuid::Uuid::new_v4().to_string();
        let session = Arc::new(Session::new(&id, config));
        sessions.insert(id.clone(), session.clone());

        tracing::info!("Created session: {}", id);

        Ok(session)
    }

    /// Get a session by ID
    pub fn get(&self, id: &str) -> Option<Arc<Session>> {
        let sessions = self.sessions.read();
        sessions.get(id).cloned()
    }

    /// Remove a session
    pub fn remove(&self, id: &str) {
        let mut sessions = self.sessions.write();
        if let Some(session) = sessions.remove(id) {
            session.close();
            tracing::info!("Removed session: {}", id);
        }
    }

    /// Get active session count
    pub fn count(&self) -> usize {
        self.sessions.read().len()
    }

    /// Cleanup expired sessions
    pub fn cleanup_expired(&self) {
        let mut sessions = self.sessions.write();
        self.cleanup_expired_internal(&mut sessions);
    }

    fn cleanup_expired_internal(&self, sessions: &mut HashMap<String, Arc<Session>>) {
        let timeout = self.session_timeout;
        let expired: Vec<String> = sessions
            .iter()
            .filter(|(_, s)| s.is_expired(timeout))
            .map(|(id, _)| id.clone())
            .collect();

        for id in expired {
            if let Some(session) = sessions.remove(&id) {
                session.close();
                tracing::info!("Expired session: {}", id);
            }
        }
    }

    /// List all session IDs
    pub fn list(&self) -> Vec<String> {
        self.sessions.read().keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_creation() {
        let manager = SessionManager::new(10);
        let session = manager.create(AgentConfig::default()).unwrap();

        assert!(session.is_active());
        assert!(!session.is_expired(Duration::from_secs(60)));
    }

    #[test]
    fn test_session_get() {
        let manager = SessionManager::new(10);
        let session = manager.create(AgentConfig::default()).unwrap();
        let id = session.id.clone();

        let retrieved = manager.get(&id);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, id);
    }

    #[test]
    fn test_session_remove() {
        let manager = SessionManager::new(10);
        let session = manager.create(AgentConfig::default()).unwrap();
        let id = session.id.clone();

        manager.remove(&id);
        assert!(manager.get(&id).is_none());
    }
}
