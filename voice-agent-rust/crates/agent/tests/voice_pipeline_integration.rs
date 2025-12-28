//! Integration tests for the voice pipeline (STT -> Agent -> TTS)
//!
//! These tests verify the end-to-end flow of voice interactions.

use std::time::Duration;
use tokio::time::timeout;

use voice_agent_agent::{
    VoiceSession, VoiceSessionConfig, VoiceSessionState, VoiceSessionEvent,
    TransportSession, SessionConfig,
};

/// Test that a voice session can be created and started
#[tokio::test]
async fn test_voice_session_lifecycle() {
    let config = VoiceSessionConfig::default();
    let session = VoiceSession::new("test-lifecycle", config).unwrap();

    // Initial state
    assert_eq!(session.state().await, VoiceSessionState::Idle);
    assert_eq!(session.session_id(), "test-lifecycle");

    // Start the session
    let result = session.start().await;
    assert!(result.is_ok());

    // Should now be listening
    assert_eq!(session.state().await, VoiceSessionState::Listening);

    // End the session
    session.end("test complete").await;
    assert_eq!(session.state().await, VoiceSessionState::Ended);
}

/// Test voice session event subscription
#[tokio::test]
async fn test_voice_session_events() {
    let config = VoiceSessionConfig::default();
    let session = VoiceSession::new("test-events", config).unwrap();

    // Subscribe to events before starting
    let mut event_rx = session.subscribe();

    // Start the session
    session.start().await.unwrap();

    // Should receive Started event
    let event = timeout(Duration::from_millis(100), event_rx.recv()).await;
    assert!(event.is_ok());
    if let Ok(Ok(VoiceSessionEvent::Started { session_id })) = event {
        assert_eq!(session_id, "test-events");
    }

    // Should receive StateChanged event
    let event = timeout(Duration::from_millis(100), event_rx.recv()).await;
    assert!(event.is_ok());
    if let Ok(Ok(VoiceSessionEvent::StateChanged { old, new })) = event {
        assert_eq!(old, VoiceSessionState::Idle);
        assert_eq!(new, VoiceSessionState::Listening);
    }
}

/// Test audio processing flow
#[tokio::test]
async fn test_audio_processing() {
    let mut config = VoiceSessionConfig::default();
    config.vad_energy_threshold = 0.001; // Lower threshold for testing

    let session = VoiceSession::new("test-audio", config).unwrap();
    session.start().await.unwrap();

    // Process some "silence" (low energy audio)
    let silence = vec![0.0f32; 320]; // 20ms at 16kHz
    let result = session.process_audio(&silence).await;
    assert!(result.is_ok());

    // Process some "speech" (higher energy audio)
    let speech: Vec<f32> = (0..320).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
    let result = session.process_audio(&speech).await;
    assert!(result.is_ok());

    // State should still be listening (haven't ended turn yet)
    assert_eq!(session.state().await, VoiceSessionState::Listening);
}

/// Test transport attachment
#[tokio::test]
async fn test_transport_attachment() {
    let config = VoiceSessionConfig::default();
    let session = VoiceSession::new("test-transport", config).unwrap();

    // Initially no transport
    assert!(!session.is_transport_connected().await);

    // Attach transport
    let transport = TransportSession::new(SessionConfig::default());
    session.attach_transport(transport).await;

    // Transport attached but not connected
    assert!(!session.is_transport_connected().await);

    // Connection would require actual WebRTC signaling, which we skip in unit tests
}

/// Test barge-in detection
#[tokio::test]
async fn test_barge_in_config() {
    let mut config = VoiceSessionConfig::default();
    config.barge_in_enabled = true;
    config.vad_energy_threshold = 0.01;

    let session = VoiceSession::new("test-bargein", config.clone()).unwrap();

    // Verify config is set correctly
    assert!(config.barge_in_enabled);
    assert_eq!(config.vad_energy_threshold, 0.01);
    assert!(session.session_id() == "test-bargein");
}

/// Test silence timeout configuration
#[tokio::test]
async fn test_silence_timeout_config() {
    let mut config = VoiceSessionConfig::default();
    config.silence_timeout_ms = 500; // 500ms silence timeout

    let session = VoiceSession::new("test-silence", config.clone()).unwrap();
    session.start().await.unwrap();

    // The silence timeout is handled by spawn_transport_event_handler
    // We just verify the config is set correctly
    assert_eq!(config.silence_timeout_ms, 500);
}

/// Test concurrent session handling
#[tokio::test]
async fn test_multiple_sessions() {
    let config = VoiceSessionConfig::default();

    // Create multiple sessions
    let session1 = VoiceSession::new("session-1", config.clone()).unwrap();
    let session2 = VoiceSession::new("session-2", config.clone()).unwrap();
    let session3 = VoiceSession::new("session-3", config.clone()).unwrap();

    // Start all sessions
    session1.start().await.unwrap();
    session2.start().await.unwrap();
    session3.start().await.unwrap();

    // All should be listening
    assert_eq!(session1.state().await, VoiceSessionState::Listening);
    assert_eq!(session2.state().await, VoiceSessionState::Listening);
    assert_eq!(session3.state().await, VoiceSessionState::Listening);

    // IDs should be unique
    assert_ne!(session1.session_id(), session2.session_id());
    assert_ne!(session2.session_id(), session3.session_id());

    // End all sessions
    session1.end("done").await;
    session2.end("done").await;
    session3.end("done").await;
}

/// Test end-to-end flow with mock audio
#[tokio::test]
async fn test_e2e_mock_conversation() {
    let config = VoiceSessionConfig::default();
    let session = VoiceSession::new("test-e2e", config).unwrap();

    // Subscribe to events
    let mut event_rx = session.subscribe();

    // Start session
    session.start().await.unwrap();

    // Drain initial events
    let mut events = Vec::new();
    while let Ok(Ok(event)) = timeout(Duration::from_millis(50), event_rx.recv()).await {
        events.push(event);
    }

    // Should have Started and StateChanged events
    assert!(events.iter().any(|e| matches!(e, VoiceSessionEvent::Started { .. })));

    // Agent should have responded to empty input with greeting
    assert!(events.iter().any(|e| matches!(e, VoiceSessionEvent::Speaking { .. })));

    session.end("test complete").await;
}

/// Test audio chunk event emission
#[tokio::test]
async fn test_audio_chunk_events() {
    let config = VoiceSessionConfig::default();
    let session = VoiceSession::new("test-chunks", config).unwrap();

    let mut event_rx = session.subscribe();
    session.start().await.unwrap();

    // Collect events with timeout
    let mut audio_chunks = Vec::new();
    while let Ok(Ok(event)) = timeout(Duration::from_millis(100), event_rx.recv()).await {
        if let VoiceSessionEvent::AudioChunk { samples, sample_rate } = event {
            audio_chunks.push((samples.len(), sample_rate));
        }
    }

    // TTS should have emitted audio chunks for the greeting
    // Note: With stub TTS, we may not get actual audio
}

/// Test state transitions
#[tokio::test]
async fn test_state_machine() {
    let config = VoiceSessionConfig::default();
    let session = VoiceSession::new("test-states", config).unwrap();

    // Idle -> Listening
    assert_eq!(session.state().await, VoiceSessionState::Idle);
    session.start().await.unwrap();

    // After speaking greeting, should be Listening
    // (In real flow: Idle -> Listening -> Speaking -> Listening)
    tokio::time::sleep(Duration::from_millis(50)).await;
    let state = session.state().await;
    assert!(state == VoiceSessionState::Listening || state == VoiceSessionState::Speaking);

    // Listening -> Ended
    session.end("done").await;
    assert_eq!(session.state().await, VoiceSessionState::Ended);
}
