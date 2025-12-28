import React from 'react'

interface MicButtonProps {
  isRecording: boolean
  onStart: () => void
  onStop: () => void
  disabled?: boolean
}

export function MicButton({ isRecording, onStart, onStop, disabled }: MicButtonProps) {
  const handleClick = () => {
    if (isRecording) {
      onStop()
    } else {
      onStart()
    }
  }

  return (
    <button
      style={{
        ...styles.button,
        ...(isRecording ? styles.recording : {}),
        ...(disabled ? styles.disabled : {}),
      }}
      onClick={handleClick}
      disabled={disabled}
    >
      <div style={styles.iconContainer}>
        {isRecording ? (
          // Stop icon
          <svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor">
            <rect x="6" y="6" width="12" height="12" rx="2" />
          </svg>
        ) : (
          // Mic icon
          <svg width="32" height="32" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
            <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
          </svg>
        )}
      </div>
      <span style={styles.label}>
        {isRecording ? 'Stop' : 'Hold to Speak'}
      </span>
      {isRecording && <div style={styles.pulse} />}
    </button>
  )
}

const styles: { [key: string]: React.CSSProperties } = {
  button: {
    position: 'relative',
    width: '120px',
    height: '120px',
    borderRadius: '50%',
    background: 'linear-gradient(145deg, #ED1C24, #c41920)',
    border: 'none',
    color: '#fff',
    cursor: 'pointer',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '0.5rem',
    transition: 'all 0.2s',
    boxShadow: '0 4px 20px rgba(237,28,36,0.4)',
  },
  recording: {
    background: 'linear-gradient(145deg, #ff4444, #cc0000)',
    transform: 'scale(1.05)',
    boxShadow: '0 4px 30px rgba(255,0,0,0.5)',
  },
  disabled: {
    opacity: 0.5,
    cursor: 'not-allowed',
  },
  iconContainer: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  label: {
    fontSize: '0.75rem',
    fontWeight: 'bold',
  },
  pulse: {
    position: 'absolute',
    inset: '-8px',
    borderRadius: '50%',
    border: '3px solid rgba(255,0,0,0.5)',
    animation: 'pulse 1.5s ease-in-out infinite',
  },
}

// Add keyframes via style tag
const styleSheet = document.createElement('style')
styleSheet.textContent = `
  @keyframes pulse {
    0% { transform: scale(1); opacity: 1; }
    100% { transform: scale(1.3); opacity: 0; }
  }
`
document.head.appendChild(styleSheet)
