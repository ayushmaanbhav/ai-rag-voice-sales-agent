import React, { useEffect, useRef } from 'react'

interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp?: number
}

interface TranscriptProps {
  messages: Message[]
}

export function Transcript({ messages }: TranscriptProps) {
  const scrollRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages])

  return (
    <div style={styles.container} ref={scrollRef}>
      {messages.length === 0 ? (
        <div style={styles.empty}>
          <p>Conversation will appear here...</p>
          <p style={styles.hint}>Press the microphone button and speak in your selected language</p>
        </div>
      ) : (
        messages.map((message, index) => (
          <div
            key={index}
            style={{
              ...styles.message,
              ...(message.role === 'user' ? styles.userMessage : styles.assistantMessage),
            }}
          >
            <div style={styles.avatar}>
              {message.role === 'user' ? '👤' : '🤖'}
            </div>
            <div style={styles.content}>
              <div style={styles.role}>
                {message.role === 'user' ? 'You' : 'Kotak Agent'}
              </div>
              <div style={styles.text}>{message.content}</div>
            </div>
          </div>
        ))
      )}
    </div>
  )
}

const styles: { [key: string]: React.CSSProperties } = {
  container: {
    flex: 1,
    background: 'rgba(255,255,255,0.03)',
    borderRadius: '12px',
    padding: '1rem',
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem',
  },
  empty: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    opacity: 0.5,
    textAlign: 'center',
  },
  hint: {
    fontSize: '0.85rem',
    marginTop: '0.5rem',
  },
  message: {
    display: 'flex',
    gap: '0.75rem',
    maxWidth: '80%',
    animation: 'fadeIn 0.3s ease-out',
  },
  userMessage: {
    alignSelf: 'flex-end',
    flexDirection: 'row-reverse',
  },
  assistantMessage: {
    alignSelf: 'flex-start',
  },
  avatar: {
    width: '36px',
    height: '36px',
    borderRadius: '50%',
    background: 'rgba(255,255,255,0.1)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '1.2rem',
    flexShrink: 0,
  },
  content: {
    background: 'rgba(255,255,255,0.08)',
    borderRadius: '12px',
    padding: '0.75rem 1rem',
  },
  role: {
    fontSize: '0.75rem',
    opacity: 0.6,
    marginBottom: '0.25rem',
  },
  text: {
    fontSize: '0.95rem',
    lineHeight: 1.5,
  },
}

// Add fadeIn animation
const styleSheet = document.createElement('style')
styleSheet.textContent = `
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }
`
document.head.appendChild(styleSheet)
