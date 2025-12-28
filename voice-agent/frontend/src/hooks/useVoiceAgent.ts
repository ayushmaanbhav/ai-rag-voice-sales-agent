import { useState, useRef, useCallback, useEffect } from 'react'

interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp: number
}

interface UseVoiceAgentReturn {
  isConnected: boolean
  isRecording: boolean
  transcript: Message[]
  startConversation: (customerId: string, language: string) => Promise<void>
  startRecording: () => void
  stopRecording: () => void
  endConversation: () => void
}

// Note: Using MediaRecorder for audio capture, sending to backend Whisper for STT

export function useVoiceAgent(): UseVoiceAgentReturn {
  const [isConnected, setIsConnected] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [transcript, setTranscript] = useState<Message[]>([])

  const wsRef = useRef<WebSocket | null>(null)
  const sessionIdRef = useRef<string | null>(null)
  const languageRef = useRef<string>('hi')
  const requestStartTimeRef = useRef<number | null>(null)

  // Audio recording refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const recordingStartTimeRef = useRef<number | null>(null)

  // Play audio from base64
  const playAudio = useCallback((base64Audio: string) => {
    if (!base64Audio) {
      // Fallback to browser TTS
      return
    }

    try {
      const audioData = atob(base64Audio)
      const audioArray = new Uint8Array(audioData.length)
      for (let i = 0; i < audioData.length; i++) {
        audioArray[i] = audioData.charCodeAt(i)
      }
      const audioBlob = new Blob([audioArray], { type: 'audio/wav' })
      const audioUrl = URL.createObjectURL(audioBlob)
      const audio = new Audio(audioUrl)
      audio.play().catch(console.error)
    } catch (error) {
      console.error('Error playing audio:', error)
    }
  }, [])

  // Browser TTS fallback
  const speakText = useCallback((text: string, language: string) => {
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text)
      const ttsStartTime = Date.now()

      // Set language
      const langMap: { [key: string]: string } = {
        hi: 'hi-IN',
        ta: 'ta-IN',
        te: 'te-IN',
        kn: 'kn-IN',
        ml: 'ml-IN',
        en: 'en-IN',
      }
      utterance.lang = langMap[language] || 'en-IN'

      utterance.onstart = () => {
        console.log(`[METRICS] Browser TTS playback started`)
      }

      utterance.onend = () => {
        const ttsDuration = Date.now() - ttsStartTime
        console.log(`[METRICS] Browser TTS completed: ${ttsDuration}ms (${text.length} chars)`)
      }

      utterance.onerror = (event) => {
        console.error(`[ERROR] Browser TTS error: ${event.error}`)
      }

      window.speechSynthesis.speak(utterance)
    }
  }, [])

  // Start conversation
  const startConversation = useCallback(async (customerId: string, language: string) => {
    languageRef.current = language

    try {
      // Initialize conversation via REST API
      const response = await fetch('/api/conversations/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ customer_id: customerId, language }),
      })

      if (!response.ok) {
        throw new Error('Failed to start conversation')
      }

      const data = await response.json()
      sessionIdRef.current = data.session_id

      // Add greeting to transcript
      setTranscript([{
        role: 'assistant',
        content: data.greeting.text,
        timestamp: Date.now(),
      }])

      // Play greeting audio
      if (data.greeting.audio_base64) {
        playAudio(data.greeting.audio_base64)
      } else {
        speakText(data.greeting.text, language)
      }

      // Connect WebSocket
      const wsUrl = `ws://${window.location.host}/ws/conversation/${data.session_id}`
      const ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        console.log('WebSocket connected')
        setIsConnected(true)
      }

      ws.onmessage = (event) => {
        const message = JSON.parse(event.data)
        const responseTime = Date.now()

        if (message.type === 'transcript') {
          // User's transcribed speech (from server-side ASR)
          console.log('[METRICS] Received transcript from server')
          setTranscript(prev => [...prev, {
            role: 'user',
            content: message.text,
            timestamp: Date.now(),
          }])
        } else if (message.type === 'response') {
          // Calculate round-trip time
          const roundTripMs = requestStartTimeRef.current
            ? responseTime - requestStartTimeRef.current
            : 0

          // Log server-side metrics if provided
          const serverMetrics = message.metrics || {}
          console.log(`[METRICS] WebSocket round-trip: ${roundTripMs}ms`)
          if (serverMetrics.asr_ms !== undefined) {
            console.log(`[METRICS] Server ASR: ${serverMetrics.asr_ms}ms`)
          }
          if (serverMetrics.llm_ms !== undefined) {
            console.log(`[METRICS] Server LLM: ${serverMetrics.llm_ms}ms`)
          }
          if (serverMetrics.total_ms !== undefined) {
            console.log(`[METRICS] Server Total: ${serverMetrics.total_ms}ms`)
          }

          // Agent's response
          setTranscript(prev => [...prev, {
            role: 'assistant',
            content: message.text,
            timestamp: Date.now(),
          }])

          // Play response audio with TTS timing
          if (message.audio_base64) {
            playAudio(message.audio_base64)
            console.log(`[METRICS] Playing server TTS audio`)
          } else {
            speakText(message.text, languageRef.current)
            // Note: Browser TTS timing is logged via utterance.onend callback
          }
        } else if (message.type === 'error') {
          console.error('[ERROR] Server error:', message.message)
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
      }

      ws.onclose = () => {
        console.log('WebSocket closed')
        setIsConnected(false)
      }

      wsRef.current = ws

    } catch (error) {
      console.error('Error starting conversation:', error)
      // Use mock mode if API not available
      setIsConnected(true)
      setTranscript([{
        role: 'assistant',
        content: 'Namaste! Main Kotak Gold Loan se bol raha hoon. Kya aapke paas 2-3 minute hain?',
        timestamp: Date.now(),
      }])
      speakText('Namaste! Main Kotak Gold Loan se bol raha hoon. Kya aapke paas 2-3 minute hain?', language)
    }
  }, [playAudio, speakText])

  // Convert blob to base64
  const blobToBase64 = (blob: Blob): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onloadend = () => {
        const base64 = reader.result as string
        // Remove data URL prefix (e.g., "data:audio/webm;base64,")
        const base64Data = base64.split(',')[1]
        resolve(base64Data)
      }
      reader.onerror = reject
      reader.readAsDataURL(blob)
    })
  }

  // Start recording using MediaRecorder (sends audio to backend Whisper)
  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })

      // Use webm format for smaller file size
      const mimeType = MediaRecorder.isTypeSupported('audio/webm')
        ? 'audio/webm'
        : 'audio/mp4'

      const mediaRecorder = new MediaRecorder(stream, { mimeType })
      mediaRecorderRef.current = mediaRecorder
      audioChunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data)
        }
      }

      mediaRecorder.onstop = async () => {
        const recordingDuration = Date.now() - (recordingStartTimeRef.current || Date.now())
        console.log(`[METRICS] Recording stopped: ${recordingDuration}ms`)

        // Combine audio chunks
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType })
        console.log(`[METRICS] Audio blob size: ${audioBlob.size} bytes`)

        // Convert to base64 and send to server
        try {
          const audioBase64 = await blobToBase64(audioBlob)

          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            requestStartTimeRef.current = Date.now()
            wsRef.current.send(JSON.stringify({
              type: 'audio',
              data: audioBase64,
              language: languageRef.current,
              format: mimeType.includes('webm') ? 'webm' : 'mp4',
            }))
            console.log('[METRICS] Audio sent to server for Whisper STT')
          }
        } catch (error) {
          console.error('Error converting audio:', error)
        }

        // Stop all tracks
        stream.getTracks().forEach(track => track.stop())
      }

      // Start recording
      recordingStartTimeRef.current = Date.now()
      mediaRecorder.start()
      setIsRecording(true)
      console.log(`[METRICS] Recording started at ${new Date().toISOString()}`)

    } catch (error) {
      console.error('Error accessing microphone:', error)
      alert('Could not access microphone. Please check permissions.')
    }
  }, [])

  // Stop recording
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop()
      mediaRecorderRef.current = null
    }
    setIsRecording(false)
  }, [])

  // End conversation
  const endConversation = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({ type: 'end' }))
      wsRef.current.close()
      wsRef.current = null
    }
    setIsConnected(false)
    setTranscript([])
    sessionIdRef.current = null
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  return {
    isConnected,
    isRecording,
    transcript,
    startConversation,
    startRecording,
    stopRecording,
    endConversation,
  }
}
