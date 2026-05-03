import { useEffect, useEffectEvent, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ThemeToggle from '../components/ThemeToggle'
import BackButton from '../components/BackButton'
import { API_BASE, ENDPOINTS, fetchApi, getNetworkErrorMessage } from '../api'

const POLL_INTERVAL_MS = 800
const FRAME_IMAGE_QUALITY = 0.82

const MODES = [
  {
    value: 'alphabet',
    label: 'Alphabets',
    detail: 'A-Z signs',
    hint: 'Alphabet mode is active. The app switches automatically between one-hand and two-hand letter models.',
  },
  {
    value: 'numeric',
    label: 'Numbers',
    detail: '0-9 signs',
    hint: 'Number mode is active. Show one clear 0-9 hand sign to build digits.',
  },
]

function getCameraErrorMessage(err, fallbackMessage = 'Cannot connect to backend. Make sure backend1 is running.') {
  return getNetworkErrorMessage(err, fallbackMessage)
}

function getBrowserCameraErrorMessage(err) {
  if (err?.name === 'NotAllowedError') {
    return 'Camera access was blocked. Allow webcam permission in your browser and try again.'
  }
  if (err?.name === 'NotFoundError') {
    return 'No camera was found on this device.'
  }
  if (err?.name === 'NotReadableError') {
    return 'Your camera is already in use by another app.'
  }
  if (err?.name === 'SecurityError') {
    return 'Camera access is only available in a secure browser context.'
  }
  return err?.message || 'Could not start your webcam.'
}

function CameraToText() {
  const lastWordRef = useRef('')
  const videoRef = useRef(null)
  const captureCanvasRef = useRef(null)
  const mediaStreamRef = useRef(null)
  const isUploadingFrameRef = useRef(false)

  const [started, setStarted] = useState(false)
  const [loading, setLoading] = useState(false)
  const [mode, setMode] = useState('alphabet')
  const [word, setWord] = useState('')
  const [lastDetected, setLastDetected] = useState('')
  const [suggestions, setSuggestions] = useState([])
  const [error, setError] = useState('')
  const [captureHint, setCaptureHint] = useState(MODES[0].hint)

  const displayWord = word.toUpperCase()
  const activeMode = MODES.find(item => item.value === mode) || MODES[0]
  const MotionButton = motion.button
  const MotionDiv = motion.div
  const MotionP = motion.p

  function stopBrowserStream() {
    const activeStream = mediaStreamRef.current
    if (activeStream) {
      activeStream.getTracks().forEach(track => track.stop())
      mediaStreamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
  }

  async function captureFrameBlob() {
    const video = videoRef.current
    const canvas = captureCanvasRef.current

    if (!video || !canvas || video.readyState < 2) return null

    const width = video.videoWidth || 960
    const height = video.videoHeight || 720
    canvas.width = width
    canvas.height = height

    const context = canvas.getContext('2d')
    if (!context) return null

    context.drawImage(video, 0, 0, width, height)
    return new Promise(resolve => {
      canvas.toBlob(resolve, 'image/jpeg', FRAME_IMAGE_QUALITY)
    })
  }

  const uploadPredictionFrame = useEffectEvent(async (activeModeValue) => {
    const blob = await captureFrameBlob()
    if (!blob) return

    const formData = new FormData()
    formData.append('file', blob, 'frame.jpg')
    formData.append('mode', activeModeValue)

    const data = await fetchApi(ENDPOINTS.predictLiveFrame, {
      method: 'POST',
      body: formData,
    }, 'Prediction polling failed')

    applyPrediction(data.current_word)
  })

  function applyPrediction(nextPrediction) {
    const nextWord = String(nextPrediction || '')
    const previousWord = lastWordRef.current

    if (nextWord !== previousWord) {
      if (nextWord.length > previousWord.length && nextWord.startsWith(previousWord)) {
        setLastDetected(nextWord.slice(previousWord.length).slice(-1).toUpperCase())
      } else {
        setLastDetected(nextWord.slice(-1).toUpperCase())
      }
      lastWordRef.current = nextWord
      setWord(nextWord)
    }
  }

  async function setBackendMode(nextMode) {
    await fetchApi(ENDPOINTS.setMode(nextMode), {}, 'Could not switch camera mode')
  }

  async function changeMode(nextMode) {
    const nextModeConfig = MODES.find(item => item.value === nextMode) || MODES[0]
    setMode(nextMode)
    setCaptureHint(nextModeConfig.hint)
    setSuggestions([])
    setError('')

    if (!started) return

    try {
      await setBackendMode(nextMode)
    } catch (err) {
      setError(getCameraErrorMessage(err))
    }
  }

  async function startCamera() {
    if (started) return
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('Camera access is not supported in this browser.')
      return
    }

    setLoading(true)
    setError('')
    try {
      await setBackendMode(mode)
      await fetchApi(ENDPOINTS.clearLastSign, { method: 'DELETE' }, 'Could not clear previous prediction').catch(() => {})
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: 'user',
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      })
      mediaStreamRef.current = mediaStream
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream
        await videoRef.current.play().catch(() => {})
      }
      lastWordRef.current = ''
      setWord('')
      setLastDetected('')
      setSuggestions([])
      setStarted(true)
      setCaptureHint(`${activeMode.hint} Hold one sign steady until it appears.`)
    } catch (err) {
      stopBrowserStream()
      if (['NotAllowedError', 'NotFoundError', 'NotReadableError', 'SecurityError'].includes(err?.name)) {
        setError(getBrowserCameraErrorMessage(err))
      } else {
        setError(getCameraErrorMessage(err))
      }
    } finally {
      setLoading(false)
    }
  }

  async function stopCamera() {
    setStarted(false)
    stopBrowserStream()
    setLastDetected('')
    setCaptureHint('Camera stopped. Choose a mode and start again when ready.')
    await fetch(ENDPOINTS.stopCamera, { method: 'POST' }).catch(() => {})
  }

  async function clearAll() {
    setError('')
    try {
      await fetchApi(ENDPOINTS.clearLastSign, { method: 'DELETE' }, 'Could not clear backend text')
    } catch (err) {
      setError(`${getCameraErrorMessage(err)} Frontend text was cleared locally.`)
    }
    lastWordRef.current = ''
    setWord('')
    setLastDetected('')
    setSuggestions([])
  }

  async function removeLastLetter() {
    if (!word) return

    setError('')
    try {
      const data = await fetchApi(ENDPOINTS.lastSignLast, { method: 'DELETE' }, 'Undo failed')
      lastWordRef.current = String(data.prediction || '')
      setWord(String(data.prediction || ''))
      setLastDetected('')
    } catch (err) {
      setError(err?.message || 'Could not undo the last sign.')
    }
  }

  async function acceptSuggestion(suggestion) {
    const nextWord = suggestion.toLowerCase()
    setError('')

    try {
      const data = await fetchApi(ENDPOINTS.lastSign, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: nextWord }),
      }, 'Could not apply suggestion')
      lastWordRef.current = String(data.prediction || nextWord)
      setWord(String(data.prediction || nextWord))
      setLastDetected('')
    } catch (err) {
      setError(err?.message || 'Could not apply suggestion.')
    }
  }

  useEffect(() => {
    if (!started) return undefined

    let cancelled = false

    async function pollPrediction() {
      if (isUploadingFrameRef.current) return

      isUploadingFrameRef.current = true
      try {
        await uploadPredictionFrame(mode)
        if (!cancelled) {
          setError('')
        }
      } catch (err) {
        if (!cancelled) {
          setError(getCameraErrorMessage(err, `Cannot read camera prediction from ${API_BASE}.`))
        }
      } finally {
        isUploadingFrameRef.current = false
      }
    }

    pollPrediction()
    const intervalId = window.setInterval(pollPrediction, POLL_INTERVAL_MS)

    return () => {
      cancelled = true
      isUploadingFrameRef.current = false
      window.clearInterval(intervalId)
    }
  }, [started, mode])

  useEffect(() => {
    if (mode !== 'alphabet' || !word) {
      setSuggestions([])
      return undefined
    }

    let cancelled = false

    async function loadSuggestions() {
      try {
        const data = await fetchApi(ENDPOINTS.suggestions(word), {}, 'Could not load suggestions')
        if (!cancelled) {
          setSuggestions(Array.isArray(data.suggestions) ? data.suggestions.slice(0, 4) : [])
        }
      } catch {
        if (!cancelled) setSuggestions([])
      }
    }

    loadSuggestions()

    return () => {
      cancelled = true
    }
  }, [mode, word])

  useEffect(() => {
    return () => {
      stopBrowserStream()
      fetch(ENDPOINTS.stopCamera, { method: 'POST' }).catch(() => {})
    }
  }, [])

  return (
    <div className="cam-fullpage">
      <BackButton />
      <ThemeToggle />

      <div className="cam-layout">
        <div className="cam-left-panel">
          <div className="cam-mode-row" aria-label="Camera detection mode">
            {MODES.map(item => (
              <MotionButton
                key={item.value}
                className={`cam-mode-btn ${mode === item.value ? 'active' : ''}`}
                onClick={() => changeMode(item.value)}
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                type="button"
              >
                <span>{item.label}</span>
                <small>{item.detail}</small>
              </MotionButton>
            ))}
          </div>

          <div className="cam-fullscreen">
            {started ? (
              <video
                ref={videoRef}
                autoPlay
                muted
                playsInline
                className="cam-fullvideo"
                style={{ transform: 'scaleX(-1)' }}
              />
            ) : (
              <div className="cam-full-placeholder">
                <div className="cam-placeholder-icon">CAM</div>
                <p style={{ color: '#6A2E3B', fontSize: 16, marginTop: 12 }}>Camera not started</p>
                <p style={{ color: '#B06A73', fontSize: 13, marginTop: 4 }}>
                  Pick Alphabets or Numbers, then click Start
                </p>
              </div>
            )}
            <canvas ref={captureCanvasRef} style={{ display: 'none' }} aria-hidden="true" />

            {started && (
              <div className="cam-live-badge">
                <span className="cam-live-dot" />
                LIVE {activeMode.detail}
              </div>
            )}

            <AnimatePresence>
              {started && lastDetected && (
                <MotionDiv
                  className="cam-big-overlay"
                  key={`${lastDetected}-${word.length}`}
                  initial={{ scale: 0.4, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ type: 'spring', stiffness: 300, damping: 20 }}
                >
                  {lastDetected}
                </MotionDiv>
              )}
            </AnimatePresence>
          </div>

          <div className="cam-button-row">
            {!started ? (
              <MotionButton
                className="cam-action-btn primary"
                onClick={startCamera}
                disabled={loading}
                whileHover={{ scale: 1.04 }}
                whileTap={{ scale: 0.96 }}
                type="button"
              >
                {loading ? 'Starting...' : 'Start Camera'}
              </MotionButton>
            ) : (
              <MotionButton
                className="cam-action-btn stop"
                onClick={stopCamera}
                whileHover={{ scale: 1.04 }}
                whileTap={{ scale: 0.96 }}
                type="button"
              >
                Stop Camera
              </MotionButton>
            )}

            <MotionButton
              className="cam-action-btn secondary"
              onClick={removeLastLetter}
              disabled={!word}
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.96 }}
              type="button"
            >
              Undo
            </MotionButton>

            <MotionButton
              className="cam-action-btn secondary"
              onClick={clearAll}
              disabled={!word && !lastDetected}
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.96 }}
              type="button"
            >
              Clear
            </MotionButton>
          </div>
        </div>

        <MotionDiv
          className="cam-bottom-card"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="cam-bottom-top">
            {displayWord ? (
              <div className="cam-word-box">
                <p className="cam-word-label">Detected Text</p>
                <MotionP className="cam-word-display" key={displayWord} initial={{ scale: 0.95 }} animate={{ scale: 1 }}>
                  {displayWord}
                </MotionP>
              </div>
            ) : (
              <div className="cam-word-empty">
                <p>Detected signs will appear here</p>
              </div>
            )}
          </div>

          <div className="cam-scroll-area">
            <div className="cam-status-card">
              <span>Mode: {activeMode.label}</span>
              <p>{captureHint}</p>
            </div>

            <AnimatePresence>
              {suggestions.length > 0 && (
                <MotionDiv
                  className="cam-suggestions"
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                >
                  <span className="cam-suggest-label">Suggestions</span>
                  {suggestions.map((suggestion, index) => (
                    <MotionButton
                      key={`${suggestion}-${index}`}
                      className="cam-suggest-btn"
                      onClick={() => acceptSuggestion(suggestion)}
                      whileHover={{ scale: 1.05, y: -2 }}
                      whileTap={{ scale: 0.95 }}
                      initial={{ opacity: 0, x: -5 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.05 }}
                      type="button"
                    >
                      {suggestion}
                    </MotionButton>
                  ))}
                </MotionDiv>
              )}
            </AnimatePresence>

            <div className="cam-letters-wrap">
              {displayWord.split('').map((letter, index) => (
                <span className="cam-letter-pill" key={`${letter}-${index}`}>
                  {letter}
                </span>
              ))}
            </div>

            {started && (
              <p className="cam-auto-info">
                <span className="cam-live-dot" style={{ display: 'inline-block', marginRight: 6 }} />
                Live sign detection is running from your webcam.
              </p>
            )}

            {error && (
              <MotionDiv className="error-box" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                {error}
              </MotionDiv>
            )}
          </div>
        </MotionDiv>
      </div>
    </div>
  )
}

export default CameraToText
