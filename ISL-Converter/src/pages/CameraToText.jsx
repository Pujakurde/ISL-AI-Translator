import { useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ThemeToggle from '../components/ThemeToggle'
import BackButton from '../components/BackButton'
import { API_BASE, ENDPOINTS, fetchApi, getNetworkErrorMessage } from '../api'

const POLL_INTERVAL_MS = 800

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

function CameraToText() {
  const lastWordRef = useRef('')

  const [started, setStarted] = useState(false)
  const [loading, setLoading] = useState(false)
  const [mode, setMode] = useState('alphabet')
  const [streamSrc, setStreamSrc] = useState('')
  const [word, setWord] = useState('')
  const [lastDetected, setLastDetected] = useState('')
  const [suggestions, setSuggestions] = useState([])
  const [error, setError] = useState('')
  const [captureHint, setCaptureHint] = useState(MODES[0].hint)

  const displayWord = word.toUpperCase()
  const activeMode = MODES.find(item => item.value === mode) || MODES[0]

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

    setLoading(true)
    setError('')
    try {
      await setBackendMode(mode)
      await fetchApi(ENDPOINTS.clearLastSign, { method: 'DELETE' }, 'Could not clear previous prediction').catch(() => {})
      lastWordRef.current = ''
      setWord('')
      setLastDetected('')
      setSuggestions([])
      setStreamSrc(`${ENDPOINTS.videoFeed}?t=${Date.now()}`)
      setStarted(true)
      setCaptureHint(`${activeMode.hint} Hold one sign steady until it appears.`)
    } catch (err) {
      setError(getCameraErrorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  async function stopCamera() {
    setStarted(false)
    setStreamSrc('')
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
      try {
        const data = await fetchApi(ENDPOINTS.lastSign, {}, 'Prediction polling failed')
        if (!cancelled) {
          applyPrediction(data.prediction)
          setError('')
        }
      } catch (err) {
        if (!cancelled) {
          setError(getCameraErrorMessage(err, `Cannot read camera prediction from ${API_BASE}.`))
        }
      }
    }

    pollPrediction()
    const intervalId = window.setInterval(pollPrediction, POLL_INTERVAL_MS)

    return () => {
      cancelled = true
      window.clearInterval(intervalId)
    }
  }, [started])

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
              <motion.button
                key={item.value}
                className={`cam-mode-btn ${mode === item.value ? 'active' : ''}`}
                onClick={() => changeMode(item.value)}
                whileHover={{ scale: 1.03 }}
                whileTap={{ scale: 0.97 }}
                type="button"
              >
                <span>{item.label}</span>
                <small>{item.detail}</small>
              </motion.button>
            ))}
          </div>

          <div className="cam-fullscreen">
            {started && streamSrc ? (
              <img
                src={streamSrc}
                alt="Live ISL camera detection"
                className="cam-fullvideo"
                onError={() => {
                  setError('Camera stream failed. Make sure backend1 can access your webcam.')
                  setStarted(false)
                  setStreamSrc('')
                }}
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

            {started && (
              <div className="cam-live-badge">
                <span className="cam-live-dot" />
                LIVE {activeMode.detail}
              </div>
            )}

            <AnimatePresence>
              {started && lastDetected && (
                <motion.div
                  className="cam-big-overlay"
                  key={`${lastDetected}-${word.length}`}
                  initial={{ scale: 0.4, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ type: 'spring', stiffness: 300, damping: 20 }}
                >
                  {lastDetected}
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          <div className="cam-button-row">
            {!started ? (
              <motion.button
                className="cam-action-btn primary"
                onClick={startCamera}
                disabled={loading}
                whileHover={{ scale: 1.04 }}
                whileTap={{ scale: 0.96 }}
                type="button"
              >
                {loading ? 'Starting...' : 'Start Camera'}
              </motion.button>
            ) : (
              <motion.button
                className="cam-action-btn stop"
                onClick={stopCamera}
                whileHover={{ scale: 1.04 }}
                whileTap={{ scale: 0.96 }}
                type="button"
              >
                Stop Camera
              </motion.button>
            )}

            <motion.button
              className="cam-action-btn secondary"
              onClick={removeLastLetter}
              disabled={!word}
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.96 }}
              type="button"
            >
              Undo
            </motion.button>

            <motion.button
              className="cam-action-btn secondary"
              onClick={clearAll}
              disabled={!word && !lastDetected}
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.96 }}
              type="button"
            >
              Clear
            </motion.button>
          </div>
        </div>

        <motion.div
          className="cam-bottom-card"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="cam-bottom-top">
            {displayWord ? (
              <div className="cam-word-box">
                <p className="cam-word-label">Detected Text</p>
                <motion.p className="cam-word-display" key={displayWord} initial={{ scale: 0.95 }} animate={{ scale: 1 }}>
                  {displayWord}
                </motion.p>
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
                <motion.div
                  className="cam-suggestions"
                  initial={{ opacity: 0, y: 5 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                >
                  <span className="cam-suggest-label">Suggestions</span>
                  {suggestions.map((suggestion, index) => (
                    <motion.button
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
                    </motion.button>
                  ))}
                </motion.div>
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
                Backend camera detection is running automatically.
              </p>
            )}

            {error && (
              <motion.div className="error-box" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                {error}
              </motion.div>
            )}
          </div>
        </motion.div>
      </div>
    </div>
  )
}

export default CameraToText
