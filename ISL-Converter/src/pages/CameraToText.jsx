import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import ThemeToggle from '../components/ThemeToggle'
import BackButton from '../components/BackButton'

const RAW_API_URL = import.meta.env.VITE_MODEL_API_URL || 'http://127.0.0.1:8000'
const API_URL = RAW_API_URL.replace(/\/+$/, '')
const GROQ_API_KEY = import.meta.env.VITE_GROQ_API_KEY
const IS_PROD = import.meta.env.PROD
const IS_LOCAL_API = /localhost|127\.0\.0\.1/.test(API_URL)
const CAPTURE_SIZE = 224

function CameraToText() {
  const navigate = useNavigate()
  const videoRef = useRef(null)
  const intervalRef = useRef(null)

  const [started, setStarted] = useState(false)
  const [autoMode, setAutoMode] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [letters, setLetters] = useState([])
  const [lastDetected, setLastDetected] = useState(null)
  const [confidence, setConfidence] = useState(null)
  const [suggestions, setSuggestions] = useState([])
  const [modelType, setModelType] = useState('onehand')
  const [mirror, setMirror] = useState(false)
  const [cropScale, setCropScale] = useState(0.6)

  const word = letters.join('')

  async function getSuggestions(currentWord) {
    if (!currentWord || currentWord.length < 1) { setSuggestions([]); return }
    if (!GROQ_API_KEY) { setSuggestions([]); return }
    try {
      const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${GROQ_API_KEY}` },
        body: JSON.stringify({
          model: 'llama-3.1-8b-instant',
          max_tokens: 50,
          temperature: 0.3,
          messages: [
            { role: 'system', content: `You are a word suggestion engine. Given letters typed so far, suggest exactly 4 common English words that start with those letters. Reply ONLY with 4 words separated by commas. No explanation. No punctuation except commas. Example: hello,help,helmet,held` },
            { role: 'user', content: currentWord.toLowerCase() }
          ]
        })
      })
      const data = await response.json()
      const text = data.choices?.[0]?.message?.content || ''
      const words = text.split(',').map(w => w.trim()).filter(w => w.length > 0).slice(0, 4)
      setSuggestions(words)
    } catch { setSuggestions([]) }
  }

  async function startCamera() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user' }
      })
      videoRef.current.srcObject = stream
      setStarted(true)
      setError('')
    } catch { setError('Could not access camera. Please allow camera permission!') }
  }

  function stopCamera() {
    const stream = videoRef.current?.srcObject
    if (stream) stream.getTracks().forEach(t => t.stop())
    setStarted(false)
    stopAuto()
    setLastDetected(null)
  }

  function startAuto() {
    setAutoMode(true)
    intervalRef.current = setInterval(() => { predictFrame() }, 2000)
  }

  function stopAuto() {
    setAutoMode(false)
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null }
  }

  async function predictFrame() {
    if (!videoRef.current || loading) return
    setLoading(true)
    try {
      if (IS_PROD && IS_LOCAL_API) {
        throw new Error('Missing VITE_MODEL_API_URL for production')
      }
      const video = videoRef.current
      const vw = video.videoWidth
      const vh = video.videoHeight
      if (!vw || !vh) throw new Error('Camera not ready')

      const minSide = Math.min(vw, vh)
      const cropSize = Math.max(1, Math.floor(minSide * cropScale))
      const sx = Math.floor((vw - cropSize) / 2)
      const sy = Math.floor((vh - cropSize) / 2)

      const canvas = document.createElement('canvas')
      canvas.width = CAPTURE_SIZE
      canvas.height = CAPTURE_SIZE
      const ctx = canvas.getContext('2d')
      if (!ctx) throw new Error('Canvas error')

      if (mirror) {
        ctx.translate(CAPTURE_SIZE, 0)
        ctx.scale(-1, 1)
      }
      ctx.drawImage(video, sx, sy, cropSize, cropSize, 0, 0, CAPTURE_SIZE, CAPTURE_SIZE)

      const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg'))
      if (!blob) throw new Error('Failed to capture frame')

      const formData = new FormData()
      formData.append('file', blob, 'frame.jpg')
      formData.append('model_type', modelType)

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData
      })
      const data = await response.json()
      if (!response.ok) {
        setError(data.detail || 'API Error')
      } else {
        const detected = String(data.prediction || '').toUpperCase()
        if (!detected) throw new Error('Empty prediction')
        setLastDetected(detected)
        setConfidence(Math.round((data.confidence || 0) * 100))
        setLetters(prev => {
          const newLetters = [...prev, detected]
          if (modelType !== 'number') getSuggestions(newLetters.join(''))
          else setSuggestions([])
          return newLetters
        })
        setError('')
      }
    } catch (err) {
      setError(err?.message || 'Cannot connect to API. Make sure backend is running!')
    }
    setLoading(false)
  }

  function removeLastLetter() {
    setLetters(prev => {
      const newLetters = prev.slice(0, -1)
      if (modelType !== 'number') getSuggestions(newLetters.join(''))
      else setSuggestions([])
      return newLetters
    })
  }

  function clearAll() {
    setLetters([])
    setLastDetected(null)
    setConfidence(null)
    setError('')
    setSuggestions([])
  }

  useEffect(() => { return () => { stopCamera() } }, [])
  useEffect(() => {
    setLetters([])
    setLastDetected(null)
    setConfidence(null)
    setSuggestions([])
    setError('')
  }, [modelType])

  return (
    <div className="cam-fullpage">
      <BackButton />
      <ThemeToggle />

      {/* FULL SCREEN CAMERA */}
      <div className="cam-fullscreen">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          className="cam-fullvideo"
          style={{
            display: started ? 'block' : 'none',
            transform: mirror ? 'scaleX(-1)' : 'none'
          }}
        />

        {started && (
          <div
            className="cam-guide-box"
            style={{ width: `${cropScale * 100}%`, height: `${cropScale * 100}%` }}
          />
        )}

        {!started && (
          <div className="cam-full-placeholder">
            <motion.div
              animate={{ scale: [1, 1.1, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
              style={{ fontSize: 64 }}
            >
              📷
            </motion.div>
            <p style={{ color: '#6A2E3B', fontSize: 16, marginTop: 12 }}>Camera not started</p>
            <p style={{ color: '#B06A73', fontSize: 13, marginTop: 4 }}>Click Start below</p>
          </div>
        )}

        {started && autoMode && (
          <div className="cam-live-badge">
            <span className="cam-live-dot" />
            LIVE
          </div>
        )}

        <AnimatePresence>
          {started && lastDetected && (
            <motion.div
              className="cam-big-overlay"
              key={lastDetected + Date.now()}
              initial={{ scale: 0.4, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ type: 'spring', stiffness: 300, damping: 20 }}
            >
              {lastDetected}
              {confidence && <span className="cam-overlay-conf">{confidence}%</span>}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* BOTTOM DETECTION CARD */}
      <motion.div
        className="cam-bottom-card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        {/* Model Type Selection */}
        <div className="dict-filters" style={{ marginBottom: 10 }}>
          {[
            { key: 'onehand', label: 'One Hand' },
            { key: 'twohand', label: 'Two Hands' },
            { key: 'number', label: 'Numbers' },
          ].map(m => (
            <button
              key={m.key}
              className={`dict-filter-btn ${modelType === m.key ? 'active' : ''}`}
              onClick={() => setModelType(m.key)}
            >
              {m.label}
            </button>
          ))}
        </div>
        <div className="cam-control-row">
          <label className="cam-toggle">
            <input
              type="checkbox"
              checked={mirror}
              onChange={e => setMirror(e.target.checked)}
            />
            <span>Mirror</span>
          </label>
          <div className="cam-slider">
            <span className="cam-slider-label">Crop</span>
            <input
              type="range"
              min="0.4"
              max="0.9"
              step="0.05"
              value={cropScale}
              onChange={e => setCropScale(parseFloat(e.target.value))}
            />
            <span className="cam-slider-value">{Math.round(cropScale * 100)}%</span>
          </div>
        </div>

        {/* ROW 1 — buttons + word box — never scrolls */}
        <div className="cam-bottom-top">
          <div className="cam-btn-row">
            {!started ? (
              <motion.button className="cam-action-btn primary" onClick={startCamera} whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.96 }}>
                📷 Start
              </motion.button>
            ) : (
              <>
                <motion.button className="cam-action-btn primary" onClick={predictFrame} disabled={loading} whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.96 }}>
                  {loading ? '...' : '📸 Capture'}
                </motion.button>
                <motion.button className={`cam-action-btn ${autoMode ? 'stop' : 'auto'}`} onClick={autoMode ? stopAuto : startAuto} whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.96 }}>
                  {autoMode ? '⏹ Stop Auto' : '▶ Auto'}
                </motion.button>
                <motion.button className="cam-action-btn secondary" onClick={stopCamera} whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.96 }}>
                  ⏹ Stop
                </motion.button>
              </>
            )}
          </div>

          {word ? (
            <div className="cam-word-box">
              <p className="cam-word-label">Detected Word</p>
              <motion.p className="cam-word-display" key={word} initial={{ scale: 0.95 }} animate={{ scale: 1 }}>
                {word}
              </motion.p>
            </div>
          ) : (
            <div className="cam-word-empty"><p>Signs will appear here</p></div>
          )}
        </div>

        {/* ROW 2 — scrollable area for suggestions + letters + edit buttons */}
        <div className="cam-scroll-area">

          {/* suggestions */}
          <AnimatePresence>
            {suggestions.length > 0 && (
              <motion.div
                className="cam-suggestions"
                initial={{ opacity: 0, y: 5 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
              >
                <span className="cam-suggest-label">💡</span>
                {suggestions.map((w, i) => (
                  <motion.button
                    key={i}
                    className="cam-suggest-btn"
                    onClick={() => { setLetters(w.toUpperCase().split('')); getSuggestions(w) }}
                    whileHover={{ scale: 1.05, y: -2 }}
                    whileTap={{ scale: 0.95 }}
                    initial={{ opacity: 0, x: -5 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: i * 0.05 }}
                  >
                    {w}
                  </motion.button>
                ))}
              </motion.div>
            )}
          </AnimatePresence>

          {letters.length > 0 && (
            <div className="cam-bottom-bottom">
              <div className="cam-letters-wrap">
                <AnimatePresence>
                  {letters.map((letter, i) => (
                    <motion.div
                      key={i}
                      className="cam-letter-pill"
                      initial={{ opacity: 0, scale: 0.5 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.5 }}
                      transition={{ type: 'spring', stiffness: 300 }}
                    >
                      {letter}
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
            </div>
          )}

              <div className="cam-edit-row">
                <motion.button className="cam-edit-btn" onClick={removeLastLetter} whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.9 }}>
                  ← Undo
                </motion.button>
                <motion.button className="cam-edit-btn danger" onClick={clearAll} whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.9 }}>
                  ✕ Clear
                </motion.button>
              </div>
            

          {autoMode && (
            <p className="cam-auto-info">
              <span className="cam-live-dot" style={{ display: 'inline-block', marginRight: 6 }} />
              Auto detecting every 2 seconds...
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
  )
}

export default CameraToText
