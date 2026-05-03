import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import ThemeToggle from '../components/ThemeToggle'
import BackButton from '../components/BackButton'
import { ENDPOINTS, fetchApi, getNetworkErrorMessage } from '../api'

const NUMBERS = '0123456789'.split('')
const ALPHABETS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('')
const ONEHAND_SET = new Set(['C', 'I', 'L', 'O', 'U', 'V'])
const ALL_SIGNS = [...NUMBERS, ...ALPHABETS, 'space']

function ImageToText() {
  const navigate = useNavigate()
  const [selected, setSelected] = useState([])
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState([])
  const [error, setError] = useState('')
  const [filter, setFilter] = useState('all')

  const filtered = ALL_SIGNS.filter(sign => {
    if (filter === 'alpha') return ALPHABETS.includes(sign)
    if (filter === 'num') return NUMBERS.includes(sign)
    if (filter === 'space') return sign === 'space'
    return true
  })

  function getModelType(sign) {
    if (NUMBERS.includes(sign)) return 'number'
    if (ONEHAND_SET.has(sign)) return 'onehand'
    if (ALPHABETS.includes(sign)) return 'twohand'
    return null
  }

  async function predictSign(sign, index) {
    setLoading(true)
    setError('')

    try {
      const imageUrl = `/Signs/${sign}.jpg`
      const response = await fetch(imageUrl)
      if (!response.ok) throw new Error(`Could not load sample sign image for ${sign}.`)
      const blob = await response.blob()

      const formData = new FormData()
      formData.append("file", blob, `${sign}.jpg`)
      const modelType = getModelType(sign)
      if (!modelType) throw new Error('Invalid sign for model')
      formData.append("model_type", modelType)

      const data = await fetchApi(ENDPOINTS.predict, {
        method: "POST",
        body: formData,
      }, 'API Error')

      setResults(prev => {
        const updated = [...prev]
        updated[index] = {
          letter: data.prediction,
          confidence: Math.round(data.confidence * 100)
        }
        return updated
      })
    } catch (err) {
      if (err?.message?.startsWith('Could not load sample sign image')) {
        setError(err.message)
      } else {
        setError(getNetworkErrorMessage(err, 'Cannot connect to API. Make sure backend is running!'))
      }
    }

    setLoading(false)
  }

  function addSign(sign) {
  const newIndex = selected.length
  setSelected(prev => [...prev, sign])
  setResults(prev => {
    if (sign === 'space') {
      return [...prev, { letter: ' ', confidence: 100 }] // directly add space
    }
    return [...prev, null]
  })

  // Only call API if not space
  if (sign !== 'space') predictSign(sign, newIndex)
}

  function removeSign(index) {
    setSelected(prev => prev.filter((_, i) => i !== index))
    setResults(prev => prev.filter((_, i) => i !== index))
  }

  function clearAll() {
    setSelected([])
    setResults([])
    setError('')
  }

  const detectedText = results
  .map(r => r?.letter ?? '?') // `??` keeps actual space
  .join('')

  return (
    <div className="dict-page">
      <BackButton />
      <ThemeToggle />
      <motion.div
        className="dict-screen"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -20 }}
        transition={{ duration: 0.3 }}
      >

        {/* HEADER */}
        <div className="dict-header">
          
          <div className="dict-title-wrap">
            <h1 className="dict-title">🖼️ Image to Text</h1>
            <p className="dict-sub">Click signs — text builds automatically!</p>
          </div>
        </div>

        {/* TWO COLUMN */}
        <div className="img2text-grid">

          {/* LEFT — sign gallery */}
          <div className="img2text-left">
            <p className="img2text-section-title">📖 ISL Signs</p>
            <p className="img2text-section-sub">Click any sign to add it</p>

            {/* Filter buttons */}
            <div className="dict-filters">
              {[
                { key: 'all', label: 'All' },
                { key: 'alpha', label: 'A–Z' },
                { key: 'num', label: '0–9' },
                { key: 'space', label: 'Space' },
              ].map(f => (
                <button
                  key={f.key}
                  className={`dict-filter-btn ${filter === f.key ? 'active' : ''}`}
                  onClick={() => setFilter(f.key)}
                >
                  {f.label}
                </button>
              ))}
            </div>

            <div className="img2text-gallery">
              <AnimatePresence>
                {filtered.map((sign, i) => (
                  <motion.div
                    key={sign}
                    className="img2text-sign-card"
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.8 }}
                    transition={{ delay: i * 0.01 }}
                    whileHover={{ scale: 1.12, y: -4 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={() => addSign(sign)}
                  >
                    <img
                      src={`/Signs/${sign}.jpg`}
                      alt={sign}
                      className="img2text-sign-img"
                      onError={e => {
                        e.target.style.display = 'none'
                        e.target.nextSibling.style.display = 'flex'
                      }}
                    />
                    <div className="img2text-sign-fallback" style={{ display: 'none' }}>
                      {sign === 'space' ? '␣' : sign}
                    </div>
                    <span className="img2text-sign-label">
                      {sign === 'space' ? '␣' : sign}
                    </span>
                    
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </div>

          {/* RIGHT — selected + result */}
          <div className="img2text-right">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <p className="img2text-section-title">🔍 Selected Signs</p>
              {selected.length > 0 && (
                <motion.button
                  className="facts-arrow"
                  onClick={clearAll}
                  whileHover={{ scale: 1.1 }}
                  whileTap={{ scale: 0.9 }}
                  style={{ fontSize: 12, width: 28, height: 28 }}
                >
                  ✕
                </motion.button>
              )}
            </div>
            <p className="img2text-section-sub">Signs you selected appear here</p>

            {/* Selected signs row */}
            {selected.length === 0 ? (
              <motion.div
                className="img2text-empty"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                <p style={{ fontSize: 40 }}>👈</p>
                <p>Click signs to translate into Text</p>
              </motion.div>
            ) : (
              <div className="img2text-selected-row">
                <AnimatePresence>
                  {selected.map((sign, i) => (
                    <motion.div
                      key={i}
                      className="img2text-selected-card"
                      initial={{ opacity: 0, scale: 0.7, y: 10 }}
                      animate={{ opacity: 1, scale: 1, y: 0 }}
                      exit={{ opacity: 0, scale: 0.7 }}
                      transition={{ type: 'spring', stiffness: 300, damping: 20 }}
                      whileHover={{ scale: 1.05 }}
                    >
                      <img
                        src={`/Signs/${sign}.jpg`}
                        alt={sign}
                        className="img2text-selected-img"
                        onError={e => { e.target.style.display = 'none' }}
                      />
                      
                      <motion.button
                        className="img2text-remove-btn"
                        onClick={() => removeSign(i)}
                        whileHover={{ scale: 1.2 }}
                        whileTap={{ scale: 0.8 }}
                      >
                        ✕
                      </motion.button>
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
            )}

            {/* Detected text result */}
            {selected.length > 0 && (
              <motion.div
                className="img2text-result"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <p className="img2text-result-label">Translated Text</p>
                <motion.p
                  className="img2text-result-text"
                  key={detectedText}
                  initial={{ scale: 0.95 }}
                  animate={{ scale: 1 }}
                >
                  {detectedText || '...'}
                </motion.p>

                {/* individual confidences */}
                <div className="img2text-conf-list">
                  {results.map((r, i) => r && (
                    <div key={i} className="img2text-conf-item">
                      <span className="img2text-conf-sign">
                        {selected[i] === 'space' ? '␣' : selected[i]}
                      </span>
                      <div className="conf-track" style={{ flex: 1 }}>
                        <motion.div
                          className="conf-fill"
                          initial={{ width: 0 }}
                          animate={{ width: `${r.confidence}%` }}
                          transition={{ duration: 0.6, ease: 'easeOut' }}
                        />
                      </div>
                      <span className="img2text-conf-pct">{r.confidence}%</span>
                    </div>
                  ))}
                </div>
              </motion.div>
            )}

            {/* Error */}
            {error && (
              <motion.div
                className="error-box"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                {error}
              </motion.div>
            )}

            {/* Loading */}
            {loading && (
              <motion.div
                className="img2text-loading"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
              >
                <span className="chat-typing">
                  <span /><span /><span />
                </span>
                <p>Translating sign...</p>
              </motion.div>
            )}

          </div>
        </div>

      </motion.div>
    </div>
  )
}

export default ImageToText
