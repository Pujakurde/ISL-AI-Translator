import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import ThemeToggle from '../components/ThemeToggle'
import BackButton from '../components/BackButton'


function TextToSign() {
  const navigate = useNavigate()
  const [text, setText] = useState('')

  const letters = text.toUpperCase().replace(/[^ A-Z0-9]/g, '').split('').map(ch => ch === ' ' ? 'space' : ch)

  return (
    <>
      <BackButton />
      <ThemeToggle />
      <div className="tts-page">
        <motion.div
          className="tts-wrap"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.3 }}
        >

          {/* HEADER */}
          <div className="tts-header">
            
            <div>
              <h1 className="tts-title">📝 Text to Sign</h1>
              <p className="tts-sub">Type text below — each letter shows its ISL sign</p>
            </div>
          </div>

          {/* INPUT */}
          <motion.div
            className="tts-input-wrap"
            whileFocus={{ scale: 1.01 }}
          >
            <input
              className="tts-input"
              type="text"
              placeholder="Type letters or words here..."
              value={text}
              onChange={e => setText(e.target.value)}
              autoFocus
            />
            {text && (
              <motion.button
                className="tts-clear"
                onClick={() => setText('')}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
              >
                ✕
              </motion.button>
            )}
          </motion.div>

          {/* COUNTER */}
          {text && (
            <motion.p
              className="tts-counter"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              {letters.filter(l => l !== 'space').length} letters • {letters.filter(l => l === 'space').length} spaces
            </motion.p>
          )}

          {/* EMPTY STATE */}
          {letters.length === 0 && (
            <motion.div
              className="tts-empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
            >
              <p style={{ fontSize: 48 }}>✋</p>
              <p>Start typing to see ISL signs</p>
            </motion.div>
          )}

          {/* SIGN GRID */}
          <div className="tts-grid">
            <AnimatePresence>
              {letters.map((letter, i) => (
                <motion.div
                  className="tts-sign-card"
                  key={i + '-' + letter}
                  initial={{ opacity: 0, scale: 0.7, y: 20 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.7 }}
                  transition={{ delay: i * 0.03, type: 'spring', stiffness: 300, damping: 20 }}
                  whileHover={{ scale: 1.1, y: -6, boxShadow: '0 12px 32px rgba(106,46,59,0.25)' }}
                  whileTap={{ scale: 0.95 }}
                >
                  <img
  src={`/Signs/${letter}.jpg`}
  alt={letter}
  className="tts-sign-img"
  onError={e => {
    e.target.style.display = 'none'
    e.target.nextSibling.style.display = 'flex'
  }}
/>
                  <div className="tts-sign-fallback" style={{ display: 'none' }}>
                    {letter}
                  </div>
                  <span className="tts-sign-label">{letter === 'space' ? 'space' : letter}</span>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>

        </motion.div>
      </div>
    </>
  )
}

export default TextToSign