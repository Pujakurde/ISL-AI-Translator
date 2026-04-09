import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import ThemeToggle from '../components/ThemeToggle'
import BackButton from '../components/BackButton'


function Translation() {
  const navigate = useNavigate()
  const [islExpanded, setIslExpanded] = useState(false)
  const subRef = useRef(null)

  function handleISLClick() {
    setIslExpanded(prev => !prev)
    if (!islExpanded) {
      setTimeout(() => {
        subRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
      }, 150)
    }
  }

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
            <h1 className="dict-title">🔄 Translation</h1>
            <p className="dict-sub">Choose your translation direction</p>
          </div>
        </div>

        {/* TWO MAIN CARDS */}
        <div className="trans-grid">

          {/* ISL TO TEXT */}
          <div className="trans-col">
            <motion.button
              className={`trans-card ${islExpanded ? 'active' : ''}`}
              onClick={handleISLClick}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.1 }}
              whileHover={{ scale: 1.03, y: -6 }}
              whileTap={{ scale: 0.97 }}
            >
              <div className="trans-icon-wrap">
                <span className="trans-icon">🤟</span>
                <span className="trans-arrow-big">→</span>
                <span className="trans-icon">📝</span>
              </div>
              <p className="trans-title">ISL to Text</p>
              <p className="trans-desc">Detect sign language and translate to text using camera or image</p>
              <div className="trans-tag">
                {islExpanded ? 'Choose input below ↓' : 'Camera • Image'}
              </div>
            </motion.button>

            {/* SUB BUTTONS */}
            <AnimatePresence>
              {islExpanded && (
                <motion.div
                  ref={subRef}
                  className="trans-sub-group"
                  initial={{ opacity: 0, y: -10, height: 0 }}
                  animate={{ opacity: 1, y: 0, height: 'auto' }}
                  exit={{ opacity: 0, y: -10, height: 0 }}
                  transition={{ type: 'spring', stiffness: 300, damping: 28 }}
                >
                  <motion.button
                    className="trans-sub-card"
                    onClick={() => navigate('/image-to-text')}
                    whileHover={{ scale: 1.04, y: -3 }}
                    whileTap={{ scale: 0.96 }}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.05 }}
                  >
                    <span className="trans-sub-icon">🖼️</span>
                    <div>
                      <p className="trans-sub-title">Image Dataset</p>
                      <p className="trans-sub-desc">Choose Sign Images to Translate</p>
                    </div>
                    <span style={{ marginLeft: 'auto', color: '#6A2E3B', fontSize: 18 }}>→</span>
                  </motion.button>

                  <motion.button
                    className="trans-sub-card"
                    onClick={() => navigate('/camera-to-text')}
                    whileHover={{ scale: 1.04, y: -3 }}
                    whileTap={{ scale: 0.96 }}
                    initial={{ opacity: 0, x: 10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.1 }}
                  >
                    <span className="trans-sub-icon">📷</span>
                    <div>
                      <p className="trans-sub-title">Live Camera</p>
                      <p className="trans-sub-desc">Use your webcam in real time</p>
                    </div>
                    <span style={{ marginLeft: 'auto', color: '#6A2E3B', fontSize: 18 }}>→</span>
                  </motion.button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* TEXT TO ISL */}
          <div className="trans-col">
            <motion.button
              className="trans-card"
              onClick={() => navigate('/text-to-sign')}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              whileHover={{ scale: 1.03, y: -6 }}
              whileTap={{ scale: 0.97 }}
            >
              <div className="trans-icon-wrap">
                <span className="trans-icon">📝</span>
                <span className="trans-arrow-big">→</span>
                <span className="trans-icon">🤟</span>
              </div>
              <p className="trans-title">Text to ISL</p>
              <p className="trans-desc">Type text and translate to Indian Sign Language Signs</p>
              <div className="trans-tag">Dictionary • Images</div>
            </motion.button>
          </div>

        </div>

      </motion.div>
    </div>
  )
}

export default Translation