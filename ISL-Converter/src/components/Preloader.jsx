import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

function CountUp() {
  const [count, setCount] = useState(0)

  useEffect(() => {
    const duration = 2500
    const interval = 30
    const steps = duration / interval
    const increment = 100 / steps
    let current = 0

    const timer = setInterval(() => {
      current += increment
      if (current >= 100) {
        setCount(100)
        clearInterval(timer)
      } else {
        setCount(Math.floor(current))
      }
    }, interval)

    return () => clearInterval(timer)
  }, [])

  return <>{count}%</>
}

function Preloader({ isLoading }) {
  return (
    <AnimatePresence>
      {isLoading && (
        <motion.div
          className="preloader"
          initial={{ opacity: 1 }}
          exit={{ opacity: 0, scale: 1.05 }}
          transition={{ duration: 0.6, ease: 'easeInOut' }}
        >
          {/* Background circles */}
          <div className="preloader-circle-1" />
          <div className="preloader-circle-2" />

          {/* Content */}
          <div className="preloader-content">

            {/* Animated hand emoji */}
            <motion.div
              className="preloader-icon"
              animate={{
                rotate: [0, 15, -15, 10, -10, 0],
                scale: [1, 1.2, 1, 1.1, 1],
              }}
              transition={{
                duration: 1.5,
                repeat: Infinity,
                repeatDelay: 0.5,
              }}
            >
              🤝
            </motion.div>

            {/* Title */}
            <motion.h1
              className="preloader-title"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              ISL Translator
            </motion.h1>

            {/* Subtitle */}
            <motion.p
              className="preloader-sub"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4 }}
            >
              Bridging Communication with Indian Sign Language
            </motion.p>

            {/* Ring loader with percentage */}
            <motion.div
              className="preloader-ring-wrap"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3 }}
            >
              <svg width="80" height="80" viewBox="0 0 80 80">
                {/* Background ring */}
                <circle
                  cx="40" cy="40" r="32"
                  stroke="rgba(242,237,238,0.15)"
                  strokeWidth="5"
                  fill="none"
                />
                {/* Progress ring */}
                <motion.circle
                  cx="40" cy="40" r="32"
                  stroke="#F2EDEE"
                  strokeWidth="5"
                  fill="none"
                  strokeLinecap="round"
                  strokeDasharray="201"
                  strokeDashoffset="201"
                  animate={{ strokeDashoffset: 0 }}
                  transition={{ duration: 2.5, ease: 'easeInOut' }}
                  style={{ rotate: -90, transformOrigin: '40px 40px' }}
                />
              </svg>

              {/* Percentage counter in center */}
              <div className="preloader-ring-center">
                <motion.span
                  className="preloader-pct"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.1 }}
                >
                  <CountUp />
                </motion.span>
              </div>
            </motion.div>

            {/* Loading dots */}
            <div className="preloader-dots">
              {[0, 1, 2].map(i => (
                <motion.div
                  key={i}
                  className="preloader-dot"
                  animate={{
                    y: [0, -10, 0],
                    opacity: [0.4, 1, 0.4],
                  }}
                  transition={{
                    duration: 0.8,
                    repeat: Infinity,
                    delay: i * 0.2,
                  }}
                />
              ))}
            </div>

          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

export default Preloader