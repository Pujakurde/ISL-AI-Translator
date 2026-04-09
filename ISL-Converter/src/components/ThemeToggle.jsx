import { useTheme } from '../ThemeContext'
import { motion } from 'framer-motion'

function ThemeToggle() {
  const { isDark, toggle } = useTheme()

  return (
    <button className="theme-toggle" onClick={toggle} title="Toggle theme">
      <motion.div
        className="toggle-track"
        animate={{ backgroundColor: isDark ? '#D9A7A7' : '#6A2E3B' }}
      >
        <motion.div
          className="toggle-thumb"
          animate={{ x: isDark ? 24 : 0 }}
          transition={{ type: 'spring', stiffness: 300, damping: 25 }}
        >
          {isDark ? '🌙' : '☀️'}
        </motion.div>
      </motion.div>
    </button>
  )
}

export default ThemeToggle