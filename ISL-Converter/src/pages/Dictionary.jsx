import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import ThemeToggle from '../components/ThemeToggle'
import BackButton from '../components/BackButton'


const ALPHABETS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('')
const NUMBERS = '0123456789'.split('')
const ALL_SIGNS = [...NUMBERS, ...ALPHABETS]

const FILTERS = [
  { key: 'all',   label: 'All 36' },
  { key: 'alpha', label: 'A–Z'    },
  { key: 'num',   label: '0–9'    },
]

function Dictionary() {
  const navigate = useNavigate()
  const [search, setSearch]   = useState('')
  const [selected, setSelected] = useState(null)
  const [filter, setFilter]   = useState('all')

  const activeIndex = FILTERS.findIndex(f => f.key === filter)

  const filtered = ALL_SIGNS.filter(sign => {
    const matchSearch = sign.toLowerCase().includes(search.toLowerCase())
    const matchFilter =
      filter === 'all' ||
      (filter === 'alpha' && ALPHABETS.includes(sign)) ||
      (filter === 'num'   && NUMBERS.includes(sign))
    return matchSearch && matchFilter
  })

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
            <h1 className="dict-title">📖 ISL Dictionary</h1>
            <p className="dict-sub">Browse all Indian Sign Language signs</p>
          </div>
        </div>

        {/* SEARCH + FILTER */}
        <div className="dict-controls">
          <input
            className="dict-search"
            type="text"
            placeholder="Search letter or number..."
            value={search}
            onChange={e => setSearch(e.target.value)}
          />

          {/* SLIDING FILTER */}
          <div className="dict-filters">
            {/* sliding pill */}
            <motion.div
              className="dict-filter-pill"
              animate={{ x: activeIndex * 100 + '%' }}
              transition={{ type: 'spring', stiffness: 400, damping: 30 }}
            />
            {FILTERS.map(f => (
              <button
                key={f.key}
                className={`dict-filter-btn ${filter === f.key ? 'active' : ''}`}
                onClick={() => setFilter(f.key)}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>

        {/* STATS */}
        <p className="dict-count">{filtered.length} signs found</p>

        {/* GRID */}
        <div className="dict-grid">
          <AnimatePresence>
            {filtered.map((sign, i) => (
              <motion.div
                key={sign}
                className={`dict-card ${selected === sign ? 'selected' : ''}`}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
                transition={{ delay: i * 0.02 }}
                whileHover={{ scale: 1.06, y: -4 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setSelected(selected === sign ? null : sign)}
              >
                <div className="dict-img-wrap">
                  <img
                    src={`/Signs/${sign}.jpg`}
                    alt={`ISL sign for ${sign}`}
                    className="dict-img"
                    onError={e => {
                      e.target.style.display = 'none'
                      e.target.nextSibling.style.display = 'flex'
                    }}
                  />
                  <div className="dict-img-placeholder" style={{ display: 'none' }}>
                    {sign}
                  </div>
                </div>
                <p className="dict-label">{sign}</p>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>

        {filtered.length === 0 && (
          <div className="dict-empty">
            <p>No signs found for "{search}" 😔</p>
            <button
              className="primary-btn"
              style={{ marginTop: 16, maxWidth: 200 }}
              onClick={() => setSearch('')}
            >
              Clear search
            </button>
          </div>
        )}

        {/* MODAL */}
        <AnimatePresence>
          {selected && (
            <motion.div
              className="dict-modal-overlay"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setSelected(null)}
            >
              <motion.div
                className="dict-modal"
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.8, opacity: 0 }}
                transition={{ type: 'spring', stiffness: 300, damping: 24 }}
                onClick={e => e.stopPropagation()}
              >
                <button className="dict-modal-close" onClick={() => setSelected(null)}>✕</button>
                <img
                  src={`/Signs/${selected}.jpg`}
                  alt={`ISL sign for ${selected}`}
                  className="dict-modal-img"
                  onError={e => {
                    e.target.style.display = 'none'
                    e.target.nextSibling.style.display = 'flex'
                  }}
                />
                <div className="dict-modal-placeholder" style={{ display: 'none' }}>
                  {selected}
                </div>
                <p className="dict-modal-label">{selected}</p>
                <p className="dict-modal-sub">
                  {ALPHABETS.includes(selected) ? 'Alphabet' : 'Number'} sign in ISL
                </p>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

      </motion.div>
    </div>
  )
}

export default Dictionary