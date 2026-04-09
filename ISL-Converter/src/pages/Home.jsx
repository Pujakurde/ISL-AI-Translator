import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import ThemeToggle from '../components/ThemeToggle'
import ISLIllustration from '../components/ISLIllustration'
import Footer from '../components/Footer'
import { useState, useRef, useEffect } from 'react'
 

const facts = [
  "ISL is a complete, natural language with its own complex grammar and syntax ✋",
  "It follows a Subject-Object-Verb (SOV) word order, unlike English 🏗️",
  "The language uses a two-handed manual alphabet for all 26 letters 👐",
  "It was officially recognized under the RPwD Act of 2016 in India 📜",
  "There are an estimated 1 to 2.7 million active ISL users nationwide 🇮🇳",
  "Facial expressions and body posture are essential grammatical components 🎭",
  "ISL is not a visual representation of spoken Hindi or English 🚫",
  "The ISLRTC was established in 2015 to lead research and training 🏛️",
  "Regional dialects exist in cities like Delhi, Mumbai, and Kolkata 🗺️",
  "The official ISL dictionary now contains over 10,000 standardized signs 📖",
  "NEP 2020 officially designated ISL as a language for school curricula 🎓",
  "Signs for 'Time' move along a spatial axis (backward for past, forward for future) ⏳",
  "India has a massive shortage with fewer than 500 certified interpreters 🗣️",
  "ISL uses Non-Manual Markers to distinguish between questions and statements ❓",
  "It shares some historical roots with British Sign Language (BSL) 🇬🇧",
  "Fingerspelling is used primarily for proper nouns like names or places 🖋️",
  "One sign in ISL can represent an entire phrase or concept in English 💡",
  "Many signs are iconic, meaning they visually resemble the object they represent 🍎",
  "ISL is the primary mode of instruction in many specialized deaf schools 🏫",
  "The language includes specific signs for Indian cultural festivals and foods 🍛",
  "Digital tools now use 3D avatars to translate text into ISL in real-time 🤖",
  "Directional Verbs change meaning based on the direction the hands move ➡️",
  "ISL is used by both the Deaf community and their hearing family members 👨‍👩‍👧",
  "It is an evolving language, with new signs added for modern technology 📱",
  "Learning ISL promotes inclusivity and breaks communication barriers 🤝",
]

function FactsCard({ facts }) {
  const [page, setPage] = useState(0)
  const perPage = 5
  const total = Math.ceil(facts.length / perPage)
  const current = facts.slice(page * perPage, page * perPage + perPage)

  const next = () => setPage(p => (p + 1) % total)
  const prev = () => setPage(p => (p - 1 + total) % total)

  return (
    <motion.div
      className="facts-card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.4 }}
      whileHover={{
        y: -8,
        scale: 1.02,
        boxShadow: '0 20px 60px rgba(106,46,59,0.25)',
        transition: { type: 'spring', stiffness: 300, damping: 20 }
      }}
    >
      <p className="facts-title">💡 Did you know?</p>

      <div className="facts-content-row">
        <motion.button
          className="facts-arrow"
          onClick={prev}
          whileHover={{ scale: 1.2, rotate: -10 }}
          whileTap={{ scale: 0.85 }}
          transition={{ type: 'spring', stiffness: 400 }}
        >
          ‹
        </motion.button>

        <div className="facts-list">
          {current.map((fact, i) => (
            <motion.div
              className="fact-item"
              key={page + '-' + i}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.08 }}
              whileHover={{ x: 6 }}
            >
              <span className="fact-dot" />
              <p className="fact-text">{fact}</p>
            </motion.div>
          ))}
        </div>

        <motion.button
          className="facts-arrow"
          onClick={next}
          whileHover={{ scale: 1.2, rotate: 10 }}
          whileTap={{ scale: 0.85 }}
          transition={{ type: 'spring', stiffness: 400 }}
        >
          ›
        </motion.button>
      </div>

      <div className="facts-dots">
        {Array.from({ length: total }).map((_, i) => (
          <motion.span
            key={i}
            className={`facts-dot-item ${i === page ? 'active' : ''}`}
            onClick={() => setPage(i)}
            whileHover={{ scale: 1.5 }}
            whileTap={{ scale: 0.8 }}
          />
        ))}
      </div>

    </motion.div>
  )
}
function AIChatCard() {
  const navigate = useNavigate()
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      text: "Hi! 👋 I'm your ISL Assistant. Ask me anything about Indian Sign Language!"
    }
  ])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [maximized, setMaximized] = useState(false)
  const bottomRef = useRef(null)
  const isFirstRender = useRef(true)

  useEffect(() => {
    if (isFirstRender.current) {
      isFirstRender.current = false
      return
    }
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }
  }, [messages])

  async function sendMessage() {
    const text = input.trim()
    if (!text || loading) return

    const userMsg = { role: 'user', text }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)

    try {
      const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${import.meta.env.VITE_GROQ_API_KEY}`
        },
        body: JSON.stringify({
          model: 'llama-3.1-8b-instant',
          max_tokens: 300,
          temperature: 0.7,
          messages: [
            {
              role: 'system',
              content: `You are a friendly ISL (Indian Sign Language) assistant built into an ISL Translator web app. 
              
Answer questions about:
- Indian Sign Language (ISL) — history, grammar, usage, facts
- How to use this ISL Translator app which has Text to Sign, Sign to Text, Live Camera, Image Upload features
- Deaf community and inclusion in India
- Sign language learning tips

If someone asks something unrelated to ISL, signs, deaf community, or this app, reply:
"That is outside my expertise! 😄 I am best at answering questions about Indian Sign Language and this app. Want to know something interesting about ISL instead?"

Keep responses short, friendly and helpful. Use emojis occasionally.`
            },
            ...messages.map(m => ({
              role: m.role === 'assistant' ? 'assistant' : 'user',
              content: m.text
            })),
            { role: 'user', content: text }
          ]
        })
      })

      const data = await response.json()

      if (data.error) {
        setMessages(prev => [...prev, {
          role: 'assistant',
          text: "Error: " + data.error.message + " 🙏"
        }])
        setLoading(false)
        return
      }

      const reply = data.choices?.[0]?.message?.content
        || "Sorry I could not understand that. Try asking about ISL! 🤟"

      setMessages(prev => [...prev, { role: 'assistant', text: reply }])

    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: "Oops! Something went wrong. Please try again 🙏"
      }])
      console.error(err)
    }

    setLoading(false)
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div
  className={`ai-card ${maximized ? 'ai-card-maximized' : ''}`}
 
>
      <div className="ai-header">
  <motion.span
    className="ai-icon"
    animate={{ rotate: [0, 10, -10, 0] }}
    transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
    whileHover={{ scale: 1.3, rotate: 20 }}
  >
    🤖
  </motion.span>
  <div>
    <p className="ai-title">ISL Assistant</p>
    <p className="ai-sub">Ask me anything about ISL Translator!</p>
  </div>
  <div className="ai-status">
    <span className="ai-status-dot" />
    <span className="ai-status-text">Online</span>
  </div>

  <motion.button
    className="ai-toggle-btn"
    onClick={() => setMaximized(prev => !prev)}
    whileHover={{ scale: 1.15 }}
    whileTap={{ scale: 0.9 }}
    title={maximized ? 'Minimize' : 'Maximize'}
  >
    {maximized ? (
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
        <path d="M9 1v4h4M5 13V9H1M1 5h4V1M13 9h-4v4" stroke="#F2EDEE" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    ) : (
      <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
        <path d="M9 1h4v4M5 13H1V9M1 1l4 4M13 13l-4-4" stroke="#F2EDEE" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    )}
  </motion.button>
</div>

      <div className="chat-messages">
        {messages.map((msg, i) => (
          <motion.div
            key={i}
            className={`chat-bubble ${msg.role}`}
            initial={{ opacity: 0, y: 8, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ type: 'spring', stiffness: 300, damping: 24 }}
            whileHover={{ scale: 1.01 }}
          >
            {msg.text}
          </motion.div>
        ))}

        {loading && (
          <div className="chat-bubble assistant">
            <span className="chat-typing">
              <span />
              <span />
              <span />
            </span>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      <div className="chat-input-row">
        <input
          className="chat-input"
          type="text"
          placeholder="Ask about ISL..."
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKey}
          disabled={loading}
        />
        <motion.button
          className="chat-send"
          onClick={sendMessage}
          disabled={!input.trim() || loading}
          whileHover={{ scale: 1.15, rotate: 15 }}
          whileTap={{ scale: 0.85, rotate: 0 }}
          transition={{ type: 'spring', stiffness: 400, damping: 17 }}
        >
          ➤
        </motion.button>
      </div>

    </div>
  )
}

function Home() {
  const navigate = useNavigate()

  return (
    <div className="home-page">
      <ThemeToggle />

      <motion.div
  className="screen"
  initial={{ opacity: 0 }}
  animate={{ opacity: 1 }}
  exit={{ opacity: 0 }}
  transition={{ duration: 0.4 }}
>
        <div className="landing">

          {/* HERO */}
          <motion.div
            className="hero"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            <motion.h1
              className="hero-heading"
              whileHover={{ scale: 1.03 }}
              transition={{ type: 'spring', stiffness: 300 }}
            >
              🤟 ISL Translator 🤟
            </motion.h1>
            <p className="hero-sub">Bridging Communication with Indian Sign Language</p>
          </motion.div>

          {/* TOP ROW */}
          <div className="top-grid">

            {/* TOP LEFT */}
            <motion.div
              className="top-left"
              initial={{ opacity: 0, x: -30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              {/* Stats */}
              <div className="stats-row">
                {[
                  { num: '36', label: 'Total Signs' },
                  { num: 'A-Z', label: 'Alphabets' },
                  { num: '0-9', label: 'Numbers' },
                ].map((stat, i) => (
                  <motion.div
                    className="stat-card"
                    key={i}
                    whileHover={{
                      scale: 1.08,
                      y: -6,
                      rotateZ: i === 1 ? 0 : i === 0 ? -2 : 2,
                      transition: { type: 'spring', stiffness: 300, damping: 18 }
                    }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <p className="stat-num">{stat.num}</p>
                    <p className="stat-label">{stat.label}</p>
                  </motion.div>
                ))}
              </div>

              {/* Buttons */}
              <div className="action-group">
                <motion.button
                  className="action-btn"
                  onClick={() => navigate('/dictionary')}
                  whileHover={{
                    scale: 1.02,
                    x: 8,
                    transition: { type: 'spring', stiffness: 300, damping: 20 }
                  }}
                  whileTap={{ scale: 0.97 }}
                >
                  <motion.span
                    className="action-icon"
                    whileHover={{ rotate: [0, -15, 15, 0] }}
                    transition={{ duration: 0.4 }}
                  >
                    📖
                  </motion.span>
                  <div className="action-info">
                    <span className="action-label">Dictionary</span>
                    <span className="action-desc">Browse all ISL signs A-Z</span>
                  </div>
                  <motion.span
                    className="action-arrow"
                    whileHover={{ x: 4 }}
                  >
                    →
                  </motion.span>
                </motion.button>

                <motion.button
                  className="action-btn"
                  onClick={() => navigate('/translation')}
                  whileHover={{
                    scale: 1.02,
                    x: 8,
                    transition: { type: 'spring', stiffness: 300, damping: 20 }
                  }}
                  whileTap={{ scale: 0.97 }}
                >
                  <motion.span
                    className="action-icon"
                    animate={{ rotate: [0, 360] }}
                    transition={{ duration: 4, repeat: Infinity, ease: 'linear' }}
                  >
                    🔄
                  </motion.span>
                  <div className="action-info">
                    <span className="action-label">Translation</span>
                    <span className="action-desc">Camera or image detection</span>
                  </div>
                  <motion.span
                    className="action-arrow"
                    whileHover={{ x: 4 }}
                  >
                    →
                  </motion.span>
                </motion.button>
              </div>
            </motion.div>

            {/* TOP RIGHT — illustration */}
            <motion.div
              className="top-right"
              initial={{ opacity: 1, x: 30 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
              whileHover={{
                scale: 1.04,
                transition: { type: 'spring', stiffness: 200, damping: 20 }
              }}
            >
              <ISLIllustration />
            </motion.div>

          </div>

          {/* BOTTOM ROW */}
          <div className="bottom-grid">
            <FactsCard facts={facts} />
            <AIChatCard />
          </div>

        </div>
      </motion.div>

      <Footer />
    </div>
  )
}

export default Home