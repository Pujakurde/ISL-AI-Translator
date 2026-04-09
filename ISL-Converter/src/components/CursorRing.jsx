import { useEffect, useState } from 'react'

const RING_SIZE = 20

function CursorRing() {
  const [position, setPosition] = useState({ x: -100, y: -100 })

  useEffect(() => {
    const moveCursor = (e) => {
      setPosition({ x: e.clientX, y: e.clientY })
    }
    window.addEventListener('mousemove', moveCursor)
    return () => window.removeEventListener('mousemove', moveCursor)
  }, [])

  return (
    <div
      className="cursor-ring"
      style={{
        top: position.y - RING_SIZE / 2 + 4,
        left: position.x - RING_SIZE / 2 + 4,
      }}
    />
  )
}

export default CursorRing