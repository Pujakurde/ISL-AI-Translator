import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import { useEffect, useState } from 'react'
import Home from './pages/Home'
import TextToSign from './pages/TextToSign'
import CameraToText from './pages/CameraToText'
import ImageToText from './pages/ImageToText'
import Dictionary from './pages/Dictionary'
import Translation from './pages/Translation'
import Preloader from './components/Preloader'
import BgCircles from './components/BgCircles'
import CursorRing from './components/CursorRing'

function ScrollToTop() {
  const { pathname } = useLocation()
  useEffect(() => {
    setTimeout(() => {
      window.scrollTo(0, 0)
      document.documentElement.scrollTop = 0
      document.body.scrollTop = 0
    }, 100)
  }, [pathname])
  return null
}

function AnimatedRoutes() {
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', 'light')
  }, [])

  const location = useLocation()

  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={<Home />} />
        <Route path="/translation" element={<Translation />} />
        <Route path="/text-to-sign" element={<TextToSign />} />
        <Route path="/camera-to-text" element={<CameraToText />} />
        <Route path="/image-to-text" element={<ImageToText />} />
        <Route path="/dictionary" element={<Dictionary />} />
      </Routes>
    </AnimatePresence>
  )
}

function App() {
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoading(false)
    }, 2500)
    return () => clearTimeout(timer)
  }, [])

  return (
    <BrowserRouter>
      <CursorRing />
      <Preloader isLoading={isLoading} />
      {!isLoading && (
        <>
          <BgCircles />
          <ScrollToTop />
          <AnimatedRoutes />
        </>
      )}
    </BrowserRouter>
  )
}

export default App