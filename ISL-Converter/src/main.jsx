import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './App.css'
import App from './App.jsx'
import { ThemeProvider } from './ThemeContext'
import CursorRing from './components/CursorRing'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ThemeProvider>
      <CursorRing />
      <App />
    </ThemeProvider>
  </StrictMode>,
)