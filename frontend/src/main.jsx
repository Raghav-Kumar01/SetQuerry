import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { AnalysisProvider } from './context/AnalysisContext'
import './index.css'
import App from './App.jsx'

// StrictMode is intentionally removed:
// React StrictMode double-invokes effects in development, which causes
// GSAP ScrollTrigger (and video.currentTime scrubbing) to initialize
// twice, creating duplicate triggers and incorrect initial scroll state.
// Remove StrictMode for any GSAP-heavy application.
createRoot(document.getElementById('root')).render(
  <BrowserRouter>
    <AnalysisProvider>
      <App />
    </AnalysisProvider>
  </BrowserRouter>
)
