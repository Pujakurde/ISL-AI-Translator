function resolveApiBase() {
  const configuredBase = import.meta.env.VITE_API_BASE?.trim()
  if (configuredBase) return configuredBase

  if (typeof window === 'undefined') {
    return 'http://127.0.0.1:8000'
  }

  const { hostname } = window.location
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    return `http://${hostname}:8000`
  }

  return 'https://isl-backend.onrender.com'
}

export const API_BASE = resolveApiBase()

export const ENDPOINTS = {
  videoFeed: `${API_BASE}/video_feed`,
  stopCamera: `${API_BASE}/stop-camera`,
  setMode: (mode) => `${API_BASE}/set-mode/${mode}`,
  lastSign: `${API_BASE}/last-prediction/sign`,
  clearLastSign: `${API_BASE}/last-prediction/sign`,
  lastSignLast: `${API_BASE}/last-prediction/sign/last`,
  predict: `${API_BASE}/predict`,
  predictImage: `${API_BASE}/predict-image`,
  textToSign: (text) => `${API_BASE}/text-to-sign/${encodeURIComponent(text)}`,
  suggestions: (prefix) => `${API_BASE}/suggestions?prefix=${encodeURIComponent(prefix)}`,
};
