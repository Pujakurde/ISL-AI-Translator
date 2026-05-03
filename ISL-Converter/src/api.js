const HOSTED_API_FALLBACK_BASE = import.meta.env.VITE_FALLBACK_API_BASE?.trim() || ''

function getRemoteApiOrigin() {
  if (!HOSTED_API_FALLBACK_BASE) return ''
  return new URL(HOSTED_API_FALLBACK_BASE).origin
}

function isLocalHostname(hostname) {
  return hostname === 'localhost' || hostname === '127.0.0.1'
}

function isRenderStaticHost(hostname) {
  if (!HOSTED_API_FALLBACK_BASE) return false
  return hostname.endsWith('.onrender.com') && hostname !== new URL(HOSTED_API_FALLBACK_BASE).hostname
}

function resolveApiBase() {
  const configuredBase = import.meta.env.VITE_API_BASE?.trim()
  if (configuredBase) return configuredBase

  if (typeof window === 'undefined') {
    return 'http://127.0.0.1:8000'
  }

  const { hostname } = window.location
  if (isLocalHostname(hostname)) {
    return `http://${hostname}:8000`
  }

  if (isRenderStaticHost(hostname)) {
    return HOSTED_API_FALLBACK_BASE
  }

  return window.location.origin
}

export const API_BASE = resolveApiBase()

export const ENDPOINTS = {
  videoFeed: `${API_BASE}/video_feed`,
  stopCamera: `${API_BASE}/stop-camera`,
  setMode: (mode) => `${API_BASE}/set-mode/${mode}`,
  predictLiveFrame: `${API_BASE}/predict-live-frame`,
  lastSign: `${API_BASE}/last-prediction/sign`,
  clearLastSign: `${API_BASE}/last-prediction/sign`,
  lastSignLast: `${API_BASE}/last-prediction/sign/last`,
  predict: `${API_BASE}/predict`,
  predictImage: `${API_BASE}/predict-image`,
  textToSign: (text) => `${API_BASE}/text-to-sign/${encodeURIComponent(text)}`,
  suggestions: (prefix) => `${API_BASE}/suggestions?prefix=${encodeURIComponent(prefix)}`,
}

function extractApiMessage(payload) {
  if (!payload) return ''
  if (typeof payload === 'string') return payload.replace(/\s+/g, ' ').trim()
  if (typeof payload === 'object') {
    return String(payload.detail || payload.error || payload.message || '').trim()
  }
  return ''
}

async function readApiPayload(response) {
  const contentType = response.headers.get('content-type') || ''

  if (contentType.includes('application/json')) {
    return response.json().catch(() => null)
  }

  return response.text().catch(() => '')
}

function buildHostedFallbackUrl(url) {
  if (typeof window === 'undefined' || !HOSTED_API_FALLBACK_BASE) return null

  const resolvedUrl = new URL(url, window.location.origin)
  const hostedOrigin = getRemoteApiOrigin()

  if (!hostedOrigin || resolvedUrl.origin === hostedOrigin || isLocalHostname(window.location.hostname)) {
    return null
  }

  return `${HOSTED_API_FALLBACK_BASE}${resolvedUrl.pathname}${resolvedUrl.search}`
}

export function formatApiError(response, payload, fallbackMessage = 'Request failed') {
  const message = extractApiMessage(payload)

  if (response.status === 503 && /service suspended|suspended by its owner/i.test(message)) {
    return `Backend service at ${API_BASE} is suspended on Render. Resume or redeploy it, then try again.`
  }

  if ([502, 503, 504].includes(response.status)) {
    return `Backend at ${API_BASE} is unavailable right now (HTTP ${response.status}). If you are using the hosted app, the Render service may be suspended or still starting.`
  }

  return message || fallbackMessage
}

export async function fetchApi(url, options = {}, fallbackMessage = 'Request failed') {
  let response = await fetch(url, options)
  let payload = await readApiPayload(response)

  if (!response.ok && response.status === 404) {
    const fallbackUrl = buildHostedFallbackUrl(url)
    if (fallbackUrl) {
      response = await fetch(fallbackUrl, options)
      payload = await readApiPayload(response)
    }
  }

  if (!response.ok) {
    throw new Error(formatApiError(response, payload, fallbackMessage))
  }

  return payload
}

export function getNetworkErrorMessage(error, fallbackMessage = `Cannot reach backend at ${API_BASE}.`) {
  if (error?.message === 'Failed to fetch') {
    return `Cannot reach backend at ${API_BASE}. If you are using the hosted app, check whether the Render backend is suspended or still starting.`
  }

  return error?.message || fallbackMessage
}
