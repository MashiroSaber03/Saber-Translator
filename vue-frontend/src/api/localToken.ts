import axios from 'axios'

interface LocalTokenResponse {
  success: boolean
  enabled?: boolean
  header?: string
  token?: string
}

let tokenLoaded = false
let tokenLoadingPromise: Promise<void> | null = null
let localTokenHeader = 'X-Saber-Local-Token'
let localToken: string | null = null
let lastLoadFailedAt = 0

const TOKEN_FETCH_TIMEOUT_MS = 5000
const TOKEN_RETRY_INTERVAL_MS = 2000

interface LocalHeaderOptions {
  forceRefresh?: boolean
}

async function ensureTokenLoaded(forceRefresh = false): Promise<void> {
  if (tokenLoaded && !forceRefresh) return
  if (tokenLoadingPromise) {
    await tokenLoadingPromise
    return
  }

  if (!forceRefresh && lastLoadFailedAt > 0 && Date.now() - lastLoadFailedAt < TOKEN_RETRY_INTERVAL_MS) {
    return
  }

  tokenLoadingPromise = (async () => {
    try {
      const response = await axios.get<LocalTokenResponse>('/api/local-token', { timeout: TOKEN_FETCH_TIMEOUT_MS })
      if (!response.data?.success) {
        tokenLoaded = false
        localToken = null
        lastLoadFailedAt = Date.now()
        return
      }

      localTokenHeader = response.data.header || localTokenHeader

      // 本机模式关闭时不需要 token，视为已加载完成
      if (response.data.enabled === false) {
        localToken = null
        tokenLoaded = true
        lastLoadFailedAt = 0
        return
      }

      const token = response.data.token || ''
      if (!token) {
        tokenLoaded = false
        localToken = null
        lastLoadFailedAt = Date.now()
        return
      }

      localToken = token
      tokenLoaded = true
      lastLoadFailedAt = 0
    } catch {
      tokenLoaded = false
      localToken = null
      lastLoadFailedAt = Date.now()
    } finally {
      tokenLoadingPromise = null
    }
  })()

  await tokenLoadingPromise
}

export async function forceRefreshLocalToken(): Promise<void> {
  tokenLoaded = false
  await ensureTokenLoaded(true)
}

export async function getLocalWriteHeaders(
  baseHeaders?: Record<string, string>,
  options?: LocalHeaderOptions
): Promise<Record<string, string>> {
  await ensureTokenLoaded(Boolean(options?.forceRefresh))

  const headers: Record<string, string> = { ...(baseHeaders || {}) }
  if (localToken) {
    headers[localTokenHeader] = localToken
  }
  return headers
}
