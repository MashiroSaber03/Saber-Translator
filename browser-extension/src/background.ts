import {
  loadSettings,
  preferenceFor,
  serializeStorageWrite,
  updateSettings,
} from './storage'
import { normalizedTaskPageUrl } from './pageIdentity'
import type {
  ActiveBrowserSession,
  BackgroundRequest,
  BackgroundResponse,
  BrowserLibraryBook,
  BrowserPageDto,
  BrowserSessionDto,
  BrowserSessionImportResult,
  ContextTranslateMessage,
  DomDetectionResult,
  ExtensionSettings,
  ResultImagePayload,
  UploadPageRequest,
} from './types'

const CONTEXT_MENU_ID = 'saber-translate-image'
const MAX_RESULT_BYTES = 45 * 1024 * 1024
const API_REQUEST_TIMEOUT_MS = 30_000
const IMAGE_TRANSFER_TIMEOUT_MS = 120_000
const ACTIVE_SESSION_KEY_PREFIX = 'saber-active-browser-session-v1:'
const FRIENDLY_API_ERRORS: Record<string, string> = {
  integration_disabled: '请先在 Saber GUI 中启用浏览器扩展连接',
  invalid_extension_token: '配对令牌无效，请从 Saber GUI 重新复制',
  loopback_required: 'Saber 只接受本机扩展连接',
  session_conflict: '当前翻译批次尚未结束，请稍后重试',
  result_not_found: '译图尚未生成或网页会话已经过期',
  browser_internal_error: 'Saber 浏览器扩展接口发生内部错误',
  request_timeout: '本机 Saber 请求超时，请确认后端仍在运行',
  source_timeout: '原始漫画图片下载超时，请重试',
}

void chrome.storage.local.setAccessLevel({
  accessLevel: 'TRUSTED_CONTEXTS',
}).catch(error => console.warn('Saber extension storage isolation failed', error))
void chrome.storage.session.setAccessLevel({
  accessLevel: 'TRUSTED_CONTEXTS',
}).catch(error => console.warn('Saber extension session isolation failed', error))

class RequestFailure extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly retryable: boolean,
  ) {
    super(message)
  }
}

function serverBase(settings: ExtensionSettings): string {
  return `http://127.0.0.1:${settings.serverPort}/api/v2/browser-extension`
}

function errorResponse(error: unknown): BackgroundResponse<never> {
  if (error instanceof RequestFailure) {
    return {
      ok: false,
      error: {
        code: error.code,
        message: error.message,
        retryable: error.retryable,
      },
    }
  }
  return {
    ok: false,
    error: {
      code: 'extension_error',
      message: error instanceof Error ? error.message : '扩展发生未知错误',
      retryable: true,
    },
  }
}

async function saberRequest<T>(
  path: string,
  init: RequestInit = {},
  settingsOverride?: ExtensionSettings,
  timeoutMs = API_REQUEST_TIMEOUT_MS,
): Promise<T> {
  const settings = settingsOverride ?? await loadSettings()
  if (!settings.token) {
    throw new RequestFailure('not_paired', '请先粘贴 Saber 配对令牌', false)
  }
  const headers = new Headers(init.headers)
  headers.set('Authorization', `Bearer ${settings.token}`)
  if (init.body && !(init.body instanceof FormData) && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }
  let response: Response
  try {
    response = await fetch(`${serverBase(settings)}${path}`, {
      ...init,
      headers,
      cache: 'no-store',
      signal: init.signal ?? AbortSignal.timeout(timeoutMs),
    })
  } catch (error) {
    if (error instanceof DOMException && ['AbortError', 'TimeoutError'].includes(error.name)) {
      throw new RequestFailure(
        'request_timeout',
        `本机 Saber 请求超过 ${Math.round(timeoutMs / 1_000)} 秒未响应`,
        true,
      )
    }
    throw new RequestFailure(
      'saber_unreachable',
      `无法连接本机 Saber（端口 ${settings.serverPort}）`,
      true,
    )
  }
  if (!response.ok) {
    let code = `http_${response.status}`
    let message = `Saber 请求失败（${response.status}）`
    let retryable = response.status >= 500 || response.status === 409
    try {
      const body = await response.json() as {
        error?: {
          code?: string
          message?: string
          retryable?: boolean
          details?: { retryable?: boolean }
        }
      }
      code = body.error?.code || code
      message = body.error?.message || message
      message = FRIENDLY_API_ERRORS[code] ?? message
      retryable = body.error?.retryable
        ?? body.error?.details?.retryable
        ?? retryable
    } catch {
      // Keep the status-derived error when a proxy returned non-JSON content.
    }
    throw new RequestFailure(code, message, retryable)
  }
  if (response.status === 204) return undefined as T
  return await response.json() as T
}

async function sourceBlob(source: UploadPageRequest['source']): Promise<Blob> {
  if (source.kind === 'data-url') {
    try {
      const response = await fetch(source.value, {
        signal: AbortSignal.timeout(IMAGE_TRANSFER_TIMEOUT_MS),
      })
      return await response.blob()
    } catch (error) {
      if (error instanceof DOMException && ['AbortError', 'TimeoutError'].includes(error.name)) {
        throw new RequestFailure('source_timeout', '页面图片读取超时', true)
      }
      throw new RequestFailure('image_decode_failed', '无法读取页面中的图片数据', false)
    }
  }
  let response: Response
  try {
    response = await fetch(source.value, {
      credentials: 'include',
      cache: 'force-cache',
      referrer: '',
      referrerPolicy: 'no-referrer',
      signal: AbortSignal.timeout(IMAGE_TRANSFER_TIMEOUT_MS),
    })
  } catch (error) {
    if (error instanceof DOMException && ['AbortError', 'TimeoutError'].includes(error.name)) {
      throw new RequestFailure('source_timeout', '原始漫画图片下载超时', true)
    }
    throw new RequestFailure('source_fetch_failed', '无法下载原始漫画图片', true)
  }
  if (!response.ok) {
    const protectedSource = response.status === 401
      || response.status === 403
      || response.status === 429
    throw new RequestFailure(
      protectedSource ? 'source_forbidden' : 'source_fetch_failed',
      protectedSource
        ? `图片源拒绝访问（${response.status}），可能需要登录或通过 Cloudflare 验证`
        : `图片下载失败（${response.status}）`,
      response.status >= 500 || response.status === 429,
    )
  }
  const blob = await response.blob()
  if (blob.type.startsWith('text/') || blob.type === 'application/json') {
    throw new RequestFailure(
      'source_forbidden',
      '图片地址返回的不是图片，可能需要登录或通过 Cloudflare 验证',
      false,
    )
  }
  return blob
}

function filenameFor(request: UploadPageRequest, blob: Blob): string {
  const fromPath = request.logicalPath.split('/').at(-1)?.trim()
  if (fromPath && /\.[a-z0-9]{1,8}$/i.test(fromPath)) return fromPath
  const extension = {
    'image/jpeg': 'jpg',
    'image/png': 'png',
    'image/webp': 'webp',
    'image/gif': 'gif',
    'image/bmp': 'bmp',
    'image/tiff': 'tiff',
  }[blob.type] ?? 'png'
  return `${fromPath || 'page'}.${extension}`
}

async function uploadPage(payload: UploadPageRequest): Promise<BrowserPageDto> {
  const blob = await sourceBlob(payload.source)
  if (!blob.size) {
    throw new RequestFailure('empty_image', '原始图片为空', false)
  }
  const form = new FormData()
  form.set('clientPageKey', payload.clientPageKey)
  form.set('ordinal', String(payload.ordinal))
  form.set('logicalPath', payload.logicalPath)
  if (payload.sourceUrl) form.set('sourceUrl', payload.sourceUrl)
  form.set('file', blob, filenameFor(payload, blob))
  return await saberRequest<BrowserPageDto>(
    `/sessions/${encodeURIComponent(payload.sessionId)}/pages`,
    {
      method: 'POST',
      body: form,
    },
    undefined,
    IMAGE_TRANSFER_TIMEOUT_MS,
  )
}

function bytesToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer)
  const chunks: string[] = []
  for (let offset = 0; offset < bytes.length; offset += 0x8000) {
    chunks.push(String.fromCharCode(...bytes.subarray(offset, offset + 0x8000)))
  }
  return btoa(chunks.join(''))
}

async function fetchResultImage(
  sessionId: string,
  browserPageId: string,
): Promise<ResultImagePayload> {
  const settings = await loadSettings()
  const result = await saberRequest<{ url: string; assetId: string; expiresAt: number }>(
    `/sessions/${encodeURIComponent(sessionId)}/pages/`
      + `${encodeURIComponent(browserPageId)}/result-capability`,
    { method: 'POST' },
    settings,
  )
  let response: Response
  try {
    response = await fetch(new URL(result.url, serverBase(settings)), {
      cache: 'no-store',
      referrer: '',
      referrerPolicy: 'no-referrer',
      signal: AbortSignal.timeout(API_REQUEST_TIMEOUT_MS),
    })
  } catch (error) {
    if (error instanceof DOMException && ['AbortError', 'TimeoutError'].includes(error.name)) {
      throw new RequestFailure('request_timeout', '读取 Saber 译图超时', true)
    }
    throw new RequestFailure('result_fetch_failed', '无法读取 Saber 译图', true)
  }
  if (!response.ok) {
    throw new RequestFailure(
      response.status === 403 ? 'result_expired' : 'result_fetch_failed',
      response.status === 403 ? '译图访问凭证已过期' : `译图读取失败（${response.status}）`,
      true,
    )
  }
  const mimeType = response.headers.get('Content-Type')?.split(';', 1)[0]?.trim() ?? ''
  if (!mimeType.startsWith('image/')) {
    throw new RequestFailure('invalid_result_image', 'Saber 返回的译图格式无效', false)
  }
  const contentLength = Number(response.headers.get('Content-Length') ?? 0)
  if (contentLength > MAX_RESULT_BYTES) {
    throw new RequestFailure('result_too_large', '译图过大，无法传回当前网页', false)
  }
  const buffer = await response.arrayBuffer()
  if (buffer.byteLength > MAX_RESULT_BYTES) {
    throw new RequestFailure('result_too_large', '译图过大，无法传回当前网页', false)
  }
  return {
    base64: bytesToBase64(buffer),
    mimeType,
  }
}

async function activeTab(): Promise<chrome.tabs.Tab | undefined> {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true })
  return tab
}

function hostnameFromUrl(url?: string): string {
  if (!url) return ''
  try {
    const parsed = new URL(url)
    return parsed.protocol === 'http:' || parsed.protocol === 'https:'
      ? parsed.hostname
      : ''
  } catch {
    return ''
  }
}

async function updateContextMenu(tab?: chrome.tabs.Tab): Promise<void> {
  const selected = tab ?? await activeTab()
  const hostname = hostnameFromUrl(selected?.url)
  const settings = await loadSettings()
  const disabled = hostname ? preferenceFor(settings, hostname).disabled : true
  try {
    await chrome.contextMenus.update(CONTEXT_MENU_ID, { enabled: !disabled })
  } catch {
    // The menu may not exist during the first service-worker startup.
  }
}

chrome.runtime.onInstalled.addListener(() => {
  void chrome.contextMenus.removeAll().then(() => chrome.contextMenus.create({
    id: CONTEXT_MENU_ID,
    title: '使用 Saber 翻译此图片',
    contexts: ['image'],
    documentUrlPatterns: ['http://*/*', 'https://*/*'],
  }))
})

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId !== CONTEXT_MENU_ID || !tab?.id || !info.srcUrl) return
  const message: ContextTranslateMessage = {
    type: 'context-translate-image',
    srcUrl: info.srcUrl,
  }
  void chrome.tabs.sendMessage(tab.id, message).catch(() => undefined)
})

chrome.tabs.onActivated.addListener(() => {
  void updateContextMenu()
})

chrome.tabs.onUpdated.addListener((_tabId, changeInfo, tab) => {
  if (tab.active && (changeInfo.url || changeInfo.status === 'complete')) {
    void updateContextMenu(tab)
  }
})

chrome.tabs.onRemoved.addListener(tabId => {
  void serializeStorageWrite(() => chrome.storage.session.remove(`${ACTIVE_SESSION_KEY_PREFIX}${tabId}`))
})

function normalizedContentPageUrl(value: string): string {
  const url = new URL(value)
  if (!['http:', 'https:'].includes(url.protocol)) {
    throw new RequestFailure('invalid_page_url', '网页地址必须使用 HTTP(S)', false)
  }
  return normalizedTaskPageUrl(url.toString())
}

function contentTabId(sender: chrome.runtime.MessageSender, pageUrl: string): number {
  const tabId = sender.tab?.id
  const senderUrl = sender.url ?? sender.tab?.url
  if (tabId === undefined || !senderUrl) {
    throw new RequestFailure('content_tab_required', '该操作只能从网页标签页执行', false)
  }
  const normalizedPageUrl = normalizedContentPageUrl(pageUrl)
  if (
    normalizedContentPageUrl(senderUrl) !== normalizedPageUrl
    || (
      sender.tab?.url
      && normalizedContentPageUrl(sender.tab.url) !== normalizedPageUrl
    )
  ) {
    throw new RequestFailure('stale_page_context', '网页已经切换，忽略过期会话操作', false)
  }
  return tabId
}

async function activeSessionForTab(
  sender: chrome.runtime.MessageSender,
  pageUrl: string,
): Promise<ActiveBrowserSession | null> {
  const tabId = contentTabId(sender, pageUrl)
  const key = `${ACTIVE_SESSION_KEY_PREFIX}${tabId}`
  const stored = (await chrome.storage.session.get(key))[key]
  if (!stored || typeof stored !== 'object' || Array.isArray(stored)) return null
  const value = stored as Partial<ActiveBrowserSession> & { pageUrl?: unknown }
  if (
    typeof value.pageUrl !== 'string'
    || typeof value.sessionId !== 'string'
    || !value.sessionId
    || normalizedContentPageUrl(value.pageUrl) !== normalizedContentPageUrl(pageUrl)
  ) {
    await chrome.storage.session.remove(key)
    return null
  }
  return {
    sessionId: value.sessionId,
    discovery: value.discovery ?? { stopped: true, usingAdapter: false, rule: null },
  }
}

async function handleRequest(
  request: BackgroundRequest,
  sender: chrome.runtime.MessageSender,
): Promise<unknown> {
  if (request.type === 'get-preference') {
    const settings = await loadSettings()
    return preferenceFor(settings, request.hostname)
  }
  if (request.type === 'set-preference') {
    await updateSettings(settings => {
      settings.domains[request.hostname] = request.preference
    })
    await updateContextMenu()
    return request.preference
  }
  if (request.type === 'get-popup-state') {
    const settings = await loadSettings()
    const tab = await activeTab()
    const hostname = hostnameFromUrl(tab?.url)
    return {
      token: settings.token,
      serverPort: settings.serverPort,
      hostname,
      preference: hostname ? preferenceFor(settings, hostname) : null,
    }
  }
  if (request.type === 'save-connection') {
    if (!Number.isInteger(request.serverPort)
      || request.serverPort < 1
      || request.serverPort > 65535) {
      throw new RequestFailure('invalid_port', '端口必须在 1 到 65535 之间', false)
    }
    const token = request.token.trim()
    if (token && (token.length < 32 || token.length > 200)) {
      throw new RequestFailure(
        'invalid_token_format',
        '配对令牌格式无效，请从 Saber GUI 重新复制',
        false,
      )
    }
    await updateSettings(settings => {
      settings.token = token
      settings.serverPort = request.serverPort
    })
    return { saved: true }
  }
  if (request.type === 'status') {
    return await saberRequest<Record<string, unknown>>('/status')
  }
  if (request.type === 'hash-source') {
    const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(request.value))
    return [...new Uint8Array(digest)].map(byte => byte.toString(16).padStart(2, '0')).join('')
  }
  if (request.type === 'get-active-session') {
    return await serializeStorageWrite(() => activeSessionForTab(sender, request.pageUrl))
  }
  if (request.type === 'set-active-session') {
    const tabId = contentTabId(sender, request.pageUrl)
    if (!request.sessionId) {
      throw new RequestFailure('invalid_session_id', '网页会话 ID 无效', false)
    }
    await serializeStorageWrite(() => chrome.storage.session.set({
      [`${ACTIVE_SESSION_KEY_PREFIX}${tabId}`]: {
        pageUrl: normalizedContentPageUrl(request.pageUrl),
        sessionId: request.sessionId,
        discovery: request.discovery,
      },
    }))
    return { saved: true }
  }
  if (request.type === 'clear-active-session') {
    const tabId = contentTabId(sender, request.pageUrl)
    const key = `${ACTIVE_SESSION_KEY_PREFIX}${tabId}`
    return await serializeStorageWrite(async () => {
      if (request.sessionId) {
        const active = await activeSessionForTab(sender, request.pageUrl)
        if (active?.sessionId !== request.sessionId) return { cleared: false }
      }
      await chrome.storage.session.remove(key)
      return { cleared: true }
    })
  }
  if (request.type === 'create-session') {
    return await saberRequest<BrowserSessionDto>('/sessions', {
      method: 'POST',
      body: JSON.stringify(request.payload),
    })
  }
  if (request.type === 'get-session') {
    return await saberRequest<BrowserSessionDto>(
      `/sessions/${encodeURIComponent(request.sessionId)}${request.touch ? '?touch=true' : ''}`,
    )
  }
  if (request.type === 'patch-session') {
    return await saberRequest<BrowserSessionDto>(
      `/sessions/${encodeURIComponent(request.sessionId)}`,
      { method: 'PATCH', body: JSON.stringify(request.payload) },
    )
  }
  if (request.type === 'start-session') {
    return await saberRequest<BrowserSessionDto>(
      `/sessions/${encodeURIComponent(request.sessionId)}/start`,
      { method: 'POST' },
    )
  }
  if (request.type === 'upload-page') return await uploadPage(request.payload)
  if (request.type === 'retry-page') {
    return await saberRequest<BrowserPageDto>(
      `/sessions/${encodeURIComponent(request.sessionId)}/pages/`
        + `${encodeURIComponent(request.browserPageId)}/retry`,
      { method: 'POST' },
    )
  }
  if (request.type === 'fetch-result') {
    return await fetchResultImage(request.sessionId, request.browserPageId)
  }
  if (request.type === 'get-terms') {
    return await saberRequest<Record<string, unknown>>(
      `/sessions/${encodeURIComponent(request.sessionId)}/terms`,
    )
  }
  if (request.type === 'cancel-session') {
    return await saberRequest<BrowserSessionDto>(
      `/sessions/${encodeURIComponent(request.sessionId)}/cancel`,
      { method: 'POST' },
    )
  }
  if (request.type === 'list-library-books') {
    return await saberRequest<{ items: BrowserLibraryBook[] }>('/library-books')
  }
  if (request.type === 'import-session') {
    return await saberRequest<BrowserSessionImportResult>(
      `/sessions/${encodeURIComponent(request.sessionId)}/import`,
      { method: 'POST', body: JSON.stringify(request.payload) },
    )
  }
  if (request.type === 'dom-detection') {
    return await saberRequest<DomDetectionResult>('/dom-detection', {
      method: 'POST',
      body: JSON.stringify(request.payload),
    }, undefined, IMAGE_TRANSFER_TIMEOUT_MS)
  }
  throw new RequestFailure('unknown_message', '不支持的扩展请求', false)
}

chrome.runtime.onMessage.addListener((message: unknown, sender, sendResponse) => {
  if (sender.id !== chrome.runtime.id || !message || typeof message !== 'object') {
    return false
  }
  const request = message as BackgroundRequest
  if (
    ['get-popup-state', 'save-connection'].includes(request.type)
    && !sender.url?.startsWith(`chrome-extension://${chrome.runtime.id}/`)
  ) {
    sendResponse(errorResponse(new RequestFailure(
      'extension_page_required',
      '该操作只能从扩展弹窗执行',
      false,
    )))
    return false
  }
  void handleRequest(request, sender)
    .then((data) => {
      const response: BackgroundResponse<unknown> = { ok: true, data }
      sendResponse(response)
    })
    .catch((error) => sendResponse(errorResponse(error)))
  return true
})
