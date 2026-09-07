import { loadSettings } from './storage'
import type { ExtensionSettings } from './types'

export const API_REQUEST_TIMEOUT_MS = 30_000

const FRIENDLY_API_ERRORS: Record<string, string> = {
  integration_disabled: '请先在 Saber GUI 中启用浏览器扩展连接',
  invalid_extension_token: '配对令牌无效，请从 Saber GUI 重新复制',
  loopback_required: 'Saber 只接受本机扩展连接',
  session_conflict: '当前翻译批次尚未结束，请稍后重试',
  revision_conflict: '配置已在其他窗口变化，请重新读取后再修改。',
  result_not_found: '译图尚未生成或网页会话已经过期',
  browser_internal_error: 'Saber 浏览器扩展接口发生内部错误',
  request_timeout: '本机 Saber 请求超时，请确认后端仍在运行',
  source_timeout: '原始漫画图片下载超时，请重试',
}

export class RequestFailure extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly retryable: boolean
  ) {
    super(message)
  }
}

export function serverBase(settings: ExtensionSettings): string {
  return `http://127.0.0.1:${settings.serverPort}/api/v2/browser-extension`
}

export async function saberRequest<T>(
  path: string,
  init: RequestInit = {},
  settingsOverride?: ExtensionSettings,
  timeoutMs = API_REQUEST_TIMEOUT_MS
): Promise<T> {
  const settings = settingsOverride ?? (await loadSettings())
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
      credentials: 'omit',
      signal: init.signal ?? AbortSignal.timeout(timeoutMs),
    })
  } catch (error) {
    if (error instanceof DOMException && ['AbortError', 'TimeoutError'].includes(error.name)) {
      throw new RequestFailure(
        'request_timeout',
        `本机 Saber 请求超过 ${Math.round(timeoutMs / 1_000)} 秒未响应`,
        true
      )
    }
    throw new RequestFailure(
      'saber_unreachable',
      `无法连接本机 Saber（端口 ${settings.serverPort}）`,
      true
    )
  }
  if (!response.ok) {
    let code = `http_${response.status}`
    let message = `Saber 请求失败（${response.status}）`
    let retryable = response.status >= 500 || response.status === 409
    try {
      const body = (await response.json()) as {
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
      retryable = body.error?.retryable ?? body.error?.details?.retryable ?? retryable
    } catch {
      // Keep the status-derived error when a proxy returned non-JSON content.
    }
    throw new RequestFailure(code, message, retryable)
  }
  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}
