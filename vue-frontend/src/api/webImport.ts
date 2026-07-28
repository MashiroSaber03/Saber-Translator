import { apiClient } from '@/api/client'
import { readApiErrorMessage } from '@/api/download'
import { readSseStream } from '@/api/sse'
import { runV2ConnectionTest } from '@/api/v2/settings'
import type {
  AgentLog,
  ComicPage,
  DownloadResult,
  ExtractResult,
  GalleryDLSupportResult,
  WebImportEngine,
  WebImportSettings,
  WebImportSettingsPayload,
} from '@/types/webImport'

const API_BASE = '/api/web-import'

function webImportEndpoint(path: string): string {
  return `${API_BASE}${path}`
}

export interface WebImportSettingsResponse {
  success: boolean
  hasStoredSettings?: boolean
  settings?: unknown
  providerConfigs?: unknown
  error?: string
}

export async function checkGalleryDLSupport(url: string): Promise<GalleryDLSupportResult> {
  return apiClient.get<GalleryDLSupportResult>(webImportEndpoint('/check-support'), {
    params: { url },
  })
}

export function getProxyImageUrl(imageUrl: string, referer?: string): string {
  const params = new URLSearchParams({ url: imageUrl })
  if (referer) {
    params.set('referer', referer)
  }
  return `${webImportEndpoint('/proxy-image')}?${params.toString()}`
}

export async function getGalleryDLImages(): Promise<{
  success: boolean
  images: Array<{ filename: string; data: string }>
  total: number
  error?: string
}> {
  return apiClient.get(webImportEndpoint('/gallery-dl-images'))
}

export async function getWebImportSettings(): Promise<WebImportSettingsResponse> {
  return apiClient.get<WebImportSettingsResponse>(webImportEndpoint('/settings'))
}

export async function saveWebImportSettings(payload: WebImportSettingsPayload): Promise<{ success: boolean; error?: string }> {
  return apiClient.post<{ success: boolean; error?: string }>(webImportEndpoint('/settings'), payload)
}

export async function extractImages(
  url: string,
  config: WebImportSettings,
  onLog: (log: AgentLog) => void,
  onResult: (result: ExtractResult) => void,
  onError: (error: string) => void,
  engine: WebImportEngine = 'auto',
  onPage?: (page: ComicPage) => void
): Promise<void> {
  const response = await fetch(webImportEndpoint('/extract'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ url, config, engine })
  })

  if (!response.ok) {
    onError(await readApiErrorMessage(response, `HTTP ${response.status}`))
    return
  }

  try {
    await readSseStream<Record<string, unknown>>(response, {
      missingBodyMessage: '无法读取响应流',
      parseErrorMessage: '解析网页导入事件失败',
      onMessage(message) {
        if (message.event === 'log') {
          onLog(message.data as AgentLog)
        } else if (message.event === 'page') {
          onPage?.(message.data as ComicPage)
        } else if (message.event === 'result') {
          onResult(message.data as ExtractResult)
        } else if (message.event === 'error') {
          onError(typeof message.data.error === 'string' ? message.data.error : '未知错误')
        }
      },
    })
  } catch (error) {
    onError(error instanceof Error ? error.message : '解析网页导入事件失败')
  }
}

export async function downloadImages(
  pages: ComicPage[],
  sourceUrl: string,
  config: WebImportSettings,
  engine: WebImportEngine = 'ai-agent'
): Promise<DownloadResult> {
  return apiClient.post<DownloadResult>(webImportEndpoint('/download'), {
    pages,
    sourceUrl,
    config,
    engine,
  })
}

export async function testFirecrawlConnection(apiKey: string): Promise<{ success: boolean; message?: string; error?: string }> {
  return runV2ConnectionTest('firecrawl', apiKey
    ? { secret: { apiKey } }
    : { domain: 'web_import_firecrawl' })
}

export async function testAgentConnection(
  provider: string,
  apiKey: string,
  customBaseUrl: string,
  modelName: string
): Promise<{ success: boolean; message?: string; error?: string }> {
  return runV2ConnectionTest('web_import_agent', {
    provider,
    baseUrl: customBaseUrl || undefined,
    model: modelName,
    ...(apiKey
      ? { secret: { apiKey } }
      : { domain: 'web_import_agent' }),
  })
}
