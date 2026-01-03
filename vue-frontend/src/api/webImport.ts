/**
 * 网页导入 API
 */

import type { WebImportSettings, ExtractResult, DownloadResult, AgentLog, ComicPage, WebImportEngine, GalleryDLSupportResult } from '@/types/webImport'

const API_BASE = '/api/web-import'

/**
 * 检查 URL 是否支持 gallery-dl
 */
export async function checkGalleryDLSupport(url: string): Promise<GalleryDLSupportResult> {
  const response = await fetch(`${API_BASE}/check-support?url=${encodeURIComponent(url)}`)
  return response.json()
}

/**
 * 获取代理图片 URL
 */
export function getProxyImageUrl(imageUrl: string, referer?: string): string {
  const params = new URLSearchParams({ url: imageUrl })
  if (referer) {
    params.set('referer', referer)
  }
  return `${API_BASE}/proxy-image?${params.toString()}`
}

/**
 * 获取 gallery-dl 临时目录中的图片（base64 格式）
 */
export async function getGalleryDLImages(): Promise<{
  success: boolean
  images: Array<{ filename: string; data: string }>
  total: number
  error?: string
}> {
  const response = await fetch(`${API_BASE}/gallery-dl-images`)
  return response.json()
}

/**
 * 提取漫画图片 (SSE 流式)
 */
export async function extractImages(
  url: string,
  config: WebImportSettings,
  onLog: (log: AgentLog) => void,
  onResult: (result: ExtractResult) => void,
  onError: (error: string) => void,
  engine: WebImportEngine = 'auto'
): Promise<void> {
  const response = await fetch(`${API_BASE}/extract`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ url, config, engine })
  })

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ error: '请求失败' }))
    onError(errorData.error || `HTTP ${response.status}`)
    return
  }

  const reader = response.body?.getReader()
  if (!reader) {
    onError('无法读取响应流')
    return
  }

  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })

      // 解析 SSE 事件
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      let eventType = ''
      let eventData = ''

      for (const line of lines) {
        if (line.startsWith('event:')) {
          eventType = line.slice(6).trim()
        } else if (line.startsWith('data:')) {
          eventData = line.slice(5).trim()
        } else if (line === '' && eventType && eventData) {
          // 处理完整事件
          try {
            const data = JSON.parse(eventData)
            if (eventType === 'log') {
              onLog(data as AgentLog)
            } else if (eventType === 'result') {
              onResult(data as ExtractResult)
            } else if (eventType === 'error') {
              onError(data.error || '未知错误')
            }
          } catch (e) {
            console.error('解析 SSE 数据失败:', e)
          }
          eventType = ''
          eventData = ''
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}

/**
 * 下载图片
 */
export async function downloadImages(
  pages: ComicPage[],
  sourceUrl: string,
  config: WebImportSettings,
  engine: WebImportEngine = 'ai-agent'
): Promise<DownloadResult> {
  const response = await fetch(`${API_BASE}/download`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ pages, sourceUrl, config, engine })
  })

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ error: '请求失败' }))
    throw new Error(errorData.error || `HTTP ${response.status}`)
  }

  return response.json()
}

/**
 * 测试 Firecrawl 连接
 */
export async function testFirecrawlConnection(apiKey: string): Promise<{ success: boolean; message?: string; error?: string }> {
  const response = await fetch(`${API_BASE}/test-firecrawl`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ apiKey })
  })

  return response.json()
}

/**
 * 测试 AI Agent 连接
 */
export async function testAgentConnection(
  provider: string,
  apiKey: string,
  customBaseUrl: string,
  modelName: string
): Promise<{ success: boolean; message?: string; error?: string }> {
  const response = await fetch(`${API_BASE}/test-agent`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ provider, apiKey, customBaseUrl, modelName })
  })

  return response.json()
}
