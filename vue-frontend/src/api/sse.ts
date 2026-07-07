export interface SseMessage<T = unknown> {
  event: string
  data: T
}

export interface ReadSseStreamOptions<T = unknown> {
  onMessage: (message: SseMessage<T>) => void
  missingBodyMessage?: string
  parseErrorMessage?: string
}

export async function readSseStream<T = unknown>(
  response: Response,
  options: ReadSseStreamOptions<T>,
): Promise<void> {
  const reader = response.body?.getReader()
  if (!reader) {
    throw new Error(options.missingBodyMessage || '无法读取响应流')
  }

  const decoder = new TextDecoder()
  let buffer = ''
  let eventType = ''
  let eventDataLines: string[] = []

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const rawLine of lines) {
        const line = rawLine.trimEnd()
        if (line.startsWith('event:')) {
          eventType = line.slice(6).trim()
        } else if (line.startsWith('data:')) {
          eventDataLines.push(line.slice(5).replace(/^ /, ''))
        } else if (line === '' && eventType && eventDataLines.length > 0) {
          const event = eventType
          const eventData = eventDataLines.join('\n')
          let data: T
          try {
            data = JSON.parse(eventData) as T
          } catch {
            throw new Error(options.parseErrorMessage || '解析事件流失败')
          } finally {
            eventType = ''
            eventDataLines = []
          }
          options.onMessage({ event, data })
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}
