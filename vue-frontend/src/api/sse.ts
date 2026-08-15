export interface SseMessage<T = unknown> {
  id?: string
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
  let eventId: string | undefined
  let eventType = ''
  let eventDataLines: string[] = []

  const dispatchEvent = (): void => {
    if (!eventType || eventDataLines.length === 0) {
      eventId = undefined
      eventType = ''
      eventDataLines = []
      return
    }
    const event = eventType
    const id = eventId
    const eventData = eventDataLines.join('\n')
    eventId = undefined
    eventType = ''
    eventDataLines = []
    let data: T
    try {
      data = JSON.parse(eventData) as T
    } catch {
      throw new Error(options.parseErrorMessage || '解析事件流失败')
    }
    options.onMessage({ ...(id === undefined ? {} : { id }), event, data })
  }

  const consumeLine = (rawLine: string): void => {
    const line = rawLine.endsWith('\r') ? rawLine.slice(0, -1) : rawLine
    if (line.startsWith('id:')) {
      eventId = line.slice(3).replace(/^ /, '')
    } else if (line.startsWith('event:')) {
      eventType = line.slice(6).trim()
    } else if (line.startsWith('data:')) {
      eventDataLines.push(line.slice(5).replace(/^ /, ''))
    } else if (line === '') {
      dispatchEvent()
    }
  }

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''

      for (const rawLine of lines) {
        consumeLine(rawLine)
      }
    }
    buffer += decoder.decode()
    if (buffer) consumeLine(buffer)
    dispatchEvent()
  } finally {
    reader.releaseLock()
  }
}
