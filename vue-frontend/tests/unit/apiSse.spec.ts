import { describe, expect, it, vi } from 'vitest'

function streamFromChunks(chunks: string[]) {
  const encoder = new TextEncoder()
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk))
      }
      controller.close()
    },
  })
}

describe('api SSE helpers', () => {
  it('reads event/data messages across chunks', async () => {
    const { readSseStream } = await import('@/api/sse')
    const onMessage = vi.fn()

    await readSseStream(new Response(streamFromChunks([
      'id: 17\nevent: log\n',
      'data: {"message":"hel',
      'lo"}\n\n',
      'event: done\ndata: {"ok":true}\n\n',
    ])), { onMessage })

    expect(onMessage).toHaveBeenCalledWith({ id: '17', event: 'log', data: { message: 'hello' } })
    expect(onMessage).toHaveBeenCalledWith({ event: 'done', data: { ok: true } })
  })

  it('joins multi-line data fields before parsing an event', async () => {
    const { readSseStream } = await import('@/api/sse')
    const onMessage = vi.fn()

    await readSseStream(new Response(streamFromChunks([
      'event: log\n',
      'data: {"message":\n',
      'data: "hello"}\n\n',
    ])), { onMessage })

    expect(onMessage).toHaveBeenCalledWith({ event: 'log', data: { message: 'hello' } })
  })

  it('preserves JSON string whitespace and dispatches a final event without a blank line', async () => {
    const { readSseStream } = await import('@/api/sse')
    const onMessage = vi.fn()

    await readSseStream(new Response(streamFromChunks([
      'event: chunk\r\n',
      'data: {"text":"tail  "}',
    ])), { onMessage })

    expect(onMessage).toHaveBeenCalledWith({ event: 'chunk', data: { text: 'tail  ' } })
  })

  it('reports parse errors and missing response bodies', async () => {
    const { readSseStream } = await import('@/api/sse')

    await expect(readSseStream(
      new Response(streamFromChunks(['event: log\ndata: not-json\n\n'])),
      { onMessage: vi.fn(), parseErrorMessage: '解析事件失败' },
    )).rejects.toThrow('解析事件失败')

    await expect(readSseStream(
      new Response(null),
      { onMessage: vi.fn(), missingBodyMessage: '无法读取响应流' },
    )).rejects.toThrow('无法读取响应流')
  })
})
