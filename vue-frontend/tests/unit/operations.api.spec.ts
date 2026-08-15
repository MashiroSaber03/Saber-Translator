import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { accessGateMock, getMock, postMock, uploadMock } = vi.hoisted(() => ({
  accessGateMock: vi.fn(),
  getMock: vi.fn(),
  postMock: vi.fn(),
  uploadMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  apiClient: {
    get: getMock,
    post: postMock,
    upload: uploadMock,
  },
}))

vi.mock('@/services/backendAccessGate', () => ({
  assertBackendActionAllowed: accessGateMock,
}))

vi.mock('@/api/v2/content', () => ({
  newIdempotencyKey: () => 'idempotency-key',
}))

import {
  createMaskRepair,
  createPageOperation,
  getOperation,
  waitForOperation,
} from '@/api/v2/operations'

const CREATED_AT = '2026-08-11T08:00:00Z'
const STARTED_AT = '2026-08-11T08:00:01Z'
const FINISHED_AT = '2026-08-11T08:00:02Z'

function operation(overrides: Record<string, unknown> = {}) {
  return {
    operationId: 'operation-1',
    kind: 'page_detect',
    executorRole: 'worker',
    status: 'completed',
    pageId: 'page-1',
    bubbleId: null,
    studioDocumentId: null,
    studioSessionId: null,
    baseRevision: 1,
    baseGeneration: null,
    request: {},
    result: { bubbleCount: 1 },
    error: null,
    createdAt: CREATED_AT,
    startedAt: STARTED_AT,
    finishedAt: FINISHED_AT,
    ...overrides,
  }
}

function event(eventId: number, overrides: Record<string, unknown> = {}) {
  return {
    eventId,
    operationId: 'operation-1',
    type: 'operation_progress',
    payload: { completed: eventId },
    createdAt: CREATED_AT,
    ...overrides,
  }
}

function sseResponse(messages: Array<{ data: unknown; event: string; id?: string }>): Response {
  const encoder = new TextEncoder()
  const body = messages
    .map(message => `${message.id ? `id: ${message.id}\n` : ''}event: ${message.event}\ndata: ${JSON.stringify(message.data)}\n\n`)
    .join('')
  return new Response(new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encoder.encode(body))
      controller.close()
    },
  }), {
    status: 200,
    headers: { 'Content-Type': 'text/event-stream; charset=utf-8' },
  })
}

describe('v2 operation API', () => {
  beforeEach(() => {
    getMock.mockReset()
    postMock.mockReset()
    uploadMock.mockReset()
    accessGateMock.mockReset()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('parses the exact accepted-operation contract for page operations and repairs', async () => {
    postMock.mockResolvedValue({
      operationId: 'operation-1',
      kind: 'page_detect',
      status: 'pending',
      executorRole: 'worker',
    })
    uploadMock.mockResolvedValue({
      operationId: 'operation-2',
      kind: 'page_repair',
      status: 'pending',
      executorRole: 'api',
      documentRevision: 2,
    })

    await expect(createPageOperation('page/1', {
      kind: 'page_detect',
      baseRevision: 1,
    })).resolves.toMatchObject({ operationId: 'operation-1', status: 'pending' })
    await expect(createMaskRepair('page/1', new Blob(['mask']), {
      baseRevision: 1,
      method: 'restore_source',
    })).resolves.toMatchObject({
      operationId: 'operation-2',
      documentRevision: 2,
    })

    expect(postMock).toHaveBeenCalledWith(
      '/api/v2/pages/page%2F1/operations',
      { kind: 'page_detect', baseRevision: 1 },
      { headers: { 'Idempotency-Key': 'idempotency-key' } },
    )
    expect(accessGateMock).toHaveBeenCalledTimes(2)
  })

  it('rejects extra accepted fields and operation identity mismatches', async () => {
    postMock.mockResolvedValue({
      operationId: 'operation-1',
      kind: 'page_detect',
      status: 'pending',
      executorRole: 'worker',
      legacyStatus: 'queued',
    })
    await expect(createPageOperation('page-1', {
      kind: 'page_detect',
      baseRevision: 1,
    })).rejects.toThrow('后端操作受理结果字段无效')

    postMock.mockResolvedValue({
      operationId: 'operation-1',
      kind: 'bubble_ocr',
      status: 'pending',
      executorRole: 'worker',
    })
    await expect(createPageOperation('page-1', {
      kind: 'page_detect',
      baseRevision: 1,
    })).rejects.toThrow('后端操作受理类型不匹配')

    getMock.mockResolvedValue(operation({ operationId: 'operation-2' }))
    await expect(getOperation('operation-1')).rejects.toThrow('后端操作身份不匹配')
  })

  it('does not submit a repair when backend actions are unavailable', async () => {
    accessGateMock.mockImplementationOnce(() => {
      throw new Error('后端设置尚未就绪')
    })

    await expect(createMaskRepair('page-1', new Blob(['mask']), {
      baseRevision: 1,
      method: 'restore_source',
    })).rejects.toThrow('后端设置尚未就绪')

    expect(uploadMock).not.toHaveBeenCalled()
  })

  it('submits fill color only for solid repair', async () => {
    uploadMock.mockResolvedValue({
      operationId: 'operation-2',
      kind: 'page_repair',
      status: 'pending',
      executorRole: 'api',
      documentRevision: 2,
    })

    await createMaskRepair('page-1', new Blob(['mask']), {
      baseRevision: 1,
      fillColor: '#112233',
      method: 'solid',
    })
    await createMaskRepair('page-1', new Blob(['mask']), {
      baseRevision: 2,
      method: 'lama_mpe',
    })

    const solid = uploadMock.mock.calls[0][1] as FormData
    const lama = uploadMock.mock.calls[1][1] as FormData
    expect(solid.get('fill_color')).toBe('#112233')
    expect(lama.has('fill_color')).toBe(false)
  })

  it('validates SSE event identity and returns only a completed durable operation', async () => {
    const payload = event(1, { type: 'operation_completed' })
    const fetchMock = vi.fn().mockResolvedValue(sseResponse([
      { id: '1', event: 'operation_completed', data: payload },
    ]))
    vi.stubGlobal('fetch', fetchMock)
    getMock.mockResolvedValue(operation())

    await expect(waitForOperation('operation-1')).resolves.toMatchObject({
      operationId: 'operation-1',
      status: 'completed',
    })
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v2/operations/operation-1/events?stream=1',
      { headers: { Accept: 'text/event-stream' }, signal: undefined },
    )
  })

  it('rejects mislabeled or cross-operation SSE events before reading final state', async () => {
    const fetchMock = vi.fn().mockResolvedValue(sseResponse([
      {
        id: '1',
        event: 'operation_progress',
        data: event(1, { operationId: 'operation-2' }),
      },
    ]))
    vi.stubGlobal('fetch', fetchMock)

    await expect(waitForOperation('operation-1')).rejects.toThrow('后端操作事件身份不匹配')
    expect(getMock).not.toHaveBeenCalled()
  })

  it('resumes a prematurely closed event stream from the last durable event', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(sseResponse([
        { id: '4', event: 'operation_progress', data: event(4) },
      ]))
      .mockResolvedValueOnce(sseResponse([
        { id: '5', event: 'operation_completed', data: event(5, { type: 'operation_completed' }) },
      ]))
    vi.stubGlobal('fetch', fetchMock)
    getMock
      .mockResolvedValueOnce(operation({
        status: 'running',
        result: null,
        error: null,
        finishedAt: null,
      }))
      .mockResolvedValueOnce(operation())

    await expect(waitForOperation('operation-1')).resolves.toMatchObject({ status: 'completed' })
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      '/api/v2/operations/operation-1/events?stream=1',
      {
        headers: { Accept: 'text/event-stream', 'Last-Event-ID': '4' },
        signal: undefined,
      },
    )
  })

  it('surfaces durable operation failures and rejects non-SSE responses', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(sseResponse([])))
    getMock.mockResolvedValue(operation({
      status: 'failed',
      result: null,
      error: { code: 'OPERATION_FAILED', message: '检测失败' },
    }))
    await expect(waitForOperation('operation-1')).rejects.toThrow('检测失败')

    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{}', {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    })))
    await expect(waitForOperation('operation-1')).rejects.toThrow('未返回事件流')
  })

  it('rejects an SSE cursor that disagrees with the durable event id', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(sseResponse([
      { id: '2', event: 'operation_progress', data: event(1) },
    ])))

    await expect(waitForOperation('operation-1')).rejects.toThrow('后端操作事件游标不一致')
    expect(getMock).not.toHaveBeenCalled()
  })

  it('rejects impossible durable status payloads', async () => {
    getMock.mockResolvedValue(operation({ status: 'running' }))
    await expect(getOperation('operation-1')).rejects.toThrow('终态与 finishedAt 不一致')
  })
})
