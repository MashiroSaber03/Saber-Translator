import { apiClient } from '@/api/client'
import { readApiErrorMessage } from '@/api/download'
import { readSseStream } from '@/api/sse'
import type { components } from '@/api/generated/v2'
import { assertBackendActionAllowed } from '@/services/backendAccessGate'
import { newIdempotencyKey } from './content'

export type OperationStatus = components['schemas']['OperationStatus']
export type PageOperationCommand = components['schemas']['PageOperationCommand']
export type PageOperationKind = PageOperationCommand['kind']
export type V2Operation = components['schemas']['Operation']
export type V2OperationAccepted = components['schemas']['OperationAccepted']
export type V2OperationEvent = components['schemas']['OperationEvent']

const OPERATION_KINDS = new Set([
  'bubble_ocr',
  'bubble_color',
  'page_detect',
  'page_repair',
  'bubble_translate',
  'studio_generate',
  'studio_chat',
  'studio_summary',
])
const OPERATION_STATUSES = new Set<OperationStatus>([
  'pending',
  'running',
  'completed',
  'failed',
  'cancelled',
])
const OPERATION_KEYS = [
  'operationId',
  'kind',
  'executorRole',
  'status',
  'pageId',
  'bubbleId',
  'studioDocumentId',
  'studioSessionId',
  'baseRevision',
  'baseGeneration',
  'request',
  'result',
  'error',
  'createdAt',
  'startedAt',
  'finishedAt',
] as const
const OPERATION_ACCEPTED_KEYS = [
  'operationId',
  'kind',
  'status',
  'executorRole',
  'baseRevision',
  'baseGeneration',
  'sessionRevision',
  'sessionGeneration',
  'userMessageId',
  'documentRevision',
] as const
const OPERATION_EVENT_KEYS = [
  'eventId',
  'operationId',
  'type',
  'payload',
  'createdAt',
] as const

function exactObject(
  value: unknown,
  allowedKeys: readonly string[],
  requiredKeys: readonly string[],
  label: string,
): Record<string, unknown> {
  const record = objectValue(value, label)
  const keys = Object.keys(record)
  if (
    keys.some(key => !allowedKeys.includes(key))
    || requiredKeys.some(key => !Object.prototype.hasOwnProperty.call(record, key))
  ) {
    throw new Error(`${label}字段无效`)
  }
  return record
}

function objectValue(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label}必须是对象`)
  }
  return value as Record<string, unknown>
}

function stringValue(value: unknown, label: string): string {
  if (typeof value !== 'string') {
    throw new Error(`${label}必须是字符串`)
  }
  return value
}

function nonEmptyString(value: unknown, label: string): string {
  const text = stringValue(value, label)
  if (text.length === 0) throw new Error(`${label}不能为空`)
  return text
}

function nullableId(value: unknown, label: string): string | null {
  return value === null ? null : nonEmptyString(value, label)
}

function positiveInteger(value: unknown, label: string): number {
  if (!Number.isInteger(value) || (value as number) < 1) {
    throw new Error(`${label}必须是正整数`)
  }
  return value as number
}

function nullablePositiveInteger(value: unknown, label: string): number | null {
  return value === null ? null : positiveInteger(value, label)
}

function optionalPositiveInteger(value: unknown, label: string): number | undefined {
  return value === undefined ? undefined : positiveInteger(value, label)
}

function dateValue(value: unknown, label: string): string {
  const date = nonEmptyString(value, label)
  if (Number.isNaN(Date.parse(date))) {
    throw new Error(`${label}必须是有效时间`)
  }
  return date
}

function nullableDate(value: unknown, label: string): string | null {
  return value === null ? null : dateValue(value, label)
}

function operationKind(value: unknown, label: string): V2Operation['kind'] {
  const kind = nonEmptyString(value, label)
  if (!OPERATION_KINDS.has(kind)) {
    throw new Error(`${label}无效`)
  }
  return kind as V2Operation['kind']
}

function operationStatus(value: unknown, label: string): OperationStatus {
  if (typeof value !== 'string' || !OPERATION_STATUSES.has(value as OperationStatus)) {
    throw new Error(`${label}无效`)
  }
  return value as OperationStatus
}

function executorRole(value: unknown, label: string): V2Operation['executorRole'] {
  if (value !== 'api' && value !== 'worker') {
    throw new Error(`${label}无效`)
  }
  return value
}

function operationError(value: unknown): V2Operation['error'] {
  if (value === null) return null
  const error = exactObject(value, ['code', 'message'], ['code', 'message'], '后端操作错误')
  return {
    code: nonEmptyString(error.code, '后端操作错误.code'),
    message: stringValue(error.message, '后端操作错误.message'),
  }
}

function operationResult(value: unknown): Record<string, unknown> | null {
  if (value === null) return null
  return objectValue(value, '后端操作结果')
}

export function parseOperation(value: unknown, expectedOperationId?: string): V2Operation {
  const payload = exactObject(value, OPERATION_KEYS, OPERATION_KEYS, '后端操作')
  const operationId = nonEmptyString(payload.operationId, '后端操作.operationId')
  if (expectedOperationId !== undefined && operationId !== expectedOperationId) {
    throw new Error(`后端操作身份不匹配：期望 ${expectedOperationId}，实际 ${operationId}`)
  }
  const status = operationStatus(payload.status, '后端操作.status')
  const result = operationResult(payload.result)
  const error = operationError(payload.error)
  const startedAt = nullableDate(payload.startedAt, '后端操作.startedAt')
  const finishedAt = nullableDate(payload.finishedAt, '后端操作.finishedAt')
  const isTerminal = status === 'completed' || status === 'failed' || status === 'cancelled'
  if (isTerminal !== (finishedAt !== null)) {
    throw new Error('后端操作终态与 finishedAt 不一致')
  }
  if (status === 'running' && startedAt === null) {
    throw new Error('运行中的后端操作缺少 startedAt')
  }
  if (status === 'completed' && (result === null || error !== null)) {
    throw new Error('已完成的后端操作结果无效')
  }
  if (status === 'failed' && (result !== null || error === null)) {
    throw new Error('失败的后端操作错误信息无效')
  }
  if ((status === 'pending' || status === 'running' || status === 'cancelled') && (result !== null || error !== null)) {
    throw new Error(`后端操作 ${status} 状态不应包含结果或错误`)
  }

  return {
    operationId,
    kind: operationKind(payload.kind, '后端操作.kind'),
    executorRole: executorRole(payload.executorRole, '后端操作.executorRole'),
    status,
    pageId: nullableId(payload.pageId, '后端操作.pageId'),
    bubbleId: nullableId(payload.bubbleId, '后端操作.bubbleId'),
    studioDocumentId: nullableId(payload.studioDocumentId, '后端操作.studioDocumentId'),
    studioSessionId: nullableId(payload.studioSessionId, '后端操作.studioSessionId'),
    baseRevision: nullablePositiveInteger(payload.baseRevision, '后端操作.baseRevision'),
    baseGeneration: nullablePositiveInteger(payload.baseGeneration, '后端操作.baseGeneration'),
    request: objectValue(payload.request, '后端操作.request'),
    result,
    error,
    createdAt: dateValue(payload.createdAt, '后端操作.createdAt'),
    startedAt,
    finishedAt,
  }
}

export function parseOperationAccepted(
  value: unknown,
  expectedKind?: V2OperationAccepted['kind'],
): V2OperationAccepted {
  const payload = exactObject(
    value,
    OPERATION_ACCEPTED_KEYS,
    ['operationId', 'kind', 'status', 'executorRole'],
    '后端操作受理结果',
  )
  if (payload.status !== 'pending') {
    throw new Error('后端操作受理结果.status必须是 pending')
  }
  const kind = operationKind(payload.kind, '后端操作受理结果.kind')
  if (expectedKind !== undefined && kind !== expectedKind) {
    throw new Error(`后端操作受理类型不匹配：期望 ${expectedKind}，实际 ${kind}`)
  }
  return {
    operationId: nonEmptyString(payload.operationId, '后端操作受理结果.operationId'),
    kind,
    status: 'pending',
    executorRole: executorRole(payload.executorRole, '后端操作受理结果.executorRole'),
    baseRevision: payload.baseRevision === undefined
      ? undefined
      : nullablePositiveInteger(payload.baseRevision, '后端操作受理结果.baseRevision'),
    baseGeneration: payload.baseGeneration === undefined
      ? undefined
      : nullablePositiveInteger(payload.baseGeneration, '后端操作受理结果.baseGeneration'),
    sessionRevision: optionalPositiveInteger(payload.sessionRevision, '后端操作受理结果.sessionRevision'),
    sessionGeneration: optionalPositiveInteger(payload.sessionGeneration, '后端操作受理结果.sessionGeneration'),
    userMessageId: payload.userMessageId === undefined
      ? undefined
      : nonEmptyString(payload.userMessageId, '后端操作受理结果.userMessageId'),
    documentRevision: optionalPositiveInteger(payload.documentRevision, '后端操作受理结果.documentRevision'),
  }
}

function parseOperationEvent(value: unknown, expectedOperationId: string): V2OperationEvent {
  const payload = exactObject(value, OPERATION_EVENT_KEYS, OPERATION_EVENT_KEYS, '后端操作事件')
  const operationId = nonEmptyString(payload.operationId, '后端操作事件.operationId')
  if (operationId !== expectedOperationId) {
    throw new Error(`后端操作事件身份不匹配：期望 ${expectedOperationId}，实际 ${operationId}`)
  }
  return {
    eventId: positiveInteger(payload.eventId, '后端操作事件.eventId'),
    operationId,
    type: nonEmptyString(payload.type, '后端操作事件.type'),
    payload: objectValue(payload.payload, '后端操作事件.payload'),
    createdAt: dateValue(payload.createdAt, '后端操作事件.createdAt'),
  }
}

function waitBeforeReconnect(signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason ?? new DOMException('操作等待已取消', 'AbortError'))
      return
    }
    const timer = globalThis.setTimeout(() => {
      signal?.removeEventListener('abort', abort)
      resolve()
    }, 250)
    const abort = () => {
      globalThis.clearTimeout(timer)
      reject(signal?.reason ?? new DOMException('操作等待已取消', 'AbortError'))
    }
    signal?.addEventListener('abort', abort, { once: true })
  })
}

export async function createPageOperation(
  pageId: string,
  command: PageOperationCommand
): Promise<V2OperationAccepted> {
  assertBackendActionAllowed()
  return parseOperationAccepted(await apiClient.post<unknown>(
    `/api/v2/pages/${encodeURIComponent(pageId)}/operations`,
    command,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } }
  ), command.kind)
}

export async function getOperation(
  operationId: string,
  signal?: AbortSignal
): Promise<V2Operation> {
  const payload = await apiClient.get<unknown>(`/api/v2/operations/${encodeURIComponent(operationId)}`, {
    signal,
  })
  return parseOperation(payload, operationId)
}

export async function waitForOperation(
  operationId: string,
  options: {
    onEvent?: (event: V2OperationEvent) => void
    signal?: AbortSignal
  } = {}
): Promise<V2Operation> {
  let lastEventId = 0
  while (true) {
    const headers: Record<string, string> = { Accept: 'text/event-stream' }
    if (lastEventId > 0) headers['Last-Event-ID'] = String(lastEventId)
    const response = await fetch(
      `/api/v2/operations/${encodeURIComponent(operationId)}/events?stream=1`,
      { headers, signal: options.signal },
    )
    if (!response.ok) {
      throw new Error(await readApiErrorMessage(response, '读取后端操作状态失败'))
    }
    if (!response.headers.get('content-type')?.toLowerCase().includes('text/event-stream')) {
      throw new Error('后端操作事件接口未返回事件流')
    }
    await readSseStream<unknown>(response, {
      missingBodyMessage: '无法读取后端操作事件流',
      parseErrorMessage: '后端操作事件格式无效',
      onMessage(message) {
        const event = parseOperationEvent(message.data, operationId)
        if (message.id !== String(event.eventId)) {
          throw new Error('后端操作事件游标不一致')
        }
        if (message.event !== event.type) {
          throw new Error('后端操作事件类型不一致')
        }
        if (event.eventId <= lastEventId) {
          throw new Error('后端操作事件顺序无效')
        }
        lastEventId = event.eventId
        options.onEvent?.(event)
      },
    })
    const operation = await getOperation(operationId, options.signal)
    if (operation.status === 'completed') return operation
    if (operation.status === 'failed' || operation.status === 'cancelled') {
      const message = operation.error?.message
        || `操作${operation.status === 'cancelled' ? '已取消' : '失败'}`
      throw new Error(message)
    }
    await waitBeforeReconnect(options.signal)
  }
}

export async function runPageOperation(
  pageId: string,
  command: PageOperationCommand,
  options: { signal?: AbortSignal } = {}
): Promise<V2Operation> {
  const accepted = await createPageOperation(pageId, command)
  return waitForOperation(accepted.operationId, options)
}

async function createBubbleRepair(
  pageId: string,
  bubbleId: string,
  baseRevision: number
): Promise<V2OperationAccepted> {
  assertBackendActionAllowed()
  const body = new FormData()
  body.append('target', 'bubble')
  body.append('bubble_id', bubbleId)
  body.append('base_revision', String(baseRevision))
  return parseOperationAccepted(await apiClient.upload<unknown>(
    `/api/v2/pages/${encodeURIComponent(pageId)}/repairs`,
    body,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } }
  ), 'page_repair')
}

export async function runBubbleRepair(
  pageId: string,
  bubbleId: string,
  baseRevision: number,
  options: { signal?: AbortSignal } = {}
): Promise<V2Operation> {
  const accepted = await createBubbleRepair(pageId, bubbleId, baseRevision)
  return waitForOperation(accepted.operationId, options)
}

export async function createMaskRepair(
  pageId: string,
  mask: Blob,
  command: {
    baseRevision: number
    fillColor: string
    method: 'solid'
  } | {
    baseRevision: number
    method: 'lama_mpe' | 'litelama' | 'restore_source'
  },
): Promise<V2OperationAccepted> {
  assertBackendActionAllowed()
  const body = new FormData()
  body.append('target', 'mask')
  body.append('base_revision', String(command.baseRevision))
  body.append('method', command.method)
  if (command.method === 'solid') body.append('fill_color', command.fillColor)
  body.append('mask', mask, 'repair-mask.png')
  return parseOperationAccepted(await apiClient.upload<unknown>(
    `/api/v2/pages/${encodeURIComponent(pageId)}/repairs`,
    body,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } }
  ), 'page_repair')
}
