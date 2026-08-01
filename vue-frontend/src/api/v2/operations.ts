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

export async function createPageOperation(
  pageId: string,
  command: PageOperationCommand
): Promise<V2Operation> {
  assertBackendActionAllowed()
  return apiClient.post<V2Operation>(
    `/api/v2/pages/${encodeURIComponent(pageId)}/operations`,
    command,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } }
  )
}

export async function getOperation(
  operationId: string,
  signal?: AbortSignal
): Promise<V2Operation> {
  return apiClient.get<V2Operation>(`/api/v2/operations/${encodeURIComponent(operationId)}`, {
    signal,
  })
}

export async function waitForOperation(
  operationId: string,
  options: { signal?: AbortSignal } = {}
): Promise<V2Operation> {
  const response = await fetch(
    `/api/v2/operations/${encodeURIComponent(operationId)}/events?stream=1`,
    {
      headers: { Accept: 'text/event-stream' },
      signal: options.signal,
    }
  )
  if (!response.ok) {
    throw new Error(await readApiErrorMessage(response, '读取后端操作状态失败'))
  }
  await readSseStream(response, {
    missingBodyMessage: '无法读取后端操作事件流',
    parseErrorMessage: '后端操作事件格式无效',
    onMessage() {},
  })
  const operation = await getOperation(operationId, options.signal)
  if (operation.status === 'completed') return operation
  const message =
    operation.error?.message || `操作${operation.status === 'cancelled' ? '已取消' : '失败'}`
  throw new Error(message)
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
): Promise<V2Operation> {
  const body = new FormData()
  body.append('target', 'bubble')
  body.append('bubble_id', bubbleId)
  body.append('base_revision', String(baseRevision))
  return apiClient.upload<V2Operation>(
    `/api/v2/pages/${encodeURIComponent(pageId)}/repairs`,
    body,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } }
  )
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
    fillColor?: string
    method: 'lama_mpe' | 'litelama' | 'restore_source' | 'solid'
  }
): Promise<V2Operation> {
  const body = new FormData()
  body.append('target', 'mask')
  body.append('base_revision', String(command.baseRevision))
  body.append('method', command.method)
  if (command.fillColor) body.append('fill_color', command.fillColor)
  body.append('mask', mask, 'repair-mask.png')
  return apiClient.upload<V2Operation>(
    `/api/v2/pages/${encodeURIComponent(pageId)}/repairs`,
    body,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } }
  )
}
