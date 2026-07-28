import { apiClient } from '@/api/client'
import { newIdempotencyKey } from './content'

export type PageOperationKind =
  | 'bubble_color'
  | 'bubble_ocr'
  | 'bubble_translate'
  | 'page_detect'

export type OperationStatus =
  | 'cancelled'
  | 'completed'
  | 'failed'
  | 'pending'
  | 'running'

export interface V2Operation {
  baseRevision: number | null
  bubbleId: string | null
  error: { code?: string; message?: string } | null
  executorRole: 'api' | 'worker'
  finishedAt: string | null
  kind: string
  operationId: string
  pageId: string | null
  result: Record<string, unknown> | null
  startedAt: string | null
  status: OperationStatus
}

export interface PageOperationCommand {
  baseRevision: number
  bubbleId?: string
  kind: PageOperationKind
}

export async function createPageOperation(
  pageId: string,
  command: PageOperationCommand,
): Promise<V2Operation> {
  return apiClient.post<V2Operation>(
    `/api/v2/pages/${encodeURIComponent(pageId)}/operations`,
    command,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export async function getOperation(
  operationId: string,
  signal?: AbortSignal,
): Promise<V2Operation> {
  return apiClient.get<V2Operation>(
    `/api/v2/operations/${encodeURIComponent(operationId)}`,
    { signal },
  )
}

export async function waitForOperation(
  operationId: string,
  options: {
    intervalMs?: number
    signal?: AbortSignal
    timeoutMs?: number
  } = {},
): Promise<V2Operation> {
  const intervalMs = options.intervalMs ?? 350
  const timeoutMs = options.timeoutMs ?? 10 * 60 * 1000
  const startedAt = Date.now()
  while (true) {
    options.signal?.throwIfAborted()
    const operation = await getOperation(operationId, options.signal)
    if (operation.status === 'completed') return operation
    if (operation.status === 'failed' || operation.status === 'cancelled') {
      const message = operation.error?.message
        || `操作${operation.status === 'failed' ? '失败' : '已取消'}`
      throw new Error(message)
    }
    if (Date.now() - startedAt >= timeoutMs) {
      throw new Error('等待后端操作完成超时；操作仍可在任务中心继续运行')
    }
    await new Promise<void>((resolve, reject) => {
      const timer = window.setTimeout(resolve, intervalMs)
      options.signal?.addEventListener('abort', () => {
        window.clearTimeout(timer)
        reject(new DOMException('Aborted', 'AbortError'))
      }, { once: true })
    })
  }
}

export async function runPageOperation(
  pageId: string,
  command: PageOperationCommand,
  options: { signal?: AbortSignal; timeoutMs?: number } = {},
): Promise<V2Operation> {
  const accepted = await createPageOperation(pageId, command)
  return waitForOperation(accepted.operationId, options)
}

export async function createBubbleRepair(
  pageId: string,
  bubbleId: string,
  baseRevision: number,
): Promise<V2Operation> {
  const body = new FormData()
  body.append('target', 'bubble')
  body.append('bubble_id', bubbleId)
  body.append('base_revision', String(baseRevision))
  return apiClient.upload<V2Operation>(
    `/api/v2/pages/${encodeURIComponent(pageId)}/repairs`,
    body,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export async function runBubbleRepair(
  pageId: string,
  bubbleId: string,
  baseRevision: number,
  options: { signal?: AbortSignal; timeoutMs?: number } = {},
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
  },
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
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}
