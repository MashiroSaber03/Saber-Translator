import { apiClient } from '@/api/client'
import type { components } from '@/api/generated/v2'
import type { InpaintMethod } from '@/types/bubble'
import { createUuid } from '@/utils/uuid'
import type { V2PageDocumentMutationResponse } from './content'

export type PageMasks = components['schemas']['PageMasks']

export function getPageMasks(
  pageId: string,
  method: InpaintMethod,
  signal?: AbortSignal
): Promise<PageMasks> {
  return apiClient.get(`/api/v2/pages/${encodeURIComponent(pageId)}/masks`, {
    params: { method },
    signal,
  })
}

export function savePageUserMask(
  pageId: string,
  baseRevision: number,
  mask: Blob,
  signal?: AbortSignal
): Promise<V2PageDocumentMutationResponse> {
  const body = new FormData()
  body.append('base_revision', String(baseRevision))
  body.append('mask', mask, 'user-mask.png')
  return apiClient.upload(
    `/api/v2/pages/${encodeURIComponent(pageId)}/masks`,
    body,
    { headers: { 'Idempotency-Key': createUuid() }, signal },
    'put'
  )
}
