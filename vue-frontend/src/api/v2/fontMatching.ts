import { apiClient } from '@/api/client'
import type { components } from '@/api/generated/v2'

export type FontMatchConfiguration = components['schemas']['FontMatchConfiguration']
export type FontMatchStatus = components['schemas']['FontMatchStatus']

export function getFontMatchConfiguration(signal?: AbortSignal): Promise<FontMatchConfiguration> {
  return apiClient.get('/api/v2/font-matching/configuration', { signal })
}

export function startFontMatch(image: Blob, idempotencyKey: string): Promise<components['schemas']['JobBatchAccepted']> {
  const body = new FormData()
  body.append('reference', image, 'reference.png')
  return apiClient.post('/api/v2/font-matching/jobs', body, {
    headers: { 'Idempotency-Key': idempotencyKey },
    timeout: 30_000,
  })
}

export function getFontMatch(jobId: string, signal?: AbortSignal): Promise<FontMatchStatus> {
  return apiClient.get(`/api/v2/font-matching/jobs/${encodeURIComponent(jobId)}`, { signal, timeout: 15_000 })
}
