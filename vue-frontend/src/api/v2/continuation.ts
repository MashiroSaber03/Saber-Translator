import { apiClient } from '@/api/client'
import type { components } from '@/api/generated/v2'
import { assertBackendActionAllowed } from '@/services/backendAccessGate'
import { newIdempotencyKey } from './content'

const ROOT = '/api/v2/insight'

export type V2ContinuationAccepted = components['schemas']['JobBatchAccepted']
export type V2ContinuationCharacter = components['schemas']['ContinuationCharacter']
export type V2ContinuationForm = components['schemas']['ContinuationForm']
export type V2ContinuationFormAdoption = components['schemas']['ContinuationFormAdoption']
export type V2ContinuationImageActivation = components['schemas']['ContinuationImageActivation']
export type V2ContinuationPage = components['schemas']['ContinuationPage']
export type V2ContinuationProject = components['schemas']['ContinuationProject']
export type V2ContinuationState = components['schemas']['ContinuationState']

type V2ContinuationFormList = components['schemas']['ContinuationFormList']

export function getV2Continuation(bookId: string): Promise<V2ContinuationState> {
  return apiClient.get(`${ROOT}/books/${encodeURIComponent(bookId)}/continuation`)
}

export function syncV2Continuation(bookId: string): Promise<V2ContinuationProject> {
  return apiClient.post(
    `${ROOT}/books/${encodeURIComponent(bookId)}/continuation/sync-analysis`,
    {},
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function updateV2ContinuationProject(
  projectId: string,
  baseRevision: number,
  config: V2ContinuationProject['config'],
): Promise<V2ContinuationProject> {
  return apiClient.patch(`${ROOT}/continuation/projects/${encodeURIComponent(projectId)}`, {
    baseRevision,
    config,
  }, { headers: { 'Idempotency-Key': newIdempotencyKey() } })
}

export function setV2ContinuationReferences(
  projectId: string,
  baseRevision: number,
  assetIds: string[],
): Promise<V2ContinuationProject> {
  return apiClient.put(
    `${ROOT}/continuation/projects/${encodeURIComponent(projectId)}/references`,
    { baseRevision, assetIds },
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export async function listAllV2ContinuationForms(projectId: string): Promise<V2ContinuationForm[]> {
  const items: V2ContinuationForm[] = []
  let cursor = 0
  do {
    const response = await apiClient.get<V2ContinuationFormList>(`${ROOT}/continuation/projects/${encodeURIComponent(projectId)}/forms`, {
      params: { cursor, limit: 200 },
    })
    items.push(...response.items)
    cursor = response.nextCursor ?? 0
  } while (cursor > 0)
  return items
}

export function createV2ContinuationCharacter(
  projectId: string,
  command: {
    aliases: string[]
    enabled: boolean
    name: string
    payload: Record<string, unknown>
  },
): Promise<V2ContinuationCharacter> {
  return apiClient.post(
    `${ROOT}/continuation/projects/${encodeURIComponent(projectId)}/characters`,
    command,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function updateV2ContinuationCharacter(
  characterId: string,
  command: {
    aliases: string[]
    baseRevision: number
    enabled: boolean
    name: string
    payload: Record<string, unknown>
  },
): Promise<V2ContinuationCharacter> {
  return apiClient.patch(
    `${ROOT}/continuation/characters/${encodeURIComponent(characterId)}`,
    command,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function deleteV2ContinuationCharacter(
  characterId: string,
  baseRevision: number,
): Promise<{ deleted: boolean }> {
  return apiClient.delete(
    `${ROOT}/continuation/characters/${encodeURIComponent(characterId)}?baseRevision=${baseRevision}`,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function createV2ContinuationForm(
  characterId: string,
  command: { name: string; payload: Record<string, unknown> },
): Promise<V2ContinuationForm> {
  return apiClient.post(
    `${ROOT}/continuation/characters/${encodeURIComponent(characterId)}/forms`,
    command,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function updateV2ContinuationForm(
  formId: string,
  command: {
    baseRevision: number
    name: string
    payload: Record<string, unknown>
  },
): Promise<V2ContinuationForm> {
  return apiClient.patch(
    `${ROOT}/continuation/forms/${encodeURIComponent(formId)}`,
    command,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function deleteV2ContinuationForm(
  formId: string,
  baseRevision: number,
): Promise<{ deleted: boolean }> {
  return apiClient.delete(
    `${ROOT}/continuation/forms/${encodeURIComponent(formId)}?baseRevision=${baseRevision}`,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function uploadV2ContinuationReference(
  formId: string,
  baseRevision: number,
  file: File,
): Promise<V2ContinuationForm> {
  const form = new FormData()
  form.append('file', file)
  form.append('baseRevision', String(baseRevision))
  return apiClient.upload(
    `${ROOT}/continuation/forms/${encodeURIComponent(formId)}/reference`,
    form,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function deleteV2ContinuationReference(
  formId: string,
  baseRevision: number,
): Promise<V2ContinuationForm> {
  return apiClient.delete(
    `${ROOT}/continuation/forms/${encodeURIComponent(formId)}/reference?baseRevision=${baseRevision}`,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function adoptV2ContinuationFormImage(
  formId: string,
  version: number,
  baseRevision: number,
): Promise<V2ContinuationFormAdoption> {
  return apiClient.post(
    `${ROOT}/continuation/forms/${encodeURIComponent(formId)}/image-versions/${version}/adopt`,
    { baseRevision },
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function updateV2ContinuationScript(
  projectId: string,
  baseRevision: number,
  content: string,
): Promise<V2ContinuationProject['script']> {
  return apiClient.patch(
    `${ROOT}/continuation/projects/${encodeURIComponent(projectId)}/script`,
    { baseRevision, content },
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function updateV2ContinuationPage(
  pageId: string,
  baseRevision: number,
  payload: Record<string, unknown>,
): Promise<V2ContinuationPage> {
  return apiClient.patch(
    `${ROOT}/continuation/pages/${encodeURIComponent(pageId)}`,
    { baseRevision, payload },
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function activateV2ContinuationImage(
  pageId: string,
  version: number,
): Promise<V2ContinuationImageActivation> {
  return apiClient.post(
    `${ROOT}/continuation/pages/${encodeURIComponent(pageId)}/image-versions/${version}/activate`,
    {},
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function createV2ContinuationJob(
  bookId: string,
  command: {
    formId?: string
    format?: 'pdf' | 'zip'
    kind: 'character_sheet' | 'export' | 'images' | 'pages' | 'script'
    ordinals?: number[]
  },
): Promise<V2ContinuationAccepted> {
  assertBackendActionAllowed()
  return apiClient.post(
    `${ROOT}/books/${encodeURIComponent(bookId)}/continuation/jobs`,
    command,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function clearV2Continuation(bookId: string): Promise<{ deleted: boolean }> {
  return apiClient.delete(
    `${ROOT}/books/${encodeURIComponent(bookId)}/continuation`,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}
