import { apiClient } from '@/api/client'
import { newIdempotencyKey } from './content'
import { jobsApi, type V2JobDetail } from './jobs'

const ROOT = '/api/v2/insight'

export interface V2ContinuationImageVersion {
  active?: boolean
  adopted?: boolean
  assetId: string
  assetUrl: string
  thumbnailUrl: string
  version: number
}

export interface V2ContinuationPage {
  continuationPageId: string
  imageVersions: V2ContinuationImageVersion[]
  ordinal: number
  payload: Record<string, unknown>
  revision: number
}

export interface V2ContinuationCharacter {
  aliases: string[]
  characterId: string
  enabled: boolean
  name: string
  payload: Record<string, unknown>
  projectId: string
  revision: number
}

export interface V2ContinuationForm {
  adoptedAssetId: string | null
  characterId: string
  formId: string
  imageVersions: V2ContinuationImageVersion[]
  name: string
  payload: Record<string, unknown>
  referenceAssetId: string | null
  referenceAssetUrl: string | null
  referenceThumbnailUrl: string | null
  revision: number
}

export interface V2ContinuationProject {
  bookId: string
  characters: V2ContinuationCharacter[]
  config: {
    direction?: string
    pageCount?: number
    styleReferencePages?: number
  }
  pages: V2ContinuationPage[]
  projectId: string
  referenceAssets: Array<{
    assetId: string
    assetUrl: string
    thumbnailUrl: string
  }>
  revision: number
  script: {
    content: string
    projectId?: string
    revision: number
    scriptId: string
  } | null
  sourceRunId: string
}

export interface V2ContinuationState {
  activeRunId: string | null
  bookId: string
  missing: string[]
  project: V2ContinuationProject | null
  ready: boolean
}

export interface V2ContinuationAccepted {
  batchId: string
  jobIds: string[]
  status: 'queued'
}

export function getV2Continuation(bookId: string): Promise<V2ContinuationState> {
  return apiClient.get(`${ROOT}/books/${encodeURIComponent(bookId)}/continuation`)
}

export function syncV2Continuation(bookId: string): Promise<V2ContinuationProject> {
  return apiClient.post(
    `${ROOT}/books/${encodeURIComponent(bookId)}/continuation/sync`,
    {},
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
  })
}

export function setV2ContinuationReferences(
  projectId: string,
  baseRevision: number,
  assetIds: string[],
): Promise<V2ContinuationProject> {
  return apiClient.put(
    `${ROOT}/continuation/projects/${encodeURIComponent(projectId)}/references`,
    { baseRevision, assetIds },
  )
}

export async function listAllV2ContinuationForms(projectId: string): Promise<V2ContinuationForm[]> {
  const items: V2ContinuationForm[] = []
  let cursor = 0
  do {
    const response = await apiClient.get<{
      items: V2ContinuationForm[]
      nextCursor: number | null
    }>(`${ROOT}/continuation/projects/${encodeURIComponent(projectId)}/forms`, {
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
  )
}

export function deleteV2ContinuationCharacter(
  characterId: string,
  baseRevision: number,
): Promise<{ deleted: boolean }> {
  return apiClient.delete(
    `${ROOT}/continuation/characters/${encodeURIComponent(characterId)}?baseRevision=${baseRevision}`,
  )
}

export function createV2ContinuationForm(
  characterId: string,
  command: { name: string; payload: Record<string, unknown> },
): Promise<V2ContinuationForm> {
  return apiClient.post(
    `${ROOT}/continuation/characters/${encodeURIComponent(characterId)}/forms`,
    command,
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
  )
}

export function deleteV2ContinuationForm(
  formId: string,
  baseRevision: number,
): Promise<{ deleted: boolean }> {
  return apiClient.delete(
    `${ROOT}/continuation/forms/${encodeURIComponent(formId)}?baseRevision=${baseRevision}`,
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
  )
}

export function deleteV2ContinuationReference(
  formId: string,
  baseRevision: number,
): Promise<V2ContinuationForm> {
  return apiClient.delete(
    `${ROOT}/continuation/forms/${encodeURIComponent(formId)}/reference?baseRevision=${baseRevision}`,
  )
}

export function adoptV2ContinuationFormImage(
  formId: string,
  version: number,
  baseRevision: number,
): Promise<V2ContinuationForm> {
  return apiClient.post(
    `${ROOT}/continuation/forms/${encodeURIComponent(formId)}/image-versions/${version}/adopt`,
    { baseRevision },
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
  )
}

export function activateV2ContinuationImage(
  pageId: string,
  version: number,
): Promise<V2ContinuationPage> {
  return apiClient.post(
    `${ROOT}/continuation/pages/${encodeURIComponent(pageId)}/image-versions/${version}/activate`,
    {},
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
  return apiClient.post(
    `${ROOT}/books/${encodeURIComponent(bookId)}/continuation/jobs`,
    command,
    { headers: { 'Idempotency-Key': newIdempotencyKey() } },
  )
}

export function clearV2Continuation(bookId: string): Promise<{ deleted: boolean }> {
  return apiClient.delete(`${ROOT}/books/${encodeURIComponent(bookId)}/continuation`)
}

export function getV2ContinuationJob(jobId: string): Promise<V2JobDetail> {
  return jobsApi.get(jobId)
}
