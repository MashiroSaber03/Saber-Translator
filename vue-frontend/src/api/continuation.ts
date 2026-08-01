import { downloadBlob } from './download'
import {
  activateV2ContinuationImage,
  adoptV2ContinuationFormImage,
  clearV2Continuation,
  createV2ContinuationCharacter,
  createV2ContinuationForm,
  createV2ContinuationJob,
  deleteV2ContinuationCharacter,
  deleteV2ContinuationForm,
  deleteV2ContinuationReference,
  getV2Continuation,
  getV2ContinuationJob,
  listAllV2ContinuationForms,
  setV2ContinuationReferences,
  syncV2Continuation,
  updateV2ContinuationCharacter,
  updateV2ContinuationForm,
  updateV2ContinuationPage,
  updateV2ContinuationProject,
  updateV2ContinuationScript,
  uploadV2ContinuationReference,
  type V2ContinuationCharacter,
  type V2ContinuationForm,
  type V2ContinuationPage,
  type V2ContinuationProject,
  type V2ContinuationState,
} from '@/api/v2/continuation'
import { listAllInsightPages } from '@/api/v2/insight'

export interface CharacterForm {
  form_id: string
  form_name: string
  description: string
  reference_image: string
  enabled?: boolean
}

export interface CharacterProfile {
  name: string
  aliases: string[]
  description: string
  forms: CharacterForm[]
  reference_image: string
  enabled?: boolean
}

export interface ChapterScript {
  chapter_title: string
  page_count: number
  script_text: string
  generated_at: string
}

export interface CharacterFormSelection {
  character: string
  form_id: string
  form_name?: string
}

export interface PageContent {
  page_number: number
  continuity_text: string
  story_text: string
  dialogue_text: string
  characters: string[]
  character_forms?: CharacterFormSelection[]
  final_prompt: string
  image_url: string
  previous_url: string
  status: 'pending' | 'generating' | 'generated' | 'failed'
}

interface SavedContinuationData {
  script: ChapterScript | null
  pages: PageContent[]
  config: {
    page_count?: number
    style_reference_pages?: number
    continuation_direction?: string
  } | null
  has_data: boolean
}

interface PrepareResponse {
  success: boolean
  ready?: boolean
  message?: string
  error?: string
  saved_data?: SavedContinuationData
}

export type SyncContinuationResponse = PrepareResponse

interface CharactersResponse {
  success: boolean
  characters?: CharacterProfile[]
  error?: string
}

export interface UploadImageResponse {
  success: boolean
  image_path?: string
  task_id?: string
  error?: string
}

interface ScriptResponse {
  success: boolean
  script?: ChapterScript
  task_id?: string
  error?: string
}

interface ImageGenerateResponse {
  success: boolean
  image_path?: string
  pages?: PageContent[]
  task_id?: string
  error?: string
}

export interface MangaImageInfo {
  token: string
  page_number: number
  path: string
  has_image: boolean
  is_placeholder?: boolean
  label?: string
}

export interface CharacterFormInfo {
  token: string
  character_name: string
  form_id: string
  form_name: string
  path: string
  has_image: boolean
  is_placeholder?: boolean
  label?: string
}

export interface AvailableImagesResponse {
  success: boolean
  original_images?: MangaImageInfo[]
  continuation_images?: MangaImageInfo[]
  character_forms?: CharacterFormInfo[]
  total_original_pages?: number
  error?: string
}

const stateCache = new Map<string, V2ContinuationState>()
const formsCache = new Map<string, V2ContinuationForm[]>()

function payloadString(payload: Record<string, unknown>, key: string): string {
  return String(payload[key] ?? '')
}

function activeImage(page: V2ContinuationPage) {
  return page.imageVersions.find(version => version.active)
}

function previousImage(page: V2ContinuationPage) {
  const active = activeImage(page)
  return page.imageVersions.find(version => version.version !== active?.version)
}

function mapPage(page: V2ContinuationPage): PageContent {
  const payload = page.payload
  const currentImage = activeImage(page)
  const oldImage = previousImage(page)
  const rawStatus = String(payload.status ?? 'pending')
  return {
    page_number: page.ordinal,
    continuity_text: payloadString(payload, 'continuityText'),
    story_text: payloadString(payload, 'storyText'),
    dialogue_text: payloadString(payload, 'dialogueText'),
    characters: Array.isArray(payload.characters) ? payload.characters.map(String) : [],
    character_forms: (
      Array.isArray(payload.characterForms) ? payload.characterForms : []
    ) as CharacterFormSelection[],
    final_prompt: payloadString(payload, 'finalPrompt'),
    image_url: currentImage?.assetUrl ?? '',
    previous_url: oldImage?.assetUrl ?? '',
    status: currentImage
      ? 'generated'
      : rawStatus === 'failed' || Boolean(payload.staleReason)
        ? 'failed'
        : rawStatus === 'generating'
          ? 'generating'
          : 'pending',
  }
}

function pagePayload(page: PageContent): Record<string, unknown> {
  return {
    continuityText: page.continuity_text,
    storyText: page.story_text,
    dialogueText: page.dialogue_text,
    characters: page.characters,
    characterForms: page.character_forms ?? [],
    finalPrompt: page.final_prompt,
    status: page.status === 'generated' ? 'ready' : page.status,
  }
}

function mapScript(project: V2ContinuationProject): ChapterScript | null {
  if (!project.script) return null
  return {
    chapter_title: '续写章节',
    page_count: Number(project.config.pageCount ?? 15),
    script_text: project.script.content,
    generated_at: '',
  }
}

function savedData(project: V2ContinuationProject | null): SavedContinuationData {
  return {
    script: project ? mapScript(project) : null,
    pages: project?.pages.map(mapPage) ?? [],
    config: project ? {
      page_count: Number(project.config.pageCount ?? 15),
      style_reference_pages: Number(project.config.styleReferencePages ?? 3),
      continuation_direction: String(project.config.direction ?? ''),
    } : null,
    has_data: Boolean(project),
  }
}

async function refreshState(bookId: string): Promise<V2ContinuationState> {
  const state = await getV2Continuation(bookId)
  stateCache.set(bookId, state)
  if (state.project) {
    formsCache.set(
      state.project.projectId,
      await listAllV2ContinuationForms(state.project.projectId),
    )
  }
  return state
}

async function ensureProject(bookId: string): Promise<V2ContinuationProject> {
  let state = stateCache.get(bookId) ?? await refreshState(bookId)
  if (!state.project) {
    if (!state.ready) {
      throw new Error(`续写前置数据未就绪：${state.missing.join('、')}`)
    }
    const project = await syncV2Continuation(bookId)
    state = { ...state, project }
    stateCache.set(bookId, state)
  }
  if (!state.project) throw new Error('续写项目同步失败')
  return state.project
}

async function refreshProject(bookId: string): Promise<V2ContinuationProject> {
  const state = await refreshState(bookId)
  if (!state.project) throw new Error('续写项目不存在')
  return state.project
}

function cacheProject(bookId: string, project: V2ContinuationProject): void {
  const previous = stateCache.get(bookId)
  stateCache.set(bookId, {
    activeRunId: previous?.activeRunId ?? project.sourceRunId,
    bookId,
    missing: previous?.missing ?? [],
    project,
    ready: true,
  })
}

function characterFor(project: V2ContinuationProject, name: string): V2ContinuationCharacter {
  const character = project.characters.find(item => item.name === name)
  if (!character) throw new Error(`角色不存在：${name}`)
  return character
}

async function formFor(
  project: V2ContinuationProject,
  characterName: string,
  formId: string,
): Promise<V2ContinuationForm> {
  const character = characterFor(project, characterName)
  let forms = formsCache.get(project.projectId)
  if (!forms) {
    forms = await listAllV2ContinuationForms(project.projectId)
    formsCache.set(project.projectId, forms)
  }
  const form = forms.find(item =>
    item.characterId === character.characterId
    && item.formId === formId
  )
  if (!form) throw new Error(`角色形态不存在：${formId}`)
  return form
}

function mapForm(form: V2ContinuationForm): CharacterForm {
  const adopted = form.imageVersions.find(version => version.adopted)
  const latestGenerated = form.imageVersions[0]
  return {
    form_id: form.formId,
    form_name: form.name,
    description: String(form.payload.description ?? ''),
    reference_image: adopted?.assetUrl
      ?? latestGenerated?.assetUrl
      ?? form.referenceAssetUrl
      ?? '',
    enabled: form.payload.enabled !== false,
  }
}

function mapCharacter(
  character: V2ContinuationCharacter,
  forms: V2ContinuationForm[],
): CharacterProfile {
  const characterForms = forms.filter(form => form.characterId === character.characterId)
  const reference = characterForms
    .map(form => form.imageVersions.find(version => version.adopted)?.assetUrl ?? form.referenceAssetUrl)
    .find(Boolean)
  return {
    name: character.name,
    aliases: character.aliases,
    description: String(character.payload.description ?? ''),
    forms: characterForms.map(mapForm),
    reference_image: String(reference ?? ''),
    enabled: character.enabled,
  }
}

export async function prepareContinuation(bookId: string): Promise<PrepareResponse> {
  let state = await refreshState(bookId)
  if (state.ready && !state.project) {
    const project = await syncV2Continuation(bookId)
    state = { ...state, project }
    stateCache.set(bookId, state)
    formsCache.set(project.projectId, [])
  }
  return {
    success: true,
    ready: state.ready,
    message: state.ready ? '续写数据已就绪' : `缺少：${state.missing.join('、')}`,
    saved_data: savedData(state.project),
  }
}

export async function syncContinuationAnalysis(bookId: string): Promise<SyncContinuationResponse> {
  const project = await syncV2Continuation(bookId)
  cacheProject(bookId, project)
  formsCache.set(project.projectId, await listAllV2ContinuationForms(project.projectId))
  return {
    success: true,
    ready: true,
    message: '分析数据同步完成',
    saved_data: savedData(project),
  }
}

export async function getCharacters(bookId: string): Promise<CharactersResponse> {
  const project = await refreshProject(bookId)
  const forms = formsCache.get(project.projectId) ?? []
  return {
    success: true,
    characters: project.characters.map(character => mapCharacter(character, forms)),
  }
}

export async function addCharacter(
  bookId: string,
  data: { name: string; aliases?: string[]; description?: string },
): Promise<{ success: boolean; character?: CharacterProfile; error?: string }> {
  const project = await ensureProject(bookId)
  await createV2ContinuationCharacter(project.projectId, {
    name: data.name,
    aliases: data.aliases ?? [],
    enabled: true,
    payload: { description: data.description ?? '' },
  })
  const refreshed = await refreshProject(bookId)
  const forms = formsCache.get(refreshed.projectId) ?? []
  const character = refreshed.characters.find(item => item.name === data.name)
  return {
    success: true,
    character: character ? mapCharacter(character, forms) : undefined,
  }
}

export async function deleteCharacter(
  bookId: string,
  characterName: string,
): Promise<{ success: boolean; message?: string; error?: string }> {
  const project = await ensureProject(bookId)
  const character = characterFor(project, characterName)
  await deleteV2ContinuationCharacter(character.characterId, character.revision)
  await refreshState(bookId)
  return { success: true, message: '角色已删除' }
}

export async function updateCharacterInfo(
  bookId: string,
  characterName: string,
  data: { name?: string; aliases?: string[]; enabled?: boolean },
): Promise<{ success: boolean; character?: CharacterProfile; error?: string }> {
  const project = await ensureProject(bookId)
  const character = characterFor(project, characterName)
  const updated = await updateV2ContinuationCharacter(character.characterId, {
    baseRevision: character.revision,
    name: data.name ?? character.name,
    aliases: data.aliases ?? character.aliases,
    enabled: data.enabled ?? character.enabled,
    payload: character.payload,
  })
  await refreshState(bookId)
  return {
    success: true,
    character: mapCharacter(updated, formsCache.get(project.projectId) ?? []),
  }
}

export async function addCharacterForm(
  bookId: string,
  characterName: string,
  data: { form_name: string; description?: string },
): Promise<{ success: boolean; form?: CharacterForm; error?: string }> {
  const project = await ensureProject(bookId)
  const character = characterFor(project, characterName)
  const form = await createV2ContinuationForm(character.characterId, {
    name: data.form_name,
    payload: {
      description: data.description ?? '',
      enabled: true,
    },
  })
  await refreshState(bookId)
  return { success: true, form: mapForm(form) }
}

export async function updateCharacterForm(
  bookId: string,
  characterName: string,
  formId: string,
  data: { form_name?: string; description?: string },
): Promise<{ success: boolean; error?: string }> {
  const project = await ensureProject(bookId)
  const form = await formFor(project, characterName, formId)
  await updateV2ContinuationForm(form.formId, {
    baseRevision: form.revision,
    name: data.form_name ?? form.name,
    payload: {
      ...form.payload,
      ...(data.description !== undefined ? { description: data.description } : {}),
    },
  })
  await refreshState(bookId)
  return { success: true }
}

export async function deleteCharacterForm(
  bookId: string,
  characterName: string,
  formId: string,
): Promise<{ success: boolean; message?: string; error?: string }> {
  const project = await ensureProject(bookId)
  const form = await formFor(project, characterName, formId)
  await deleteV2ContinuationForm(form.formId, form.revision)
  await refreshState(bookId)
  return { success: true, message: '形态已删除' }
}

export async function toggleCharacterEnabled(
  bookId: string,
  characterName: string,
  enabled: boolean,
): Promise<{ success: boolean; enabled?: boolean; error?: string }> {
  const result = await updateCharacterInfo(bookId, characterName, { enabled })
  return { success: result.success, enabled, error: result.error }
}

export async function toggleFormEnabled(
  bookId: string,
  characterName: string,
  formId: string,
  enabled: boolean,
): Promise<{ success: boolean; enabled?: boolean; error?: string }> {
  const project = await ensureProject(bookId)
  const form = await formFor(project, characterName, formId)
  await updateV2ContinuationForm(form.formId, {
    baseRevision: form.revision,
    name: form.name,
    payload: { ...form.payload, enabled },
  })
  await refreshState(bookId)
  return { success: true, enabled }
}

export async function uploadFormImage(
  bookId: string,
  characterName: string,
  formId: string,
  formData: FormData,
): Promise<UploadImageResponse> {
  const project = await ensureProject(bookId)
  const form = await formFor(project, characterName, formId)
  const file = formData.get('file')
  if (!(file instanceof File)) return { success: false, error: '请选择参考图片' }
  const updated = await uploadV2ContinuationReference(form.formId, form.revision, file)
  await refreshState(bookId)
  return { success: true, image_path: updated.referenceAssetUrl ?? undefined }
}

export async function deleteFormImage(
  bookId: string,
  characterName: string,
  formId: string,
): Promise<{ success: boolean; error?: string }> {
  const project = await ensureProject(bookId)
  const form = await formFor(project, characterName, formId)
  await deleteV2ContinuationReference(form.formId, form.revision)
  await refreshState(bookId)
  return { success: true }
}

export async function generateFormOrtho(
  bookId: string,
  characterName: string,
  formId: string,
  sourceImages: File[],
): Promise<UploadImageResponse> {
  const project = await ensureProject(bookId)
  let form = await formFor(project, characterName, formId)
  if (sourceImages[0]) {
    form = await uploadV2ContinuationReference(form.formId, form.revision, sourceImages[0])
  }
  const accepted = await createV2ContinuationJob(bookId, {
    kind: 'character_sheet',
    formId: form.formId,
  })
  return { success: true, task_id: accepted.jobIds[0] }
}

export async function setFormReference(
  bookId: string,
  characterName: string,
  formId: string,
  imagePath: string,
): Promise<{ success: boolean; error?: string }> {
  const project = await ensureProject(bookId)
  const form = await formFor(project, characterName, formId)
  const version = form.imageVersions.find(item =>
    item.assetUrl === imagePath || item.assetId === imagePath
  )
  if (!version) return { success: false, error: '未找到生成结果版本' }
  await adoptV2ContinuationFormImage(form.formId, version.version, form.revision)
  await refreshState(bookId)
  return { success: true }
}

export async function saveScript(
  bookId: string,
  script: ChapterScript,
): Promise<ScriptResponse> {
  const project = await ensureProject(bookId)
  await updateV2ContinuationScript(
    project.projectId,
    project.script?.revision ?? 0,
    script.script_text,
  )
  const refreshed = await refreshProject(bookId)
  return { success: true, script: mapScript(refreshed) ?? script }
}

export async function savePages(
  bookId: string,
  pages: PageContent[],
): Promise<{ success: boolean; error?: string }> {
  const project = await ensureProject(bookId)
  const byOrdinal = new Map(project.pages.map(page => [page.ordinal, page]))
  await Promise.all(
    pages.flatMap(page => {
      const stored = byOrdinal.get(page.page_number)
      return stored
        ? [updateV2ContinuationPage(
            stored.continuationPageId,
            stored.revision,
            pagePayload(page),
          )]
        : []
    }),
  )
  await refreshState(bookId)
  return { success: true }
}

export async function saveConfig(
  bookId: string,
  config: {
    page_count: number
    style_reference_pages: number
    continuation_direction: string
  },
): Promise<{ success: boolean; error?: string }> {
  const project = await ensureProject(bookId)
  const updated = await updateV2ContinuationProject(project.projectId, project.revision, {
    pageCount: config.page_count,
    styleReferencePages: config.style_reference_pages,
    direction: config.continuation_direction,
  })
  cacheProject(bookId, updated)
  return { success: true }
}

export async function clearContinuationData(
  bookId: string,
): Promise<{ success: boolean; message?: string; error?: string }> {
  await clearV2Continuation(bookId)
  stateCache.delete(bookId)
  return { success: true, message: '续写数据已清空' }
}

export async function generateSinglePageDetails(
  bookId: string,
  _script: ChapterScript,
  pageNumber: number,
): Promise<{ success: boolean; page?: PageContent; task_id?: string; error?: string }> {
  const accepted = await createV2ContinuationJob(bookId, {
    kind: 'pages',
    ordinals: [pageNumber],
  })
  return { success: true, task_id: accepted.jobIds[0] }
}

export async function getStyleReferences(
  bookId: string,
  count = 3,
): Promise<{ success: boolean; tokens?: string[]; error?: string }> {
  const project = await ensureProject(bookId)
  return {
    success: true,
    tokens: project.referenceAssets.slice(-count).map(asset => asset.assetId),
  }
}

async function savePageBeforeImage(bookId: string, pageNumber: number, page: PageContent): Promise<void> {
  const project = await ensureProject(bookId)
  const stored = project.pages.find(item => item.ordinal === pageNumber)
  if (!stored) throw new Error('请先生成页面剧情')
  await updateV2ContinuationPage(
    stored.continuationPageId,
    stored.revision,
    pagePayload(page),
  )
  await refreshState(bookId)
}

export async function generatePageImage(
  bookId: string,
  pageNumber: number,
  page: PageContent,
  _styleReferenceTokens: string[],
  _sessionId?: string,
  _styleRefCount = 3,
): Promise<ImageGenerateResponse> {
  await savePageBeforeImage(bookId, pageNumber, page)
  const accepted = await createV2ContinuationJob(bookId, {
    kind: 'images',
    ordinals: [pageNumber],
  })
  return { success: true, task_id: accepted.jobIds[0] }
}

export const regeneratePageImage = generatePageImage

async function exportContinuation(bookId: string, format: 'pdf' | 'zip'): Promise<Blob> {
  const accepted = await createV2ContinuationJob(bookId, { kind: 'export', format })
  const job = await waitForContinuationJob(accepted.jobIds[0])
  const artifact = job.artifacts?.[0]
  if (!artifact?.assetId) throw new Error('导出任务未生成文件')
  const { blob } = await downloadBlob({
    url: `/api/v2/assets/${artifact.assetId}`,
    fallbackFilename: format === 'pdf'
      ? `${bookId}.continuation.pdf`
      : `${bookId}.continuation-images.zip`,
    fallbackErrorMessage: '导出失败',
  })
  return blob
}

export function exportAsImages(bookId: string): Promise<Blob> {
  return exportContinuation(bookId, 'zip')
}

export function exportAsPdf(bookId: string): Promise<Blob> {
  return exportContinuation(bookId, 'pdf')
}

export async function getAvailableImages(
  bookId: string,
  _mode: 'script' | 'image' = 'script',
): Promise<AvailableImagesResponse> {
  const [project, sourcePages] = await Promise.all([
    ensureProject(bookId),
    listAllInsightPages(bookId),
  ])
  const forms = formsCache.get(project.projectId) ?? []
  return {
    success: true,
    original_images: sourcePages.map(page => ({
      token: page.sourceAssetId,
      page_number: page.displayPageNumber,
      path: page.thumbnailUrl ?? '',
      has_image: Boolean(page.thumbnailUrl),
      is_placeholder: !page.thumbnailUrl,
      label: `原作第 ${page.displayPageNumber} 页`,
    })),
    continuation_images: project.pages.flatMap(page => {
      const image = activeImage(page)
      return image ? [{
        token: image.assetId,
        page_number: page.ordinal,
        path: image.thumbnailUrl,
        has_image: true,
        label: `续写第 ${page.ordinal} 页`,
      }] : []
    }),
    character_forms: forms.flatMap(form => {
      const image = form.imageVersions.find(version => version.adopted)
      const path = image?.thumbnailUrl ?? form.referenceThumbnailUrl
      return path ? [{
        token: image?.assetId ?? form.referenceAssetId ?? '',
        character_name: project.characters.find(
          character => character.characterId === form.characterId,
        )?.name ?? '',
        form_id: form.formId,
        form_name: form.name,
        path,
        has_image: true,
      }] : []
    }),
    total_original_pages: sourcePages.length,
  }
}

export async function generateScriptWithRefs(
  bookId: string,
  direction: string,
  pageCount: number,
  referenceTokens?: string[],
  _referenceImageCount = 5,
): Promise<ScriptResponse> {
  let project = await ensureProject(bookId)
  project = await updateV2ContinuationProject(project.projectId, project.revision, {
    ...project.config,
    direction,
    pageCount,
  })
  if (referenceTokens) {
    project = await setV2ContinuationReferences(
      project.projectId,
      project.revision,
      referenceTokens,
    )
  }
  cacheProject(bookId, project)
  const accepted = await createV2ContinuationJob(bookId, { kind: 'script' })
  return { success: true, task_id: accepted.jobIds[0] }
}

export async function generateAllPageDetails(bookId: string): Promise<string> {
  const accepted = await createV2ContinuationJob(bookId, { kind: 'pages' })
  return accepted.jobIds[0]
}

export async function generateAllPageImages(
  bookId: string,
  ordinals?: number[],
): Promise<string> {
  const accepted = await createV2ContinuationJob(bookId, {
    kind: 'images',
    ...(ordinals ? { ordinals } : {}),
  })
  return accepted.jobIds[0]
}

export async function setContinuationReferenceTokens(
  bookId: string,
  assetIds: string[],
): Promise<void> {
  const project = await ensureProject(bookId)
  const updated = await setV2ContinuationReferences(
    project.projectId,
    project.revision,
    assetIds,
  )
  cacheProject(bookId, updated)
}

export async function activatePageImageVersion(
  bookId: string,
  pageNumber: number,
  imagePath: string,
): Promise<{ success: boolean }> {
  const project = await ensureProject(bookId)
  const page = project.pages.find(item => item.ordinal === pageNumber)
  const version = page?.imageVersions.find(item => item.assetUrl === imagePath)
  if (!page || !version) return { success: false }
  await activateV2ContinuationImage(page.continuationPageId, version.version)
  await refreshState(bookId)
  return { success: true }
}

export async function waitForContinuationJob(
  jobId: string,
  pollIntervalMs = 800,
  onProgress?: (progress: Record<string, unknown>) => void,
): Promise<Awaited<ReturnType<typeof getV2ContinuationJob>>> {
  while (true) {
    const job = await getV2ContinuationJob(jobId)
    onProgress?.(job.progress as Record<string, unknown>)
    if (job.status === 'completed' || job.status === 'completed_with_errors') return job
    if (['failed', 'cancelled', 'interrupted'].includes(job.status)) {
      const progress = job.progress as Record<string, unknown>
      const error = progress.error as Record<string, unknown> | undefined
      throw new Error(String(error?.message ?? '续写任务失败'))
    }
    await new Promise(resolve => setTimeout(resolve, pollIntervalMs))
  }
}
