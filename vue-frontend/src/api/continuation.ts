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
  listV2ContinuationForms,
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
import { listInsightChapters, listInsightPages } from '@/api/v2/insight'

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
  status: 'pending' | 'generating' | 'generated' | 'stale' | 'failed'
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

interface ContinuationPreparation {
  ready: boolean
  message: string
  saved_data: SavedContinuationData
}

export type SyncContinuationResponse = ContinuationPreparation

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

export interface AvailableImages {
  original_images: MangaImageInfo[]
  continuation_images: MangaImageInfo[]
  character_forms: CharacterFormInfo[]
  total_original_pages: number
  original_cursor: number
  has_older_original_images: boolean
  has_more_character_forms: boolean
}

const stateCache = new Map<string, V2ContinuationState>()
interface ContinuationFormCache {
  items: V2ContinuationForm[]
  nextCursor: number | null
}

const FORM_PAGE_SIZE = 100
const CONTINUATION_CACHE_SIZE = 4
const formsCache = new Map<string, ContinuationFormCache>()

function cachedState(bookId: string): V2ContinuationState | undefined {
  const state = stateCache.get(bookId)
  if (!state) return undefined
  stateCache.delete(bookId)
  stateCache.set(bookId, state)
  return state
}

function cacheState(bookId: string, state: V2ContinuationState): V2ContinuationState {
  stateCache.delete(bookId)
  stateCache.set(bookId, state)
  while (stateCache.size > CONTINUATION_CACHE_SIZE) {
    const oldestBookId = stateCache.keys().next().value as string | undefined
    if (!oldestBookId) break
    const evicted = stateCache.get(oldestBookId)
    stateCache.delete(oldestBookId)
    const projectId = evicted?.project?.projectId
    if (projectId) formsCache.delete(projectId)
  }
  return state
}

function cachedForms(projectId: string): ContinuationFormCache | undefined {
  const entry = formsCache.get(projectId)
  if (!entry) return undefined
  formsCache.delete(projectId)
  formsCache.set(projectId, entry)
  return entry
}

function cacheForms(projectId: string, entry: ContinuationFormCache): ContinuationFormCache {
  formsCache.delete(projectId)
  formsCache.set(projectId, entry)
  while (formsCache.size > CONTINUATION_CACHE_SIZE) {
    const oldestProjectId = formsCache.keys().next().value as string | undefined
    if (!oldestProjectId) break
    formsCache.delete(oldestProjectId)
  }
  return entry
}

async function loadFirstFormPage(projectId: string): Promise<ContinuationFormCache> {
  const response = await listV2ContinuationForms(projectId, {
    cursor: 0,
    limit: FORM_PAGE_SIZE,
  })
  const entry = { items: response.items, nextCursor: response.nextCursor }
  return cacheForms(projectId, entry)
}

async function ensureFirstFormPage(projectId: string): Promise<ContinuationFormCache> {
  return cachedForms(projectId) ?? loadFirstFormPage(projectId)
}

async function loadNextFormPage(projectId: string): Promise<ContinuationFormCache> {
  const current = await ensureFirstFormPage(projectId)
  if (current.nextCursor === null) return current
  const response = await listV2ContinuationForms(projectId, {
    cursor: current.nextCursor,
    limit: FORM_PAGE_SIZE,
  })
  const known = new Set(current.items.map(form => form.formId))
  const entry = {
    items: [
      ...current.items,
      ...response.items.filter(form => !known.has(form.formId)),
    ],
    nextCursor: response.nextCursor,
  }
  return cacheForms(projectId, entry)
}

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
    character_forms: (Array.isArray(payload.characterForms)
      ? payload.characterForms
      : []) as CharacterFormSelection[],
    final_prompt: payloadString(payload, 'finalPrompt'),
    image_url: currentImage?.assetUrl ?? '',
    previous_url: oldImage?.assetUrl ?? '',
    status: Boolean(payload.staleReason) || rawStatus === 'stale'
      ? 'stale'
      : currentImage
        ? 'generated'
        : rawStatus === 'failed'
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
    config: project
      ? {
          page_count: Number(project.config.pageCount ?? 15),
          style_reference_pages: Number(project.config.styleReferencePages ?? 3),
          continuation_direction: String(project.config.direction ?? ''),
        }
      : null,
    has_data: Boolean(project),
  }
}

async function refreshState(bookId: string): Promise<V2ContinuationState> {
  const previousProjectId = cachedState(bookId)?.project?.projectId
  const state = await getV2Continuation(bookId)
  cacheState(bookId, state)
  if (previousProjectId && previousProjectId !== state.project?.projectId) {
    formsCache.delete(previousProjectId)
  }
  return state
}

async function ensureProject(bookId: string): Promise<V2ContinuationProject> {
  let state = cachedState(bookId) ?? (await refreshState(bookId))
  if (!state.project) {
    if (!state.ready) {
      throw new Error(`续写前置数据未就绪：${state.missing.join('、')}`)
    }
    const project = await syncV2Continuation(bookId)
    state = { ...state, project }
    cacheState(bookId, state)
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
  const previous = cachedState(bookId)
  cacheState(bookId, {
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
  formId: string
): Promise<V2ContinuationForm> {
  const character = characterFor(project, characterName)
  let entry = await ensureFirstFormPage(project.projectId)
  let form = entry.items.find(
    item => item.characterId === character.characterId && item.formId === formId
  )
  while (!form && entry.nextCursor !== null) {
    entry = await loadNextFormPage(project.projectId)
    form = entry.items.find(
      item => item.characterId === character.characterId && item.formId === formId
    )
  }
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
    reference_image: adopted?.assetUrl ?? latestGenerated?.assetUrl ?? form.referenceAssetUrl ?? '',
    enabled: form.payload.enabled !== false,
  }
}

function mapCharacter(
  character: V2ContinuationCharacter,
  forms: V2ContinuationForm[]
): CharacterProfile {
  const characterForms = forms.filter(form => form.characterId === character.characterId)
  const reference = characterForms
    .map(
      form =>
        form.imageVersions.find(version => version.adopted)?.assetUrl ?? form.referenceAssetUrl
    )
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

export async function prepareContinuation(bookId: string): Promise<ContinuationPreparation> {
  let state = await refreshState(bookId)
  if (state.ready && !state.project) {
    const project = await syncV2Continuation(bookId)
    state = { ...state, project }
    cacheState(bookId, state)
    formsCache.delete(project.projectId)
  }
  return {
    ready: state.ready,
    message: state.ready ? '续写数据已就绪' : `缺少：${state.missing.join('、')}`,
    saved_data: savedData(state.project),
  }
}

export async function syncContinuationAnalysis(bookId: string): Promise<SyncContinuationResponse> {
  const project = await syncV2Continuation(bookId)
  cacheProject(bookId, project)
  formsCache.delete(project.projectId)
  return {
    ready: true,
    message: '分析数据同步完成',
    saved_data: savedData(project),
  }
}

export async function getCharacters(bookId: string): Promise<CharacterProfile[]> {
  const project = await ensureProject(bookId)
  const forms = await loadFirstFormPage(project.projectId)
  return project.characters.map(character => mapCharacter(character, forms.items))
}

export function hasMoreCharacterForms(bookId: string): boolean {
  const projectId = cachedState(bookId)?.project?.projectId
  if (!projectId) return false
  const entry = cachedForms(projectId)
  return Boolean(entry && entry.nextCursor !== null)
}

export async function loadMoreCharacterForms(bookId: string): Promise<CharacterProfile[]> {
  const project = await ensureProject(bookId)
  const forms = await loadNextFormPage(project.projectId)
  return project.characters.map(character => mapCharacter(character, forms.items))
}

export async function addCharacter(
  bookId: string,
  data: { name: string; aliases?: string[]; description?: string }
): Promise<CharacterProfile> {
  const project = await ensureProject(bookId)
  await createV2ContinuationCharacter(project.projectId, {
    name: data.name,
    aliases: data.aliases ?? [],
    enabled: true,
    payload: { description: data.description ?? '' },
  })
  const refreshed = await refreshProject(bookId)
  const forms = cachedForms(refreshed.projectId)?.items ?? []
  const character = refreshed.characters.find(item => item.name === data.name)
  if (!character) throw new Error('角色创建后未能重新加载')
  return mapCharacter(character, forms)
}

export async function deleteCharacter(bookId: string, characterName: string): Promise<void> {
  const project = await ensureProject(bookId)
  const character = characterFor(project, characterName)
  await deleteV2ContinuationCharacter(character.characterId, character.revision)
  formsCache.delete(project.projectId)
  await refreshState(bookId)
}

export async function updateCharacterInfo(
  bookId: string,
  characterName: string,
  data: { name?: string; aliases?: string[]; enabled?: boolean }
): Promise<CharacterProfile> {
  const project = await ensureProject(bookId)
  const character = characterFor(project, characterName)
  const updated = await updateV2ContinuationCharacter(character.characterId, {
    baseRevision: character.revision,
    name: data.name ?? character.name,
    aliases: data.aliases ?? character.aliases,
    enabled: data.enabled ?? character.enabled,
    payload: character.payload,
  })
  const forms = cachedForms(project.projectId)?.items ?? []
  formsCache.delete(project.projectId)
  await refreshState(bookId)
  return mapCharacter(updated, forms)
}

export async function addCharacterForm(
  bookId: string,
  characterName: string,
  data: { form_name: string; description?: string }
): Promise<CharacterForm> {
  const project = await ensureProject(bookId)
  const character = characterFor(project, characterName)
  const form = await createV2ContinuationForm(character.characterId, {
    name: data.form_name,
    payload: {
      description: data.description ?? '',
      enabled: true,
    },
  })
  formsCache.delete(project.projectId)
  await refreshState(bookId)
  return mapForm(form)
}

export async function updateCharacterForm(
  bookId: string,
  characterName: string,
  formId: string,
  data: { form_name?: string; description?: string }
): Promise<void> {
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
  formsCache.delete(project.projectId)
  await refreshState(bookId)
}

export async function deleteCharacterForm(
  bookId: string,
  characterName: string,
  formId: string
): Promise<void> {
  const project = await ensureProject(bookId)
  const form = await formFor(project, characterName, formId)
  await deleteV2ContinuationForm(form.formId, form.revision)
  formsCache.delete(project.projectId)
  await refreshState(bookId)
}

export async function toggleFormEnabled(
  bookId: string,
  characterName: string,
  formId: string,
  enabled: boolean
): Promise<void> {
  const project = await ensureProject(bookId)
  const form = await formFor(project, characterName, formId)
  await updateV2ContinuationForm(form.formId, {
    baseRevision: form.revision,
    name: form.name,
    payload: { ...form.payload, enabled },
  })
  formsCache.delete(project.projectId)
  await refreshState(bookId)
}

export async function uploadFormImage(
  bookId: string,
  characterName: string,
  formId: string,
  file: File
): Promise<string | null> {
  const project = await ensureProject(bookId)
  const form = await formFor(project, characterName, formId)
  const updated = await uploadV2ContinuationReference(form.formId, form.revision, file)
  formsCache.delete(project.projectId)
  await refreshState(bookId)
  return updated.referenceAssetUrl ?? null
}

export async function deleteFormImage(
  bookId: string,
  characterName: string,
  formId: string
): Promise<void> {
  const project = await ensureProject(bookId)
  const form = await formFor(project, characterName, formId)
  await deleteV2ContinuationReference(form.formId, form.revision)
  formsCache.delete(project.projectId)
  await refreshState(bookId)
}

export async function generateFormOrtho(
  bookId: string,
  characterName: string,
  formId: string,
  sourceImages: File[]
): Promise<string> {
  const project = await ensureProject(bookId)
  let form = await formFor(project, characterName, formId)
  if (sourceImages[0]) {
    form = await uploadV2ContinuationReference(form.formId, form.revision, sourceImages[0])
    formsCache.delete(project.projectId)
  }
  const accepted = await createV2ContinuationJob(bookId, {
    kind: 'character_sheet',
    formId: form.formId,
  })
  return accepted.jobIds[0]
}

export async function setFormReference(
  bookId: string,
  characterName: string,
  formId: string,
  imagePath: string
): Promise<void> {
  const project = await ensureProject(bookId)
  const form = await formFor(project, characterName, formId)
  const version = form.imageVersions.find(
    item => item.assetUrl === imagePath || item.assetId === imagePath
  )
  if (!version) throw new Error('未找到生成结果版本')
  await adoptV2ContinuationFormImage(form.formId, version.version, form.revision)
  formsCache.delete(project.projectId)
  await refreshState(bookId)
}

export async function saveScript(bookId: string, script: ChapterScript): Promise<ChapterScript> {
  const project = await ensureProject(bookId)
  await updateV2ContinuationScript(
    project.projectId,
    project.script?.revision ?? 0,
    script.script_text
  )
  const refreshed = await refreshProject(bookId)
  return mapScript(refreshed) ?? script
}

export async function savePages(bookId: string, pages: PageContent[]): Promise<void> {
  const project = await ensureProject(bookId)
  const byOrdinal = new Map(project.pages.map(page => [page.ordinal, page]))
  for (const page of pages) {
    const stored = byOrdinal.get(page.page_number)
    if (!stored) continue
    await updateV2ContinuationPage(
      stored.continuationPageId,
      stored.revision,
      pagePayload(page),
    )
  }
  await refreshState(bookId)
}

export async function saveConfig(
  bookId: string,
  config: {
    page_count: number
    style_reference_pages: number
    continuation_direction: string
  }
): Promise<void> {
  const project = await ensureProject(bookId)
  const updated = await updateV2ContinuationProject(project.projectId, project.revision, {
    pageCount: config.page_count,
    styleReferencePages: config.style_reference_pages,
    direction: config.continuation_direction,
  })
  cacheProject(bookId, updated)
}

export async function clearContinuationData(bookId: string): Promise<void> {
  const projectId = cachedState(bookId)?.project?.projectId
  await clearV2Continuation(bookId)
  stateCache.delete(bookId)
  if (projectId) formsCache.delete(projectId)
}

async function savePageBeforeImage(
  bookId: string,
  pageNumber: number,
  page: PageContent
): Promise<void> {
  const project = await ensureProject(bookId)
  const stored = project.pages.find(item => item.ordinal === pageNumber)
  if (!stored) throw new Error('请先生成页面剧情')
  await updateV2ContinuationPage(stored.continuationPageId, stored.revision, pagePayload(page))
  await refreshState(bookId)
}

export async function regeneratePageImage(
  bookId: string,
  pageNumber: number,
  page: PageContent
): Promise<string> {
  await savePageBeforeImage(bookId, pageNumber, page)
  const accepted = await createV2ContinuationJob(bookId, {
    kind: 'images',
    ordinals: [pageNumber],
  })
  return accepted.jobIds[0]
}

export async function createContinuationExportJob(
  bookId: string,
  format: 'pdf' | 'zip'
): Promise<string> {
  const accepted = await createV2ContinuationJob(bookId, { kind: 'export', format })
  return accepted.jobIds[0]
}

export async function downloadContinuationExport(
  assetId: string,
  bookId: string,
  format: 'pdf' | 'zip'
): Promise<Blob> {
  const { blob } = await downloadBlob({
    url: `/api/v2/assets/${assetId}`,
    fallbackFilename:
      format === 'pdf' ? `${bookId}.continuation.pdf` : `${bookId}.continuation-images.zip`,
    fallbackErrorMessage: '导出失败',
  })
  return blob
}

export async function getAvailableImages(
  bookId: string,
  originalCursor?: number
): Promise<AvailableImages> {
  const [project, chapterPage] = await Promise.all([
    ensureProject(bookId),
    listInsightChapters(bookId),
  ])
  const totalOriginalPages = chapterPage.items.reduce(
    (total, chapter) => total + chapter.pageCount,
    0
  )
  const cursor = originalCursor ?? Math.max(0, totalOriginalPages - 100)
  const sourcePage = await listInsightPages(bookId, { cursor, limit: 100 })
  const sourcePages = sourcePage.items
  const forms = await ensureFirstFormPage(project.projectId)
  return {
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
      return image
        ? [
            {
              token: image.assetId,
              page_number: page.ordinal,
              path: image.thumbnailUrl,
              has_image: true,
              label: `续写第 ${page.ordinal} 页`,
            },
          ]
        : []
    }),
    character_forms: mapAvailableCharacterForms(project, forms.items),
    total_original_pages: totalOriginalPages,
    original_cursor: cursor,
    has_older_original_images: cursor > 0,
    has_more_character_forms: forms.nextCursor !== null,
  }
}

function mapAvailableCharacterForms(
  project: V2ContinuationProject,
  forms: V2ContinuationForm[],
): CharacterFormInfo[] {
  return forms.flatMap(form => {
      const image = form.imageVersions.find(version => version.adopted)
      const path = image?.thumbnailUrl ?? form.referenceThumbnailUrl
      return path
        ? [
            {
              token: image?.assetId ?? form.referenceAssetId ?? '',
              character_name:
                project.characters.find(character => character.characterId === form.characterId)
                  ?.name ?? '',
              form_id: form.formId,
              form_name: form.name,
              path,
              has_image: true,
            },
          ]
        : []
  })
}

export async function loadMoreAvailableCharacterForms(
  bookId: string,
): Promise<{ character_forms: CharacterFormInfo[]; has_more_character_forms: boolean }> {
  const project = await ensureProject(bookId)
  const forms = await loadNextFormPage(project.projectId)
  return {
    character_forms: mapAvailableCharacterForms(project, forms.items),
    has_more_character_forms: forms.nextCursor !== null,
  }
}

export async function generateScriptWithRefs(
  bookId: string,
  direction: string,
  pageCount: number,
  referenceTokens?: string[],
  referenceImageCount = 5
): Promise<string> {
  let project = await ensureProject(bookId)
  project = await updateV2ContinuationProject(project.projectId, project.revision, {
    ...project.config,
    direction,
    pageCount,
    styleReferencePages: referenceImageCount,
  })
  if (referenceTokens) {
    project = await setV2ContinuationReferences(
      project.projectId,
      project.revision,
      referenceTokens
    )
  }
  cacheProject(bookId, project)
  const accepted = await createV2ContinuationJob(bookId, { kind: 'script' })
  return accepted.jobIds[0]
}

export async function generateAllPageDetails(bookId: string): Promise<string> {
  const accepted = await createV2ContinuationJob(bookId, { kind: 'pages' })
  return accepted.jobIds[0]
}

export async function generateAllPageImages(bookId: string, ordinals?: number[]): Promise<string> {
  const accepted = await createV2ContinuationJob(bookId, {
    kind: 'images',
    ...(ordinals ? { ordinals } : {}),
  })
  return accepted.jobIds[0]
}

export async function setContinuationReferenceTokens(
  bookId: string,
  assetIds: string[]
): Promise<void> {
  const project = await ensureProject(bookId)
  const updated = await setV2ContinuationReferences(project.projectId, project.revision, assetIds)
  cacheProject(bookId, updated)
}

export async function activatePageImageVersion(
  bookId: string,
  pageNumber: number,
  imagePath: string
): Promise<void> {
  const project = await ensureProject(bookId)
  const page = project.pages.find(item => item.ordinal === pageNumber)
  const version = page?.imageVersions.find(item => item.assetUrl === imagePath)
  if (!page || !version) throw new Error('未找到要启用的图片版本')
  await activateV2ContinuationImage(page.continuationPageId, version.version)
  await refreshState(bookId)
}
