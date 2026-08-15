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

export interface CharacterProfilePage {
  items: CharacterProfile[]
  nextCursor: number | null
}

export interface ChapterScript {
  chapter_title: string
  page_count: number
  script_text: string
  generated_at: string
}

export interface PageContent {
  page_number: number
  continuity_text: string
  story_text: string
  dialogue_text: string
  characters: string[]
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

export interface OriginalReferenceImages {
  original_images: MangaImageInfo[]
  original_cursor: number
}

export interface AvailableImages extends OriginalReferenceImages {
  continuation_images: MangaImageInfo[]
  character_forms: CharacterFormInfo[]
  character_forms_cursor: number | null
}

const FORM_PAGE_SIZE = 100

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
  const rawStatus = payload.status
  return {
    page_number: page.ordinal,
    continuity_text: payload.continuityText,
    story_text: payload.storyText,
    dialogue_text: payload.dialogueText,
    characters: [...payload.characters],
    final_prompt: payload.finalPrompt,
    image_url: currentImage?.assetUrl ?? '',
    previous_url: oldImage?.assetUrl ?? '',
    status: Boolean(payload.staleReason) || rawStatus === 'stale'
      ? 'stale'
      : rawStatus === 'ready'
        ? 'generated'
        : rawStatus === 'failed'
          ? 'failed'
          : rawStatus === 'generating'
            ? 'generating'
            : 'pending',
  }
}

function pagePayload(page: PageContent): V2ContinuationPage['payload'] {
  return {
    continuityText: page.continuity_text,
    storyText: page.story_text,
    dialogueText: page.dialogue_text,
    characters: page.characters,
    finalPrompt: page.final_prompt,
    status: page.status === 'generated' ? 'ready' : page.status,
  }
}

function mapScript(project: V2ContinuationProject): ChapterScript | null {
  if (!project.script) return null
  return {
    chapter_title: '续写章节',
    page_count: project.config.pageCount,
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
          page_count: project.config.pageCount,
          style_reference_pages: project.config.styleReferencePages,
          continuation_direction: project.config.direction,
        }
      : null,
    has_data: Boolean(project),
  }
}

async function ensureProject(bookId: string): Promise<V2ContinuationProject> {
  const state = await getV2Continuation(bookId)
  if (state.project) return state.project
  if (!state.ready) {
    throw new Error(`续写前置数据未就绪：${state.missing.join('、')}`)
  }
  return syncV2Continuation(bookId)
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
  let cursor: number | null = 0
  while (cursor !== null) {
    const page = await listV2ContinuationForms(project.projectId, {
      cursor,
      limit: FORM_PAGE_SIZE,
    })
    const form = page.items.find(
      item => item.characterId === character.characterId && item.formId === formId
    )
    if (form) return form
    cursor = page.nextCursor
  }
  throw new Error(`角色形态不存在：${formId}`)
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
  let state = await getV2Continuation(bookId)
  if (state.ready && !state.project) {
    const project = await syncV2Continuation(bookId)
    state = { ...state, project }
  }
  return {
    ready: state.ready,
    message: state.ready ? '续写数据已就绪' : `缺少：${state.missing.join('、')}`,
    saved_data: savedData(state.project),
  }
}

export async function syncContinuationAnalysis(bookId: string): Promise<SyncContinuationResponse> {
  const project = await syncV2Continuation(bookId)
  return {
    ready: true,
    message: '分析数据同步完成',
    saved_data: savedData(project),
  }
}

export async function getCharacters(
  bookId: string,
  cursor = 0,
): Promise<CharacterProfilePage> {
  const project = await ensureProject(bookId)
  const forms = await listV2ContinuationForms(project.projectId, {
    cursor,
    limit: FORM_PAGE_SIZE,
  })
  return {
    items: project.characters.map(character => mapCharacter(character, forms.items)),
    nextCursor: forms.nextCursor,
  }
}

export async function addCharacter(
  bookId: string,
  data: { name: string; aliases?: string[]; description?: string }
): Promise<CharacterProfile> {
  const project = await ensureProject(bookId)
  const created = await createV2ContinuationCharacter(project.projectId, {
    name: data.name,
    aliases: data.aliases ?? [],
    enabled: true,
    payload: { description: data.description ?? '' },
  })
  return mapCharacter(created, [])
}

export async function deleteCharacter(bookId: string, characterName: string): Promise<void> {
  const project = await ensureProject(bookId)
  const character = characterFor(project, characterName)
  await deleteV2ContinuationCharacter(character.characterId, character.revision)
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
  return mapCharacter(updated, [])
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
}

export async function deleteCharacterForm(
  bookId: string,
  characterName: string,
  formId: string
): Promise<void> {
  const project = await ensureProject(bookId)
  const form = await formFor(project, characterName, formId)
  await deleteV2ContinuationForm(form.formId, form.revision)
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
}

export async function generateFormOrtho(
  bookId: string,
  characterName: string,
  formId: string,
  sourceImage: File
): Promise<string> {
  const project = await ensureProject(bookId)
  let form = await formFor(project, characterName, formId)
  form = await uploadV2ContinuationReference(form.formId, form.revision, sourceImage)
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
}

export async function saveScript(bookId: string, script: ChapterScript): Promise<ChapterScript> {
  const project = await ensureProject(bookId)
  const updated = await updateV2ContinuationScript(
    project.projectId,
    project.script?.revision ?? 0,
    script.script_text
  )
  if (!updated) return script
  return {
    chapter_title: script.chapter_title,
    page_count: project.config.pageCount,
    script_text: updated.content,
    generated_at: script.generated_at,
  }
}

export async function savePages(bookId: string, pages: PageContent[]): Promise<void> {
  let project = await ensureProject(bookId)
  for (const page of pages) {
    const stored = project.pages.find(item => item.ordinal === page.page_number)
    if (!stored) continue
    const payload = pagePayload(page)
    if (JSON.stringify(pagePayload(mapPage(stored))) === JSON.stringify(payload)) continue

    const updated = await updateV2ContinuationPage(
      stored.continuationPageId,
      stored.revision,
      payload,
    )
    project = {
      ...project,
      pages: project.pages.map(item => (
        item.continuationPageId === updated.continuationPageId ? updated : item
      )),
    }
  }
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
  await updateV2ContinuationProject(project.projectId, project.revision, {
    pageCount: config.page_count,
    styleReferencePages: config.style_reference_pages,
    direction: config.continuation_direction,
  })
}

export async function clearContinuationData(bookId: string): Promise<void> {
  await clearV2Continuation(bookId)
}

async function savePageBeforeImage(
  bookId: string,
  pageNumber: number,
  page: PageContent
): Promise<void> {
  const project = await ensureProject(bookId)
  const stored = project.pages.find(item => item.ordinal === pageNumber)
  if (!stored) throw new Error('请先生成页面剧情')
  await updateV2ContinuationPage(
    stored.continuationPageId,
    stored.revision,
    pagePayload(page),
  )
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
  const [project, originalImages] = await Promise.all([
    ensureProject(bookId),
    getOriginalReferenceImages(bookId, originalCursor),
  ])
  const forms = await listV2ContinuationForms(project.projectId, {
    cursor: 0,
    limit: FORM_PAGE_SIZE,
  })
  return {
    ...originalImages,
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
    character_forms_cursor: forms.nextCursor,
  }
}

export async function getOriginalReferenceImages(
  bookId: string,
  requestedCursor?: number,
): Promise<OriginalReferenceImages> {
  const chapterPage = await listInsightChapters(bookId)
  const totalOriginalPages = chapterPage.items.reduce(
    (total, chapter) => total + chapter.pageCount,
    0
  )
  const cursor = requestedCursor ?? Math.max(0, totalOriginalPages - 100)
  const sourcePage = await listInsightPages(bookId, { cursor, limit: 100 })
  return {
    original_images: sourcePage.items.map(page => ({
      token: page.sourceAssetId,
      page_number: page.displayPageNumber,
      path: page.thumbnailUrl ?? '',
      has_image: Boolean(page.thumbnailUrl),
      is_placeholder: !page.thumbnailUrl,
      label: `原作第 ${page.displayPageNumber} 页`,
    })),
    original_cursor: cursor,
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
  cursor: number,
): Promise<{ character_forms: CharacterFormInfo[]; next_cursor: number | null }> {
  const project = await ensureProject(bookId)
  const forms = await listV2ContinuationForms(project.projectId, {
    cursor,
    limit: FORM_PAGE_SIZE,
  })
  return {
    character_forms: mapAvailableCharacterForms(project, forms.items),
    next_cursor: forms.nextCursor,
  }
}

export async function generateScriptWithRefs(
  bookId: string,
  direction: string,
  pageCount: number,
  referenceTokens?: string[],
  referenceImageCount = 5
): Promise<string> {
  const project = await ensureProject(bookId)
  const updatedProject = await updateV2ContinuationProject(project.projectId, project.revision, {
    ...project.config,
    direction,
    pageCount,
    styleReferencePages: referenceImageCount,
  })
  if (referenceTokens) {
    await setV2ContinuationReferences(
      updatedProject.projectId,
      updatedProject.revision,
      referenceTokens
    )
  }
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
  await setV2ContinuationReferences(project.projectId, project.revision, assetIds)
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
}
