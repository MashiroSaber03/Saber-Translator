import { apiClient } from './client'
import { downloadBlob } from './download'

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
  story_summary_ready?: boolean
  timeline_ready?: boolean
  characters_added?: number
  total_characters?: number
  synced_at?: string
  error?: string
  saved_data?: SavedContinuationData
}

export interface SyncContinuationResponse {
  success: boolean
  ready?: boolean
  message?: string
  story_summary_ready?: boolean
  timeline_ready?: boolean
  characters_added?: number
  total_characters?: number
  synced_at?: string
  error?: string
}

interface CharactersResponse {
  success: boolean
  characters?: CharacterProfile[]
  error?: string
}

interface UploadImageResponse {
  success: boolean
  image_path?: string
  error?: string
}

interface ScriptResponse {
  success: boolean
  script?: ChapterScript
  error?: string
}

interface ImageGenerateResponse {
  success: boolean
  image_path?: string
  pages?: PageContent[]
  session_id?: string
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

function continuationPathSegment(value: string): string {
  return encodeURIComponent(value)
}

function continuationEndpoint(bookId: string, suffix = ''): string {
  return `/api/manga-insight/${continuationPathSegment(bookId)}/continuation${suffix}`
}

function characterEndpoint(bookId: string, characterName: string, suffix = ''): string {
  return continuationEndpoint(bookId, `/characters/${encodeURIComponent(characterName)}${suffix}`)
}

function formEndpoint(
  bookId: string,
  characterName: string,
  formId: string,
  suffix = '',
): string {
  return characterEndpoint(
    bookId,
    characterName,
    `/forms/${encodeURIComponent(formId)}${suffix}`,
  )
}

function emptyJsonPostInit(): RequestInit {
  return {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({}),
  }
}

export async function prepareContinuation(bookId: string): Promise<PrepareResponse> {
  return apiClient.get(continuationEndpoint(bookId, '/prepare'))
}

export async function syncContinuationAnalysis(bookId: string): Promise<SyncContinuationResponse> {
  return apiClient.post(continuationEndpoint(bookId, '/sync'), {})
}

export async function getCharacters(bookId: string): Promise<CharactersResponse> {
  return apiClient.get(continuationEndpoint(bookId, '/characters'))
}

export async function addCharacter(
  bookId: string,
  data: { name: string; aliases?: string[]; description?: string },
): Promise<{ success: boolean; character?: CharacterProfile; error?: string }> {
  return apiClient.post(continuationEndpoint(bookId, '/characters'), data)
}

export async function deleteCharacter(
  bookId: string,
  characterName: string,
): Promise<{ success: boolean; message?: string; error?: string }> {
  return apiClient.delete(characterEndpoint(bookId, characterName))
}

export async function updateCharacterInfo(
  bookId: string,
  characterName: string,
  data: { name?: string; aliases?: string[]; enabled?: boolean },
): Promise<{ success: boolean; character?: CharacterProfile; error?: string }> {
  return apiClient.put(characterEndpoint(bookId, characterName), data)
}

export async function addCharacterForm(
  bookId: string,
  characterName: string,
  data: { form_id: string; form_name: string; description?: string },
): Promise<{ success: boolean; form?: CharacterForm; error?: string }> {
  return apiClient.post(characterEndpoint(bookId, characterName, '/forms'), data)
}

export async function updateCharacterForm(
  bookId: string,
  characterName: string,
  formId: string,
  data: { form_name?: string; description?: string },
): Promise<{ success: boolean; error?: string }> {
  return apiClient.put(formEndpoint(bookId, characterName, formId), data)
}

export async function deleteCharacterForm(
  bookId: string,
  characterName: string,
  formId: string,
): Promise<{ success: boolean; message?: string; error?: string }> {
  return apiClient.delete(formEndpoint(bookId, characterName, formId))
}

export async function toggleCharacterEnabled(
  bookId: string,
  characterName: string,
  enabled: boolean,
): Promise<{ success: boolean; enabled?: boolean; error?: string }> {
  return apiClient.post(characterEndpoint(bookId, characterName, '/toggle'), { enabled })
}

export async function toggleFormEnabled(
  bookId: string,
  characterName: string,
  formId: string,
  enabled: boolean,
): Promise<{ success: boolean; enabled?: boolean; error?: string }> {
  return apiClient.post(formEndpoint(bookId, characterName, formId, '/toggle'), { enabled })
}

export async function uploadFormImage(
  bookId: string,
  characterName: string,
  formId: string,
  formData: FormData,
): Promise<UploadImageResponse> {
  return apiClient.upload(formEndpoint(bookId, characterName, formId, '/image'), formData)
}

export async function deleteFormImage(
  bookId: string,
  characterName: string,
  formId: string,
): Promise<{ success: boolean; error?: string }> {
  return apiClient.delete(formEndpoint(bookId, characterName, formId, '/image'))
}

export async function generateFormOrtho(
  bookId: string,
  characterName: string,
  formId: string,
  sourceImages: File[],
): Promise<UploadImageResponse> {
  const formData = new FormData()
  sourceImages.forEach(file => formData.append('images', file))

  return apiClient.post(formEndpoint(bookId, characterName, formId, '/orthographic'), formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 0,
  })
}

export async function setFormReference(
  bookId: string,
  characterName: string,
  formId: string,
  imagePath: string,
): Promise<{ success: boolean; error?: string }> {
  return apiClient.post(formEndpoint(bookId, characterName, formId, '/set-reference'), {
    image_path: imagePath,
  })
}

export async function saveScript(
  bookId: string,
  script: ChapterScript,
): Promise<ScriptResponse> {
  return apiClient.post(continuationEndpoint(bookId, '/save-script'), { script })
}

export async function savePages(
  bookId: string,
  pages: PageContent[],
): Promise<{ success: boolean; error?: string }> {
  return apiClient.post(continuationEndpoint(bookId, '/save-pages'), { pages })
}

export async function saveConfig(
  bookId: string,
  config: {
    page_count: number
    style_reference_pages: number
    continuation_direction: string
  },
): Promise<{ success: boolean; error?: string }> {
  return apiClient.post(continuationEndpoint(bookId, '/save-config'), config)
}

export async function clearContinuationData(
  bookId: string,
): Promise<{ success: boolean; message?: string; error?: string }> {
  return apiClient.delete(continuationEndpoint(bookId, '/clear'))
}

export async function generateSinglePageDetails(
  bookId: string,
  script: ChapterScript,
  pageNumber: number,
): Promise<{ success: boolean; page?: PageContent; error?: string }> {
  return apiClient.post(
    continuationEndpoint(bookId, `/pages/${pageNumber}`),
    { script },
    { timeout: 0 },
  )
}

export async function getStyleReferences(
  bookId: string,
  count = 3,
): Promise<{ success: boolean; tokens?: string[]; error?: string }> {
  return apiClient.get(continuationEndpoint(bookId, `/style-references?count=${count}`))
}

export async function generatePageImage(
  bookId: string,
  pageNumber: number,
  page: PageContent,
  styleReferenceTokens: string[],
  sessionId?: string,
  styleRefCount = 3,
): Promise<ImageGenerateResponse> {
  return apiClient.post(continuationEndpoint(bookId, `/generate/${pageNumber}`), {
    page,
    style_reference_tokens: styleReferenceTokens,
    session_id: sessionId,
    style_ref_count: styleRefCount,
  }, {
    timeout: 0,
  })
}

export async function regeneratePageImage(
  bookId: string,
  pageNumber: number,
  page: PageContent,
  styleReferenceTokens: string[],
  sessionId?: string,
  styleRefCount = 3,
): Promise<ImageGenerateResponse> {
  return apiClient.post(continuationEndpoint(bookId, `/regenerate/${pageNumber}`), {
    page,
    style_reference_tokens: styleReferenceTokens,
    session_id: sessionId,
    style_ref_count: styleRefCount,
  }, {
    timeout: 0,
  })
}

export async function exportAsImages(bookId: string): Promise<Blob> {
  const { blob } = await downloadBlob({
    url: continuationEndpoint(bookId, '/export/images'),
    fallbackFilename: `${bookId}.continuation-images.zip`,
    fallbackErrorMessage: '导出失败',
    init: emptyJsonPostInit(),
  })
  return blob
}

export async function exportAsPdf(bookId: string): Promise<Blob> {
  const { blob } = await downloadBlob({
    url: continuationEndpoint(bookId, '/export/pdf'),
    fallbackFilename: `${bookId}.continuation.pdf`,
    fallbackErrorMessage: '导出失败',
    init: emptyJsonPostInit(),
  })
  return blob
}

export async function getAvailableImages(
  bookId: string,
  mode: 'script' | 'image' = 'script',
): Promise<AvailableImagesResponse> {
  const params = new URLSearchParams({ mode })
  return apiClient.get(continuationEndpoint(bookId, `/available-images?${params}`))
}

export async function generateScriptWithRefs(
  bookId: string,
  direction: string,
  pageCount: number,
  referenceTokens?: string[],
  referenceImageCount = 5,
): Promise<ScriptResponse> {
  return apiClient.post(continuationEndpoint(bookId, '/script'), {
    direction,
    page_count: pageCount,
    reference_tokens: referenceTokens || null,
    reference_image_count: referenceImageCount,
  }, {
    timeout: 0,
  })
}
