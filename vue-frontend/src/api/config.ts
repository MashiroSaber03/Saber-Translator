import textStyleDefaultsJson from '../../../src/shared/text_style_defaults_factory.json'
import { apiClient } from './client'
import { normalizeProviderId } from '@/config/aiProviders'
import {
  createV2Prompt,
  deleteV2Prompt,
  fetchV2ModelCatalog,
  getV2Settings,
  listV2Fonts,
  listV2Prompts,
  resetV2Prompt,
  runV2ConnectionTest,
  saveV2SettingsTransaction,
  updateV2Prompt,
  updateV2WorkflowPreferences,
  uploadV2Font,
  type V2Prompt,
} from '@/api/v2/settings'
import type {
  ApiResponse,
  ConnectionTestResponse,
  FetchModelsResponse,
  FontListResponse,
  PromptListResponse,
} from '@/types'
import type { TextStyleSettings } from '@/types/settings'
import type { WorkflowMode } from '@/types/workflow'

export interface PromptContentResponse {
  success?: boolean
  prompt_content?: string
  error?: string
}

export interface AiVisionOcrTestParams {
  provider: string
  apiKey: string
  modelName: string
  customBaseUrl?: string
  prompt?: string
  domain?: string
}

export interface AiTranslateTestParams {
  provider: string
  apiKey: string
  modelName?: string
  baseUrl?: string
  domain?: string
}

export interface FontUploadResponse {
  success: boolean
  fontPath?: string
  error?: string
}

export interface UserSettingsResponse {
  success: boolean
  settings?: Record<string, unknown>
  error?: string
}

export interface TextStyleDefaultsResponse {
  success: boolean
  defaults?: TextStyleSettings
  error?: string
}

export interface SaveTextStyleDefaultsResponse {
  success: boolean
  defaults?: TextStyleSettings
  error?: string
}

export interface TranslateWorkflowPreferences {
  rememberWorkflowModeEnabled: boolean
  lastWorkflowMode: WorkflowMode
}

export interface TranslateWorkflowPreferencesResponse {
  success: boolean
  preferences?: TranslateWorkflowPreferences
  error?: string
}

const promptCache = new Map<string, V2Prompt[]>()
const settingRevisions = new Map<string, number>()

async function promptsFor(type: string, refresh = false): Promise<V2Prompt[]> {
  if (!refresh && promptCache.has(type)) return promptCache.get(type)!
  const prompts = await listV2Prompts(type)
  promptCache.set(type, prompts)
  return prompts
}

async function promptByName(type: string, name: string): Promise<V2Prompt> {
  const prompt = (await promptsFor(type, true)).find(item => item.name === name)
  if (!prompt) throw new Error(`提示词不存在：${name}`)
  return prompt
}

function cachePrompt(type: string, prompt: V2Prompt): void {
  const current = promptCache.get(type) || []
  promptCache.set(type, [
    ...current.filter(item => item.id !== prompt.id),
    prompt,
  ].sort((left, right) => left.name.localeCompare(right.name)))
}

async function loadSetting(domain: string): Promise<Record<string, unknown>> {
  const document = await getV2Settings([domain])
  const row = document.settings.find(item => item.domain === domain)
  settingRevisions.set(domain, row?.revision ?? 0)
  return row?.payload || {}
}

async function saveSetting(
  domain: string,
  payload: Record<string, unknown>,
  schemaVersion = 1,
): Promise<void> {
  if (!settingRevisions.has(domain)) await loadSetting(domain)
  const result = await saveV2SettingsTransaction({
    settings: [{
      domain,
      payload,
      baseRevision: settingRevisions.get(domain) ?? 0,
      schemaVersion,
    }],
  })
  const mutation = result.settings.find(item => item.domain === domain)
  if (mutation) settingRevisions.set(domain, mutation.revision)
}

function secretOrDomain(
  domain: string,
  secret: Record<string, string>,
): { domain?: string; secret?: Record<string, string> } {
  const present = Object.fromEntries(
    Object.entries(secret).filter(([, value]) => value.trim().length > 0),
  )
  return Object.keys(present).length > 0 ? { secret: present } : { domain }
}

export async function getPrompts(type = 'translate'): Promise<PromptListResponse> {
  const prompts = await promptsFor(type, true)
  return {
    success: true,
    prompt_names: prompts.map(item => item.name),
    default_prompt_content: prompts.find(item => item.isFactoryDefault)?.content,
  }
}

export async function getPromptContent(type: string, name: string): Promise<PromptContentResponse> {
  const prompt = await promptByName(type, name)
  return { success: true, prompt_content: prompt.content }
}

export async function savePrompt(type: string, name: string, content: string): Promise<ApiResponse> {
  const existing = (await promptsFor(type, true)).find(item => item.name === name)
  const saved = existing
    ? await updateV2Prompt({ ...existing, content })
    : await createV2Prompt(type, name, content)
  cachePrompt(type, saved)
  return { success: true }
}

export async function deletePrompt(type: string, name: string): Promise<ApiResponse> {
  const prompt = await promptByName(type, name)
  await deleteV2Prompt(prompt.id)
  promptCache.set(type, (promptCache.get(type) || []).filter(item => item.id !== prompt.id))
  return { success: true }
}

export async function resetPromptToDefault(name: string): Promise<PromptContentResponse> {
  const prompt = await promptByName('translate', name)
  const reset = await resetV2Prompt(prompt)
  cachePrompt('translate', reset)
  return { success: true, prompt_content: reset.content }
}

export function getTextboxPrompts(): Promise<PromptListResponse> {
  return getPrompts('textbox')
}

export function getTextboxPromptContent(name: string): Promise<PromptContentResponse> {
  return getPromptContent('textbox', name)
}

export function saveTextboxPrompt(name: string, content: string): Promise<ApiResponse> {
  return savePrompt('textbox', name, content)
}

export function deleteTextboxPrompt(name: string): Promise<ApiResponse> {
  return deletePrompt('textbox', name)
}

export async function resetTextboxPromptToDefault(name: string): Promise<PromptContentResponse> {
  const prompt = await promptByName('textbox', name)
  const reset = await resetV2Prompt(prompt)
  cachePrompt('textbox', reset)
  return { success: true, prompt_content: reset.content }
}

export async function fetchModels(
  provider: string,
  apiKey: string,
  baseUrl?: string,
  domain = 'translation',
): Promise<FetchModelsResponse> {
  return fetchV2ModelCatalog({
    provider: normalizeProviderId(provider),
    baseUrl: baseUrl || undefined,
    ...secretOrDomain(domain, { apiKey }),
  })
}

export function testOllamaConnection(baseUrl?: string): Promise<ConnectionTestResponse> {
  return runV2ConnectionTest('ollama', { baseUrl, domain: 'translation' })
}

export function testSakuraConnection(baseUrl?: string): Promise<ConnectionTestResponse> {
  return runV2ConnectionTest('sakura', { baseUrl, domain: 'translation' })
}

export function testBaiduOcrConnection(
  apiKey: string,
  secretKey: string,
): Promise<ConnectionTestResponse> {
  return runV2ConnectionTest('baidu_ocr', {
    ...secretOrDomain('ocr', { apiKey, secretKey }),
  })
}

export function testLamaRepair(): Promise<ConnectionTestResponse> {
  return runV2ConnectionTest('lama_repair')
}

export function testAiVisionOcrConnection(
  params: AiVisionOcrTestParams,
): Promise<ConnectionTestResponse> {
  return runV2ConnectionTest('ai_vision_ocr', {
    provider: normalizeProviderId(params.provider),
    model: params.modelName,
    baseUrl: params.customBaseUrl || undefined,
    prompt: params.prompt || undefined,
    ...secretOrDomain(params.domain || 'ai_vision_ocr', { apiKey: params.apiKey }),
  })
}

export function testAiTranslateConnection(
  params: AiTranslateTestParams,
): Promise<ConnectionTestResponse> {
  return runV2ConnectionTest('ai_translate', {
    provider: normalizeProviderId(params.provider),
    model: params.modelName || undefined,
    baseUrl: params.baseUrl || undefined,
    ...secretOrDomain(params.domain || 'translation', { apiKey: params.apiKey }),
  })
}

export function testBaiduTranslateConnection(
  appId: string,
  appKey: string,
): Promise<ConnectionTestResponse> {
  return runV2ConnectionTest('baidu_translate', {
    provider: 'baidu_translate',
    ...secretOrDomain('translation', { appId, appKey }),
  })
}

export function testYoudaoTranslateConnection(
  appKey: string,
  appSecret: string,
): Promise<ConnectionTestResponse> {
  return runV2ConnectionTest('youdao_translate', {
    provider: 'youdao_translate',
    ...secretOrDomain('translation', { appKey, appSecret }),
  })
}

export async function getFontList(): Promise<FontListResponse> {
  const fonts = await listV2Fonts()
  return {
    success: true,
    fonts: fonts.map(font => ({
      id: font.id,
      kind: font.kind,
      file_name: font.builtinKey || font.displayName,
      display_name: font.displayName,
      path: font.id,
      is_default: font.kind === 'builtin',
    })),
    default_fonts: Object.fromEntries(
      fonts
        .filter(font => font.kind === 'builtin')
        .map(font => [font.displayName, font.id]),
    ),
  }
}

export async function uploadFont(file: File): Promise<FontUploadResponse> {
  const uploaded = await uploadV2Font(file)
  return { success: true, fontPath: uploaded.id }
}

export async function getUserSettings(): Promise<UserSettingsResponse> {
  return { success: true, settings: await loadSetting('translation') }
}

export async function getTextStyleDefaults(): Promise<TextStyleDefaultsResponse> {
  return {
    success: true,
    defaults: await loadSetting('text_style_defaults') as unknown as TextStyleSettings,
  }
}

export async function saveTextStyleDefaults(
  defaults: TextStyleSettings,
): Promise<SaveTextStyleDefaultsResponse> {
  await saveSetting(
    'text_style_defaults',
    defaults as unknown as Record<string, unknown>,
  )
  return { success: true, defaults }
}

export async function resetTextStyleDefaults(): Promise<SaveTextStyleDefaultsResponse> {
  const defaults = structuredClone(textStyleDefaultsJson) as TextStyleSettings
  return saveTextStyleDefaults(defaults)
}

export async function getTranslateWorkflowPreferences(): Promise<TranslateWorkflowPreferencesResponse> {
  const payload = await loadSetting('workflow_preferences')
  return {
    success: true,
    preferences: {
      rememberWorkflowModeEnabled: Boolean(payload.rememberWorkflowModeEnabled),
      lastWorkflowMode: String(payload.lastWorkflowMode || 'standard') as WorkflowMode,
    },
  }
}

export async function saveTranslateWorkflowPreferences(
  preferences: TranslateWorkflowPreferences,
): Promise<TranslateWorkflowPreferencesResponse> {
  if (!settingRevisions.has('workflow_preferences')) {
    await loadSetting('workflow_preferences')
  }
  const updated = await updateV2WorkflowPreferences(
    preferences as unknown as Record<string, unknown>,
    settingRevisions.get('workflow_preferences') ?? 0,
  )
  settingRevisions.set('workflow_preferences', updated.revision)
  return { success: true, preferences }
}

export async function saveUserSettings(
  settings: Record<string, unknown>,
): Promise<ApiResponse> {
  await saveSetting('translation', settings, 3)
  return { success: true }
}

export const configApi = {
  getPrompts,
  getPromptContent,
  savePrompt,
  deletePrompt,
  resetPromptToDefault,
  getTextboxPrompts,
  getTextboxPromptContent,
  saveTextboxPrompt,
  deleteTextboxPrompt,
  resetTextboxPromptToDefault,
  fetchModels,
  testOllamaConnection,
  testSakuraConnection,
  testBaiduOcrConnection,
  testLamaRepair,
  testAiVisionOcrConnection,
  testAiTranslateConnection,
  testBaiduTranslateConnection,
  testYoudaoTranslateConnection,
  getFontList,
  uploadFont,
  getUserSettings,
  getTextStyleDefaults,
  saveTextStyleDefaults,
  resetTextStyleDefaults,
  getTranslateWorkflowPreferences,
  saveTranslateWorkflowPreferences,
  saveUserSettings,
}
