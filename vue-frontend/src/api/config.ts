import { apiClient } from './client'
import { normalizeProviderId } from '@/config/aiProviders'
import type {
  ApiResponse,
  ConnectionTestResponse,
  FetchModelsResponse,
  FontListResponse,
  PromptListResponse,
} from '@/types'
import type { TextStyleSettings } from '@/types/settings'
import type { WorkflowMode } from '@/types/workflow'

const CONFIG_ENDPOINTS = {
  prompts: '/api/get_prompts',
  promptContent: '/api/get_prompt_content',
  savePrompt: '/api/save_prompt',
  deletePrompt: '/api/delete_prompt',
  resetPrompt: '/api/reset_prompt_to_default',
  textboxPrompts: '/api/get_textbox_prompts',
  textboxPromptContent: '/api/get_textbox_prompt_content',
  saveTextboxPrompt: '/api/save_textbox_prompt',
  deleteTextboxPrompt: '/api/delete_textbox_prompt',
  resetTextboxPrompt: '/api/reset_textbox_prompt_to_default',
  fetchModels: '/api/fetch_models',
  ollamaConnection: '/api/test_ollama_connection',
  sakuraConnection: '/api/test_sakura_connection',
  baiduOcrConnection: '/api/test_baidu_ocr_connection',
  lamaRepair: '/api/test_lama_repair',
  aiVisionOcrConnection: '/api/test_ai_vision_ocr',
  aiTranslateConnection: '/api/test_ai_translate_connection',
  baiduTranslateConnection: '/api/test_baidu_translate_connection',
  youdaoTranslateConnection: '/api/test_youdao_translate',
  fontList: '/api/get_font_list',
  uploadFont: '/api/upload_font',
  testParams: '/api/test_params',
  userSettings: '/api/get_settings',
  saveUserSettings: '/api/save_settings',
  textStyleDefaults: '/api/config/text-style-defaults',
  resetTextStyleDefaults: '/api/config/text-style-defaults/reset',
  translateWorkflowPreferences: '/api/config/translate-workflow-preferences',
} as const

interface PromptPayload {
  type?: string
  prompt_name: string
  prompt_content?: string
}

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
}

export interface AiTranslateTestParams {
  provider: string
  apiKey: string
  modelName?: string
  baseUrl?: string
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

function promptPayload(name: string, content?: string, type?: string): PromptPayload {
  return {
    ...(type ? { type } : {}),
    prompt_name: name,
    ...(content !== undefined ? { prompt_content: content } : {}),
  }
}

export async function getPrompts(type?: string): Promise<PromptListResponse> {
  return apiClient.get<PromptListResponse>(CONFIG_ENDPOINTS.prompts, {
    params: type ? { type } : {},
  })
}

export async function getPromptContent(type: string, name: string): Promise<PromptContentResponse> {
  return apiClient.get<PromptContentResponse>(CONFIG_ENDPOINTS.promptContent, {
    params: { type, prompt_name: name },
  })
}

export async function savePrompt(type: string, name: string, content: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(CONFIG_ENDPOINTS.savePrompt, promptPayload(name, content, type))
}

export async function deletePrompt(type: string, name: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(CONFIG_ENDPOINTS.deletePrompt, promptPayload(name, undefined, type))
}

export async function resetPromptToDefault(name: string): Promise<PromptContentResponse> {
  return apiClient.post<PromptContentResponse>(CONFIG_ENDPOINTS.resetPrompt, promptPayload(name))
}

export async function getTextboxPrompts(): Promise<PromptListResponse> {
  return apiClient.get<PromptListResponse>(CONFIG_ENDPOINTS.textboxPrompts)
}

export async function getTextboxPromptContent(name: string): Promise<PromptContentResponse> {
  return apiClient.get<PromptContentResponse>(CONFIG_ENDPOINTS.textboxPromptContent, {
    params: { prompt_name: name },
  })
}

export async function saveTextboxPrompt(name: string, content: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(CONFIG_ENDPOINTS.saveTextboxPrompt, promptPayload(name, content))
}

export async function deleteTextboxPrompt(name: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(CONFIG_ENDPOINTS.deleteTextboxPrompt, promptPayload(name))
}

export async function resetTextboxPromptToDefault(name: string): Promise<PromptContentResponse> {
  return apiClient.post<PromptContentResponse>(CONFIG_ENDPOINTS.resetTextboxPrompt, promptPayload(name))
}

export async function fetchModels(
  provider: string,
  apiKey: string,
  baseUrl?: string
): Promise<FetchModelsResponse> {
  return apiClient.post<FetchModelsResponse>(CONFIG_ENDPOINTS.fetchModels, {
    provider: normalizeProviderId(provider),
    api_key: apiKey,
    base_url: baseUrl || '',
  })
}

export async function testOllamaConnection(baseUrl?: string): Promise<ConnectionTestResponse> {
  return apiClient.post<ConnectionTestResponse>(CONFIG_ENDPOINTS.ollamaConnection, {
    base_url: baseUrl,
  })
}

export async function testSakuraConnection(baseUrl?: string): Promise<ConnectionTestResponse> {
  return apiClient.post<ConnectionTestResponse>(CONFIG_ENDPOINTS.sakuraConnection, {
    base_url: baseUrl,
  })
}

export async function testBaiduOcrConnection(
  apiKey: string,
  secretKey: string
): Promise<ConnectionTestResponse> {
  return apiClient.post<ConnectionTestResponse>(CONFIG_ENDPOINTS.baiduOcrConnection, {
    api_key: apiKey,
    secret_key: secretKey,
  })
}

export async function testLamaRepair(): Promise<ConnectionTestResponse> {
  return apiClient.post<ConnectionTestResponse>(CONFIG_ENDPOINTS.lamaRepair)
}

export async function testAiVisionOcrConnection(
  params: AiVisionOcrTestParams
): Promise<ConnectionTestResponse> {
  return apiClient.post<ConnectionTestResponse>(CONFIG_ENDPOINTS.aiVisionOcrConnection, {
    provider: normalizeProviderId(params.provider),
    api_key: params.apiKey,
    model_name: params.modelName,
    custom_ai_vision_base_url: params.customBaseUrl,
    prompt: params.prompt || '',
  })
}

export async function testAiTranslateConnection(
  params: AiTranslateTestParams
): Promise<ConnectionTestResponse> {
  return apiClient.post<ConnectionTestResponse>(CONFIG_ENDPOINTS.aiTranslateConnection, {
    provider: normalizeProviderId(params.provider),
    api_key: params.apiKey,
    model_name: params.modelName || '',
    base_url: params.baseUrl || '',
  })
}

export async function testBaiduTranslateConnection(
  appId: string,
  appKey: string
): Promise<ConnectionTestResponse> {
  return apiClient.post<ConnectionTestResponse>(CONFIG_ENDPOINTS.baiduTranslateConnection, {
    app_id: appId,
    app_key: appKey,
  })
}

export async function testYoudaoTranslateConnection(
  appKey: string,
  appSecret: string
): Promise<ConnectionTestResponse> {
  return apiClient.post<ConnectionTestResponse>(CONFIG_ENDPOINTS.youdaoTranslateConnection, {
    appKey,
    appSecret,
  })
}

export async function getFontList(): Promise<FontListResponse> {
  return apiClient.get<FontListResponse>(CONFIG_ENDPOINTS.fontList)
}

export async function uploadFont(file: File): Promise<FontUploadResponse> {
  const formData = new FormData()
  formData.append('font', file)
  return apiClient.upload<FontUploadResponse>(CONFIG_ENDPOINTS.uploadFont, formData)
}

export async function testParams(params: Record<string, unknown>): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(CONFIG_ENDPOINTS.testParams, params)
}

export async function getUserSettings(): Promise<UserSettingsResponse> {
  return apiClient.get<UserSettingsResponse>(CONFIG_ENDPOINTS.userSettings)
}

export async function getTextStyleDefaults(): Promise<TextStyleDefaultsResponse> {
  return apiClient.get<TextStyleDefaultsResponse>(CONFIG_ENDPOINTS.textStyleDefaults)
}

export async function saveTextStyleDefaults(
  defaults: TextStyleSettings
): Promise<SaveTextStyleDefaultsResponse> {
  return apiClient.post<SaveTextStyleDefaultsResponse>(CONFIG_ENDPOINTS.textStyleDefaults, { defaults })
}

export async function resetTextStyleDefaults(): Promise<SaveTextStyleDefaultsResponse> {
  return apiClient.post<SaveTextStyleDefaultsResponse>(CONFIG_ENDPOINTS.resetTextStyleDefaults)
}

export async function getTranslateWorkflowPreferences(): Promise<TranslateWorkflowPreferencesResponse> {
  return apiClient.get<TranslateWorkflowPreferencesResponse>(CONFIG_ENDPOINTS.translateWorkflowPreferences)
}

export async function saveTranslateWorkflowPreferences(
  preferences: TranslateWorkflowPreferences
): Promise<TranslateWorkflowPreferencesResponse> {
  return apiClient.post<TranslateWorkflowPreferencesResponse>(
    CONFIG_ENDPOINTS.translateWorkflowPreferences,
    preferences
  )
}

export async function saveUserSettings(settings: Record<string, unknown>): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(CONFIG_ENDPOINTS.saveUserSettings, { settings })
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
  testParams,
  getUserSettings,
  getTextStyleDefaults,
  saveTextStyleDefaults,
  resetTextStyleDefaults,
  getTranslateWorkflowPreferences,
  saveTranslateWorkflowPreferences,
  saveUserSettings,
}
