/**
 * 配置 API
 * 包含提示词管理、模型信息、服务连接测试、字体管理等功能
 */

import { apiClient } from './client'
import type {
  ApiResponse,
  FontListResponse,
  PromptListResponse,
  ModelInfoResponse,
  ConnectionTestResponse,
} from '@/types'

// ==================== 提示词内容响应 ====================

/**
 * 提示词内容响应
 */
export interface PromptContentResponse {
  success: boolean
  content?: string
  error?: string
}

// ==================== 翻译提示词 API ====================

/**
 * 获取翻译提示词列表
 * @param type 提示词类型（可选，默认为translate）
 */
export async function getPrompts(type?: string): Promise<PromptListResponse> {
  const params = type ? { type } : {}
  return apiClient.get<PromptListResponse>('/api/get_prompts', { params })
}

/**
 * 获取翻译提示词内容
 * @param type 提示词类型
 * @param name 提示词名称
 */
export async function getPromptContent(type: string, name: string): Promise<PromptContentResponse> {
  return apiClient.get<PromptContentResponse>('/api/get_prompt_content', {
    params: { type, name },
  })
}

/**
 * 保存翻译提示词
 * @param type 提示词类型
 * @param name 提示词名称
 * @param content 提示词内容
 */
export async function savePrompt(type: string, name: string, content: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>('/api/save_prompt', { type, name, content })
}

/**
 * 删除翻译提示词
 * @param type 提示词类型
 * @param name 提示词名称
 */
export async function deletePrompt(type: string, name: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>('/api/delete_prompt', { type, name })
}

/**
 * 重置翻译提示词为默认值
 * @param name 提示词名称
 */
export async function resetPromptToDefault(name: string): Promise<PromptContentResponse> {
  return apiClient.post<PromptContentResponse>('/api/reset_prompt_to_default', { name })
}

// ==================== 文本框提示词 API ====================

/**
 * 获取文本框提示词列表
 */
export async function getTextboxPrompts(): Promise<PromptListResponse> {
  return apiClient.get<PromptListResponse>('/api/get_textbox_prompts')
}

/**
 * 获取文本框提示词内容
 * @param name 提示词名称
 */
export async function getTextboxPromptContent(name: string): Promise<PromptContentResponse> {
  return apiClient.get<PromptContentResponse>('/api/get_textbox_prompt_content', {
    params: { name },
  })
}

/**
 * 保存文本框提示词
 * @param name 提示词名称
 * @param content 提示词内容
 */
export async function saveTextboxPrompt(name: string, content: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>('/api/save_textbox_prompt', { name, content })
}

/**
 * 删除文本框提示词
 * @param name 提示词名称
 */
export async function deleteTextboxPrompt(name: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>('/api/delete_textbox_prompt', { name })
}

/**
 * 重置文本框提示词为默认值
 * @param name 提示词名称
 */
export async function resetTextboxPromptToDefault(name: string): Promise<PromptContentResponse> {
  return apiClient.post<PromptContentResponse>('/api/reset_textbox_prompt_to_default', { name })
}

// ==================== 模型信息 API ====================

/**
 * 获取模型信息/模型列表
 * @param provider 服务商
 * @param apiKey API Key（可选，用于获取云服务商的模型列表）
 */
export async function getModelInfo(provider: string, apiKey?: string): Promise<ModelInfoResponse> {
  const params: Record<string, string> = { provider }
  if (apiKey) {
    params.api_key = apiKey
  }
  return apiClient.get<ModelInfoResponse>('/api/get_model_info', { params })
}

/**
 * 保存模型信息
 * @param provider 服务商
 * @param model 模型名称
 */
export async function saveModelInfo(provider: string, model: string): Promise<ApiResponse> {
  // 后端期望 camelCase: modelProvider, modelName
  return apiClient.post<ApiResponse>('/api/save_model_info', { modelProvider: provider, modelName: model })
}

/**
 * 获取已使用的模型列表
 * @param provider 服务商
 */
export async function getUsedModels(provider: string): Promise<ModelInfoResponse> {
  // 后端期望 snake_case: model_provider
  return apiClient.get<ModelInfoResponse>('/api/get_used_models', {
    params: { model_provider: provider },
  })
}

// ==================== 服务连接测试 API ====================

/**
 * 测试 Ollama 连接
 * @param baseUrl Ollama 服务地址
 */
export async function testOllamaConnection(baseUrl?: string): Promise<ConnectionTestResponse> {
  return apiClient.post<ConnectionTestResponse>('/api/test_ollama_connection', {
    base_url: baseUrl,
  })
}

/**
 * 测试 Sakura 连接
 * @param baseUrl Sakura 服务地址
 */
export async function testSakuraConnection(baseUrl?: string): Promise<ConnectionTestResponse> {
  return apiClient.post<ConnectionTestResponse>('/api/test_sakura_connection', {
    base_url: baseUrl,
  })
}

/**
 * 测试百度 OCR 连接
 * 使用当前设置中的 API Key 和 Secret Key
 */
export async function testBaiduOcrConnection(): Promise<ConnectionTestResponse> {
  return apiClient.post<ConnectionTestResponse>('/api/test_baidu_ocr_connection')
}

/**
 * 测试 LAMA 修复功能
 */
export async function testLamaRepair(): Promise<ConnectionTestResponse> {
  return apiClient.post<ConnectionTestResponse>('/api/test_lama_repair')
}

/**
 * AI视觉OCR连接测试参数
 */
export interface AiVisionOcrTestParams {
  provider: string
  apiKey: string
  modelName: string
  customBaseUrl?: string
}

/**
 * 测试 AI 视觉 OCR 连接
 * @param params 测试参数
 */
export async function testAiVisionOcrConnection(
  params: AiVisionOcrTestParams
): Promise<ConnectionTestResponse> {
  return apiClient.post<ConnectionTestResponse>('/api/test_ai_vision_ocr', {
    provider: params.provider,
    api_key: params.apiKey,
    model: params.modelName,
    custom_base_url: params.customBaseUrl,
  })
}

// ==================== 字体管理 API ====================

/**
 * 获取系统字体列表
 */
export async function getFontList(): Promise<FontListResponse> {
  return apiClient.get<FontListResponse>('/api/get_font_list')
}

/**
 * 字体上传响应
 */
export interface FontUploadResponse {
  success: boolean
  fontPath?: string
  error?: string
}

/**
 * 上传自定义字体
 * @param file 字体文件（.ttf/.ttc/.otf）
 */
export async function uploadFont(file: File): Promise<FontUploadResponse> {
  const formData = new FormData()
  formData.append('font', file)
  return apiClient.upload<FontUploadResponse>('/api/upload_font', formData)
}

// ==================== 参数测试 API ====================

/**
 * 参数测试（用于调试）
 * @param params 测试参数
 */
export async function testParams(params: Record<string, unknown>): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>('/api/test_params', params)
}

// ==================== 用户设置 API ====================

/**
 * 用户设置响应
 */
export interface UserSettingsResponse {
  success: boolean
  settings?: Record<string, unknown>
  error?: string
}

/**
 * 获取用户设置（从后端 config/user_settings.json 加载）
 */
export async function getUserSettings(): Promise<UserSettingsResponse> {
  console.log('[API] 正在调用 /api/get_settings ...')
  const response = await apiClient.get<UserSettingsResponse>('/api/get_settings')
  console.log('[API] /api/get_settings 响应:', response)
  return response
}

/**
 * 保存用户设置（保存到后端 config/user_settings.json）
 * @param settings 用户设置对象
 */
export async function saveUserSettings(settings: Record<string, unknown>): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>('/api/save_settings', { settings })
}


// ==================== 别名导出（兼容旧调用方式） ====================

/** 获取字体列表（别名） */
export const getFontListApi = getFontList

/** 上传字体（别名） */
export const uploadFontApi = uploadFont

// ==================== 导出 API 对象 ====================

/**
 * 配置 API 对象
 * 提供统一的 API 调用入口
 */
export const configApi = {
  // 翻译提示词
  getPrompts,
  getPromptContent,
  savePrompt,
  deletePrompt,
  resetPromptToDefault,
  
  // 文本框提示词
  getTextboxPrompts,
  getTextboxPromptContent,
  saveTextboxPrompt,
  deleteTextboxPrompt,
  resetTextboxPromptToDefault,
  
  // 模型信息
  getModelInfo,
  saveModelInfo,
  getUsedModels,
  
  // 服务连接测试
  testOllamaConnection,
  testSakuraConnection,
  testBaiduOcrConnection,
  testLamaRepair,
  testAiVisionOcrConnection,
  
  // 字体管理
  getFontList,
  uploadFont,
  getFontListApi,
  uploadFontApi,
  
  // 参数测试
  testParams,
  
  // 用户设置
  getUserSettings,
  saveUserSettings,
}
