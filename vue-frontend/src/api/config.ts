/**
 * 配置 API
 * 包含提示词管理、模型信息、服务连接测试、字体管理等功能
 */

import { apiClient } from './client'
import type {
  ApiResponse,
  FontListResponse,
  PromptListResponse,
  FetchModelsResponse,
  ConnectionTestResponse,
} from '@/types'

// ==================== 提示词内容响应 ====================

/**
 * 提示词内容响应
 */
export interface PromptContentResponse {
  success?: boolean
  prompt_content?: string
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
    params: { type, prompt_name: name },
  })
}

/**
 * 保存翻译提示词
 * @param type 提示词类型
 * @param name 提示词名称
 * @param content 提示词内容
 */
export async function savePrompt(type: string, name: string, content: string): Promise<ApiResponse> {
  // 后端期望 prompt_name 和 prompt_content
  return apiClient.post<ApiResponse>('/api/save_prompt', {
    type,
    prompt_name: name,
    prompt_content: content
  })
}

/**
 * 删除翻译提示词
 * @param type 提示词类型
 * @param name 提示词名称
 */
export async function deletePrompt(type: string, name: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>('/api/delete_prompt', {
    type,
    prompt_name: name
  })
}

/**
 * 重置翻译提示词为默认值
 * @param name 提示词名称
 */
export async function resetPromptToDefault(name: string): Promise<PromptContentResponse> {
  return apiClient.post<PromptContentResponse>('/api/reset_prompt_to_default', {
    prompt_name: name
  })
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
    params: { prompt_name: name },
  })
}

/**
 * 保存文本框提示词
 * @param name 提示词名称
 * @param content 提示词内容
 */
export async function saveTextboxPrompt(name: string, content: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>('/api/save_textbox_prompt', {
    prompt_name: name,
    prompt_content: content
  })
}

/**
 * 删除文本框提示词
 * @param name 提示词名称
 */
export async function deleteTextboxPrompt(name: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>('/api/delete_textbox_prompt', {
    prompt_name: name
  })
}

/**
 * 重置文本框提示词为默认值
 * @param name 提示词名称
 */
export async function resetTextboxPromptToDefault(name: string): Promise<PromptContentResponse> {
  return apiClient.post<PromptContentResponse>('/api/reset_textbox_prompt_to_default', {
    prompt_name: name
  })
}

// ==================== 模型信息 API ====================




/**
 * 获取模型列表响应类型（/api/fetch_models）
 */


/**
 * 从云服务商获取可用模型列表（复刻原版 doFetchModels 逻辑）
 * @param provider 服务商 (siliconflow, deepseek, volcano, gemini, custom_openai 等)
 * @param apiKey API Key
 * @param baseUrl 自定义服务的 Base URL（仅 custom_openai 需要）
 */
export async function fetchModels(
  provider: string,
  apiKey: string,
  baseUrl?: string
): Promise<FetchModelsResponse> {
  // 将 custom_openai_vision 映射为 custom_openai 发送给后端（与原版一致）
  const apiProvider = provider === 'custom_openai_vision' ? 'custom_openai' : provider

  return apiClient.post<FetchModelsResponse>('/api/fetch_models', {
    provider: apiProvider,
    api_key: apiKey,
    base_url: baseUrl || ''
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
 * @param apiKey 百度OCR API Key
 * @param secretKey 百度OCR Secret Key
 */
export async function testBaiduOcrConnection(
  apiKey: string,
  secretKey: string
): Promise<ConnectionTestResponse> {
  return apiClient.post<ConnectionTestResponse>('/api/test_baidu_ocr_connection', {
    api_key: apiKey,
    secret_key: secretKey
  })
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
  prompt?: string
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
    model_name: params.modelName,  // 后端期望 model_name
    custom_ai_vision_base_url: params.customBaseUrl,  // 后端期望 custom_ai_vision_base_url
    prompt: params.prompt || '', // 增加：将提示词发送到后端
  })
}

/**
 * AI翻译服务连接测试参数
 */
export interface AiTranslateTestParams {
  provider: string
  apiKey: string
  modelName?: string
  baseUrl?: string
}

/**
 * 测试 AI 翻译服务连接（通用接口）
 * 支持: SiliconFlow, DeepSeek, 火山引擎, Gemini, 彩云小译, 自定义OpenAI兼容服务
 * 复刻原版 testTranslationConnection 逻辑
 * @param params 测试参数
 */
export async function testAiTranslateConnection(
  params: AiTranslateTestParams
): Promise<ConnectionTestResponse> {
  return apiClient.post<ConnectionTestResponse>('/api/test_ai_translate_connection', {
    provider: params.provider,
    api_key: params.apiKey,
    model_name: params.modelName || '',
    base_url: params.baseUrl || ''
  })
}

/**
 * 测试百度翻译连接
 * @param appId 应用ID
 * @param appKey 应用密钥
 */
export async function testBaiduTranslateConnection(
  appId: string,
  appKey: string
): Promise<ConnectionTestResponse> {
  return apiClient.post<ConnectionTestResponse>('/api/test_baidu_translate_connection', {
    app_id: appId,
    app_key: appKey
  })
}

/**
 * 测试有道翻译连接
 * @param appKey 应用Key
 * @param appSecret 应用密钥
 */
export async function testYoudaoTranslateConnection(
  appKey: string,
  appSecret: string
): Promise<ConnectionTestResponse> {
  return apiClient.post<ConnectionTestResponse>('/api/test_youdao_translate', {
    appKey: appKey,
    appSecret: appSecret
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
  fetchModels,  // 新增：获取云服务商模型列表

  // 服务连接测试
  testOllamaConnection,
  testSakuraConnection,
  testBaiduOcrConnection,
  testLamaRepair,
  testAiVisionOcrConnection,
  testAiTranslateConnection,        // 新增：测试AI翻译服务连接
  testBaiduTranslateConnection,     // 新增：测试百度翻译连接
  testYoudaoTranslateConnection,    // 新增：测试有道翻译连接

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
