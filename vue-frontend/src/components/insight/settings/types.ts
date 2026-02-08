/**
 * InsightSettings 共享类型定义
 *
 * 所有模型类型（VLM、LLM、Embedding、Reranker、生图）共用统一的服务商列表
 */

/** 自定义层级类型 */
export interface CustomLayer {
  name: string
  units: number
  align: boolean
}

/** 模型信息 */
export interface ModelInfo {
  id: string
  name: string
}

/**
 * 统一的 API 服务商列表
 * 所有模型类型共用此列表
 */
export const API_PROVIDER_OPTIONS = [
  { value: 'openai', label: 'OpenAI' },
  { value: 'gemini', label: 'Google Gemini' },
  { value: 'qwen', label: '通义千问' },
  { value: 'siliconflow', label: 'SiliconFlow' },
  { value: 'deepseek', label: 'DeepSeek' },
  { value: 'volcano', label: '火山引擎' },
  { value: 'custom', label: '自定义 API' }
]

/**
 * 服务商能力配置
 * 标记每个服务商支持哪些功能
 */
export const PROVIDER_CAPABILITIES: Record<string, {
  vlm: boolean
  embedding: boolean
  rerank: boolean
  imageGen: boolean
}> = {
  openai: { vlm: true, embedding: true, rerank: false, imageGen: true },
  gemini: { vlm: true, embedding: true, rerank: false, imageGen: false },
  qwen: { vlm: true, embedding: true, rerank: true, imageGen: true },
  siliconflow: { vlm: true, embedding: true, rerank: true, imageGen: true },
  deepseek: { vlm: true, embedding: true, rerank: true, imageGen: false },
  volcano: { vlm: true, embedding: true, rerank: true, imageGen: true },
  custom: { vlm: true, embedding: true, rerank: true, imageGen: true }
}

/** 根据能力过滤服务商列表 */
export function getProvidersForCapability(capability: 'vlm' | 'embedding' | 'rerank' | 'imageGen') {
  return API_PROVIDER_OPTIONS.filter(p => PROVIDER_CAPABILITIES[p.value]?.[capability])
}

/** VLM/LLM 服务商选项（统一使用完整列表） */
export const VLM_PROVIDER_OPTIONS = API_PROVIDER_OPTIONS

/** Embedding 服务商选项（统一使用完整列表） */
export const EMBEDDING_PROVIDER_OPTIONS = API_PROVIDER_OPTIONS

/** Reranker 服务商选项（统一使用完整列表） */
export const RERANKER_PROVIDER_OPTIONS = API_PROVIDER_OPTIONS

/** 生图服务商选项（统一使用完整列表） */
export const IMAGE_GEN_PROVIDER_OPTIONS = API_PROVIDER_OPTIONS

/** 分析架构选项 */
export const ARCHITECTURE_OPTIONS = [
  { value: 'simple', label: '简洁模式 - 批量分析 → 全书总结（短篇）' },
  { value: 'standard', label: '标准模式 - 批量分析 → 段落总结 → 全书总结' },
  { value: 'chapter_based', label: '章节模式 - 批量分析 → 章节总结 → 全书总结' },
  { value: 'full', label: '完整模式 - 批量分析 → 小总结 → 章节总结 → 全书总结' },
  { value: 'custom', label: '自定义模式 - 完全自定义层级架构' }
]

/** 提示词类型选项 */
export const PROMPT_TYPE_OPTIONS = [
  { value: 'batch_analysis', label: '📄 批量分析提示词' },
  { value: 'segment_summary', label: '📑 段落总结提示词' },
  { value: 'chapter_summary', label: '📖 章节总结提示词' },
  { value: 'qa_response', label: '💬 问答响应提示词' }
]

/**
 * 统一的默认模型配置
 * 按服务商组织，包含所有模型类型
 */
export const PROVIDER_DEFAULT_MODELS: Record<string, {
  vlm?: string
  chat?: string
  embedding?: string
  reranker?: string
  imageGen?: string
}> = {
  openai: {
    vlm: 'gpt-4o',
    chat: 'gpt-4o-mini',
    embedding: 'text-embedding-3-small',
    imageGen: 'dall-e-3'
  },
  gemini: {
    vlm: 'gemini-2.0-flash',
    chat: 'gemini-2.0-flash',
    embedding: 'text-embedding-004'
  },
  qwen: {
    vlm: 'qwen-vl-max',
    chat: 'qwen-turbo',
    embedding: 'text-embedding-v3',
    imageGen: 'wanx-v1'
  },
  siliconflow: {
    vlm: 'Qwen/Qwen2.5-VL-72B-Instruct',
    chat: 'Qwen/Qwen2.5-72B-Instruct',
    embedding: 'BAAI/bge-m3',
    reranker: 'BAAI/bge-reranker-v2-m3',
    imageGen: 'stabilityai/stable-diffusion-3-5-large'
  },
  deepseek: {
    vlm: 'deepseek-chat',
    chat: 'deepseek-chat'
  },
  volcano: {
    vlm: 'doubao-1.5-vision-pro-32k',
    chat: 'doubao-1.5-pro-32k',
    imageGen: 'high_aes_general_v21'
  },
  jina: {
    reranker: 'jina-reranker-v2-base-multilingual'
  },
  cohere: {
    reranker: 'rerank-multilingual-v3.0'
  }
}

/** 获取默认模型 */
export function getDefaultModel(provider: string, modelType: 'vlm' | 'chat' | 'embedding' | 'reranker' | 'imageGen'): string {
  return PROVIDER_DEFAULT_MODELS[provider]?.[modelType] || ''
}

/** VLM 默认模型映射（向后兼容） */
export const VLM_DEFAULT_MODELS: Record<string, string> = Object.fromEntries(
  Object.entries(PROVIDER_DEFAULT_MODELS)
    .filter(([_, v]) => v.vlm)
    .map(([k, v]) => [k, v.vlm!])
)

/** LLM 默认模型映射（向后兼容） */
export const LLM_DEFAULT_MODELS: Record<string, string> = Object.fromEntries(
  Object.entries(PROVIDER_DEFAULT_MODELS)
    .filter(([_, v]) => v.chat)
    .map(([k, v]) => [k, v.chat!])
)

/** Embedding 默认模型映射（向后兼容） */
export const EMBEDDING_DEFAULT_MODELS: Record<string, string> = Object.fromEntries(
  Object.entries(PROVIDER_DEFAULT_MODELS)
    .filter(([_, v]) => v.embedding)
    .map(([k, v]) => [k, v.embedding!])
)

/** Reranker 默认模型映射（向后兼容） */
export const RERANKER_DEFAULT_MODELS: Record<string, string> = Object.fromEntries(
  Object.entries(PROVIDER_DEFAULT_MODELS)
    .filter(([_, v]) => v.reranker)
    .map(([k, v]) => [k, v.reranker!])
)

/** 生图默认模型映射（向后兼容） */
export const IMAGE_GEN_DEFAULT_MODELS: Record<string, string> = Object.fromEntries(
  Object.entries(PROVIDER_DEFAULT_MODELS)
    .filter(([_, v]) => v.imageGen)
    .map(([k, v]) => [k, v.imageGen!])
)

/** 架构预设数据 */
export const ARCHITECTURE_PRESETS: Record<string, { name: string; description: string; layers: CustomLayer[] }> = {
  simple: {
    name: "简洁模式",
    description: "适合100页以内的短篇漫画",
    layers: [
      { name: "批量分析", units: 5, align: false },
      { name: "全书总结", units: 0, align: false }
    ]
  },
  standard: {
    name: "标准模式",
    description: "适合大多数漫画，平衡效果与速度",
    layers: [
      { name: "批量分析", units: 5, align: false },
      { name: "段落总结", units: 5, align: false },
      { name: "全书总结", units: 0, align: false }
    ]
  },
  chapter_based: {
    name: "章节模式",
    description: "适合有明确章节划分的漫画，会在章节边界处切分",
    layers: [
      { name: "批量分析", units: 5, align: true },
      { name: "章节总结", units: 0, align: true },
      { name: "全书总结", units: 0, align: false }
    ]
  },
  full: {
    name: "完整模式",
    description: "适合长篇连载，提供最详细的分层总结",
    layers: [
      { name: "批量分析", units: 5, align: false },
      { name: "小总结", units: 5, align: false },
      { name: "章节总结", units: 0, align: true },
      { name: "全书总结", units: 0, align: false }
    ]
  }
}

/** 支持获取模型列表的服务商 */
export const SUPPORTED_FETCH_PROVIDERS = ['siliconflow', 'deepseek', 'volcano', 'gemini', 'qwen', 'openai', 'custom']

/**
 * 统一的服务商 Base URL 配置
 */
export const PROVIDER_BASE_URLS: Record<string, {
  base?: string
  imageGen?: string  // 部分服务商生图使用不同的 base_url
}> = {
  openai: {
    base: 'https://api.openai.com/v1'
  },
  gemini: {
    base: 'https://generativelanguage.googleapis.com/v1beta/openai/'
  },
  qwen: {
    base: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    imageGen: 'https://dashscope.aliyuncs.com/api/v1'
  },
  siliconflow: {
    base: 'https://api.siliconflow.cn/v1'
  },
  deepseek: {
    base: 'https://api.deepseek.com/v1'
  },
  volcano: {
    base: 'https://ark.cn-beijing.volces.com/api/v3',
    imageGen: 'https://visual.volcengineapi.com'
  },
  jina: {
    base: 'https://api.jina.ai/v1'
  },
  cohere: {
    base: 'https://api.cohere.ai/v1'
  }
}

/** 获取 Base URL */
export function getBaseUrl(provider: string, forImageGen = false): string {
  const config = PROVIDER_BASE_URLS[provider]
  if (!config) return ''
  if (forImageGen && config.imageGen) return config.imageGen
  return config.base || ''
}

/** 生图服务商默认 Base URL（向后兼容） */
export const IMAGE_GEN_DEFAULT_BASE_URLS: Record<string, string> = Object.fromEntries(
  Object.entries(PROVIDER_BASE_URLS)
    .filter(([k]) => PROVIDER_CAPABILITIES[k]?.imageGen)
    .map(([k, v]) => [k, v.imageGen || v.base || ''])
)

/** 生图尺寸选项 */
export const IMAGE_SIZE_OPTIONS = [
  { value: '1024x1024', label: '1024×1024（方形）' },
  { value: '1024x1536', label: '1024×1536（竖版漫画推荐）' },
  { value: '1536x1024', label: '1536×1024（横版）' },
  { value: '768x1024', label: '768×1024（竖版）' },
  { value: '1024x768', label: '1024×768（横版）' }
]
