/**
 * InsightSettings 共享类型定义
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

/** VLM/LLM 服务商选项 */
export const VLM_PROVIDER_OPTIONS = [
  { value: 'gemini', label: 'Google Gemini' },
  { value: 'openai', label: 'OpenAI' },
  { value: 'qwen', label: '阿里通义千问' },
  { value: 'siliconflow', label: 'SiliconFlow' },
  { value: 'deepseek', label: 'DeepSeek' },
  { value: 'volcano', label: '火山引擎' },
  { value: 'custom', label: '自定义 OpenAI 兼容' }
]

/** Embedding 服务商选项 */
export const EMBEDDING_PROVIDER_OPTIONS = [
  { value: 'openai', label: 'OpenAI' },
  { value: 'siliconflow', label: 'SiliconFlow' },
  { value: 'custom', label: '自定义' }
]

/** Reranker 服务商选项 */
export const RERANKER_PROVIDER_OPTIONS = [
  { value: 'jina', label: 'Jina AI' },
  { value: 'cohere', label: 'Cohere' },
  { value: 'siliconflow', label: 'SiliconFlow' },
  { value: 'custom', label: '自定义' }
]

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

/** VLM 默认模型映射 */
export const VLM_DEFAULT_MODELS: Record<string, string> = {
  'gemini': 'gemini-2.0-flash',
  'openai': 'gpt-4o',
  'qwen': 'qwen-vl-max',
  'deepseek': 'deepseek-chat',
  'siliconflow': 'Qwen/Qwen2.5-VL-72B-Instruct',
  'volcano': 'doubao-1.5-vision-pro-32k'
}

/** LLM 默认模型映射 */
export const LLM_DEFAULT_MODELS: Record<string, string> = {
  'gemini': 'gemini-2.0-flash',
  'openai': 'gpt-4o-mini',
  'qwen': 'qwen-turbo',
  'deepseek': 'deepseek-chat',
  'siliconflow': 'Qwen/Qwen2.5-72B-Instruct',
  'volcano': 'doubao-1.5-pro-32k'
}

/** Embedding 默认模型映射 */
export const EMBEDDING_DEFAULT_MODELS: Record<string, string> = {
  'openai': 'text-embedding-3-small',
  'siliconflow': 'BAAI/bge-m3'
}

/** Reranker 默认模型映射 */
export const RERANKER_DEFAULT_MODELS: Record<string, string> = {
  'jina': 'jina-reranker-v2-base-multilingual',
  'cohere': 'rerank-multilingual-v3.0',
  'siliconflow': 'BAAI/bge-reranker-v2-m3'
}

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
