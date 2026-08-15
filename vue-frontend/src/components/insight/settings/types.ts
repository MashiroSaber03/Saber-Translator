import {
  AI_PROVIDER_MANIFEST,
  getProviderOptionsForCapability,
} from '@/config/aiProviders'

export interface CustomLayer {
  name: string
  units: number
  align: boolean
}

export const VLM_PROVIDER_OPTIONS = getProviderOptionsForCapability('vlm')

export const LLM_PROVIDER_OPTIONS = getProviderOptionsForCapability('chat')

export const EMBEDDING_PROVIDER_OPTIONS = getProviderOptionsForCapability('embedding')

export const RERANKER_PROVIDER_OPTIONS = getProviderOptionsForCapability('rerank')

export const IMAGE_GEN_PROVIDER_OPTIONS = getProviderOptionsForCapability('imageGen')

export const ARCHITECTURE_OPTIONS = [
  { value: 'simple', label: '简洁模式 - 批量分析 → 全书总结（短篇）' },
  { value: 'standard', label: '标准模式 - 批量分析 → 段落总结 → 全书总结' },
  { value: 'chapter_based', label: '章节模式 - 批量分析 → 章节总结 → 全书总结' },
  { value: 'full', label: '完整模式 - 批量分析 → 小总结 → 章节总结 → 全书总结' },
  { value: 'custom', label: '自定义模式 - 完全自定义层级架构' },
]

export const PROMPT_TYPE_OPTIONS = [
  { value: 'batch_analysis', label: '批量分析提示词' },
  { value: 'segment_summary', label: '段落总结提示词' },
  { value: 'chapter_summary', label: '章节总结提示词' },
  { value: 'qa_response', label: '问答响应提示词' },
]

export const ARCHITECTURE_PRESETS: Record<string, { description: string; layers: CustomLayer[] }> = {
  simple: {
    description: "适合100页以内的短篇漫画",
    layers: [
      { name: "批量分析", units: 5, align: false },
      { name: "全书总结", units: 0, align: false },
    ],
  },
  standard: {
    description: "适合大多数漫画，平衡效果与速度",
    layers: [
      { name: "批量分析", units: 5, align: false },
      { name: "段落总结", units: 5, align: false },
      { name: "全书总结", units: 0, align: false },
    ],
  },
  chapter_based: {
    description: "适合有明确章节划分的漫画，会在章节边界处切分",
    layers: [
      { name: "批量分析", units: 5, align: true },
      { name: "章节总结", units: 0, align: true },
      { name: "全书总结", units: 0, align: false },
    ],
  },
  full: {
    description: "适合长篇连载，提供最详细的分层总结",
    layers: [
      { name: "批量分析", units: 5, align: false },
      { name: "小总结", units: 5, align: false },
      { name: "章节总结", units: 0, align: true },
      { name: "全书总结", units: 0, align: false },
    ],
  },
}

export const SUPPORTED_FETCH_PROVIDERS = AI_PROVIDER_MANIFEST
  .filter(entry => entry.capabilities.includes('modelFetch') && entry.kind !== 'adapter')
  .map(entry => entry.id)
