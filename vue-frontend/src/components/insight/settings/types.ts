/**
 * InsightSettings 共享类型定义
 *
 * 所有模型类型（VLM、LLM、Embedding、Reranker、生图）共用统一的服务商清单，
 * 并从全局 AI provider manifest 派生。
 */

import {
  AI_PROVIDER_MANIFEST,
  getProviderDefaultModel,
  getProviderOptionsForCapability,
  providerSupportsCapability,
} from '@/config/aiProviders'

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
 * Insight 相关 provider 选项（只包含 Insight 使用到的模型服务商）
 */
export const API_PROVIDER_OPTIONS = AI_PROVIDER_MANIFEST
  .filter(entry => entry.capabilities.some(cap => ['vlm', 'chat', 'embedding', 'rerank', 'imageGen'].includes(cap)))
  .map(entry => ({ value: entry.id, label: entry.label }))

/**
 * 服务商能力配置
 */
export const PROVIDER_CAPABILITIES: Record<string, {
  vlm: boolean
  embedding: boolean
  rerank: boolean
  imageGen: boolean
}> = Object.fromEntries(
  API_PROVIDER_OPTIONS.map(option => [
    option.value,
    {
      vlm: providerSupportsCapability(option.value, 'vlm'),
      embedding: providerSupportsCapability(option.value, 'embedding'),
      rerank: providerSupportsCapability(option.value, 'rerank'),
      imageGen: providerSupportsCapability(option.value, 'imageGen'),
    },
  ])
)

/** 根据能力过滤服务商列表 */
export function getProvidersForCapability(capability: 'vlm' | 'embedding' | 'rerank' | 'imageGen') {
  return getProviderOptionsForCapability(capability)
}

/** VLM/LLM 服务商选项 */
export const VLM_PROVIDER_OPTIONS = getProvidersForCapability('vlm')

/** Embedding 服务商选项 */
export const EMBEDDING_PROVIDER_OPTIONS = getProvidersForCapability('embedding')

/** Reranker 服务商选项 */
export const RERANKER_PROVIDER_OPTIONS = getProvidersForCapability('rerank')

/** 生图服务商选项 */
export const IMAGE_GEN_PROVIDER_OPTIONS = getProvidersForCapability('imageGen')

/** 分析架构选项 */
export const ARCHITECTURE_OPTIONS = [
  { value: 'simple', label: '简洁模式 - 批量分析 → 全书总结（短篇）' },
  { value: 'standard', label: '标准模式 - 批量分析 → 段落总结 → 全书总结' },
  { value: 'chapter_based', label: '章节模式 - 批量分析 → 章节总结 → 全书总结' },
  { value: 'full', label: '完整模式 - 批量分析 → 小总结 → 章节总结 → 全书总结' },
  { value: 'custom', label: '自定义模式 - 完全自定义层级架构' },
]

/** 提示词类型选项 */
export const PROMPT_TYPE_OPTIONS = [
  { value: 'batch_analysis', label: '📄 批量分析提示词' },
  { value: 'segment_summary', label: '📑 段落总结提示词' },
  { value: 'chapter_summary', label: '📖 章节总结提示词' },
  { value: 'qa_response', label: '💬 问答响应提示词' },
]

/**
 * 统一的默认模型配置
 */
export const PROVIDER_DEFAULT_MODELS: Record<string, {
  vlm?: string
  chat?: string
  embedding?: string
  reranker?: string
  imageGen?: string
}> = Object.fromEntries(
  API_PROVIDER_OPTIONS.map(option => [
    option.value,
    {
      vlm: getProviderDefaultModel(option.value, 'vlm') || undefined,
      chat: getProviderDefaultModel(option.value, 'chat') || undefined,
      embedding: getProviderDefaultModel(option.value, 'embedding') || undefined,
      reranker: getProviderDefaultModel(option.value, 'reranker') || undefined,
      imageGen: getProviderDefaultModel(option.value, 'imageGen') || undefined,
    },
  ])
)

/** VLM 默认模型映射（向后兼容） */
export const VLM_DEFAULT_MODELS: Record<string, string> = Object.fromEntries(
  Object.entries(PROVIDER_DEFAULT_MODELS)
    .filter(([_, value]) => value.vlm)
    .map(([key, value]) => [key, value.vlm!])
)

/** LLM 默认模型映射（向后兼容） */
export const LLM_DEFAULT_MODELS: Record<string, string> = Object.fromEntries(
  Object.entries(PROVIDER_DEFAULT_MODELS)
    .filter(([_, value]) => value.chat)
    .map(([key, value]) => [key, value.chat!])
)

/** Embedding 默认模型映射（向后兼容） */
export const EMBEDDING_DEFAULT_MODELS: Record<string, string> = Object.fromEntries(
  Object.entries(PROVIDER_DEFAULT_MODELS)
    .filter(([_, value]) => value.embedding)
    .map(([key, value]) => [key, value.embedding!])
)

/** Reranker 默认模型映射（向后兼容） */
export const RERANKER_DEFAULT_MODELS: Record<string, string> = Object.fromEntries(
  Object.entries(PROVIDER_DEFAULT_MODELS)
    .filter(([_, value]) => value.reranker)
    .map(([key, value]) => [key, value.reranker!])
)

/** 架构预设数据 */
export const ARCHITECTURE_PRESETS: Record<string, { name: string; description: string; layers: CustomLayer[] }> = {
  simple: {
    name: "简洁模式",
    description: "适合100页以内的短篇漫画",
    layers: [
      { name: "批量分析", units: 5, align: false },
      { name: "全书总结", units: 0, align: false },
    ],
  },
  standard: {
    name: "标准模式",
    description: "适合大多数漫画，平衡效果与速度",
    layers: [
      { name: "批量分析", units: 5, align: false },
      { name: "段落总结", units: 5, align: false },
      { name: "全书总结", units: 0, align: false },
    ],
  },
  chapter_based: {
    name: "章节模式",
    description: "适合有明确章节划分的漫画，会在章节边界处切分",
    layers: [
      { name: "批量分析", units: 5, align: true },
      { name: "章节总结", units: 0, align: true },
      { name: "全书总结", units: 0, align: false },
    ],
  },
  full: {
    name: "完整模式",
    description: "适合长篇连载，提供最详细的分层总结",
    layers: [
      { name: "批量分析", units: 5, align: false },
      { name: "小总结", units: 5, align: false },
      { name: "章节总结", units: 0, align: true },
      { name: "全书总结", units: 0, align: false },
    ],
  },
}

/** 支持获取模型列表的服务商 */
export const SUPPORTED_FETCH_PROVIDERS = AI_PROVIDER_MANIFEST
  .filter(entry => entry.capabilities.includes('modelFetch') && entry.kind !== 'adapter')
  .map(entry => entry.id)
