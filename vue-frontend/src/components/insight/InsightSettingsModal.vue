<script setup lang="ts">
/**
 * 漫画分析设置模态框组件
 * 配置VLM、LLM、Embedding、Reranker等模型参数
 */

import { ref, computed, watch, onMounted } from 'vue'
import BaseModal from '@/components/common/BaseModal.vue'
import CustomSelect from '@/components/common/CustomSelect.vue'
import { useInsightStore } from '@/stores/insightStore'
import * as insightApi from '@/api/insight'
import type { PromptType, SavedPromptItem } from '@/api/insight'

// ============================================================
// 事件定义
// ============================================================

const emit = defineEmits<{
  /** 关闭事件 */
  (e: 'close'): void
}>()

// ============================================================
// Store
// ============================================================

const insightStore = useInsightStore()

// ============================================================
// 状态
// ============================================================

/** 当前设置选项卡 */
const activeSettingsTab = ref<'vlm' | 'llm' | 'batch' | 'embedding' | 'reranker' | 'prompts'>('vlm')

/** 是否正在保存 */
const isSaving = ref(false)

/** 是否正在测试连接 */
const isTesting = ref(false)

/** 测试结果消息 */
const testMessage = ref('')

/** 测试结果类型 */
const testMessageType = ref<'success' | 'error' | ''>('')

// ============================================================
// 模型获取状态
// ============================================================

/** 模型列表 */
const vlmModels = ref<Array<{ id: string; name: string }>>([])
const llmModels = ref<Array<{ id: string; name: string }>>([])
const embeddingModels = ref<Array<{ id: string; name: string }>>([])
const rerankerModels = ref<Array<{ id: string; name: string }>>([])

/** 模型下拉框是否可见 */
const vlmModelSelectVisible = ref(false)
const llmModelSelectVisible = ref(false)
const embeddingModelSelectVisible = ref(false)
const rerankerModelSelectVisible = ref(false)

/** 是否正在获取模型 */
const isFetchingVlmModels = ref(false)
const isFetchingLlmModels = ref(false)
const isFetchingEmbeddingModels = ref(false)
const isFetchingRerankerModels = ref(false)

/** 是否正在测试 LLM 连接 */
const isTestingLlm = ref(false)

// VLM 设置（从 store 同步）
const vlmProvider = ref(insightStore.config.vlm.provider)
const vlmApiKey = ref(insightStore.config.vlm.apiKey)
const vlmModel = ref(insightStore.config.vlm.model)
const vlmBaseUrl = ref(insightStore.config.vlm.baseUrl)
const vlmRpm = ref(insightStore.config.vlm.rpmLimit)
const vlmTemperature = ref(insightStore.config.vlm.temperature)
const vlmForceJson = ref(insightStore.config.vlm.forceJson)
const vlmUseStream = ref(insightStore.config.vlm.useStream)
const vlmImageMaxSize = ref(insightStore.config.vlm.imageMaxSize)

// LLM 设置（独立配置，不再支持 "使用与 VLM 相同的配置"）
const llmProvider = ref(insightStore.config.llm.provider)
const llmApiKey = ref(insightStore.config.llm.apiKey)
const llmModel = ref(insightStore.config.llm.model)
const llmBaseUrl = ref(insightStore.config.llm.baseUrl)
const llmUseStream = ref(insightStore.config.llm.useStream)

// 批量分析设置
const pagesPerBatch = ref(insightStore.config.batch.pagesPerBatch)
const contextBatchCount = ref(insightStore.config.batch.contextBatchCount)
const architecturePreset = ref(insightStore.config.batch.architecturePreset)

// 自定义层级类型
interface CustomLayer {
  name: string
  units: number
  align: boolean
}

// 架构预设数据
const ARCHITECTURE_PRESETS: Record<string, { name: string; description: string; layers: CustomLayer[] }> = {
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

// 自定义层级数据
const customLayers = ref<CustomLayer[]>(
  insightStore.config.batch.customLayers?.length > 0
    ? insightStore.config.batch.customLayers.map((l: any) => ({
        name: l.name,
        units: l.units_per_group ?? l.units ?? 5,
        align: l.align_to_chapter ?? l.align ?? false
      }))
    : [
        { name: "批量分析", units: 5, align: false },
        { name: "段落总结", units: 5, align: false },
        { name: "全书总结", units: 0, align: false }
      ]
)

// Embedding 设置
const embeddingProvider = ref(insightStore.config.embedding.provider)
const embeddingApiKey = ref(insightStore.config.embedding.apiKey)
const embeddingModel = ref(insightStore.config.embedding.model)
const embeddingBaseUrl = ref(insightStore.config.embedding.baseUrl)
const embeddingRpmLimit = ref(insightStore.config.embedding.rpmLimit)

// Reranker 设置
const rerankerProvider = ref(insightStore.config.reranker.provider)
const rerankerApiKey = ref(insightStore.config.reranker.apiKey)
const rerankerModel = ref(insightStore.config.reranker.model)
const rerankerBaseUrl = ref(insightStore.config.reranker.baseUrl)
const rerankerTopK = ref(insightStore.config.reranker.topK)

// 提示词设置
/** 当前编辑的提示词类型 */
const currentPromptType = ref<PromptType>('batch_analysis')
/** 当前提示词内容 */
const currentPromptContent = ref('')
/** 自定义提示词（用户修改过的） */
const customPrompts = ref<Record<string, string>>({})
/** 保存的提示词库 */
const savedPromptsLibrary = ref<SavedPromptItem[]>([])
/** 是否正在加载提示词库 */
const isLoadingPrompts = ref(false)
/** 默认提示词（从后端获取） */
const defaultPrompts = ref<Record<PromptType, string>>({
  batch_analysis: '',
  segment_summary: '',
  chapter_summary: '',
  qa_response: ''
})

// ============================================================
// 服务商选项
// ============================================================

/** VLM/LLM 服务商选项 */
const vlmProviderOptions = [
  { value: 'gemini', label: 'Google Gemini' },
  { value: 'openai', label: 'OpenAI' },
  { value: 'qwen', label: '阿里通义千问' },
  { value: 'siliconflow', label: 'SiliconFlow' },
  { value: 'deepseek', label: 'DeepSeek' },
  { value: 'volcano', label: '火山引擎' },
  { value: 'custom', label: '自定义 OpenAI 兼容' }
]

/** Embedding 服务商选项 */
const embeddingProviderOptions = [
  { value: 'openai', label: 'OpenAI' },
  { value: 'siliconflow', label: 'SiliconFlow' },
  { value: 'custom', label: '自定义' }
]

/** Reranker 服务商选项 */
const rerankerProviderOptions = [
  { value: 'jina', label: 'Jina AI' },
  { value: 'cohere', label: 'Cohere' },
  { value: 'siliconflow', label: 'SiliconFlow' },
  { value: 'custom', label: '自定义' }
]

/** 分析架构选项 */
const architectureOptions = [
  { value: 'simple', label: '简洁模式 - 批量分析 → 全书总结（短篇）' },
  { value: 'standard', label: '标准模式 - 批量分析 → 段落总结 → 全书总结' },
  { value: 'chapter_based', label: '章节模式 - 批量分析 → 章节总结 → 全书总结' },
  { value: 'full', label: '完整模式 - 批量分析 → 小总结 → 章节总结 → 全书总结' },
  { value: 'custom', label: '自定义模式 - 完全自定义层级架构' }
]

/** 提示词类型选项 */
const promptTypeOptions = [
  { value: 'batch_analysis', label: '📄 批量分析提示词' },
  { value: 'segment_summary', label: '📑 段落总结提示词' },
  { value: 'chapter_summary', label: '📖 章节总结提示词' },
  { value: 'qa_response', label: '💬 问答响应提示词' }
]

/** VLM 默认模型映射 */
const vlmDefaultModels: Record<string, string> = {
  'gemini': 'gemini-2.0-flash',
  'openai': 'gpt-4o',
  'qwen': 'qwen-vl-max',
  'deepseek': 'deepseek-chat',
  'siliconflow': 'Qwen/Qwen2.5-VL-72B-Instruct',
  'volcano': 'doubao-1.5-vision-pro-32k'
}

/** LLM 默认模型映射 */
const llmDefaultModels: Record<string, string> = {
  'gemini': 'gemini-2.0-flash',
  'openai': 'gpt-4o-mini',
  'qwen': 'qwen-turbo',
  'deepseek': 'deepseek-chat',
  'siliconflow': 'Qwen/Qwen2.5-72B-Instruct',
  'volcano': 'doubao-1.5-pro-32k'
}

/** Embedding 默认模型映射 */
const embeddingDefaultModels: Record<string, string> = {
  'openai': 'text-embedding-3-small',
  'siliconflow': 'BAAI/bge-m3'
}

/** Reranker 默认模型映射 */
const rerankerDefaultModels: Record<string, string> = {
  'jina': 'jina-reranker-v2-base-multilingual',
  'cohere': 'rerank-multilingual-v3.0',
  'siliconflow': 'BAAI/bge-reranker-v2-m3'
}

// ============================================================
// 计算属性
// ============================================================

/** 是否显示 VLM Base URL 输入框 */
const showVlmBaseUrl = computed(() => vlmProvider.value === 'custom')

/** 是否显示 LLM Base URL 输入框 */
const showLlmBaseUrl = computed(() => llmProvider.value === 'custom')

/** 是否显示 Embedding Base URL 输入框 */
const showEmbeddingBaseUrl = computed(() => embeddingProvider.value === 'custom')

/** 是否显示 Reranker Base URL 输入框 */
const showRerankerBaseUrl = computed(() => rerankerProvider.value === 'custom')

/** 批量分析估算信息 */
const batchEstimate = computed(() => {
  const pages = pagesPerBatch.value || 5
  return `每批次分析 ${pages} 页`
})

/** 是否显示自定义层级编辑器 */
const showCustomLayersEditor = computed(() => architecturePreset.value === 'custom')

/** 架构描述 */
const architectureDescription = computed(() => {
  if (architecturePreset.value === 'custom') {
    return '完全自定义层级架构，灵活配置分析流程'
  }
  return ARCHITECTURE_PRESETS[architecturePreset.value]?.description || '根据漫画类型选择合适的分析架构'
})

/** 当前预览的层级列表 */
const previewLayers = computed(() => {
  if (architecturePreset.value === 'custom') {
    return customLayers.value
  }
  const preset = ARCHITECTURE_PRESETS[architecturePreset.value]
  if (preset) return preset.layers
  return ARCHITECTURE_PRESETS['standard']!.layers
})

// ============================================================
// 方法
// ============================================================

/**
 * 切换设置选项卡
 * @param tab - 选项卡名称
 */
function switchSettingsTab(tab: typeof activeSettingsTab.value): void {
  activeSettingsTab.value = tab
  // 清除测试消息
  testMessage.value = ''
  testMessageType.value = ''
}

/**
 * 关闭模态框
 */
function close(): void {
  emit('close')
}

/**
 * 添加自定义层级
 */
function addCustomLayer(): void {
  // 在倒数第二个位置插入新层级（最后一层是全书总结）
  const insertIdx = customLayers.value.length - 1
  customLayers.value.splice(insertIdx, 0, {
    name: `汇总层${insertIdx}`,
    units: 5,
    align: false
  })
}

/**
 * 删除自定义层级
 */
function removeCustomLayer(idx: number): void {
  if (idx > 0 && idx < customLayers.value.length - 1) {
    customLayers.value.splice(idx, 1)
  }
}

/**
 * 更新自定义层级
 */
function updateCustomLayer(idx: number, field: keyof CustomLayer, value: string | number | boolean): void {
  if (customLayers.value[idx]) {
    (customLayers.value[idx] as any)[field] = value
    
    // 如果是修改第一层的单元数，同步到"每批次分析页数"
    if (idx === 0 && field === 'units') {
      pagesPerBatch.value = value as number
    }
  }
}

/**
 * 每批次分析页数变更处理
 */
function onPagesPerBatchChange(): void {
  // 同步到自定义层级的第一层
  if (customLayers.value.length > 0 && customLayers.value[0]) {
    customLayers.value[0].units = pagesPerBatch.value
  }
}

/**
 * 判断层级是否可删除
 */
function canDeleteLayer(idx: number): boolean {
  return idx > 0 && idx < customLayers.value.length - 1 && customLayers.value.length > 2
}

/**
 * 判断层级名称是否可编辑
 */
function canEditLayerName(idx: number): boolean {
  return idx > 0 && idx < customLayers.value.length - 1
}

/**
 * 判断层级单元数是否可编辑
 */
function canEditLayerUnits(idx: number): boolean {
  return idx < customLayers.value.length - 1
}

/**
 * 获取层级单元数提示
 */
function getLayerUnitsTitle(idx: number): string {
  if (idx === 0) return '每批分析的页数'
  return '每组包含单元数（0=全部汇总）'
}

/**
 * VLM服务商变更处理
 */
function onVlmProviderChange(): void {
  const newProvider = vlmProvider.value
  const oldProvider = insightStore.config.vlm.provider
  
  // 【关键】在切换服务商之前，先将当前本地 ref 的值同步到 store
  // 这样 setVlmProvider 才能正确保存旧服务商的配置
  if (oldProvider !== newProvider) {
    insightStore.config.vlm.apiKey = vlmApiKey.value
    insightStore.config.vlm.model = vlmModel.value
    insightStore.config.vlm.baseUrl = vlmBaseUrl.value
    insightStore.config.vlm.rpmLimit = vlmRpm.value
    insightStore.config.vlm.temperature = vlmTemperature.value
    insightStore.config.vlm.forceJson = vlmForceJson.value
    insightStore.config.vlm.useStream = vlmUseStream.value
    insightStore.config.vlm.imageMaxSize = vlmImageMaxSize.value
  }
  
  // 调用 store 方法切换服务商（自动保存旧配置并恢复新配置）
  insightStore.setVlmProvider(newProvider)
  
  // 从 store 同步恢复的配置到本地状态
  vlmApiKey.value = insightStore.config.vlm.apiKey
  vlmModel.value = insightStore.config.vlm.model
  vlmBaseUrl.value = insightStore.config.vlm.baseUrl
  vlmRpm.value = insightStore.config.vlm.rpmLimit
  vlmTemperature.value = insightStore.config.vlm.temperature
  vlmForceJson.value = insightStore.config.vlm.forceJson
  vlmUseStream.value = insightStore.config.vlm.useStream
  vlmImageMaxSize.value = insightStore.config.vlm.imageMaxSize
  
  // 如果恢复的配置没有模型名称，设置默认模型
  if (!vlmModel.value) {
    const defaultModel = vlmDefaultModels[newProvider]
    if (defaultModel) {
      vlmModel.value = defaultModel
    }
  }
}

/**
 * LLM服务商变更处理
 */
function onLlmProviderChange(): void {
  const newProvider = llmProvider.value
  const oldProvider = insightStore.config.llm.provider
  
  // 【关键】在切换服务商之前，先将当前本地 ref 的值同步到 store
  if (oldProvider !== newProvider) {
    insightStore.config.llm.apiKey = llmApiKey.value
    insightStore.config.llm.model = llmModel.value
    insightStore.config.llm.baseUrl = llmBaseUrl.value
    insightStore.config.llm.useStream = llmUseStream.value
  }
  
  // 调用 store 方法切换服务商（自动保存旧配置并恢复新配置）
  insightStore.setLlmProvider(newProvider)
  
  // 从 store 同步恢复的配置到本地状态
  llmApiKey.value = insightStore.config.llm.apiKey
  llmModel.value = insightStore.config.llm.model
  llmBaseUrl.value = insightStore.config.llm.baseUrl
  llmUseStream.value = insightStore.config.llm.useStream
  
  // 如果恢复的配置没有模型名称，设置默认模型
  if (!llmModel.value) {
    const defaultModel = llmDefaultModels[newProvider]
    if (defaultModel) {
      llmModel.value = defaultModel
    }
  }
}

// 已移除: onLlmUseSameChange 函数
// 用户必须手动配置 LLM，避免从 VLM 复制错误的 baseUrl

/**
 * Embedding服务商变更处理
 */
function onEmbeddingProviderChange(): void {
  const newProvider = embeddingProvider.value
  const oldProvider = insightStore.config.embedding.provider
  
  // 【关键】在切换服务商之前，先将当前本地 ref 的值同步到 store
  if (oldProvider !== newProvider) {
    insightStore.config.embedding.apiKey = embeddingApiKey.value
    insightStore.config.embedding.model = embeddingModel.value
    insightStore.config.embedding.baseUrl = embeddingBaseUrl.value
    insightStore.config.embedding.rpmLimit = embeddingRpmLimit.value
  }
  
  // 调用 store 方法切换服务商（自动保存旧配置并恢复新配置）
  insightStore.setEmbeddingProvider(newProvider)
  
  // 从 store 同步恢复的配置到本地状态
  embeddingApiKey.value = insightStore.config.embedding.apiKey
  embeddingModel.value = insightStore.config.embedding.model
  embeddingBaseUrl.value = insightStore.config.embedding.baseUrl
  embeddingRpmLimit.value = insightStore.config.embedding.rpmLimit
  
  // 如果恢复的配置没有模型名称，设置默认模型
  if (!embeddingModel.value) {
    const defaultModel = embeddingDefaultModels[newProvider]
    if (defaultModel) {
      embeddingModel.value = defaultModel
    }
  }
}

/**
 * Reranker服务商变更处理
 */
function onRerankerProviderChange(): void {
  const newProvider = rerankerProvider.value
  const oldProvider = insightStore.config.reranker.provider
  
  // 【关键】在切换服务商之前，先将当前本地 ref 的值同步到 store
  if (oldProvider !== newProvider) {
    insightStore.config.reranker.apiKey = rerankerApiKey.value
    insightStore.config.reranker.model = rerankerModel.value
    insightStore.config.reranker.baseUrl = rerankerBaseUrl.value
    insightStore.config.reranker.topK = rerankerTopK.value
  }
  
  // 调用 store 方法切换服务商（自动保存旧配置并恢复新配置）
  insightStore.setRerankerProvider(newProvider)
  
  // 从 store 同步恢复的配置到本地状态
  rerankerApiKey.value = insightStore.config.reranker.apiKey
  rerankerModel.value = insightStore.config.reranker.model
  rerankerBaseUrl.value = insightStore.config.reranker.baseUrl
  rerankerTopK.value = insightStore.config.reranker.topK
  
  // 如果恢复的配置没有模型名称，设置默认模型
  if (!rerankerModel.value) {
    const defaultModel = rerankerDefaultModels[newProvider]
    if (defaultModel) {
      rerankerModel.value = defaultModel
    }
  }
}

/**
 * 显示测试结果消息
 * @param message - 消息内容
 * @param type - 消息类型
 */
function showTestMessage(message: string, type: 'success' | 'error'): void {
  testMessage.value = message
  testMessageType.value = type
  // 3秒后自动清除
  setTimeout(() => {
    testMessage.value = ''
    testMessageType.value = ''
  }, 3000)
}

/**
 * 测试VLM连接
 */
async function testVlmConnection(): Promise<void> {
  if (isTesting.value) return
  
  isTesting.value = true
  testMessage.value = ''
  
  try {
    const response = await insightApi.testVlmConnection({
      provider: vlmProvider.value,
      api_key: vlmApiKey.value,
      model: vlmModel.value,
      base_url: vlmBaseUrl.value || undefined
    })
    
    if (response.success) {
      showTestMessage('VLM 连接成功', 'success')
    } else {
      showTestMessage('连接失败: ' + (response.error || '未知错误'), 'error')
    }
  } catch (error) {
    showTestMessage('测试失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    isTesting.value = false
  }
}

/**
 * 测试Embedding连接
 */
async function testEmbeddingConnection(): Promise<void> {
  if (isTesting.value) return
  
  isTesting.value = true
  testMessage.value = ''
  
  try {
    const response = await insightApi.testEmbeddingConnection({
      provider: embeddingProvider.value,
      api_key: embeddingApiKey.value,
      model: embeddingModel.value,
      base_url: embeddingBaseUrl.value || undefined
    })
    
    if (response.success) {
      showTestMessage('Embedding 连接成功', 'success')
    } else {
      showTestMessage('连接失败: ' + (response.error || '未知错误'), 'error')
    }
  } catch (error) {
    showTestMessage('测试失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    isTesting.value = false
  }
}

/**
 * 测试Reranker连接
 */
async function testRerankerConnection(): Promise<void> {
  if (isTesting.value) return
  
  isTesting.value = true
  testMessage.value = ''
  
  try {
    const response = await insightApi.testRerankerConnection({
      provider: rerankerProvider.value,
      api_key: rerankerApiKey.value,
      model: rerankerModel.value,
      base_url: rerankerBaseUrl.value || undefined
    })
    
    if (response.success) {
      showTestMessage('Reranker 连接成功', 'success')
    } else {
      showTestMessage('连接失败: ' + (response.error || '未知错误'), 'error')
    }
  } catch (error) {
    showTestMessage('测试失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    isTesting.value = false
  }
}

/**
 * 测试LLM连接
 */
async function testLlmConnection(): Promise<void> {
  if (isTestingLlm.value) return
  
  isTestingLlm.value = true
  testMessage.value = ''
  
  try {
    const response = await insightApi.testLlmConnection({
      provider: llmProvider.value,
      api_key: llmApiKey.value,
      model: llmModel.value,
      base_url: llmBaseUrl.value || undefined
    })
    
    if (response.success) {
      showTestMessage('LLM 连接成功', 'success')
    } else {
      showTestMessage('连接失败: ' + (response.error || '未知错误'), 'error')
    }
  } catch (error) {
    showTestMessage('测试失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    isTestingLlm.value = false
  }
}

// ============================================================
// 模型获取方法
// ============================================================

/** 支持获取模型列表的服务商 */
const SUPPORTED_FETCH_PROVIDERS = ['siliconflow', 'deepseek', 'volcano', 'gemini', 'qwen', 'openai', 'custom']

/**
 * 获取模型列表
 * @param type 模型类型
 */
async function fetchModelsFor(type: 'vlm' | 'llm' | 'embedding' | 'reranker'): Promise<void> {
  // 获取对应类型的配置
  let provider: string
  let apiKey: string
  let baseUrl: string
  let setFetching: (v: boolean) => void
  let setModels: (models: Array<{ id: string; name: string }>) => void
  let setVisible: (v: boolean) => void
  
  switch (type) {
    case 'vlm':
      provider = vlmProvider.value
      apiKey = vlmApiKey.value
      baseUrl = vlmBaseUrl.value
      setFetching = (v) => { isFetchingVlmModels.value = v }
      setModels = (models) => { vlmModels.value = models }
      setVisible = (v) => { vlmModelSelectVisible.value = v }
      break
    case 'llm':
      provider = llmProvider.value
      apiKey = llmApiKey.value
      baseUrl = llmBaseUrl.value
      setFetching = (v) => { isFetchingLlmModels.value = v }
      setModels = (models) => { llmModels.value = models }
      setVisible = (v) => { llmModelSelectVisible.value = v }
      break
    case 'embedding':
      provider = embeddingProvider.value
      apiKey = embeddingApiKey.value
      baseUrl = embeddingBaseUrl.value
      setFetching = (v) => { isFetchingEmbeddingModels.value = v }
      setModels = (models) => { embeddingModels.value = models }
      setVisible = (v) => { embeddingModelSelectVisible.value = v }
      break
    case 'reranker':
      provider = rerankerProvider.value
      apiKey = rerankerApiKey.value
      baseUrl = rerankerBaseUrl.value
      setFetching = (v) => { isFetchingRerankerModels.value = v }
      setModels = (models) => { rerankerModels.value = models }
      setVisible = (v) => { rerankerModelSelectVisible.value = v }
      break
  }
  
  // 验证
  if (!apiKey) {
    showTestMessage('请先填写 API Key', 'error')
    return
  }
  
  // 检查是否支持模型获取
  if (!SUPPORTED_FETCH_PROVIDERS.includes(provider)) {
    showTestMessage(`${provider} 不支持自动获取模型列表`, 'error')
    return
  }
  
  // 自定义服务需要 base_url
  if (provider === 'custom' && !baseUrl) {
    showTestMessage('自定义服务需要先填写 Base URL', 'error')
    return
  }
  
  // 映射服务商名称
  const apiProvider = provider === 'custom' ? 'custom_openai' : provider
  
  setFetching(true)
  
  try {
    const response = await insightApi.fetchModels(apiProvider, apiKey, baseUrl || undefined)
    
    if (response.success && response.models && response.models.length > 0) {
      setModels(response.models)
      setVisible(true)
      showTestMessage(`获取到 ${response.models.length} 个模型`, 'success')
    } else {
      showTestMessage(response.message || '未获取到模型列表', 'error')
      setVisible(false)
    }
  } catch (error) {
    showTestMessage('获取模型列表失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
    setVisible(false)
  } finally {
    setFetching(false)
  }
}

/**
 * 模型选择事件
 * @param type 模型类型
 * @param modelId 选中的模型 ID
 */
function onModelSelected(type: 'vlm' | 'llm' | 'embedding' | 'reranker', modelId: string): void {
  if (!modelId) return
  
  switch (type) {
    case 'vlm':
      vlmModel.value = modelId
      break
    case 'llm':
      llmModel.value = modelId
      break
    case 'embedding':
      embeddingModel.value = modelId
      break
    case 'reranker':
      rerankerModel.value = modelId
      break
  }
}

// ============================================================
// 提示词管理方法
// ============================================================

/**
 * 加载默认提示词
 */
async function loadDefaultPrompts(): Promise<void> {
  try {
    const response = await insightApi.getDefaultPrompts()
    if (response.success && response.prompts) {
      defaultPrompts.value = response.prompts
    } else {
      console.warn('获取默认提示词失败，将使用空白提示词')
    }
  } catch (error) {
    console.error('加载默认提示词失败:', error)
    // 失败时提示用户（可选）
    // showTestMessage('加载默认提示词失败，请检查网络连接', 'error')
  }
}

/**
 * 加载提示词库
 */
async function loadPromptsLibrary(): Promise<void> {
  isLoadingPrompts.value = true
  try {
    const response = await insightApi.getPromptsLibrary()
    if (response.success && response.library) {
      savedPromptsLibrary.value = response.library
    }
  } catch (error) {
    console.error('加载提示词库失败:', error)
    savedPromptsLibrary.value = []
  } finally {
    isLoadingPrompts.value = false
  }
}


/**
 * 重置当前提示词为默认值
 */
function resetCurrentPrompt(): void {
  if (confirm('确定要重置为默认提示词吗？当前编辑的内容将丢失。')) {
    const promptType = currentPromptType.value
    currentPromptContent.value = defaultPrompts.value[promptType] || ''
    // 清空自定义，使用默认
    delete customPrompts.value[promptType]
    showTestMessage('已重置为默认提示词', 'success')
  }
}

/**
 * 复制提示词到剪贴板
 */
async function copyPromptToClipboard(): Promise<void> {
  try {
    await navigator.clipboard.writeText(currentPromptContent.value)
    showTestMessage('已复制到剪贴板', 'success')
  } catch (error) {
    showTestMessage('复制失败', 'error')
  }
}

/**
 * 保存提示词到库
 */
async function savePromptToLibrary(): Promise<void> {
  const content = currentPromptContent.value.trim()
  if (!content) {
    showTestMessage('提示词内容不能为空', 'error')
    return
  }
  
  const name = prompt('请输入提示词名称：')
  if (!name || !name.trim()) return
  
  const newPrompt: SavedPromptItem = {
    id: Date.now().toString(),
    name: name.trim(),
    type: currentPromptType.value,
    content: content,
    created_at: new Date().toISOString()
  }
  
  try {
    const response = await insightApi.savePromptToLibrary(newPrompt)
    if (response.success) {
      savedPromptsLibrary.value.push(newPrompt)
      showTestMessage('提示词已保存到库', 'success')
    } else {
      showTestMessage('保存失败: ' + (response.error || '未知错误'), 'error')
    }
  } catch (error) {
    showTestMessage('保存失败', 'error')
  }
}

/**
 * 从库加载提示词
 * @param promptItem 提示词项
 */
function loadPromptFromLibrary(promptItem: SavedPromptItem): void {
  // 切换到对应类型
  currentPromptType.value = promptItem.type
  // 填入内容
  currentPromptContent.value = promptItem.content
  customPrompts.value[promptItem.type] = promptItem.content
  showTestMessage(`已加载提示词: ${promptItem.name}`, 'success')
}

/**
 * 从库删除提示词
 * @param promptId 提示词 ID
 */
async function deletePromptFromLibrary(promptId: string): Promise<void> {
  if (!confirm('确定要删除这个提示词吗？')) return
  
  try {
    const response = await insightApi.deletePromptFromLibrary(promptId)
    if (response.success) {
      savedPromptsLibrary.value = savedPromptsLibrary.value.filter(p => p.id !== promptId)
      showTestMessage('提示词已删除', 'success')
    } else {
      showTestMessage('删除失败', 'error')
    }
  } catch (error) {
    showTestMessage('删除失败', 'error')
  }
}

/**
 * 导出所有提示词
 */
function exportAllPrompts(): void {
  // 保存当前编辑的
  if (currentPromptContent.value) {
    customPrompts.value[currentPromptType.value] = currentPromptContent.value
  }
  
  const exportData = {
    version: '1.0',
    exported_at: new Date().toISOString(),
    prompts: customPrompts.value,
    library: savedPromptsLibrary.value
  }
  
  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `manga-insight-prompts-${new Date().toISOString().slice(0, 10)}.json`
  a.click()
  URL.revokeObjectURL(url)
  
  showTestMessage('提示词已导出', 'success')
}

/**
 * 触发导入文件选择
 */
function triggerImportPrompts(): void {
  const fileInput = document.getElementById('promptsFileInput') as HTMLInputElement
  if (fileInput) {
    fileInput.click()
  }
}

/**
 * 处理导入文件
 * @param event 文件选择事件
 */
async function handlePromptsFileImport(event: Event): Promise<void> {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (!file) return
  
  try {
    const text = await file.text()
    const importData = JSON.parse(text)
    
    // 导入自定义提示词
    if (importData.prompts) {
      customPrompts.value = { ...customPrompts.value, ...importData.prompts }
    }
    
    // 导入提示词库
    if (importData.library && Array.isArray(importData.library)) {
      const existingIds = new Set(savedPromptsLibrary.value.map(p => p.id))
      for (const promptItem of importData.library) {
        if (!existingIds.has(promptItem.id)) {
          savedPromptsLibrary.value.push(promptItem)
        }
      }
      
      // 保存到服务器
      await insightApi.importPromptsLibrary(savedPromptsLibrary.value)
    }
    
    showTestMessage('提示词导入成功', 'success')
  } catch (error) {
    console.error('导入失败:', error)
    showTestMessage('导入失败，请检查文件格式', 'error')
  }
  
  // 清空文件输入
  target.value = ''
}

/**
 * 保存设置到 Store 和后端
 */
async function saveSettings(): Promise<void> {
  if (isSaving.value) return
  
  isSaving.value = true
  
  try {
    // 更新 Store 中的配置
    // 注意：仅在 custom 服务商时保存 baseUrl，其他服务商使用预设地址
    insightStore.updateVlmConfig({
      provider: vlmProvider.value,
      apiKey: vlmApiKey.value,
      model: vlmModel.value,
      baseUrl: vlmProvider.value === 'custom' ? vlmBaseUrl.value : '',
      rpmLimit: vlmRpm.value,
      temperature: vlmTemperature.value,
      forceJson: vlmForceJson.value,
      useStream: vlmUseStream.value,
      imageMaxSize: vlmImageMaxSize.value
    })
    
    insightStore.updateLlmConfig({
      useSameAsVlm: false, // 始终独立配置
      provider: llmProvider.value,
      apiKey: llmApiKey.value,
      model: llmModel.value,
      baseUrl: llmProvider.value === 'custom' ? llmBaseUrl.value : '',
      useStream: llmUseStream.value
    })

    insightStore.updateEmbeddingConfig({
      provider: embeddingProvider.value,
      apiKey: embeddingApiKey.value,
      model: embeddingModel.value,
      baseUrl: embeddingProvider.value === 'custom' ? embeddingBaseUrl.value : '',
      rpmLimit: embeddingRpmLimit.value
    })
    
    insightStore.updateRerankerConfig({
      provider: rerankerProvider.value,
      apiKey: rerankerApiKey.value,
      model: rerankerModel.value,
      baseUrl: rerankerProvider.value === 'custom' ? rerankerBaseUrl.value : '',
      topK: rerankerTopK.value
    })
    
    insightStore.updateBatchConfig({
      pagesPerBatch: pagesPerBatch.value,
      contextBatchCount: contextBatchCount.value,
      architecturePreset: architecturePreset.value
    })
    
    // 保存当前编辑的提示词
    if (currentPromptContent.value) {
      customPrompts.value[currentPromptType.value] = currentPromptContent.value
    }
    
    // 保存提示词配置
    insightStore.updatePrompts(customPrompts.value)
    
    // 保存到后端
    const apiConfig = insightStore.getConfigForApi()
    const response = await insightApi.saveGlobalConfig(apiConfig as insightApi.AnalysisConfig)
    
    if (response.success) {
      showTestMessage('设置已保存', 'success')
      setTimeout(() => {
        close()
      }, 500)
    } else {
      showTestMessage('保存失败: ' + (response.error || '未知错误'), 'error')
    }
  } catch (error) {
    showTestMessage('保存失败: ' + (error instanceof Error ? error.message : '网络错误'), 'error')
  } finally {
    isSaving.value = false
  }
}

/**
 * 加载配置
 */
async function loadConfig(): Promise<void> {
  try {
    // 先从 localStorage 加载
    insightStore.loadConfigFromStorage()
    
    // 尝试从后端加载
    const response = await insightApi.getGlobalConfig()
    if (response.success && response.config) {
      insightStore.setConfigFromApi(response.config as Record<string, unknown>)
    }
    
    // 同步到本地状态
    syncFromStore()
  } catch (error) {
    console.error('加载配置失败:', error)
    // 使用 localStorage 中的配置
    syncFromStore()
  }
}

/**
 * 从 Store 同步配置到本地状态
 */
function syncFromStore(): void {
  // VLM
  vlmProvider.value = insightStore.config.vlm.provider
  vlmApiKey.value = insightStore.config.vlm.apiKey
  vlmModel.value = insightStore.config.vlm.model
  vlmBaseUrl.value = insightStore.config.vlm.baseUrl
  vlmRpm.value = insightStore.config.vlm.rpmLimit
  vlmTemperature.value = insightStore.config.vlm.temperature
  vlmForceJson.value = insightStore.config.vlm.forceJson
  vlmUseStream.value = insightStore.config.vlm.useStream
  vlmImageMaxSize.value = insightStore.config.vlm.imageMaxSize
  
  // LLM（独立配置）
  llmProvider.value = insightStore.config.llm.provider
  llmApiKey.value = insightStore.config.llm.apiKey
  llmModel.value = insightStore.config.llm.model
  llmBaseUrl.value = insightStore.config.llm.baseUrl
  llmUseStream.value = insightStore.config.llm.useStream
  
  // Embedding
  embeddingProvider.value = insightStore.config.embedding.provider
  embeddingApiKey.value = insightStore.config.embedding.apiKey
  embeddingModel.value = insightStore.config.embedding.model
  embeddingBaseUrl.value = insightStore.config.embedding.baseUrl
  embeddingRpmLimit.value = insightStore.config.embedding.rpmLimit
  
  // Reranker
  rerankerProvider.value = insightStore.config.reranker.provider
  rerankerApiKey.value = insightStore.config.reranker.apiKey
  rerankerModel.value = insightStore.config.reranker.model
  rerankerBaseUrl.value = insightStore.config.reranker.baseUrl
  rerankerTopK.value = insightStore.config.reranker.topK
  
  // Batch
  pagesPerBatch.value = insightStore.config.batch.pagesPerBatch
  contextBatchCount.value = insightStore.config.batch.contextBatchCount
  architecturePreset.value = insightStore.config.batch.architecturePreset
  
  // Prompts（提示词配置）
  if (insightStore.config.prompts) {
    customPrompts.value = { ...insightStore.config.prompts }
  } else {
    customPrompts.value = {}
  }
  
  // 加载当前类型的提示词到编辑器（统一处理）
  const promptType = currentPromptType.value
  currentPromptContent.value = customPrompts.value[promptType] || defaultPrompts.value[promptType] || ''
}

// ============================================================
// 监听器
// ============================================================

/**
 * 监听提示词类型变化
 * 在类型切换时自动保存旧类型的内容并加载新类型的内容
 */
watch(currentPromptType, (newType, oldType) => {
  // 保存旧类型的内容（如果有修改）
  if (oldType && currentPromptContent.value) {
    customPrompts.value[oldType] = currentPromptContent.value
  }
  
  // 加载新类型的内容
  if (newType) {
    currentPromptContent.value = customPrompts.value[newType] || defaultPrompts.value[newType] || ''
  }
})

// ============================================================
// 生命周期
// ============================================================

onMounted(async () => {
  // 先加载默认提示词（从后端获取），确保 syncFromStore 时有默认值可用
  await loadDefaultPrompts()
  // 加载配置并同步到本地状态（内部会调用 syncFromStore，已包含提示词初始化）
  await loadConfig()
  // 加载提示词库
  await loadPromptsLibrary()
})
</script>

<template>
  <BaseModal title="漫画分析设置" size="large" customClass="insight-settings-modal" @close="close">
    <!-- 选项卡导航 -->
    <div class="settings-tabs">
      <button 
        class="settings-tab" 
        :class="{ active: activeSettingsTab === 'vlm' }"
        @click="switchSettingsTab('vlm')"
      >
        🖼️ VLM 多模态
      </button>
      <button 
        class="settings-tab" 
        :class="{ active: activeSettingsTab === 'llm' }"
        @click="switchSettingsTab('llm')"
      >
        💬 LLM 对话
      </button>
      <button 
        class="settings-tab" 
        :class="{ active: activeSettingsTab === 'batch' }"
        @click="switchSettingsTab('batch')"
      >
        📊 批量分析
      </button>
      <button 
        class="settings-tab" 
        :class="{ active: activeSettingsTab === 'embedding' }"
        @click="switchSettingsTab('embedding')"
      >
        🔢 向量模型
      </button>
      <button 
        class="settings-tab" 
        :class="{ active: activeSettingsTab === 'reranker' }"
        @click="switchSettingsTab('reranker')"
      >
        🔄 重排序
      </button>
      <button 
        class="settings-tab" 
        :class="{ active: activeSettingsTab === 'prompts' }"
        @click="switchSettingsTab('prompts')"
      >
        📝 提示词
      </button>
    </div>

    <!-- 测试结果消息 -->
    <div v-if="testMessage" class="test-message" :class="testMessageType">
      {{ testMessage }}
    </div>

    <!-- VLM 设置 -->
    <div v-if="activeSettingsTab === 'vlm'" class="insight-settings-content">
      <p class="settings-hint">VLM（视觉语言模型）用于分析漫画图片内容，提取对话和场景信息。</p>
      
      <div class="form-group">
        <label>服务商</label>
        <CustomSelect
          v-model="vlmProvider"
          :options="vlmProviderOptions"
          @change="onVlmProviderChange"
        />
      </div>
      
      <div class="form-group">
        <label>API Key</label>
        <input v-model="vlmApiKey" type="password" placeholder="输入 API Key">
      </div>
      
      <div class="form-group">
        <label>模型</label>
        <div class="model-input-row">
          <input v-model="vlmModel" type="text" placeholder="例如: gemini-2.0-flash">
          <button 
            class="btn btn-secondary btn-sm fetch-btn" 
            :disabled="isFetchingVlmModels"
            @click="fetchModelsFor('vlm')"
          >
            {{ isFetchingVlmModels ? '获取中...' : '🔍 获取模型' }}
          </button>
        </div>
        <!-- 模型下拉选择 -->
        <div v-if="vlmModelSelectVisible && vlmModels.length > 0" class="model-select-container">
          <select 
            class="model-select"
            :value="vlmModel"
            @change="onModelSelected('vlm', ($event.target as HTMLSelectElement).value)"
          >
            <option value="">-- 选择模型 --</option>
            <option v-for="model in vlmModels" :key="model.id" :value="model.id">
              {{ model.name || model.id }}
            </option>
          </select>
          <span class="model-count">共 {{ vlmModels.length }} 个模型</span>
        </div>
      </div>
      
      <div v-if="showVlmBaseUrl" class="form-group">
        <label>Base URL</label>
        <input v-model="vlmBaseUrl" type="text" placeholder="自定义 API 地址">
      </div>
      
      <div class="form-row">
        <div class="form-group">
          <label>RPM 限制</label>
          <input v-model.number="vlmRpm" type="number" min="1" max="100">
          <p class="form-hint">每分钟最大请求数</p>
        </div>
        <div class="form-group">
          <label>温度</label>
          <input v-model.number="vlmTemperature" type="number" min="0" max="1" step="0.1">
          <p class="form-hint">0-1，越低越确定</p>
        </div>
      </div>
      
      <div class="form-group">
        <label class="checkbox-label">
          <input v-model="vlmForceJson" type="checkbox">
          <span>强制 JSON 输出</span>
        </label>
        <p class="form-hint">对 OpenAI 兼容 API 启用 response_format: json_object</p>
      </div>
      
      <div class="form-group">
        <label class="checkbox-label">
          <input v-model="vlmUseStream" type="checkbox">
          <span>使用流式请求</span>
        </label>
        <p class="form-hint">流式请求可避免长时间等待导致的超时问题</p>
      </div>
      
      <div class="form-group">
        <label>图片压缩（最大边长）</label>
        <input v-model.number="vlmImageMaxSize" type="number" min="0" max="4096" step="128" placeholder="0 表示不压缩">
        <p class="form-hint">发送前将图片等比例缩放到指定最大边长（像素），0 表示不压缩</p>
      </div>
      
      <button class="btn btn-secondary" :disabled="isTesting" @click="testVlmConnection">
        {{ isTesting ? '测试中...' : '测试连接' }}
      </button>
    </div>

    <!-- LLM 设置 -->
    <div v-if="activeSettingsTab === 'llm'" class="insight-settings-content">
      <p class="settings-hint">LLM（对话模型）用于生成故事概要、智能问答等文本生成任务。</p>
      
      <div class="form-group">
        <label>服务商</label>
        <CustomSelect
          v-model="llmProvider"
          :options="vlmProviderOptions"
          @change="onLlmProviderChange"
        />
      </div>
      
      <div class="form-group">
        <label>API Key</label>
        <input v-model="llmApiKey" type="password" placeholder="输入 API Key">
      </div>
      
      <div class="form-group">
        <label>模型</label>
        <div class="model-input-row">
          <input v-model="llmModel" type="text" placeholder="例如: gpt-4o-mini">
          <button 
            class="btn btn-secondary btn-sm fetch-btn" 
            :disabled="isFetchingLlmModels"
            @click="fetchModelsFor('llm')"
          >
            {{ isFetchingLlmModels ? '获取中...' : '🔍 获取模型' }}
          </button>
        </div>
        <!-- 模型下拉选择 -->
        <div v-if="llmModelSelectVisible && llmModels.length > 0" class="model-select-container">
          <select 
            class="model-select"
            :value="llmModel"
            @change="onModelSelected('llm', ($event.target as HTMLSelectElement).value)"
          >
            <option value="">-- 选择模型 --</option>
            <option v-for="model in llmModels" :key="model.id" :value="model.id">
              {{ model.name || model.id }}
            </option>
          </select>
          <span class="model-count">共 {{ llmModels.length }} 个模型</span>
        </div>
      </div>
      
      <div v-if="showLlmBaseUrl" class="form-group">
        <label>Base URL</label>
        <input v-model="llmBaseUrl" type="text" placeholder="自定义 API 地址">
      </div>
      
      <div class="form-group">
        <label class="checkbox-label">
          <input v-model="llmUseStream" type="checkbox">
          <span>使用流式请求</span>
        </label>
      </div>
      
      <button class="btn btn-secondary" :disabled="isTestingLlm" @click="testLlmConnection">
        {{ isTestingLlm ? '测试中...' : '测试连接' }}
      </button>
    </div>

    <!-- 批量分析设置 -->
    <div v-if="activeSettingsTab === 'batch'" class="insight-settings-content">
      <p class="settings-hint">配置批量分析的参数，影响分析速度和质量。</p>
      
      <div class="form-group">
        <label>每批次分析页数</label>
        <input v-model.number="pagesPerBatch" type="number" min="1" max="10" @change="onPagesPerBatchChange">
        <p class="form-hint">每次发送给 VLM 的图片数量，建议 3-5 张。{{ batchEstimate }}</p>
      </div>
      
      <div class="form-group">
        <label>上文参考批次数</label>
        <input v-model.number="contextBatchCount" type="number" min="0" max="5">
        <p class="form-hint">每批分析时参考前几批的结果作为上下文，0 表示不参考</p>
      </div>
      
      <div class="form-group">
        <label>分析架构</label>
        <CustomSelect
          v-model="architecturePreset"
          :options="architectureOptions"
        />
        <p class="form-hint">{{ architectureDescription }}</p>
      </div>
      
      <!-- 自定义层级编辑器 -->
      <div v-if="showCustomLayersEditor" style="margin-top: 16px;">
        <label style="display: block; margin-bottom: 8px; font-weight: 500; font-size: 14px;">自定义层级</label>
        <div style="margin-bottom: 8px;">
          <div 
            v-for="(layer, idx) in customLayers" 
            :key="idx"
            style="display: flex; flex-direction: row; gap: 8px; align-items: center; margin-bottom: 8px; padding: 12px; background: #f5f5f5; border-radius: 8px; border: 1px solid #e0e0e0;"
          >
            <span style="min-width: 50px; color: #666; font-size: 13px;">第{{ idx + 1 }}层</span>
            <input 
              type="text" 
              :value="layer.name"
              :disabled="!canEditLayerName(idx)"
              placeholder="层级名称"
              style="flex: 1; padding: 8px 12px; border: 1px solid #e0e0e0; border-radius: 6px; font-size: 14px;"
              @change="updateCustomLayer(idx, 'name', ($event.target as HTMLInputElement).value)"
            >
            <input 
              type="number" 
              :value="layer.units"
              :disabled="!canEditLayerUnits(idx)"
              :title="getLayerUnitsTitle(idx)"
              min="0" 
              max="20"
              style="width: 70px; padding: 8px 12px; border: 1px solid #e0e0e0; border-radius: 6px; font-size: 14px;"
              @change="updateCustomLayer(idx, 'units', parseInt(($event.target as HTMLInputElement).value) || 0)"
            >
            <label style="display: flex; flex-direction: column; align-items: center; gap: 2px; font-size: 11px; cursor: pointer; min-width: 40px; text-align: center;">
              <input 
                type="checkbox" 
                :checked="layer.align"
                style="width: 16px; height: 16px;"
                @change="updateCustomLayer(idx, 'align', ($event.target as HTMLInputElement).checked)"
              >
              <span style="line-height: 1.2;">章节<br>对齐</span>
            </label>
            <button 
              v-if="canDeleteLayer(idx)"
              type="button" 
              style="padding: 6px 12px; background: #ef4444; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500;"
              @click="removeCustomLayer(idx)"
            >
              删除
            </button>
          </div>
        </div>
        <button type="button" class="btn btn-sm" style="margin-top: 4px; border: 1px solid #e0e0e0;" @click="addCustomLayer">
          + 添加层级
        </button>
        <p class="form-hint">第一层固定为批量分析，最后一层固定为全书总结。中间可添加任意汇总层级。</p>
      </div>
      
      <!-- 当前架构预览 -->
      <div class="batch-info-box">
        <h4>当前架构预览</h4>
        <ul class="layers-preview-list">
          <li v-for="(layer, idx) in previewLayers" :key="idx">
            <strong>第{{ idx + 1 }}层 - {{ layer.name }}</strong>
            {{ layer.units > 0 ? ` - 每${layer.units}个单元汇总` : ' - 汇总全部' }}
            <span v-if="layer.align" class="align-badge">(按章节对齐)</span>
          </li>
        </ul>
      </div>
      
      <!-- 当前配置信息 -->
      <div class="batch-estimate-box">
        <p>当前配置：每 <strong>{{ pagesPerBatch }}</strong> 页一批</p>
      </div>
    </div>

    <!-- Embedding 设置 -->
    <div v-if="activeSettingsTab === 'embedding'" class="insight-settings-content">
      <p class="settings-hint">Embedding（向量化模型）用于将文本转换为向量，支持语义搜索和问答功能。</p>
      
      <div class="form-group">
        <label>服务商</label>
        <CustomSelect
          v-model="embeddingProvider"
          :options="embeddingProviderOptions"
          @change="onEmbeddingProviderChange"
        />
      </div>
      
      <div class="form-group">
        <label>API Key</label>
        <input v-model="embeddingApiKey" type="password" placeholder="输入 API Key">
      </div>
      
      <div class="form-group">
        <label>模型</label>
        <div class="model-input-row">
          <input v-model="embeddingModel" type="text" placeholder="例如: text-embedding-3-small">
          <button 
            class="btn btn-secondary btn-sm fetch-btn" 
            :disabled="isFetchingEmbeddingModels"
            @click="fetchModelsFor('embedding')"
          >
            {{ isFetchingEmbeddingModels ? '获取中...' : '🔍 获取模型' }}
          </button>
        </div>
        <!-- 模型下拉选择 -->
        <div v-if="embeddingModelSelectVisible && embeddingModels.length > 0" class="model-select-container">
          <select 
            class="model-select"
            :value="embeddingModel"
            @change="onModelSelected('embedding', ($event.target as HTMLSelectElement).value)"
          >
            <option value="">-- 选择模型 --</option>
            <option v-for="model in embeddingModels" :key="model.id" :value="model.id">
              {{ model.name || model.id }}
            </option>
          </select>
          <span class="model-count">共 {{ embeddingModels.length }} 个模型</span>
        </div>
      </div>
      
      <div v-if="showEmbeddingBaseUrl" class="form-group">
        <label>Base URL</label>
        <input v-model="embeddingBaseUrl" type="text" placeholder="自定义 API 地址">
      </div>
      
      <div class="form-group">
        <label>RPM 限制</label>
        <input v-model.number="embeddingRpmLimit" type="number" min="0" max="1000">
        <p class="form-hint">每分钟最大请求数，0 表示不限制</p>
      </div>
      
      <button class="btn btn-secondary" :disabled="isTesting" @click="testEmbeddingConnection">
        {{ isTesting ? '测试中...' : '测试连接' }}
      </button>
    </div>

    <!-- Reranker 设置 -->
    <div v-if="activeSettingsTab === 'reranker'" class="insight-settings-content">
      <p class="settings-hint">Reranker（重排序模型）用于对搜索结果进行重新排序，提高问答准确性。</p>
      
      <div class="form-group">
        <label>服务商</label>
        <CustomSelect
          v-model="rerankerProvider"
          :options="rerankerProviderOptions"
          @change="onRerankerProviderChange"
        />
      </div>
      
      <div class="form-group">
        <label>API Key</label>
        <input v-model="rerankerApiKey" type="password" placeholder="输入 API Key">
      </div>
      
      <div class="form-group">
        <label>模型</label>
        <div class="model-input-row">
          <input v-model="rerankerModel" type="text" placeholder="例如: jina-reranker-v2-base-multilingual">
          <button 
            class="btn btn-secondary btn-sm fetch-btn" 
            :disabled="isFetchingRerankerModels"
            @click="fetchModelsFor('reranker')"
          >
            {{ isFetchingRerankerModels ? '获取中...' : '🔍 获取模型' }}
          </button>
        </div>
        <!-- 模型下拉选择 -->
        <div v-if="rerankerModelSelectVisible && rerankerModels.length > 0" class="model-select-container">
          <select 
            class="model-select"
            :value="rerankerModel"
            @change="onModelSelected('reranker', ($event.target as HTMLSelectElement).value)"
          >
            <option value="">-- 选择模型 --</option>
            <option v-for="model in rerankerModels" :key="model.id" :value="model.id">
              {{ model.name || model.id }}
            </option>
          </select>
          <span class="model-count">共 {{ rerankerModels.length }} 个模型</span>
        </div>
      </div>
      
      <div v-if="showRerankerBaseUrl" class="form-group">
        <label>Base URL</label>
        <input v-model="rerankerBaseUrl" type="text" placeholder="自定义 API 地址">
      </div>
      
      <div class="form-group">
        <label>Top K</label>
        <input v-model.number="rerankerTopK" type="number" min="1" max="20">
        <p class="form-hint">重排序后返回的结果数量</p>
      </div>
      
      <button class="btn btn-secondary" :disabled="isTesting" @click="testRerankerConnection">
        {{ isTesting ? '测试中...' : '测试连接' }}
      </button>
    </div>

    <!-- 提示词设置 -->
    <div v-if="activeSettingsTab === 'prompts'" class="insight-settings-content prompts-settings">
      <p class="settings-hint">自定义分析过程中使用的提示词模板。</p>
      
      <!-- 提示词类型选择器 -->
      <div class="form-group">
        <label>提示词类型</label>
        <CustomSelect
          v-model="currentPromptType"
          :options="promptTypeOptions"
        />
        <p class="form-hint">{{ insightApi.PROMPT_METADATA[currentPromptType]?.hint }}</p>
      </div>
      
      <!-- 提示词编辑器 -->
      <div class="form-group">
        <label>提示词内容</label>
        <textarea 
          v-model="currentPromptContent" 
          class="prompt-editor"
          rows="12"
          placeholder="输入提示词内容..."
        ></textarea>
      </div>
      
      <!-- 提示词操作按钮 -->
      <div class="prompt-actions-bar">
        <button class="btn btn-secondary btn-sm" @click="resetCurrentPrompt" title="重置为默认">
          🔄 重置
        </button>
        <button class="btn btn-secondary btn-sm" @click="copyPromptToClipboard" title="复制到剪贴板">
          📋 复制
        </button>
        <button class="btn btn-primary btn-sm" @click="savePromptToLibrary" title="保存到库">
          💾 保存到库
        </button>
      </div>
      
      <!-- 分隔线 -->
      <hr class="section-divider">
      
      <!-- 提示词库 -->
      <div class="prompts-library-section">
        <div class="library-header">
          <h4>📚 提示词库</h4>
          <div class="library-actions">
            <button class="btn btn-secondary btn-sm" @click="exportAllPrompts" title="导出所有提示词">
              📤 导出
            </button>
            <button class="btn btn-secondary btn-sm" @click="triggerImportPrompts" title="导入提示词">
              📥 导入
            </button>
            <input 
              id="promptsFileInput" 
              type="file" 
              accept=".json" 
              style="display: none"
              @change="handlePromptsFileImport"
            >
          </div>
        </div>
        
        <!-- 提示词库列表 -->
        <div class="saved-prompts-list">
          <div v-if="isLoadingPrompts" class="loading-text">加载中...</div>
          <div v-else-if="savedPromptsLibrary.length === 0" class="placeholder-text">
            暂无保存的提示词
          </div>
          <div 
            v-else
            v-for="promptItem in savedPromptsLibrary" 
            :key="promptItem.id"
            class="saved-prompt-item"
            @click="loadPromptFromLibrary(promptItem)"
          >
            <span class="prompt-name">{{ promptItem.name }}</span>
            <span class="prompt-type-badge">{{ insightApi.PROMPT_METADATA[promptItem.type]?.label || promptItem.type }}</span>
            <button 
              class="btn-icon-sm" 
              @click.stop="deletePromptFromLibrary(promptItem.id)" 
              title="删除"
            >
              🗑️
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- 底部按钮 -->
    <template #footer>
      <button class="btn btn-secondary" @click="close">取消</button>
      <button class="btn btn-primary" :disabled="isSaving" @click="saveSettings">
        {{ isSaving ? '保存中...' : '保存' }}
      </button>
    </template>
  </BaseModal>
</template>

<style>
/* 
 * InsightSettingsModal 样式
 * 注意：不使用 scoped，因为 BaseModal 使用 Teleport 将内容传送到 body
 * 样式使用 .insight-settings- 前缀避免全局污染
 */

/* 表单基础样式 */
.insight-settings-modal .form-group {
  margin-bottom: 16px;
}

.insight-settings-modal .form-group label {
  display: block;
  margin-bottom: 6px;
  font-weight: 500;
  font-size: 14px;
  color: var(--text-primary, #333);
}

.insight-settings-modal .form-group input[type="text"],
.insight-settings-modal .form-group input[type="password"],
.insight-settings-modal .form-group input[type="number"],
.insight-settings-modal .form-group select,
.insight-settings-modal .form-group textarea {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 6px;
  font-size: 14px;
  background: var(--input-bg-color, #fff);
  color: var(--text-primary, #333);
  transition: border-color 0.2s, box-shadow 0.2s;
}

.insight-settings-modal .form-group input:focus,
.insight-settings-modal .form-group select:focus,
.insight-settings-modal .form-group textarea:focus {
  outline: none;
  border-color: var(--primary, #6366f1);
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

.insight-settings-modal .form-hint {
  margin-top: 4px;
  font-size: 12px;
  color: var(--text-secondary, #666);
}

.insight-settings-modal .checkbox-label {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  font-weight: normal;
}

.insight-settings-modal .checkbox-label input[type="checkbox"] {
  width: 16px;
  height: 16px;
  cursor: pointer;
}

/* 按钮样式 */
.insight-settings-modal .btn {
  padding: 10px 16px;
  border: none;
  border-radius: 6px;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.insight-settings-modal .btn-primary {
  background: var(--primary, #6366f1);
  color: white;
}

.insight-settings-modal .btn-primary:hover:not(:disabled) {
  background: var(--primary-dark, #4f46e5);
}

.insight-settings-modal .btn-primary:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.insight-settings-modal .btn-secondary {
  background: var(--bg-secondary, #f3f4f6);
  color: var(--text-primary, #333);
  border: 1px solid var(--border-color, #e0e0e0);
}

.insight-settings-modal .btn-secondary:hover:not(:disabled) {
  background: var(--bg-hover, #e5e7eb);
}

.insight-settings-modal .settings-tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  margin-bottom: 16px;
  border-bottom: 1px solid var(--border-color, #e0e0e0);
  padding-bottom: 8px;
}

.insight-settings-modal .settings-tab {
  padding: 8px 12px;
  border: none;
  background: none;
  cursor: pointer;
  border-radius: 4px;
  transition: all 0.2s;
  font-size: 13px;
  color: var(--text-primary, #333);
}

.insight-settings-modal .settings-tab:hover {
  background: var(--bg-hover, #f3f4f6);
}

.insight-settings-modal .settings-tab.active {
  background: var(--primary, #6366f1);
  color: white;
}

.insight-settings-modal .insight-settings-content {
  padding: 16px 0;
  min-height: 300px;
}

.insight-settings-modal .settings-hint {
  color: var(--text-secondary, #666);
  font-size: 13px;
  margin-bottom: 16px;
  padding: 8px 12px;
  background: var(--bg-secondary, #f3f4f6);
  border-radius: 4px;
}

.insight-settings-modal .form-row {
  display: flex;
  gap: 16px;
}

.insight-settings-modal .form-row .form-group {
  flex: 1;
}

.insight-settings-modal .test-message {
  padding: 8px 12px;
  border-radius: 4px;
  margin-bottom: 12px;
  font-size: 13px;
}

.insight-settings-modal .test-message.success {
  background: var(--success-bg, #d4edda);
  color: var(--success-text, #155724);
  border: 1px solid var(--success-border, #c3e6cb);
}

.insight-settings-modal .test-message.error {
  background: var(--error-bg, #f8d7da);
  color: var(--error-text, #721c24);
  border: 1px solid var(--error-border, #f5c6cb);
}

.insight-settings-modal .placeholder-text {
  color: var(--text-secondary, #666);
  text-align: center;
  padding: 40px;
}

/* 提示词编辑器样式 */
.insight-settings-modal .prompts-settings {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.insight-settings-modal .prompt-editor {
  width: 100%;
  min-height: 200px;
  font-family: 'Consolas', 'Monaco', monospace;
  font-size: 13px;
  line-height: 1.5;
  padding: 12px;
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 4px;
  background: var(--bg-secondary, #f3f4f6);
  color: var(--text-primary, #333);
  resize: vertical;
}

.insight-settings-modal .prompt-editor:focus {
  outline: none;
  border-color: var(--primary, #6366f1);
}

.insight-settings-modal .prompt-actions-bar {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
}

.insight-settings-modal .btn-sm {
  padding: 4px 12px;
  font-size: 12px;
}

.insight-settings-modal .section-divider {
  border: none;
  border-top: 1px solid var(--border-color, #e0e0e0);
  margin: 16px 0;
}

.insight-settings-modal .prompts-library-section {
  margin-top: 8px;
}

.insight-settings-modal .library-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.insight-settings-modal .library-header h4 {
  margin: 0;
  font-size: 14px;
  font-weight: 500;
}

.insight-settings-modal .library-actions {
  display: flex;
  gap: 8px;
}

.insight-settings-modal .saved-prompts-list {
  max-height: 200px;
  overflow-y: auto;
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 4px;
  background: var(--bg-secondary, #f3f4f6);
}

.insight-settings-modal .saved-prompt-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  cursor: pointer;
  border-bottom: 1px solid var(--border-color, #e0e0e0);
  transition: background 0.2s;
}

.insight-settings-modal .saved-prompt-item:last-child {
  border-bottom: none;
}

.insight-settings-modal .saved-prompt-item:hover {
  background: var(--bg-hover, #e5e7eb);
}

.insight-settings-modal .prompt-name {
  flex: 1;
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.insight-settings-modal .prompt-type-badge {
  font-size: 11px;
  padding: 2px 6px;
  background: rgba(99, 102, 241, 0.1);
  color: var(--primary, #6366f1);
  border-radius: 4px;
  white-space: nowrap;
}

.insight-settings-modal .btn-icon-sm {
  padding: 2px 6px;
  background: none;
  border: none;
  cursor: pointer;
  opacity: 0.6;
  transition: opacity 0.2s;
}

.insight-settings-modal .btn-icon-sm:hover {
  opacity: 1;
}

.insight-settings-modal .loading-text {
  text-align: center;
  padding: 20px;
  color: var(--text-secondary, #666);
}

/* 架构预览样式 */
.insight-settings-modal .batch-info-box {
  margin-top: 16px;
  padding: 12px;
  background: var(--bg-secondary, #f5f5f5);
  border-radius: 8px;
  border: 1px solid var(--border-color, #e0e0e0);
}

.insight-settings-modal .batch-info-box h4 {
  margin: 0 0 8px 0;
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary, #333);
}

.insight-settings-modal .layers-preview-list {
  margin: 0;
  padding-left: 20px;
  font-size: 13px;
  line-height: 1.6;
}

.insight-settings-modal .layers-preview-list li {
  margin-bottom: 4px;
}

.insight-settings-modal .align-badge {
  color: var(--primary, #6366f1);
  font-size: 12px;
}

/* 当前配置信息 */
.insight-settings-modal .batch-estimate-box {
  margin-top: 12px;
  padding: 10px 12px;
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(99, 102, 241, 0.05));
  border-radius: 6px;
  border: 1px solid rgba(99, 102, 241, 0.2);
}

.insight-settings-modal .batch-estimate-box p {
  margin: 0;
  font-size: 13px;
  color: var(--text-primary, #333);
}

.insight-settings-modal .batch-estimate-box strong {
  color: var(--primary, #6366f1);
}

.insight-settings-modal .btn-sm {
  padding: 6px 12px;
  font-size: 13px;
}

/* 模型输入行样式 */
.insight-settings-modal .model-input-row {
  display: flex;
  gap: 8px;
  align-items: center;
}

.insight-settings-modal .model-input-row input {
  flex: 1;
}

.insight-settings-modal .fetch-btn {
  white-space: nowrap;
  flex-shrink: 0;
}

/* 模型下拉选择容器 */
.insight-settings-modal .model-select-container {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 8px;
  padding: 8px 12px;
  background: var(--bg-secondary, #f5f5f5);
  border-radius: 6px;
  border: 1px solid var(--border-color, #e0e0e0);
}

.insight-settings-modal .model-select {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid var(--border-color, #e0e0e0);
  border-radius: 4px;
  font-size: 13px;
  background: var(--input-bg-color, #fff);
  color: var(--text-primary, #333);
  cursor: pointer;
}

.insight-settings-modal .model-select:focus {
  outline: none;
  border-color: var(--primary, #6366f1);
}

.insight-settings-modal .model-count {
  font-size: 12px;
  color: var(--text-secondary, #666);
  white-space: nowrap;
}
</style>
