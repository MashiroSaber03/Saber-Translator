/**
 * 漫画分析状态管理 Store
 * 管理漫画分析状态、进度跟踪、问答和笔记
 *
 * 重构后使用拆分的 composables:
 * - useInsightNotes: 笔记管理
 * - useInsightQA: 问答管理
 * - useInsightConfigManager: 服务商配置管理
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

// 导入拆分的 composables
import { useInsightNotes } from './insight/useInsightNotes'
import { useInsightQA } from './insight/useInsightQA'
import { useInsightConfigManager, type ProviderConfigsCache } from './insight/useInsightConfigManager'

// 从统一类型导入
import type {
  AnalysisStatus, AnalysisMode, OverviewTemplateType, NoteType,
  StoreAnalysisProgress, PageData, ChapterInfo, OverviewData, TimelineEvent,
  QAMessage, NoteData, StoreVlmConfig, StoreLlmConfig, StoreEmbeddingConfig,
  StoreRerankerConfig, StoreImageGenConfig, BatchConfig, StoreInsightConfig
} from '@/types/insight'
import {
  normalizeOpenAiOptions
} from '@/utils/openaiOptions'
import { getProviderBaseUrl, getProviderDefaultModel, normalizeProviderId } from '@/config/aiProviders'

// 重新导出类型（保持向后兼容）
export type {
  AnalysisStatus, AnalysisMode, OverviewTemplateType, NoteType,
  PageData, ChapterInfo, OverviewData, TimelineEvent,
  QAMessage, NoteData, BatchConfig
}

// 为了兼容性，导出 Store 类型的别名
export type AnalysisProgress = StoreAnalysisProgress
export type VlmConfig = StoreVlmConfig
export type LlmConfig = StoreLlmConfig
export type EmbeddingConfig = StoreEmbeddingConfig
export type RerankerConfig = StoreRerankerConfig
export type ImageGenConfig = StoreImageGenConfig
export type InsightConfig = StoreInsightConfig

export const useInsightStore = defineStore('insight', () => {
  function coerceLegacyRetryValue(value: unknown, fallback: number): number {
    return value === undefined || value === null ? fallback : Number(value)
  }

  function syncVlmAliases(target: StoreVlmConfig): StoreVlmConfig {
    return target
  }

  function syncLlmAliases(target: StoreLlmConfig): StoreLlmConfig {
    return target
  }

  function normalizeRerankerConfig(
    source?: Partial<StoreRerankerConfig> | null,
    previous?: StoreRerankerConfig
  ): StoreRerankerConfig {
    const provider = normalizeProviderId(source?.provider || previous?.provider || 'jina') || 'jina'
    return {
      provider,
      apiKey: source?.apiKey ?? previous?.apiKey ?? '',
      model: source?.model ?? previous?.model ?? 'jina-reranker-v2-base-multilingual',
      baseUrl: source?.baseUrl ?? previous?.baseUrl ?? '',
      topK: source?.topK ?? previous?.topK ?? 5,
      transportRetries: source?.transportRetries ?? previous?.transportRetries ?? 10,
      businessRetries: source?.businessRetries ?? previous?.businessRetries ?? 10,
      timeoutSeconds: source?.timeoutSeconds ?? previous?.timeoutSeconds ?? 0,
    }
  }

  function normalizeImageGenConfig(
    source?: Partial<StoreImageGenConfig> | null,
    previous?: StoreImageGenConfig
  ): StoreImageGenConfig {
    const legacyMaxRetries = (source as Record<string, unknown> | null | undefined)?.maxRetries
    const normalizedProvider = normalizeProviderId(source?.provider || previous?.provider || 'gpt2api') || 'gpt2api'
    const previousProvider = normalizeProviderId(previous?.provider || '') || 'gpt2api'
    const providerChanged = normalizedProvider !== previousProvider
    const providerDefaultModel = getProviderDefaultModel(normalizedProvider, 'imageGen')
    const defaultModel = providerDefaultModel || (normalizedProvider === 'gpt2api' ? 'gpt-image-2' : '')
    const defaultBaseUrl = getProviderBaseUrl(normalizedProvider, 'imageGen')
    const base = previous ?? {
      provider: normalizedProvider,
      apiKey: '',
      model: defaultModel,
      baseUrl: defaultBaseUrl,
      transportRetries: 10,
      businessRetries: 10,
      timeoutSeconds: 0,
    }
    const model = source?.model ?? (providerChanged ? providerDefaultModel : base.model || defaultModel)
    const baseUrl = source?.baseUrl ?? (providerChanged ? defaultBaseUrl : (base.baseUrl || defaultBaseUrl))
    const businessRetries = source?.businessRetries ?? (typeof legacyMaxRetries === 'number' ? legacyMaxRetries : base.businessRetries ?? 10)

    return {
      provider: normalizedProvider,
      apiKey: source?.apiKey ?? base.apiKey,
      model,
      baseUrl,
      transportRetries: source?.transportRetries ?? base.transportRetries ?? 10,
      businessRetries,
      timeoutSeconds: source?.timeoutSeconds ?? base.timeoutSeconds ?? 0
    }
  }

  // ============================================================
  // 核心状态
  // ============================================================

  const currentBookId = ref<string | null>(null)
  const currentTaskId = ref<string | null>(null)
  const analysisStatus = ref<AnalysisStatus>('idle')
  const progress = ref<AnalysisProgress>({ current: 0, total: 0, status: 'idle' })
  const bookTotalPages = ref(0)
  const analyzedPagesCount = ref(0)
  const analysisMode = ref<AnalysisMode>('full')
  const incrementalAnalysis = ref(true)
  const chapters = ref<ChapterInfo[]>([])
  const pages = ref<Map<number, PageData>>(new Map())
  const overview = ref<OverviewData | null>(null)
  const generatedTemplates = ref<OverviewTemplateType[]>([])
  const timeline = ref<TimelineEvent[]>([])
  const selectedPageNum = ref<number | null>(null)
  const isLoading = ref(false)
  const error = ref<string | null>(null)
  const dataRefreshKey = ref(0)

  // ============================================================
  // Composables 初始化
  // ============================================================

  const notesComposable = useInsightNotes({ currentBookId })
  const qaComposable = useInsightQA({ currentBookId })

  // ============================================================
  // 配置管理
  // ============================================================

  const config = ref<InsightConfig>({
    vlm: {
      provider: 'gemini',
      apiKey: '',
      model: 'gemini-2.0-flash',
      baseUrl: '',
      openaiOptions: {
        request: { forceJsonOutput: false, temperature: 0.3 },
        execution: { useStream: true, rpmLimit: 0, transportRetries: 10, businessRetries: 10 }
      },
      imageMaxSize: 1280
    },
    llm: {
      useSameAsVlm: false,
      provider: 'gemini',
      apiKey: '',
      model: 'gemini-2.0-flash',
      baseUrl: '',
      openaiOptions: {
        request: { forceJsonOutput: false },
        execution: { useStream: true, rpmLimit: 0, transportRetries: 10, businessRetries: 10 }
      }
    },
    embedding: {
      provider: 'openai',
      apiKey: '',
      model: 'text-embedding-3-small',
      baseUrl: '',
      rpmLimit: 0,
      transportRetries: 10,
      businessRetries: 10,
      timeoutSeconds: 0
    },
    reranker: normalizeRerankerConfig(),
    imageGen: normalizeImageGenConfig(),
    batch: { pagesPerBatch: 5, contextBatchCount: 3, architecturePreset: 'standard', customLayers: [] },
    prompts: {}
  })

  const providerConfigs = ref<ProviderConfigsCache>({ vlm: {}, llm: {}, embedding: {}, reranker: {}, imageGen: {} })
  const configManager = useInsightConfigManager(providerConfigs)

  // ============================================================
  // 计算属性
  // ============================================================

  const progressPercent = computed(() => progress.value.total === 0 ? 0 : Math.round((progress.value.current / progress.value.total) * 100))
  const isAnalyzing = computed(() => analysisStatus.value === 'running')
  const isAnalysisCompleted = computed(() => analysisStatus.value === 'completed')
  const analyzedPageCount = computed(() => {
    if (analyzedPagesCount.value > 0) return analyzedPagesCount.value
    let count = 0
    pages.value.forEach(page => { if (page.analyzed) count++ })
    return count
  })
  const totalPageCount = computed(() => bookTotalPages.value || pages.value.size)
  const selectedPage = computed(() => selectedPageNum.value === null ? null : pages.value.get(selectedPageNum.value) || null)

  // ============================================================
  // 状态管理方法
  // ============================================================

  function setCurrentBook(bookId: string | null): void {
    currentBookId.value = bookId
    bookId ? notesComposable.loadNotes() : notesComposable.clearNotes()
  }
  function setAnalysisStatus(status: AnalysisStatus): void { analysisStatus.value = status; progress.value.status = status }
  function setCurrentTaskId(taskId: string | null): void { currentTaskId.value = taskId }
  function updateProgress(current: number, total: number, message?: string): void { progress.value = { current, total, status: analysisStatus.value, message } }
  function setAnalysisMode(mode: AnalysisMode): void { analysisMode.value = mode }
  function setIncrementalAnalysis(incremental: boolean): void { incrementalAnalysis.value = incremental }
  function setBookTotalPages(totalPages: number): void { bookTotalPages.value = totalPages }
  function setAnalyzedPagesCount(count: number): void { analyzedPagesCount.value = count }
  function setChapters(chapterList: ChapterInfo[]): void { chapters.value = chapterList }
  function setPageData(pageNum: number, data: PageData): void { pages.value.set(pageNum, data) }
  function setPages(pageDataList: PageData[]): void { pages.value.clear(); pageDataList.forEach(p => pages.value.set(p.pageNum, p)) }
  function selectPage(pageNum: number | null): void { selectedPageNum.value = pageNum }
  function setOverview(data: OverviewData | null): void { overview.value = data; if (data && !generatedTemplates.value.includes(data.type)) generatedTemplates.value.push(data.type) }
  function setGeneratedTemplates(templates: OverviewTemplateType[]): void { generatedTemplates.value = templates }
  function setTimeline(events: TimelineEvent[]): void { timeline.value = events }
  function triggerDataRefresh(): void { dataRefreshKey.value = Date.now() }

  // 问答管理
  function addQAMessage(message: QAMessage): void { qaComposable.qaHistory.value.push(message) }
  function updateLastAssistantMessage(content: string): void { const h = qaComposable.qaHistory.value; const m = h[h.length - 1]; if (m?.role === 'assistant') m.content = content }
  function clearQAHistory(): void { qaComposable.clearHistory() }
  function removeLoadingMessages(): void { qaComposable.qaHistory.value = qaComposable.qaHistory.value.filter(m => !m.isLoading) }
  function setStreaming(streaming: boolean): void { qaComposable.setStreaming(streaming) }
  function setCurrentPage(pageNum: number): void { selectedPageNum.value = pageNum }

  // 笔记管理
  async function addNote(note: NoteData): Promise<void> { if (!await notesComposable.addNote({ type: note.type, content: note.content, pageNum: note.pageNum, title: note.title, tags: note.tags, question: note.question, answer: note.answer, citations: note.citations, comment: note.comment })) throw new Error('保存笔记失败') }
  async function updateNote(noteId: string, updates: Partial<NoteData>): Promise<void> { await notesComposable.updateNote(noteId, updates) }
  async function deleteNote(noteId: string): Promise<void> { await notesComposable.deleteNote(noteId) }
  function setNoteTypeFilter(type: NoteType | 'all'): void { notesComposable.setNoteTypeFilter(type) }
  async function loadNotesFromAPI(): Promise<void> { await notesComposable.loadNotes() }

  function setLoading(loading: boolean): void { isLoading.value = loading }
  function setError(message: string | null): void { error.value = message }

  // ============================================================
  // 配置管理 (使用 configManager)
  // ============================================================

  function updateVlmConfig(c: Partial<VlmConfig>): void {
    config.value.vlm = syncVlmAliases({ ...config.value.vlm, ...c })
    if ((c as Record<string, unknown>).rpmLimit !== undefined) config.value.vlm.openaiOptions.execution.rpmLimit = (c as Record<string, any>).rpmLimit
    if ((c as Record<string, unknown>).temperature !== undefined) config.value.vlm.openaiOptions.request.temperature = (c as Record<string, any>).temperature
    if ((c as Record<string, unknown>).forceJson !== undefined) config.value.vlm.openaiOptions.request.forceJsonOutput = Boolean((c as Record<string, any>).forceJson)
    if (Object.prototype.hasOwnProperty.call(c, 'extraBody')) {
      config.value.vlm.openaiOptions.request.extraBody = (c as Record<string, any>).extraBody
    }
    if ((c as Record<string, unknown>).useStream !== undefined) config.value.vlm.openaiOptions.execution.useStream = Boolean((c as Record<string, any>).useStream)
    config.value.vlm = syncVlmAliases(config.value.vlm)
    configManager.vlmManager.save(config.value.vlm.provider, config.value.vlm)
    saveConfigToStorage()
  }
  function updateLlmConfig(c: Partial<LlmConfig>): void {
    config.value.llm = syncLlmAliases({ ...config.value.llm, ...c })
    if ((c as Record<string, unknown>).forceJson !== undefined) config.value.llm.openaiOptions.request.forceJsonOutput = Boolean((c as Record<string, any>).forceJson)
    if (Object.prototype.hasOwnProperty.call(c, 'extraBody')) {
      config.value.llm.openaiOptions.request.extraBody = (c as Record<string, any>).extraBody
    }
    if ((c as Record<string, unknown>).useStream !== undefined) config.value.llm.openaiOptions.execution.useStream = Boolean((c as Record<string, any>).useStream)
    config.value.llm = syncLlmAliases(config.value.llm)
    configManager.llmManager.save(config.value.llm.provider, config.value.llm)
    saveConfigToStorage()
  }
  function updateEmbeddingConfig(c: Partial<EmbeddingConfig>): void { config.value.embedding = { ...config.value.embedding, ...c }; configManager.embeddingManager.save(config.value.embedding.provider, config.value.embedding); saveConfigToStorage() }
  function updateRerankerConfig(c: Partial<RerankerConfig>): void { config.value.reranker = normalizeRerankerConfig(c, config.value.reranker); configManager.rerankerManager.save(config.value.reranker.provider, config.value.reranker); saveConfigToStorage() }
  function updateImageGenConfig(c: Partial<ImageGenConfig>): void { config.value.imageGen = normalizeImageGenConfig(c, config.value.imageGen); configManager.imageGenManager.save(config.value.imageGen.provider, config.value.imageGen); saveConfigToStorage() }
  function updateBatchConfig(c: Partial<BatchConfig>): void { config.value.batch = { ...config.value.batch, ...c }; saveConfigToStorage() }
  function updatePrompts(prompts: Record<string, string>): void { config.value.prompts = { ...config.value.prompts, ...prompts }; saveConfigToStorage() }

  function setVlmProvider(p: string): void { if (config.value.vlm.provider === p) return; configManager.vlmManager.switch(config.value.vlm.provider, p, config.value.vlm); config.value.vlm.provider = p; saveConfigToStorage() }
  function setLlmProvider(p: string): void { if (config.value.llm.provider === p) return; configManager.llmManager.switch(config.value.llm.provider, p, config.value.llm); config.value.llm.provider = p; saveConfigToStorage() }
  function setEmbeddingProvider(p: string): void { if (config.value.embedding.provider === p) return; configManager.embeddingManager.switch(config.value.embedding.provider, p, config.value.embedding); config.value.embedding.provider = p; saveConfigToStorage() }
  function setRerankerProvider(p: string): void { if (config.value.reranker.provider === p) return; configManager.rerankerManager.switch(config.value.reranker.provider, p, config.value.reranker); config.value.reranker.provider = p; saveConfigToStorage() }
  function setImageGenProvider(p: string): void { if (config.value.imageGen.provider === p) return; configManager.imageGenManager.switch(config.value.imageGen.provider, p, config.value.imageGen); config.value.imageGen.provider = p; saveConfigToStorage() }

  function setConfig(newConfig: InsightConfig): void {
    config.value = {
      ...newConfig,
      reranker: normalizeRerankerConfig(newConfig.reranker, config.value.reranker),
      imageGen: normalizeImageGenConfig(newConfig.imageGen, config.value.imageGen),
    }
    saveConfigToStorage()
  }
  function saveConfigToStorage(): void { localStorage.setItem('manga_insight_config', JSON.stringify(config.value)) }
  function loadConfigFromStorage(): void {
    configManager.loadFromStorage()
    const stored = localStorage.getItem('manga_insight_config')
      if (stored) { try { const p = JSON.parse(stored); config.value = { vlm: syncVlmAliases({ ...config.value.vlm, ...p.vlm, openaiOptions: normalizeOpenAiOptions(p?.vlm?.openaiOptions, { rpmLimit: p?.vlm?.rpmLimit, temperature: p?.vlm?.temperature, forceJsonOutput: p?.vlm?.forceJson, extraBody: p?.vlm?.extraBody, useStream: p?.vlm?.useStream }, config.value.vlm.openaiOptions) }), llm: syncLlmAliases({ ...config.value.llm, ...p.llm, openaiOptions: normalizeOpenAiOptions(p?.llm?.openaiOptions, { forceJsonOutput: p?.llm?.forceJson, extraBody: p?.llm?.extraBody, useStream: p?.llm?.useStream }, config.value.llm.openaiOptions) }), embedding: { ...config.value.embedding, ...p.embedding }, reranker: normalizeRerankerConfig(p?.reranker, config.value.reranker), imageGen: normalizeImageGenConfig(p?.imageGen, config.value.imageGen), batch: { ...config.value.batch, ...p.batch }, prompts: p.prompts || {} } } catch (e) { console.error('加载配置失败:', e) } }
  }

  function getConfigForApi(): Record<string, unknown> {
    const mapProvider = <T>(cache: Record<string, T>, mapper: (c: T) => Record<string, unknown>) => Object.fromEntries(Object.entries(cache).map(([p, c]) => [p, mapper(c)]))
    return {
      vlm: { provider: config.value.vlm.provider, api_key: config.value.vlm.apiKey, model: config.value.vlm.model, base_url: config.value.vlm.baseUrl || null, openai_options: { request: { force_json_output: config.value.vlm.openaiOptions.request.forceJsonOutput, temperature: config.value.vlm.openaiOptions.request.temperature, ...(config.value.vlm.openaiOptions.request.extraBody !== undefined ? { extra_body: config.value.vlm.openaiOptions.request.extraBody } : {}) }, execution: { use_stream: config.value.vlm.openaiOptions.execution.useStream, rpm_limit: config.value.vlm.openaiOptions.execution.rpmLimit, transport_retries: config.value.vlm.openaiOptions.execution.transportRetries, business_retries: config.value.vlm.openaiOptions.execution.businessRetries } }, image_max_size: config.value.vlm.imageMaxSize },
      chat_llm: { use_same_as_vlm: config.value.llm.useSameAsVlm, provider: config.value.llm.provider, api_key: config.value.llm.apiKey, model: config.value.llm.model, base_url: config.value.llm.baseUrl || null, openai_options: { request: { force_json_output: config.value.llm.openaiOptions.request.forceJsonOutput, temperature: config.value.llm.openaiOptions.request.temperature, ...(config.value.llm.openaiOptions.request.extraBody !== undefined ? { extra_body: config.value.llm.openaiOptions.request.extraBody } : {}) }, execution: { use_stream: config.value.llm.openaiOptions.execution.useStream, rpm_limit: config.value.llm.openaiOptions.execution.rpmLimit, transport_retries: config.value.llm.openaiOptions.execution.transportRetries, business_retries: config.value.llm.openaiOptions.execution.businessRetries } } },
      embedding: {
        provider: config.value.embedding.provider,
        api_key: config.value.embedding.apiKey,
        model: config.value.embedding.model,
        base_url: config.value.embedding.baseUrl || null,
        rpm_limit: config.value.embedding.rpmLimit,
        transport_retries: config.value.embedding.transportRetries ?? 10,
        business_retries: config.value.embedding.businessRetries ?? 10,
        timeout_seconds: config.value.embedding.timeoutSeconds ?? 0
      },
      reranker: {
        provider: config.value.reranker.provider,
        api_key: config.value.reranker.apiKey,
        model: config.value.reranker.model,
        base_url: config.value.reranker.baseUrl || null,
        top_k: config.value.reranker.topK,
        transport_retries: config.value.reranker.transportRetries ?? 10,
        business_retries: config.value.reranker.businessRetries ?? 10,
        timeout_seconds: config.value.reranker.timeoutSeconds ?? 0,
      },
      image_gen: {
        provider: config.value.imageGen.provider,
        api_key: config.value.imageGen.apiKey,
        model: config.value.imageGen.model,
        base_url: config.value.imageGen.baseUrl || null,
        transport_retries: config.value.imageGen.transportRetries ?? 10,
        business_retries: config.value.imageGen.businessRetries ?? 10,
        timeout_seconds: config.value.imageGen.timeoutSeconds ?? 0,
      },
      analysis: { batch: { pages_per_batch: config.value.batch.pagesPerBatch, context_batch_count: config.value.batch.contextBatchCount, architecture_preset: config.value.batch.architecturePreset, custom_layers: config.value.batch.customLayers.map(l => ({ name: l.name, units_per_group: l.units, align_to_chapter: l.align })) } },
      prompts: config.value.prompts,
      providerSettings: {
        vlmProvider: mapProvider(providerConfigs.value.vlm, c => ({ api_key: c.apiKey || '', model: c.model || '', base_url: c.baseUrl || '', openai_options: { request: { force_json_output: (c.openaiOptions as any)?.request?.forceJsonOutput ?? false, temperature: (c.openaiOptions as any)?.request?.temperature ?? 0.3, ...((c.openaiOptions as any)?.request?.extraBody !== undefined ? { extra_body: (c.openaiOptions as any)?.request?.extraBody } : {}) }, execution: { use_stream: (c.openaiOptions as any)?.execution?.useStream ?? true, rpm_limit: (c.openaiOptions as any)?.execution?.rpmLimit ?? 0, transport_retries: (c.openaiOptions as any)?.execution?.transportRetries ?? 10, business_retries: (c.openaiOptions as any)?.execution?.businessRetries ?? 10 } }, image_max_size: c.imageMaxSize ?? 1280 })),
        llmProvider: mapProvider(providerConfigs.value.llm, c => ({ api_key: c.apiKey || '', model: c.model || '', base_url: c.baseUrl || '', openai_options: { request: { force_json_output: (c.openaiOptions as any)?.request?.forceJsonOutput ?? false, temperature: (c.openaiOptions as any)?.request?.temperature, ...((c.openaiOptions as any)?.request?.extraBody !== undefined ? { extra_body: (c.openaiOptions as any)?.request?.extraBody } : {}) }, execution: { use_stream: (c.openaiOptions as any)?.execution?.useStream ?? true, rpm_limit: (c.openaiOptions as any)?.execution?.rpmLimit ?? 0, transport_retries: (c.openaiOptions as any)?.execution?.transportRetries ?? 10, business_retries: (c.openaiOptions as any)?.execution?.businessRetries ?? 10 } } })),
        embeddingProvider: mapProvider(providerConfigs.value.embedding, c => ({
          api_key: c.apiKey || '',
          model: c.model || '',
          base_url: c.baseUrl || '',
          rpm_limit: c.rpmLimit ?? 0,
          transport_retries: c.transportRetries ?? 10,
          business_retries: c.businessRetries ?? 10,
          timeout_seconds: c.timeoutSeconds ?? 0
        })),
        rerankerProvider: mapProvider(providerConfigs.value.reranker, c => ({ api_key: c.apiKey || '', model: c.model || '', base_url: c.baseUrl || '', top_k: c.topK ?? 5, transport_retries: c.transportRetries ?? 10, business_retries: c.businessRetries ?? 10, timeout_seconds: c.timeoutSeconds ?? 0 })),
        imageGenProvider: mapProvider(providerConfigs.value.imageGen, c => ({ api_key: c.apiKey || '', model: c.model || '', base_url: c.baseUrl || '', transport_retries: c.transportRetries ?? 10, business_retries: c.businessRetries ?? 10, timeout_seconds: c.timeoutSeconds ?? 0 }))
      }
    }
  }

  function setConfigFromApi(apiConfig: Record<string, unknown>): void {
    const vlm = apiConfig.vlm as Record<string, unknown> | undefined
    const chatLlm = apiConfig.chat_llm as Record<string, unknown> | undefined
    const embedding = apiConfig.embedding as Record<string, unknown> | undefined
    const reranker = apiConfig.reranker as Record<string, unknown> | undefined
    const batch = (apiConfig.analysis as Record<string, unknown> | undefined)?.batch as Record<string, unknown> | undefined
    const imageGen = apiConfig.image_gen as Record<string, unknown> | undefined

    if (vlm) config.value.vlm = syncVlmAliases({ provider: (vlm.provider as string) || 'gemini', apiKey: (vlm.api_key as string) || '', model: (vlm.model as string) || '', baseUrl: (vlm.base_url as string) || '', openaiOptions: normalizeOpenAiOptions((vlm.openai_options as any), undefined, { request: { forceJsonOutput: false, temperature: 0.3 }, execution: { useStream: true, rpmLimit: 0, transportRetries: 10, businessRetries: 10 } }), imageMaxSize: vlm.image_max_size !== undefined && vlm.image_max_size !== null ? Number(vlm.image_max_size) : 1280 })
    if (chatLlm) config.value.llm = syncLlmAliases({ useSameAsVlm: chatLlm.use_same_as_vlm === true ? true : false, provider: (chatLlm.provider as string) || config.value.vlm.provider, apiKey: (chatLlm.api_key as string) || config.value.vlm.apiKey, model: (chatLlm.model as string) || config.value.vlm.model, baseUrl: (chatLlm.base_url as string) || config.value.vlm.baseUrl || '', openaiOptions: normalizeOpenAiOptions((chatLlm.openai_options as any), undefined, { request: { forceJsonOutput: false }, execution: { useStream: true, rpmLimit: 0, transportRetries: 10, businessRetries: 10 } }) })
    if (embedding) config.value.embedding = {
      provider: (embedding.provider as string) || 'openai',
      apiKey: (embedding.api_key as string) || '',
      model: (embedding.model as string) || '',
      baseUrl: (embedding.base_url as string) || '',
      rpmLimit: (embedding.rpm_limit as number) ?? 0,
      transportRetries: (embedding.transport_retries as number) ?? 10,
      businessRetries: (embedding.business_retries as number) ?? 10,
      timeoutSeconds: (embedding.timeout_seconds as number) ?? 0
    }
    if (reranker) config.value.reranker = normalizeRerankerConfig({
      provider: (reranker.provider as string) || 'jina',
      apiKey: (reranker.api_key as string) || '',
      model: (reranker.model as string) || '',
      baseUrl: (reranker.base_url as string) || '',
      topK: (reranker.top_k as number) || 5,
      transportRetries: (reranker.transport_retries as number) ?? 10,
      businessRetries: (reranker.business_retries as number) ?? 10,
      timeoutSeconds: (reranker.timeout_seconds as number) ?? 0,
    }, config.value.reranker)
    if (batch) { const cl = batch.custom_layers as Array<Record<string, unknown>> | undefined; config.value.batch = { pagesPerBatch: (batch.pages_per_batch as number) || 5, contextBatchCount: (batch.context_batch_count as number) ?? 3, architecturePreset: (batch.architecture_preset as string) || 'standard', customLayers: cl?.map(l => ({ name: (l.name as string) || '', units: (l.units_per_group as number) || 1, align: (l.align_to_chapter as boolean) || false })) || [] } }
    if (imageGen) config.value.imageGen = normalizeImageGenConfig({
      provider: imageGen.provider as string | undefined,
      apiKey: (imageGen.api_key as string) || '',
      model: imageGen.model as string | undefined,
      baseUrl: (imageGen.base_url as string) || '',
      transportRetries: (imageGen.transport_retries as number) ?? 10,
      businessRetries: (imageGen.business_retries as number) ?? coerceLegacyRetryValue(imageGen.max_retries, 10),
      timeoutSeconds: (imageGen.timeout_seconds as number) ?? 0,
    }, config.value.imageGen)

    const ps = apiConfig.providerSettings as Record<string, Record<string, Record<string, unknown>>> | undefined
    if (ps) {
      if (ps.vlmProvider) for (const [p, c] of Object.entries(ps.vlmProvider)) providerConfigs.value.vlm[p] = { apiKey: (c.api_key as string) || '', model: (c.model as string) || '', baseUrl: (c.base_url as string) || '', openaiOptions: normalizeOpenAiOptions((c.openai_options as any), undefined, { request: { forceJsonOutput: false, temperature: 0.3 }, execution: { useStream: true, rpmLimit: 0, transportRetries: 10, businessRetries: 10 } }), imageMaxSize: (c.image_max_size as number) ?? 1280 }
      if (ps.llmProvider) for (const [p, c] of Object.entries(ps.llmProvider)) providerConfigs.value.llm[p] = { apiKey: (c.api_key as string) || '', model: (c.model as string) || '', baseUrl: (c.base_url as string) || '', openaiOptions: normalizeOpenAiOptions((c.openai_options as any), undefined, { request: { forceJsonOutput: false }, execution: { useStream: true, rpmLimit: 0, transportRetries: 10, businessRetries: 10 } }) }
      if (ps.embeddingProvider) for (const [p, c] of Object.entries(ps.embeddingProvider)) providerConfigs.value.embedding[p] = {
        apiKey: (c.api_key as string) || '',
        model: (c.model as string) || '',
        baseUrl: (c.base_url as string) || '',
        rpmLimit: (c.rpm_limit as number) ?? 0,
        transportRetries: (c.transport_retries as number) ?? 10,
        businessRetries: (c.business_retries as number) ?? 10,
        timeoutSeconds: (c.timeout_seconds as number) ?? 0
      }
      if (ps.rerankerProvider) for (const [p, c] of Object.entries(ps.rerankerProvider)) providerConfigs.value.reranker[p] = { apiKey: (c.api_key as string) || '', model: (c.model as string) || '', baseUrl: (c.base_url as string) || '', topK: (c.top_k as number) ?? 5, transportRetries: (c.transport_retries as number) ?? 10, businessRetries: (c.business_retries as number) ?? 10, timeoutSeconds: (c.timeout_seconds as number) ?? 0 }
      if (ps.imageGenProvider) for (const [p, c] of Object.entries(ps.imageGenProvider)) providerConfigs.value.imageGen[p] = { apiKey: (c.api_key as string) || '', model: (c.model as string) || '', baseUrl: (c.base_url as string) || '', transportRetries: (c.transport_retries as number) ?? 10, businessRetries: (c.business_retries as number) ?? coerceLegacyRetryValue(c.max_retries, 10), timeoutSeconds: (c.timeout_seconds as number) ?? 0 }
      configManager.saveToStorage()
    }
    if (apiConfig.prompts) config.value.prompts = apiConfig.prompts as Record<string, string>
    saveConfigToStorage()
    configManager.vlmManager.save(config.value.vlm.provider, config.value.vlm)
    configManager.llmManager.save(config.value.llm.provider, config.value.llm)
    configManager.embeddingManager.save(config.value.embedding.provider, config.value.embedding)
    configManager.rerankerManager.save(config.value.reranker.provider, config.value.reranker)
    configManager.imageGenManager.save(config.value.imageGen.provider, config.value.imageGen)
  }

  function resetAnalysis(): void { analysisStatus.value = 'idle'; progress.value = { current: 0, total: 0, status: 'idle' }; pages.value.clear(); overview.value = null; timeline.value = [] }
  function reset(): void { currentBookId.value = null; analysisStatus.value = 'idle'; progress.value = { current: 0, total: 0, status: 'idle' }; analysisMode.value = 'full'; incrementalAnalysis.value = true; chapters.value = []; pages.value.clear(); overview.value = null; generatedTemplates.value = []; timeline.value = []; qaComposable.clearHistory(); notesComposable.clearNotes(); selectedPageNum.value = null; notesComposable.setNoteTypeFilter('all'); isLoading.value = false; qaComposable.setStreaming(false); error.value = null }

  return {
    currentBookId, currentTaskId, analysisStatus, progress, analysisMode, incrementalAnalysis, chapters, pages, overview, generatedTemplates, timeline, qaHistory: qaComposable.qaHistory, notes: notesComposable.notes, selectedPageNum, noteTypeFilter: notesComposable.noteTypeFilter, isLoading, isStreaming: qaComposable.isStreaming, error, config,
    progressPercent, isAnalyzing, isAnalysisCompleted, analyzedPageCount, totalPageCount, filteredNotes: notesComposable.filteredNotes, selectedPage,
    setCurrentBook, setCurrentTaskId, setAnalysisStatus, updateProgress, setAnalysisMode, setIncrementalAnalysis, setBookTotalPages, setAnalyzedPagesCount, setChapters, setPageData, setPages, selectPage, setOverview, setGeneratedTemplates, setTimeline, dataRefreshKey, triggerDataRefresh,
    addQAMessage, updateLastAssistantMessage, clearQAHistory, removeLoadingMessages, setStreaming, setCurrentPage, addNote, updateNote, deleteNote, setNoteTypeFilter, loadNotesFromAPI, setLoading, setError,
    updateVlmConfig, updateLlmConfig, updateEmbeddingConfig, updateRerankerConfig, updateImageGenConfig, updateBatchConfig, updatePrompts, setConfig, saveConfigToStorage, loadConfigFromStorage, getConfigForApi, setConfigFromApi, setVlmProvider, setLlmProvider, setEmbeddingProvider, setRerankerProvider, setImageGenProvider,
    resetAnalysis, reset
  }
})
