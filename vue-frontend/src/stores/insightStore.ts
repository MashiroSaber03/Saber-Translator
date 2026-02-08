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
    vlm: { provider: 'gemini', apiKey: '', model: 'gemini-2.0-flash', baseUrl: '', rpmLimit: 10, temperature: 0.3, forceJson: false, useStream: true, imageMaxSize: 0 },
    llm: { useSameAsVlm: true, provider: 'gemini', apiKey: '', model: 'gemini-2.0-flash', baseUrl: '', useStream: true },
    embedding: { provider: 'openai', apiKey: '', model: 'text-embedding-3-small', baseUrl: '', rpmLimit: 0 },
    reranker: { provider: 'jina', apiKey: '', model: 'jina-reranker-v2-base-multilingual', baseUrl: '', topK: 5 },
    imageGen: { provider: 'siliconflow', apiKey: '', model: 'stabilityai/stable-diffusion-3-5-large', baseUrl: '', maxRetries: 3 },
    batch: { pagesPerBatch: 5, contextBatchCount: 1, architecturePreset: 'standard', customLayers: [] },
    prompts: {}
  })

  const providerConfigs = ref<ProviderConfigsCache>({ vlm: {}, llm: {}, embedding: {}, reranker: {} })
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

  function updateVlmConfig(c: Partial<VlmConfig>): void { config.value.vlm = { ...config.value.vlm, ...c }; configManager.vlmManager.save(config.value.vlm.provider, config.value.vlm); saveConfigToStorage() }
  function updateLlmConfig(c: Partial<LlmConfig>): void { config.value.llm = { ...config.value.llm, ...c }; configManager.llmManager.save(config.value.llm.provider, config.value.llm); saveConfigToStorage() }
  function updateEmbeddingConfig(c: Partial<EmbeddingConfig>): void { config.value.embedding = { ...config.value.embedding, ...c }; configManager.embeddingManager.save(config.value.embedding.provider, config.value.embedding); saveConfigToStorage() }
  function updateRerankerConfig(c: Partial<RerankerConfig>): void { config.value.reranker = { ...config.value.reranker, ...c }; configManager.rerankerManager.save(config.value.reranker.provider, config.value.reranker); saveConfigToStorage() }
  function updateImageGenConfig(c: Partial<ImageGenConfig>): void { config.value.imageGen = { ...config.value.imageGen, ...c }; saveConfigToStorage() }
  function updateBatchConfig(c: Partial<BatchConfig>): void { config.value.batch = { ...config.value.batch, ...c }; saveConfigToStorage() }
  function updatePrompts(prompts: Record<string, string>): void { config.value.prompts = { ...config.value.prompts, ...prompts }; saveConfigToStorage() }

  function setVlmProvider(p: string): void { if (config.value.vlm.provider === p) return; configManager.vlmManager.switch(config.value.vlm.provider, p, config.value.vlm); config.value.vlm.provider = p; saveConfigToStorage() }
  function setLlmProvider(p: string): void { if (config.value.llm.provider === p) return; configManager.llmManager.switch(config.value.llm.provider, p, config.value.llm); config.value.llm.provider = p; saveConfigToStorage() }
  function setEmbeddingProvider(p: string): void { if (config.value.embedding.provider === p) return; configManager.embeddingManager.switch(config.value.embedding.provider, p, config.value.embedding); config.value.embedding.provider = p; saveConfigToStorage() }
  function setRerankerProvider(p: string): void { if (config.value.reranker.provider === p) return; configManager.rerankerManager.switch(config.value.reranker.provider, p, config.value.reranker); config.value.reranker.provider = p; saveConfigToStorage() }

  function setConfig(newConfig: InsightConfig): void { config.value = newConfig; saveConfigToStorage() }
  function saveConfigToStorage(): void { localStorage.setItem('manga_insight_config', JSON.stringify(config.value)) }
  function loadConfigFromStorage(): void {
    configManager.loadFromStorage()
    const stored = localStorage.getItem('manga_insight_config')
    if (stored) { try { const p = JSON.parse(stored); config.value = { vlm: { ...config.value.vlm, ...p.vlm }, llm: { ...config.value.llm, ...p.llm }, embedding: { ...config.value.embedding, ...p.embedding }, reranker: { ...config.value.reranker, ...p.reranker }, imageGen: { ...config.value.imageGen, ...p.imageGen }, batch: { ...config.value.batch, ...p.batch }, prompts: p.prompts || {} } } catch (e) { console.error('加载配置失败:', e) } }
  }

  function getConfigForApi(): Record<string, unknown> {
    const mapProvider = <T>(cache: Record<string, T>, mapper: (c: T) => Record<string, unknown>) => Object.fromEntries(Object.entries(cache).map(([p, c]) => [p, mapper(c)]))
    return {
      vlm: { provider: config.value.vlm.provider, api_key: config.value.vlm.apiKey, model: config.value.vlm.model, base_url: config.value.vlm.baseUrl || null, rpm_limit: config.value.vlm.rpmLimit, temperature: config.value.vlm.temperature, force_json: config.value.vlm.forceJson, use_stream: config.value.vlm.useStream, image_max_size: config.value.vlm.imageMaxSize },
      chat_llm: { use_same_as_vlm: config.value.llm.useSameAsVlm, provider: config.value.llm.provider, api_key: config.value.llm.apiKey, model: config.value.llm.model, base_url: config.value.llm.baseUrl || null, use_stream: config.value.llm.useStream },
      embedding: { provider: config.value.embedding.provider, api_key: config.value.embedding.apiKey, model: config.value.embedding.model, base_url: config.value.embedding.baseUrl || null, rpm_limit: config.value.embedding.rpmLimit },
      reranker: { provider: config.value.reranker.provider, api_key: config.value.reranker.apiKey, model: config.value.reranker.model, base_url: config.value.reranker.baseUrl || null, top_k: config.value.reranker.topK },
      image_gen: { provider: config.value.imageGen.provider, api_key: config.value.imageGen.apiKey, model: config.value.imageGen.model, base_url: config.value.imageGen.baseUrl || null, max_retries: config.value.imageGen.maxRetries },
      analysis: { batch: { pages_per_batch: config.value.batch.pagesPerBatch, context_batch_count: config.value.batch.contextBatchCount, architecture_preset: config.value.batch.architecturePreset, custom_layers: config.value.batch.customLayers.map(l => ({ name: l.name, units_per_group: l.units, align_to_chapter: l.align })) } },
      prompts: config.value.prompts,
      providerSettings: {
        vlmProvider: mapProvider(providerConfigs.value.vlm, c => ({ api_key: c.apiKey || '', model: c.model || '', base_url: c.baseUrl || '', rpm_limit: c.rpmLimit ?? 10, temperature: c.temperature ?? 0.3, force_json: c.forceJson ?? false, use_stream: c.useStream ?? true, image_max_size: c.imageMaxSize ?? 0 })),
        llmProvider: mapProvider(providerConfigs.value.llm, c => ({ api_key: c.apiKey || '', model: c.model || '', base_url: c.baseUrl || '', use_stream: c.useStream ?? true })),
        embeddingProvider: mapProvider(providerConfigs.value.embedding, c => ({ api_key: c.apiKey || '', model: c.model || '', base_url: c.baseUrl || '', rpm_limit: c.rpmLimit ?? 0 })),
        rerankerProvider: mapProvider(providerConfigs.value.reranker, c => ({ api_key: c.apiKey || '', model: c.model || '', base_url: c.baseUrl || '', top_k: c.topK ?? 5 }))
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

    if (vlm) config.value.vlm = { provider: (vlm.provider as string) || 'gemini', apiKey: (vlm.api_key as string) || '', model: (vlm.model as string) || '', baseUrl: (vlm.base_url as string) || '', rpmLimit: (vlm.rpm_limit as number) || 10, temperature: (vlm.temperature as number) || 0.3, forceJson: (vlm.force_json as boolean) || false, useStream: vlm.use_stream !== false, imageMaxSize: (vlm.image_max_size as number) || 0 }
    if (chatLlm) config.value.llm = { useSameAsVlm: chatLlm.use_same_as_vlm !== false, provider: (chatLlm.provider as string) || config.value.vlm.provider, apiKey: (chatLlm.api_key as string) || config.value.vlm.apiKey, model: (chatLlm.model as string) || config.value.vlm.model, baseUrl: (chatLlm.base_url as string) || config.value.vlm.baseUrl, useStream: chatLlm.use_stream !== false }
    if (embedding) config.value.embedding = { provider: (embedding.provider as string) || 'openai', apiKey: (embedding.api_key as string) || '', model: (embedding.model as string) || '', baseUrl: (embedding.base_url as string) || '', rpmLimit: (embedding.rpm_limit as number) ?? 0 }
    if (reranker) config.value.reranker = { provider: (reranker.provider as string) || 'jina', apiKey: (reranker.api_key as string) || '', model: (reranker.model as string) || '', baseUrl: (reranker.base_url as string) || '', topK: (reranker.top_k as number) || 5 }
    if (batch) { const cl = batch.custom_layers as Array<Record<string, unknown>> | undefined; config.value.batch = { pagesPerBatch: (batch.pages_per_batch as number) || 5, contextBatchCount: (batch.context_batch_count as number) ?? 1, architecturePreset: (batch.architecture_preset as string) || 'standard', customLayers: cl?.map(l => ({ name: (l.name as string) || '', units: (l.units_per_group as number) || 1, align: (l.align_to_chapter as boolean) || false })) || [] } }
    if (imageGen) config.value.imageGen = { provider: (imageGen.provider as string) || 'siliconflow', apiKey: (imageGen.api_key as string) || '', model: (imageGen.model as string) || 'stabilityai/stable-diffusion-3-5-large', baseUrl: (imageGen.base_url as string) || '', maxRetries: (imageGen.max_retries as number) || 3 }

    const ps = apiConfig.providerSettings as Record<string, Record<string, Record<string, unknown>>> | undefined
    if (ps) {
      if (ps.vlmProvider) for (const [p, c] of Object.entries(ps.vlmProvider)) providerConfigs.value.vlm[p] = { apiKey: (c.api_key as string) || '', model: (c.model as string) || '', baseUrl: (c.base_url as string) || '', rpmLimit: (c.rpm_limit as number) ?? 10, temperature: (c.temperature as number) ?? 0.3, forceJson: (c.force_json as boolean) ?? false, useStream: (c.use_stream as boolean) ?? true, imageMaxSize: (c.image_max_size as number) ?? 0 }
      if (ps.llmProvider) for (const [p, c] of Object.entries(ps.llmProvider)) providerConfigs.value.llm[p] = { apiKey: (c.api_key as string) || '', model: (c.model as string) || '', baseUrl: (c.base_url as string) || '', useStream: (c.use_stream as boolean) ?? true }
      if (ps.embeddingProvider) for (const [p, c] of Object.entries(ps.embeddingProvider)) providerConfigs.value.embedding[p] = { apiKey: (c.api_key as string) || '', model: (c.model as string) || '', baseUrl: (c.base_url as string) || '', rpmLimit: (c.rpm_limit as number) ?? 0 }
      if (ps.rerankerProvider) for (const [p, c] of Object.entries(ps.rerankerProvider)) providerConfigs.value.reranker[p] = { apiKey: (c.api_key as string) || '', model: (c.model as string) || '', baseUrl: (c.base_url as string) || '', topK: (c.top_k as number) ?? 5 }
      configManager.saveToStorage()
    }
    if (apiConfig.prompts) config.value.prompts = apiConfig.prompts as Record<string, string>
    saveConfigToStorage()
    configManager.vlmManager.save(config.value.vlm.provider, config.value.vlm)
    configManager.llmManager.save(config.value.llm.provider, config.value.llm)
    configManager.embeddingManager.save(config.value.embedding.provider, config.value.embedding)
    configManager.rerankerManager.save(config.value.reranker.provider, config.value.reranker)
  }

  function resetAnalysis(): void { analysisStatus.value = 'idle'; progress.value = { current: 0, total: 0, status: 'idle' }; pages.value.clear(); overview.value = null; timeline.value = [] }
  function reset(): void { currentBookId.value = null; analysisStatus.value = 'idle'; progress.value = { current: 0, total: 0, status: 'idle' }; analysisMode.value = 'full'; incrementalAnalysis.value = true; chapters.value = []; pages.value.clear(); overview.value = null; generatedTemplates.value = []; timeline.value = []; qaComposable.clearHistory(); notesComposable.clearNotes(); selectedPageNum.value = null; notesComposable.setNoteTypeFilter('all'); isLoading.value = false; qaComposable.setStreaming(false); error.value = null }

  return {
    currentBookId, currentTaskId, analysisStatus, progress, analysisMode, incrementalAnalysis, chapters, pages, overview, generatedTemplates, timeline, qaHistory: qaComposable.qaHistory, notes: notesComposable.notes, selectedPageNum, noteTypeFilter: notesComposable.noteTypeFilter, isLoading, isStreaming: qaComposable.isStreaming, error, config,
    progressPercent, isAnalyzing, isAnalysisCompleted, analyzedPageCount, totalPageCount, filteredNotes: notesComposable.filteredNotes, selectedPage,
    setCurrentBook, setCurrentTaskId, setAnalysisStatus, updateProgress, setAnalysisMode, setIncrementalAnalysis, setBookTotalPages, setAnalyzedPagesCount, setChapters, setPageData, setPages, selectPage, setOverview, setGeneratedTemplates, setTimeline, dataRefreshKey, triggerDataRefresh,
    addQAMessage, updateLastAssistantMessage, clearQAHistory, removeLoadingMessages, setStreaming, setCurrentPage, addNote, updateNote, deleteNote, setNoteTypeFilter, loadNotesFromAPI, setLoading, setError,
    updateVlmConfig, updateLlmConfig, updateEmbeddingConfig, updateRerankerConfig, updateImageGenConfig, updateBatchConfig, updatePrompts, setConfig, saveConfigToStorage, loadConfigFromStorage, getConfigForApi, setConfigFromApi, setVlmProvider, setLlmProvider, setEmbeddingProvider, setRerankerProvider,
    resetAnalysis, reset
  }
})
