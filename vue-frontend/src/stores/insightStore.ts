import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

import { useInsightNotes } from './insight/useInsightNotes'
import { useInsightQA } from './insight/useInsightQA'
import { useInsightConfigManager, type ProviderConfigsCache } from './insight/useInsightConfigManager'
import { buildInsightConfigApiPayload } from './insight/insightConfigApiPayload'
import {
  normalizeInsightImageGenConfig,
  normalizeInsightRerankerConfig,
} from './insight/insightConfigDefaults'
import { applyInsightProviderSettingsFromApi } from './insight/insightProviderSettingsHydration'
import { applyActiveInsightConfigFromApi } from './insight/insightConfigApiHydration'

import type {
  AnalysisStatus, AnalysisMode, OverviewTemplateType, NoteType,
  StoreAnalysisProgress, PageData, ChapterInfo, OverviewData, TimelineEvent,
  QAMessage, NoteData, StoreVlmConfig, StoreLlmConfig, StoreEmbeddingConfig,
  StoreRerankerConfig, StoreImageGenConfig, BatchConfig, StoreInsightConfig
} from '@/types/insight'

export type {
  AnalysisStatus, AnalysisMode, OverviewTemplateType, NoteType,
  PageData, ChapterInfo, OverviewData, TimelineEvent,
  QAMessage, NoteData, BatchConfig
}

export type AnalysisProgress = StoreAnalysisProgress
export type VlmConfig = StoreVlmConfig
export type LlmConfig = StoreLlmConfig
export type EmbeddingConfig = StoreEmbeddingConfig
export type RerankerConfig = StoreRerankerConfig
export type ImageGenConfig = StoreImageGenConfig
export type InsightConfig = StoreInsightConfig

export const useInsightStore = defineStore('insight', () => {
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

  const notesComposable = useInsightNotes({ currentBookId })
  const qaComposable = useInsightQA()

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
    reranker: normalizeInsightRerankerConfig(),
    imageGen: normalizeInsightImageGenConfig(),
    batch: { pagesPerBatch: 5, contextBatchCount: 3, architecturePreset: 'standard', customLayers: [] },
    prompts: {}
  })

  const providerConfigs = ref<ProviderConfigsCache>({ vlm: {}, llm: {}, embedding: {}, reranker: {}, imageGen: {} })
  const configManager = useInsightConfigManager(providerConfigs)

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

  function setCurrentBook(bookId: string | null): void {
    const previousBookId = currentBookId.value
    currentBookId.value = bookId
    if (bookId) {
      if (previousBookId !== bookId) {
        notesComposable.clearNotes()
      }
      notesComposable.loadNotes()
    } else {
      notesComposable.clearNotes()
    }
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

  function addQAMessage(message: QAMessage): void { qaComposable.qaHistory.value.push(message) }
  function updateLastAssistantMessage(content: string): void { const h = qaComposable.qaHistory.value; const m = h[h.length - 1]; if (m?.role === 'assistant') m.content = content }
  function clearQAHistory(): void { qaComposable.clearHistory() }
  function removeLoadingMessages(): void { qaComposable.qaHistory.value = qaComposable.qaHistory.value.filter(m => !m.isLoading) }
  function setStreaming(streaming: boolean): void { qaComposable.setStreaming(streaming) }
  function setCurrentPage(pageNum: number): void { selectedPageNum.value = pageNum }

  function addNote(note: NoteData): Promise<void> {
    return notesComposable.addNote(note).then(result => {
      if (!result) throw new Error('保存笔记失败')
    })
  }
  async function updateNote(noteId: string, updates: Partial<NoteData>): Promise<void> { await notesComposable.updateNote(noteId, updates) }
  async function deleteNote(noteId: string): Promise<void> { await notesComposable.deleteNote(noteId) }
  function setNoteTypeFilter(type: NoteType | 'all'): void { notesComposable.setNoteTypeFilter(type) }
  async function loadNotesFromAPI(): Promise<void> { await notesComposable.loadNotes() }

  function setLoading(loading: boolean): void { isLoading.value = loading }
  function setError(message: string | null): void { error.value = message }

  function updateVlmConfig(c: Partial<VlmConfig>): void {
    config.value.vlm = {
      provider: c.provider ?? config.value.vlm.provider,
      apiKey: c.apiKey ?? config.value.vlm.apiKey,
      model: c.model ?? config.value.vlm.model,
      baseUrl: c.baseUrl ?? config.value.vlm.baseUrl,
      openaiOptions: c.openaiOptions ?? config.value.vlm.openaiOptions,
      imageMaxSize: c.imageMaxSize ?? config.value.vlm.imageMaxSize,
    }
    configManager.vlmManager.save(config.value.vlm.provider, config.value.vlm)
  }
  function updateLlmConfig(c: Partial<LlmConfig>): void {
    config.value.llm = {
      useSameAsVlm: c.useSameAsVlm ?? config.value.llm.useSameAsVlm,
      provider: c.provider ?? config.value.llm.provider,
      apiKey: c.apiKey ?? config.value.llm.apiKey,
      model: c.model ?? config.value.llm.model,
      baseUrl: c.baseUrl ?? config.value.llm.baseUrl,
      openaiOptions: c.openaiOptions ?? config.value.llm.openaiOptions,
    }
    configManager.llmManager.save(config.value.llm.provider, config.value.llm)
  }
  function updateEmbeddingConfig(c: Partial<EmbeddingConfig>): void { config.value.embedding = { ...config.value.embedding, ...c }; configManager.embeddingManager.save(config.value.embedding.provider, config.value.embedding) }
  function updateRerankerConfig(c: Partial<RerankerConfig>): void { config.value.reranker = normalizeInsightRerankerConfig(c, config.value.reranker); configManager.rerankerManager.save(config.value.reranker.provider, config.value.reranker) }
  function updateImageGenConfig(c: Partial<ImageGenConfig>): void { config.value.imageGen = normalizeInsightImageGenConfig(c, config.value.imageGen); configManager.imageGenManager.save(config.value.imageGen.provider, config.value.imageGen) }
  function updateBatchConfig(c: Partial<BatchConfig>): void { config.value.batch = { ...config.value.batch, ...c } }
  function updatePrompts(prompts: Record<string, string>): void { config.value.prompts = { ...config.value.prompts, ...prompts } }

  function setVlmProvider(p: string): void { if (config.value.vlm.provider === p) return; configManager.vlmManager.switch(config.value.vlm.provider, p, config.value.vlm); config.value.vlm.provider = p }
  function setLlmProvider(p: string): void { if (config.value.llm.provider === p) return; configManager.llmManager.switch(config.value.llm.provider, p, config.value.llm); config.value.llm.provider = p }
  function setEmbeddingProvider(p: string): void { if (config.value.embedding.provider === p) return; configManager.embeddingManager.switch(config.value.embedding.provider, p, config.value.embedding); config.value.embedding.provider = p }
  function setRerankerProvider(p: string): void { if (config.value.reranker.provider === p) return; configManager.rerankerManager.switch(config.value.reranker.provider, p, config.value.reranker); config.value.reranker.provider = p }
  function setImageGenProvider(p: string): void { if (config.value.imageGen.provider === p) return; configManager.imageGenManager.switch(config.value.imageGen.provider, p, config.value.imageGen); config.value.imageGen.provider = p }

  function switchVlmProviderDraft(draft: VlmConfig): VlmConfig {
    const previousProvider = config.value.vlm.provider
    const nextProvider = draft.provider
    if (previousProvider === nextProvider) return config.value.vlm

    config.value.vlm = { ...draft, provider: previousProvider }
    configManager.vlmManager.switch(previousProvider, nextProvider, config.value.vlm)
    config.value.vlm.provider = nextProvider
    return config.value.vlm
  }

  function switchLlmProviderDraft(draft: LlmConfig): LlmConfig {
    const previousProvider = config.value.llm.provider
    const nextProvider = draft.provider
    if (previousProvider === nextProvider) return config.value.llm

    config.value.llm = { ...draft, provider: previousProvider }
    configManager.llmManager.switch(previousProvider, nextProvider, config.value.llm)
    config.value.llm.provider = nextProvider
    return config.value.llm
  }

  function switchEmbeddingProviderDraft(draft: EmbeddingConfig): EmbeddingConfig {
    const previousProvider = config.value.embedding.provider
    const nextProvider = draft.provider
    if (previousProvider === nextProvider) return config.value.embedding

    config.value.embedding = { ...draft, provider: previousProvider }
    configManager.embeddingManager.switch(previousProvider, nextProvider, config.value.embedding)
    config.value.embedding.provider = nextProvider
    return config.value.embedding
  }

  function switchRerankerProviderDraft(draft: RerankerConfig): RerankerConfig {
    const previousProvider = config.value.reranker.provider
    const nextProvider = draft.provider
    if (previousProvider === nextProvider) return config.value.reranker

    config.value.reranker = normalizeInsightRerankerConfig({ ...draft, provider: previousProvider }, config.value.reranker)
    configManager.rerankerManager.switch(previousProvider, nextProvider, config.value.reranker)
    config.value.reranker.provider = nextProvider
    return config.value.reranker
  }

  function switchImageGenProviderDraft(draft: ImageGenConfig): ImageGenConfig {
    const previousProvider = config.value.imageGen.provider
    const nextProvider = draft.provider
    if (previousProvider === nextProvider) return config.value.imageGen

    config.value.imageGen = normalizeInsightImageGenConfig({ ...draft, provider: previousProvider }, config.value.imageGen)
    configManager.imageGenManager.switch(previousProvider, nextProvider, config.value.imageGen)
    config.value.imageGen.provider = nextProvider
    return config.value.imageGen
  }

  function setConfig(newConfig: InsightConfig): void {
    config.value = {
      ...newConfig,
      reranker: normalizeInsightRerankerConfig(newConfig.reranker, config.value.reranker),
      imageGen: normalizeInsightImageGenConfig(newConfig.imageGen, config.value.imageGen),
    }
  }

  function getConfigForApi(): Record<string, unknown> {
    return buildInsightConfigApiPayload(config.value, providerConfigs.value)
  }

  function setConfigFromApi(apiConfig: Record<string, unknown>): void {
    applyActiveInsightConfigFromApi(config.value, apiConfig)
    applyInsightProviderSettingsFromApi(providerConfigs.value, apiConfig.provider_settings)
    if (apiConfig.prompts) config.value.prompts = apiConfig.prompts as Record<string, string>
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
    updateVlmConfig, updateLlmConfig, updateEmbeddingConfig, updateRerankerConfig, updateImageGenConfig, updateBatchConfig, updatePrompts, setConfig, getConfigForApi, setConfigFromApi, setVlmProvider, setLlmProvider, setEmbeddingProvider, setRerankerProvider, setImageGenProvider,
    switchVlmProviderDraft, switchLlmProviderDraft, switchEmbeddingProviderDraft, switchRerankerProviderDraft, switchImageGenProviderDraft,
    resetAnalysis, reset
  }
})
