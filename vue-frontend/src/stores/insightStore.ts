/**
 * 漫画分析状态管理 Store
 * 管理漫画分析状态、进度跟踪、问答和笔记
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

// ============================================================
// 类型定义
// ============================================================

/**
 * 分析状态
 */
export type AnalysisStatus = 'idle' | 'running' | 'paused' | 'completed' | 'failed'

/**
 * 分析模式
 */
export type AnalysisMode = 'full' | 'chapter' | 'page'

/**
 * 概览模板类型
 */
export type OverviewTemplateType =
  | 'no_spoiler'      // 无剧透简介
  | 'story_summary'   // 故事概要
  | 'recap'           // 前情回顾
  | 'character_guide' // 角色指南
  | 'world_setting'   // 世界设定
  | 'highlights'      // 名场面盘点
  | 'reading_notes'   // 阅读笔记

/**
 * 笔记类型
 */
export type NoteType = 'text' | 'qa'

/**
 * 分析进度
 */
export interface AnalysisProgress {
  current: number
  total: number
  status: AnalysisStatus
  message?: string
}

/**
 * 页面数据
 */
export interface PageData {
  pageNum: number
  summary?: string
  dialogues?: Array<{
    character: string
    text: string
  }>
  analyzed: boolean
}

/**
 * 章节数据
 */
export interface ChapterInfo {
  id: string
  title: string
  startPage: number
  endPage: number
}

/**
 * 概览数据
 */
export interface OverviewData {
  type: OverviewTemplateType
  content: string
  generatedAt: string
}

/**
 * 时间线事件
 */
export interface TimelineEvent {
  event: string
  page: number
  description: string
}

/**
 * 问答消息
 */
export interface QAMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  /** 是否为加载状态 */
  isLoading?: boolean
  /** 回答模式标识 */
  mode?: string
  /** 引用页码列表 */
  citations?: Array<{ page: number }>
  /** 是否已保存为笔记 */
  saved?: boolean
}

/**
 * 笔记数据
 */
export interface NoteData {
  id: string
  type: NoteType
  title?: string
  content: string
  pageNum?: number
  tags?: string[]
  createdAt: string
  updatedAt: string
  /** 问答笔记的问题 */
  question?: string
  /** 问答笔记的回答 */
  answer?: string
  /** 问答笔记的引用页码 */
  citations?: Array<{ page: number }>
  /** 问答笔记的补充说明 */
  comment?: string
}

/**
 * VLM 配置
 */
export interface VlmConfig {
  provider: string
  apiKey: string
  model: string
  baseUrl: string
  rpmLimit: number
  temperature: number
  forceJson: boolean
  useStream: boolean
  imageMaxSize: number
}

/**
 * LLM（对话模型）配置
 */
export interface LlmConfig {
  useSameAsVlm: boolean
  provider: string
  apiKey: string
  model: string
  baseUrl: string
  useStream: boolean
}

/**
 * Embedding 配置
 */
export interface EmbeddingConfig {
  provider: string
  apiKey: string
  model: string
  baseUrl: string
  rpmLimit: number
}

/**
 * Reranker 配置
 */
export interface RerankerConfig {
  provider: string
  apiKey: string
  model: string
  baseUrl: string
  topK: number
}

/**
 * 批量分析配置
 */
export interface BatchConfig {
  pagesPerBatch: number
  contextBatchCount: number
  architecturePreset: string
  customLayers: Array<{
    name: string
    units: number
    align: boolean
  }>
}

// ============================================================
// 服务商配置缓存类型定义
// ============================================================

/** VLM 服务商配置缓存项 */
interface VlmProviderConfig {
  apiKey?: string
  model?: string
  baseUrl?: string
  rpmLimit?: number
  temperature?: number
  forceJson?: boolean
  useStream?: boolean
  imageMaxSize?: number
}

/** LLM 服务商配置缓存项 */
interface LlmProviderConfig {
  apiKey?: string
  model?: string
  baseUrl?: string
  useStream?: boolean
}

/** Embedding 服务商配置缓存项 */
interface EmbeddingProviderConfig {
  apiKey?: string
  model?: string
  baseUrl?: string
  rpmLimit?: number
}

/** Reranker 服务商配置缓存项 */
interface RerankerProviderConfig {
  apiKey?: string
  model?: string
  baseUrl?: string
  topK?: number
}

/** Insight 服务商配置缓存结构 */
interface InsightProviderConfigsCache {
  vlm: Record<string, VlmProviderConfig>
  llm: Record<string, LlmProviderConfig>
  embedding: Record<string, EmbeddingProviderConfig>
  reranker: Record<string, RerankerProviderConfig>
}

/** localStorage 存储键 */
const STORAGE_KEY_INSIGHT_PROVIDER_CONFIGS = 'insight_provider_configs'

/**
 * 分析配置
 */
export interface InsightConfig {
  vlm: VlmConfig
  llm: LlmConfig
  embedding: EmbeddingConfig
  reranker: RerankerConfig
  batch: BatchConfig
  prompts: Record<string, string>
}

// ============================================================
// Store 定义
// ============================================================

export const useInsightStore = defineStore('insight', () => {
  // ============================================================
  // 状态定义
  // ============================================================

  /** 当前书籍ID */
  const currentBookId = ref<string | null>(null)

  /** 当前分析任务ID（用于暂停/恢复/取消操作） */
  const currentTaskId = ref<string | null>(null)

  /** 分析状态 */
  const analysisStatus = ref<AnalysisStatus>('idle')

  /** 分析进度 */
  const progress = ref<AnalysisProgress>({
    current: 0,
    total: 0,
    status: 'idle'
  })

  /** 书籍总页数（从书籍信息获取） */
  const bookTotalPages = ref(0)

  /** 已分析页数（从API获取） */
  const analyzedPagesCount = ref(0)

  /** 分析模式 */
  const analysisMode = ref<AnalysisMode>('full')

  /** 是否增量分析 */
  const incrementalAnalysis = ref(true)

  /** 章节列表 */
  const chapters = ref<ChapterInfo[]>([])

  /** 页面数据映射 */
  const pages = ref<Map<number, PageData>>(new Map())

  /** 概览数据 */
  const overview = ref<OverviewData | null>(null)

  /** 已生成的概览模板列表 */
  const generatedTemplates = ref<OverviewTemplateType[]>([])

  /** 时间线数据 */
  const timeline = ref<TimelineEvent[]>([])

  /** 问答历史 */
  const qaHistory = ref<QAMessage[]>([])

  /** 笔记列表 */
  const notes = ref<NoteData[]>([])

  /** 当前选中的页码 */
  const selectedPageNum = ref<number | null>(null)

  /** 笔记类型筛选 */
  const noteTypeFilter = ref<NoteType | 'all'>('all')

  /** 是否正在加载 */
  const isLoading = ref(false)

  /** 是否正在流式响应 */
  const isStreaming = ref(false)

  /** 错误信息 */
  const error = ref<string | null>(null)

  /** 数据刷新触发器（时间戳，用于通知面板组件重新加载数据） */
  const dataRefreshKey = ref(0)

  /** 分析配置 */
  const config = ref<InsightConfig>({
    vlm: {
      provider: 'gemini',
      apiKey: '',
      model: 'gemini-2.0-flash',
      baseUrl: '',
      rpmLimit: 10,
      temperature: 0.3,
      forceJson: false,
      useStream: true,
      imageMaxSize: 0
    },
    llm: {
      useSameAsVlm: true,
      provider: 'gemini',
      apiKey: '',
      model: 'gemini-2.0-flash',
      baseUrl: '',
      useStream: true
    },
    embedding: {
      provider: 'openai',
      apiKey: '',
      model: 'text-embedding-3-small',
      baseUrl: '',
      rpmLimit: 0
    },
    reranker: {
      provider: 'jina',
      apiKey: '',
      model: 'jina-reranker-v2-base-multilingual',
      baseUrl: '',
      topK: 5
    },
    batch: {
      pagesPerBatch: 5,
      contextBatchCount: 1,
      architecturePreset: 'standard',
      customLayers: []
    },
    prompts: {}
  })

  /** 服务商配置分组存储（用于切换服务商时保存/恢复配置） */
  const providerConfigs = ref<InsightProviderConfigsCache>({
    vlm: {},
    llm: {},
    embedding: {},
    reranker: {}
  })

  // ============================================================
  // 计算属性
  // ============================================================

  /** 分析进度百分比 */
  const progressPercent = computed(() => {
    if (progress.value.total === 0) return 0
    return Math.round((progress.value.current / progress.value.total) * 100)
  })

  /** 是否正在分析 */
  const isAnalyzing = computed(() => {
    return analysisStatus.value === 'running'
  })

  /** 是否已完成分析 */
  const isAnalysisCompleted = computed(() => {
    return analysisStatus.value === 'completed'
  })

  /** 已分析的页面数量 - 优先使用API返回的值 */
  const analyzedPageCount = computed(() => {
    if (analyzedPagesCount.value > 0) return analyzedPagesCount.value
    let count = 0
    pages.value.forEach(page => {
      if (page.analyzed) count++
    })
    return count
  })

  /** 总页面数量（优先使用bookTotalPages，否则使用pages.value.size） */
  const totalPageCount = computed(() => bookTotalPages.value || pages.value.size)

  /** 过滤后的笔记列表 */
  const filteredNotes = computed(() => {
    if (noteTypeFilter.value === 'all') {
      return notes.value
    }
    return notes.value.filter(note => note.type === noteTypeFilter.value)
  })

  /** 当前选中的页面数据 */
  const selectedPage = computed(() => {
    if (selectedPageNum.value === null) return null
    return pages.value.get(selectedPageNum.value) || null
  })

  // ============================================================
  // 书籍和分析管理
  // ============================================================

  /**
   * 设置当前书籍
   * @param bookId - 书籍ID
   */
  function setCurrentBook(bookId: string | null): void {
    currentBookId.value = bookId
    if (bookId) {
      console.log(`当前分析书籍: ${bookId}`)
      // 加载该书籍的笔记
      console.log('准备加载笔记...')
      loadNotesFromAPI()
    } else {
      notes.value = []
    }
  }

  /**
   * 设置分析状态
   * @param status - 分析状态
   */
  function setAnalysisStatus(status: AnalysisStatus): void {
    analysisStatus.value = status
    progress.value.status = status
    console.log(`分析状态: ${status}`)
  }

  /**
   * 设置当前任务ID
   * @param taskId - 任务ID，传 null 清除
   */
  function setCurrentTaskId(taskId: string | null): void {
    currentTaskId.value = taskId
  }

  /**
   * 更新分析进度
   * @param current - 当前进度
   * @param total - 总数
   * @param message - 可选消息
   */
  function updateProgress(current: number, total: number, message?: string): void {
    progress.value = {
      current,
      total,
      status: analysisStatus.value,
      message
    }
  }

  /**
   * 设置分析模式
   * @param mode - 分析模式
   */
  function setAnalysisMode(mode: AnalysisMode): void {
    analysisMode.value = mode
  }

  /**
   * 设置增量分析开关
   * @param incremental - 是否增量分析
   */
  function setIncrementalAnalysis(incremental: boolean): void {
    incrementalAnalysis.value = incremental
  }

  // ============================================================
  // 章节管理
  // ============================================================

  /**
   * 设置书籍总页数
   * @param totalPages - 总页数
   */
  function setBookTotalPages(totalPages: number): void {
    bookTotalPages.value = totalPages
    console.log(`书籍总页数: ${totalPages}`)
  }

  /**
   * 设置已分析页数
   * @param count - 已分析页数
   */
  function setAnalyzedPagesCount(count: number): void {
    analyzedPagesCount.value = count
    console.log(`已分析页数: ${count}`)
  }

  /**
   * 设置章节列表
   * @param chapterList - 章节列表
   */
  function setChapters(chapterList: ChapterInfo[]): void {
    chapters.value = chapterList
    console.log(`章节列表已设置，共 ${chapterList.length} 章`)
  }

  // ============================================================
  // 页面数据管理
  // ============================================================

  /**
   * 设置页面数据
   * @param pageNum - 页码
   * @param data - 页面数据
   */
  function setPageData(pageNum: number, data: PageData): void {
    pages.value.set(pageNum, data)
  }

  /**
   * 批量设置页面数据
   * @param pageDataList - 页面数据列表
   */
  function setPages(pageDataList: PageData[]): void {
    pages.value.clear()
    for (const page of pageDataList) {
      pages.value.set(page.pageNum, page)
    }
    console.log(`页面数据已设置，共 ${pageDataList.length} 页`)
  }

  /**
   * 选择页面
   * @param pageNum - 页码
   */
  function selectPage(pageNum: number | null): void {
    selectedPageNum.value = pageNum
  }

  // ============================================================
  // 概览管理
  // ============================================================

  /**
   * 设置概览数据
   * @param data - 概览数据
   */
  function setOverview(data: OverviewData | null): void {
    overview.value = data
    if (data && !generatedTemplates.value.includes(data.type)) {
      generatedTemplates.value.push(data.type)
    }
  }

  /**
   * 设置已生成的模板列表
   * @param templates - 模板类型列表
   */
  function setGeneratedTemplates(templates: OverviewTemplateType[]): void {
    generatedTemplates.value = templates
  }

  // ============================================================
  // 时间线管理
  // ============================================================

  /**
   * 设置时间线数据
   * @param events - 时间线事件列表
   */
  function setTimeline(events: TimelineEvent[]): void {
    timeline.value = events
    console.log(`时间线已设置，共 ${events.length} 个事件`)
  }

  // ============================================================
  // 问答管理
  // ============================================================

  /**
   * 添加问答消息
   * @param message - 消息
   */
  function addQAMessage(message: QAMessage): void {
    qaHistory.value.push(message)
  }

  /**
   * 更新最后一条助手消息（用于流式响应）
   * @param content - 新内容
   */
  function updateLastAssistantMessage(content: string): void {
    const lastMessage = qaHistory.value[qaHistory.value.length - 1]
    if (lastMessage && lastMessage.role === 'assistant') {
      lastMessage.content = content
    }
  }

  /**
   * 清除问答历史
   */
  function clearQAHistory(): void {
    qaHistory.value = []
  }

  /**
   * 移除加载中的消息
   */
  function removeLoadingMessages(): void {
    qaHistory.value = qaHistory.value.filter(m => !m.isLoading)
  }

  /**
   * 设置流式响应状态
   * @param streaming - 是否正在流式响应
   */
  function setStreaming(streaming: boolean): void {
    isStreaming.value = streaming
  }

  /**
   * 设置当前选中的页码
   * @param pageNum - 页码
   */
  function setCurrentPage(pageNum: number): void {
    selectedPageNum.value = pageNum
  }

  // ============================================================
  // 笔记管理
  // ============================================================

  /**
   * 添加笔记
   * @param note - 笔记数据
   */
  async function addNote(note: NoteData): Promise<void> {
    const success = await saveNoteToAPI(note)
    if (success) {
      notes.value.unshift(note)
      console.log(`已添加笔记: ${note.id}`)
    } else {
      throw new Error('保存笔记到后端失败')
    }
  }

  /**
   * 更新笔记
   * @param noteId - 笔记ID
   * @param updates - 更新数据
   */
  async function updateNote(noteId: string, updates: Partial<NoteData>): Promise<void> {
    const note = notes.value.find(n => n.id === noteId)
    if (note) {
      Object.assign(note, updates, { updatedAt: new Date().toISOString() })
      await updateNoteToAPI(noteId, updates)
      console.log(`已更新笔记: ${noteId}`)
    }
  }

  /**
   * 删除笔记
   * @param noteId - 笔记ID
   */
  async function deleteNote(noteId: string): Promise<void> {
    const index = notes.value.findIndex(n => n.id === noteId)
    if (index >= 0) {
      const success = await deleteNoteFromAPI(noteId)
      if (success) {
        notes.value.splice(index, 1)
        console.log(`已删除笔记: ${noteId}`)
      } else {
        console.error(`删除笔记失败: ${noteId}`)
      }
    }
  }

  /**
   * 设置笔记类型筛选
   * @param type - 笔记类型或 'all'
   */
  function setNoteTypeFilter(type: NoteType | 'all'): void {
    noteTypeFilter.value = type
  }

  /**
   * 从后端API加载笔记
   */
  async function loadNotesFromAPI(): Promise<void> {
    console.log('loadNotesFromAPI 被调用, bookId:', currentBookId.value)
    if (!currentBookId.value) return

    try {
      const response = await fetch(`/api/manga-insight/${currentBookId.value}/notes`)
      const data = await response.json()
      console.log('笔记API响应:', data)

      if (data.success && data.notes) {
        notes.value = data.notes
        console.log(`已从API加载 ${notes.value.length} 条笔记`)
      } else {
        console.log('API返回无笔记数据')
        notes.value = []
      }
    } catch (e) {
      console.error('从API加载笔记失败:', e)
      notes.value = []
    }
  }

  /**
   * 保存笔记到后端API
   */
  async function saveNoteToAPI(note: NoteData): Promise<boolean> {
    if (!currentBookId.value) return false

    try {
      const response = await fetch(`/api/manga-insight/${currentBookId.value}/notes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(note)
      })
      const data = await response.json()
      return data.success
    } catch (e) {
      console.error('保存笔记到API失败:', e)
      return false
    }
  }

  /**
   * 更新笔记到后端API
   */
  async function updateNoteToAPI(noteId: string, updates: Partial<NoteData>): Promise<boolean> {
    if (!currentBookId.value) return false

    try {
      const response = await fetch(`/api/manga-insight/${currentBookId.value}/notes/${noteId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updates)
      })
      const data = await response.json()
      return data.success
    } catch (e) {
      console.error('更新笔记到API失败:', e)
      return false
    }
  }

  /**
   * 从后端API删除笔记
   */
  async function deleteNoteFromAPI(noteId: string): Promise<boolean> {
    if (!currentBookId.value) return false

    try {
      const response = await fetch(`/api/manga-insight/${currentBookId.value}/notes/${noteId}`, {
        method: 'DELETE'
      })
      const data = await response.json()
      return data.success
    } catch (e) {
      console.error('从API删除笔记失败:', e)
      return false
    }
  }

  // ============================================================
  // 数据刷新触发器
  // ============================================================

  /**
   * 触发数据刷新
   * 更新 dataRefreshKey 以通知面板组件重新加载数据
   * 用于分析完成后自动刷新概览和时间线
   */
  function triggerDataRefresh(): void {
    dataRefreshKey.value = Date.now()
    console.log('触发数据刷新，key:', dataRefreshKey.value)
  }

  // ============================================================
  // 加载状态管理
  // ============================================================

  /**
   * 设置加载状态
   * @param loading - 是否正在加载
   */
  function setLoading(loading: boolean): void {
    isLoading.value = loading
  }

  /**
   * 设置错误信息
   * @param message - 错误信息
   */
  function setError(message: string | null): void {
    error.value = message
  }

  // ============================================================
  // 配置管理
  // ============================================================

  /**
   * 更新 VLM 配置
   * @param vlmConfig - VLM 配置
   */
  function updateVlmConfig(vlmConfig: Partial<VlmConfig>): void {
    config.value.vlm = { ...config.value.vlm, ...vlmConfig }
    // 同时保存当前服务商配置到缓存
    saveVlmProviderConfig(config.value.vlm.provider)
    saveConfigToStorage()
  }

  /**
   * 更新 LLM 配置
   * @param llmConfig - LLM 配置
   */
  function updateLlmConfig(llmConfig: Partial<LlmConfig>): void {
    config.value.llm = { ...config.value.llm, ...llmConfig }
    // 同时保存当前服务商配置到缓存
    saveLlmProviderConfig(config.value.llm.provider)
    saveConfigToStorage()
  }

  /**
   * 更新 Embedding 配置
   * @param embeddingConfig - Embedding 配置
   */
  function updateEmbeddingConfig(embeddingConfig: Partial<EmbeddingConfig>): void {
    config.value.embedding = { ...config.value.embedding, ...embeddingConfig }
    // 同时保存当前服务商配置到缓存
    saveEmbeddingProviderConfig(config.value.embedding.provider)
    saveConfigToStorage()
  }

  /**
   * 更新 Reranker 配置
   * @param rerankerConfig - Reranker 配置
   */
  function updateRerankerConfig(rerankerConfig: Partial<RerankerConfig>): void {
    config.value.reranker = { ...config.value.reranker, ...rerankerConfig }
    // 同时保存当前服务商配置到缓存
    saveRerankerProviderConfig(config.value.reranker.provider)
    saveConfigToStorage()
  }

  /**
   * 更新批量分析配置
   * @param batchConfig - 批量分析配置
   */
  function updateBatchConfig(batchConfig: Partial<BatchConfig>): void {
    config.value.batch = { ...config.value.batch, ...batchConfig }
    saveConfigToStorage()
  }

  /**
   * 更新提示词配置
   * @param prompts - 提示词配置（部分或全部）
   */
  function updatePrompts(prompts: Record<string, string>): void {
    config.value.prompts = { ...config.value.prompts, ...prompts }
    saveConfigToStorage()
  }

  // ============================================================
  // 服务商切换方法（支持多服务商配置持久化）
  // ============================================================

  /**
   * 设置 VLM 服务商（自动保存旧配置并恢复新配置）
   * @param provider - 新服务商名称
   */
  function setVlmProvider(provider: string): void {
    const oldProvider = config.value.vlm.provider
    if (oldProvider === provider) return

    // 保存当前服务商配置
    saveVlmProviderConfig(oldProvider)

    // 切换服务商
    config.value.vlm.provider = provider

    // 恢复目标服务商配置（如果有）
    restoreVlmProviderConfig(provider)

    saveConfigToStorage()
    console.log(`[Insight] VLM 服务商已切换为: ${provider}`)
  }

  /**
   * 设置 LLM 服务商（自动保存旧配置并恢复新配置）
   * @param provider - 新服务商名称
   */
  function setLlmProvider(provider: string): void {
    const oldProvider = config.value.llm.provider
    if (oldProvider === provider) return

    // 保存当前服务商配置
    saveLlmProviderConfig(oldProvider)

    // 切换服务商
    config.value.llm.provider = provider

    // 恢复目标服务商配置（如果有）
    restoreLlmProviderConfig(provider)

    saveConfigToStorage()
    console.log(`[Insight] LLM 服务商已切换为: ${provider}`)
  }

  /**
   * 设置 Embedding 服务商（自动保存旧配置并恢复新配置）
   * @param provider - 新服务商名称
   */
  function setEmbeddingProvider(provider: string): void {
    const oldProvider = config.value.embedding.provider
    if (oldProvider === provider) return

    // 保存当前服务商配置
    saveEmbeddingProviderConfig(oldProvider)

    // 切换服务商
    config.value.embedding.provider = provider

    // 恢复目标服务商配置（如果有）
    restoreEmbeddingProviderConfig(provider)

    saveConfigToStorage()
    console.log(`[Insight] Embedding 服务商已切换为: ${provider}`)
  }

  /**
   * 设置 Reranker 服务商（自动保存旧配置并恢复新配置）
   * @param provider - 新服务商名称
   */
  function setRerankerProvider(provider: string): void {
    const oldProvider = config.value.reranker.provider
    if (oldProvider === provider) return

    // 保存当前服务商配置
    saveRerankerProviderConfig(oldProvider)

    // 切换服务商
    config.value.reranker.provider = provider

    // 恢复目标服务商配置（如果有）
    restoreRerankerProviderConfig(provider)

    saveConfigToStorage()
    console.log(`[Insight] Reranker 服务商已切换为: ${provider}`)
  }

  // ============================================================
  // 服务商配置保存/恢复方法
  // ============================================================

  /**
   * 保存服务商配置缓存到 localStorage
   */
  function saveProviderConfigsToStorage(): void {
    localStorage.setItem(STORAGE_KEY_INSIGHT_PROVIDER_CONFIGS, JSON.stringify(providerConfigs.value))
  }

  /**
   * 从 localStorage 加载服务商配置缓存
   */
  function loadProviderConfigsFromStorage(): void {
    const stored = localStorage.getItem(STORAGE_KEY_INSIGHT_PROVIDER_CONFIGS)
    if (stored) {
      try {
        const parsed = JSON.parse(stored)
        providerConfigs.value = {
          vlm: parsed.vlm || {},
          llm: parsed.llm || {},
          embedding: parsed.embedding || {},
          reranker: parsed.reranker || {}
        }
        console.log('[Insight] 已加载服务商配置缓存')
      } catch (e) {
        console.error('[Insight] 加载服务商配置缓存失败:', e)
      }
    }
  }

  /** 保存 VLM 服务商配置到缓存 */
  function saveVlmProviderConfig(provider: string): void {
    if (!provider) return
    providerConfigs.value.vlm[provider] = {
      apiKey: config.value.vlm.apiKey,
      model: config.value.vlm.model,
      baseUrl: config.value.vlm.baseUrl,
      rpmLimit: config.value.vlm.rpmLimit,
      temperature: config.value.vlm.temperature,
      forceJson: config.value.vlm.forceJson,
      useStream: config.value.vlm.useStream,
      imageMaxSize: config.value.vlm.imageMaxSize
    }
    saveProviderConfigsToStorage()
    console.log(`[Insight] 保存 VLM 服务商配置: ${provider}`)
  }

  /** 恢复 VLM 服务商配置从缓存 */
  function restoreVlmProviderConfig(provider: string): void {
    if (!provider) return
    const cached = providerConfigs.value.vlm[provider]
    if (cached) {
      if (cached.apiKey !== undefined) config.value.vlm.apiKey = cached.apiKey
      if (cached.model !== undefined) config.value.vlm.model = cached.model
      if (cached.baseUrl !== undefined) config.value.vlm.baseUrl = cached.baseUrl
      if (cached.rpmLimit !== undefined) config.value.vlm.rpmLimit = cached.rpmLimit
      if (cached.temperature !== undefined) config.value.vlm.temperature = cached.temperature
      if (cached.forceJson !== undefined) config.value.vlm.forceJson = cached.forceJson
      if (cached.useStream !== undefined) config.value.vlm.useStream = cached.useStream
      if (cached.imageMaxSize !== undefined) config.value.vlm.imageMaxSize = cached.imageMaxSize
      console.log(`[Insight] 恢复 VLM 服务商配置: ${provider}`)
    } else {
      // 无缓存时清空配置
      config.value.vlm.apiKey = ''
      config.value.vlm.model = ''
      config.value.vlm.baseUrl = ''
      console.log(`[Insight] ${provider} 无缓存配置，使用默认值`)
    }
  }

  /** 保存 LLM 服务商配置到缓存 */
  function saveLlmProviderConfig(provider: string): void {
    if (!provider) return
    providerConfigs.value.llm[provider] = {
      apiKey: config.value.llm.apiKey,
      model: config.value.llm.model,
      baseUrl: config.value.llm.baseUrl,
      useStream: config.value.llm.useStream
    }
    saveProviderConfigsToStorage()
    console.log(`[Insight] 保存 LLM 服务商配置: ${provider}`)
  }

  /** 恢复 LLM 服务商配置从缓存 */
  function restoreLlmProviderConfig(provider: string): void {
    if (!provider) return
    const cached = providerConfigs.value.llm[provider]
    if (cached) {
      if (cached.apiKey !== undefined) config.value.llm.apiKey = cached.apiKey
      if (cached.model !== undefined) config.value.llm.model = cached.model
      if (cached.baseUrl !== undefined) config.value.llm.baseUrl = cached.baseUrl
      if (cached.useStream !== undefined) config.value.llm.useStream = cached.useStream
      console.log(`[Insight] 恢复 LLM 服务商配置: ${provider}`)
    } else {
      config.value.llm.apiKey = ''
      config.value.llm.model = ''
      config.value.llm.baseUrl = ''
      console.log(`[Insight] ${provider} 无缓存配置，使用默认值`)
    }
  }

  /** 保存 Embedding 服务商配置到缓存 */
  function saveEmbeddingProviderConfig(provider: string): void {
    if (!provider) return
    providerConfigs.value.embedding[provider] = {
      apiKey: config.value.embedding.apiKey,
      model: config.value.embedding.model,
      baseUrl: config.value.embedding.baseUrl,
      rpmLimit: config.value.embedding.rpmLimit
    }
    saveProviderConfigsToStorage()
    console.log(`[Insight] 保存 Embedding 服务商配置: ${provider}`)
  }

  /** 恢复 Embedding 服务商配置从缓存 */
  function restoreEmbeddingProviderConfig(provider: string): void {
    if (!provider) return
    const cached = providerConfigs.value.embedding[provider]
    if (cached) {
      if (cached.apiKey !== undefined) config.value.embedding.apiKey = cached.apiKey
      if (cached.model !== undefined) config.value.embedding.model = cached.model
      if (cached.baseUrl !== undefined) config.value.embedding.baseUrl = cached.baseUrl
      if (cached.rpmLimit !== undefined) config.value.embedding.rpmLimit = cached.rpmLimit
      console.log(`[Insight] 恢复 Embedding 服务商配置: ${provider}`)
    } else {
      config.value.embedding.apiKey = ''
      config.value.embedding.model = ''
      config.value.embedding.baseUrl = ''
      console.log(`[Insight] ${provider} 无缓存配置，使用默认值`)
    }
  }

  /** 保存 Reranker 服务商配置到缓存 */
  function saveRerankerProviderConfig(provider: string): void {
    if (!provider) return
    providerConfigs.value.reranker[provider] = {
      apiKey: config.value.reranker.apiKey,
      model: config.value.reranker.model,
      baseUrl: config.value.reranker.baseUrl,
      topK: config.value.reranker.topK
    }
    saveProviderConfigsToStorage()
    console.log(`[Insight] 保存 Reranker 服务商配置: ${provider}`)
  }

  /** 恢复 Reranker 服务商配置从缓存 */
  function restoreRerankerProviderConfig(provider: string): void {
    if (!provider) return
    const cached = providerConfigs.value.reranker[provider]
    if (cached) {
      if (cached.apiKey !== undefined) config.value.reranker.apiKey = cached.apiKey
      if (cached.model !== undefined) config.value.reranker.model = cached.model
      if (cached.baseUrl !== undefined) config.value.reranker.baseUrl = cached.baseUrl
      if (cached.topK !== undefined) config.value.reranker.topK = cached.topK
      console.log(`[Insight] 恢复 Reranker 服务商配置: ${provider}`)
    } else {
      config.value.reranker.apiKey = ''
      config.value.reranker.model = ''
      config.value.reranker.baseUrl = ''
      console.log(`[Insight] ${provider} 无缓存配置，使用默认值`)
    }
  }

  /**
   * 设置完整配置
   * @param newConfig - 完整配置
   */
  function setConfig(newConfig: InsightConfig): void {
    config.value = newConfig
    saveConfigToStorage()
  }

  /**
   * 保存配置到 localStorage
   */
  function saveConfigToStorage(): void {
    const key = 'manga_insight_config'
    localStorage.setItem(key, JSON.stringify(config.value))
  }

  /**
   * 从 localStorage 加载配置
   */
  function loadConfigFromStorage(): void {
    // 先加载服务商配置缓存
    loadProviderConfigsFromStorage()

    const key = 'manga_insight_config'
    const stored = localStorage.getItem(key)
    if (stored) {
      try {
        const parsed = JSON.parse(stored)
        // 合并配置，保留默认值
        config.value = {
          vlm: { ...config.value.vlm, ...parsed.vlm },
          llm: { ...config.value.llm, ...parsed.llm },
          embedding: { ...config.value.embedding, ...parsed.embedding },
          reranker: { ...config.value.reranker, ...parsed.reranker },
          batch: { ...config.value.batch, ...parsed.batch },
          prompts: parsed.prompts || {}
        }
        console.log('已加载漫画分析配置')
      } catch (e) {
        console.error('加载漫画分析配置失败:', e)
      }
    }
  }

  /**
   * 将配置转换为 API 格式
   */
  function getConfigForApi(): Record<string, unknown> {
    return {
      vlm: {
        provider: config.value.vlm.provider,
        api_key: config.value.vlm.apiKey,
        model: config.value.vlm.model,
        base_url: config.value.vlm.baseUrl || null,
        rpm_limit: config.value.vlm.rpmLimit,
        temperature: config.value.vlm.temperature,
        force_json: config.value.vlm.forceJson,
        use_stream: config.value.vlm.useStream,
        image_max_size: config.value.vlm.imageMaxSize
      },
      chat_llm: {
        use_same_as_vlm: config.value.llm.useSameAsVlm,
        provider: config.value.llm.provider,
        api_key: config.value.llm.apiKey,
        model: config.value.llm.model,
        base_url: config.value.llm.baseUrl || null,
        use_stream: config.value.llm.useStream
      },
      embedding: {
        provider: config.value.embedding.provider,
        api_key: config.value.embedding.apiKey,
        model: config.value.embedding.model,
        base_url: config.value.embedding.baseUrl || null,
        rpm_limit: config.value.embedding.rpmLimit
      },
      reranker: {
        provider: config.value.reranker.provider,
        api_key: config.value.reranker.apiKey,
        model: config.value.reranker.model,
        base_url: config.value.reranker.baseUrl || null,
        top_k: config.value.reranker.topK
      },
      analysis: {
        batch: {
          pages_per_batch: config.value.batch.pagesPerBatch,
          context_batch_count: config.value.batch.contextBatchCount,
          architecture_preset: config.value.batch.architecturePreset,
          custom_layers: config.value.batch.customLayers.map(l => ({
            name: l.name,
            units_per_group: l.units,
            align_to_chapter: l.align
          }))
        }
      },
      prompts: config.value.prompts,
      // ===== 服务商分组配置缓存（复刻翻译设置页面的 providerSettings）=====
      // 保存所有服务商的配置，实现切换服务商时的配置记忆
      providerSettings: buildProviderSettingsForBackend()
    }
  }

  /**
   * 构建服务商分组配置用于保存到后端
   * 复刻翻译设置页面的 buildProviderSettingsForBackend 结构
   */
  function buildProviderSettingsForBackend(): Record<string, Record<string, Record<string, unknown>>> {
    // 初始化结果对象
    const vlmProviderConfigs: Record<string, Record<string, unknown>> = {}
    const llmProviderConfigs: Record<string, Record<string, unknown>> = {}
    const embeddingProviderConfigs: Record<string, Record<string, unknown>> = {}
    const rerankerProviderConfigs: Record<string, Record<string, unknown>> = {}

    // VLM 服务商配置
    for (const [provider, cfg] of Object.entries(providerConfigs.value.vlm)) {
      vlmProviderConfigs[provider] = {
        api_key: cfg.apiKey || '',
        model: cfg.model || '',
        base_url: cfg.baseUrl || '',
        rpm_limit: cfg.rpmLimit ?? 10,
        temperature: cfg.temperature ?? 0.3,
        force_json: cfg.forceJson ?? false,
        use_stream: cfg.useStream ?? true,
        image_max_size: cfg.imageMaxSize ?? 0
      }
    }

    // LLM 服务商配置
    for (const [provider, cfg] of Object.entries(providerConfigs.value.llm)) {
      llmProviderConfigs[provider] = {
        api_key: cfg.apiKey || '',
        model: cfg.model || '',
        base_url: cfg.baseUrl || '',
        use_stream: cfg.useStream ?? true
      }
    }

    // Embedding 服务商配置
    for (const [provider, cfg] of Object.entries(providerConfigs.value.embedding)) {
      embeddingProviderConfigs[provider] = {
        api_key: cfg.apiKey || '',
        model: cfg.model || '',
        base_url: cfg.baseUrl || '',
        rpm_limit: cfg.rpmLimit ?? 0
      }
    }

    // Reranker 服务商配置
    for (const [provider, cfg] of Object.entries(providerConfigs.value.reranker)) {
      rerankerProviderConfigs[provider] = {
        api_key: cfg.apiKey || '',
        model: cfg.model || '',
        base_url: cfg.baseUrl || '',
        top_k: cfg.topK ?? 5
      }
    }

    return {
      vlmProvider: vlmProviderConfigs,
      llmProvider: llmProviderConfigs,
      embeddingProvider: embeddingProviderConfigs,
      rerankerProvider: rerankerProviderConfigs
    }
  }

  /**
   * 从 API 响应设置配置
   * @param apiConfig - API 返回的配置
   */
  function setConfigFromApi(apiConfig: Record<string, unknown>): void {
    const vlm = apiConfig.vlm as Record<string, unknown> | undefined
    const chatLlm = apiConfig.chat_llm as Record<string, unknown> | undefined
    const embedding = apiConfig.embedding as Record<string, unknown> | undefined
    const reranker = apiConfig.reranker as Record<string, unknown> | undefined
    const analysis = apiConfig.analysis as Record<string, unknown> | undefined
    const batch = analysis?.batch as Record<string, unknown> | undefined

    if (vlm) {
      config.value.vlm = {
        provider: (vlm.provider as string) || 'gemini',
        apiKey: (vlm.api_key as string) || '',
        model: (vlm.model as string) || '',
        baseUrl: (vlm.base_url as string) || '',
        rpmLimit: (vlm.rpm_limit as number) || 10,
        temperature: (vlm.temperature as number) || 0.3,
        forceJson: (vlm.force_json as boolean) || false,
        useStream: vlm.use_stream !== false,
        imageMaxSize: (vlm.image_max_size as number) || 0
      }
    }

    if (chatLlm) {
      config.value.llm = {
        useSameAsVlm: chatLlm.use_same_as_vlm !== false,
        provider: (chatLlm.provider as string) || config.value.vlm.provider,
        apiKey: (chatLlm.api_key as string) || config.value.vlm.apiKey,
        model: (chatLlm.model as string) || config.value.vlm.model,
        baseUrl: (chatLlm.base_url as string) || config.value.vlm.baseUrl,
        useStream: chatLlm.use_stream !== false
      }
    }

    if (embedding) {
      config.value.embedding = {
        provider: (embedding.provider as string) || 'openai',
        apiKey: (embedding.api_key as string) || '',
        model: (embedding.model as string) || '',
        baseUrl: (embedding.base_url as string) || '',
        rpmLimit: (embedding.rpm_limit as number) ?? 0
      }
    }

    if (reranker) {
      config.value.reranker = {
        provider: (reranker.provider as string) || 'jina',
        apiKey: (reranker.api_key as string) || '',
        model: (reranker.model as string) || '',
        baseUrl: (reranker.base_url as string) || '',
        topK: (reranker.top_k as number) || 5
      }
    }

    if (batch) {
      const customLayers = batch.custom_layers as Array<Record<string, unknown>> | undefined
      config.value.batch = {
        pagesPerBatch: (batch.pages_per_batch as number) || 5,
        contextBatchCount: (batch.context_batch_count as number) ?? 1,
        architecturePreset: (batch.architecture_preset as string) || 'standard',
        customLayers: customLayers?.map(l => ({
          name: (l.name as string) || '',
          units: (l.units_per_group as number) || 1,
          align: (l.align_to_chapter as boolean) || false
        })) || []
      }
    }

    // ===== 解析并恢复后端保存的服务商配置缓存（复刻翻译设置页面逻辑）=====
    const providerSettings = apiConfig.providerSettings as Record<string, Record<string, Record<string, unknown>>> | undefined
    if (providerSettings) {
      // VLM 服务商配置
      if (providerSettings.vlmProvider) {
        for (const [provider, cfg] of Object.entries(providerSettings.vlmProvider)) {
          providerConfigs.value.vlm[provider] = {
            apiKey: (cfg.api_key as string) || '',
            model: (cfg.model as string) || '',
            baseUrl: (cfg.base_url as string) || '',
            rpmLimit: (cfg.rpm_limit as number) ?? 10,
            temperature: (cfg.temperature as number) ?? 0.3,
            forceJson: (cfg.force_json as boolean) ?? false,
            useStream: (cfg.use_stream as boolean) ?? true,
            imageMaxSize: (cfg.image_max_size as number) ?? 0
          }
        }
      }

      // LLM 服务商配置
      if (providerSettings.llmProvider) {
        for (const [provider, cfg] of Object.entries(providerSettings.llmProvider)) {
          providerConfigs.value.llm[provider] = {
            apiKey: (cfg.api_key as string) || '',
            model: (cfg.model as string) || '',
            baseUrl: (cfg.base_url as string) || '',
            useStream: (cfg.use_stream as boolean) ?? true
          }
        }
      }

      // Embedding 服务商配置
      if (providerSettings.embeddingProvider) {
        for (const [provider, cfg] of Object.entries(providerSettings.embeddingProvider)) {
          providerConfigs.value.embedding[provider] = {
            apiKey: (cfg.api_key as string) || '',
            model: (cfg.model as string) || '',
            baseUrl: (cfg.base_url as string) || '',
            rpmLimit: (cfg.rpm_limit as number) ?? 0
          }
        }
      }

      // Reranker 服务商配置
      if (providerSettings.rerankerProvider) {
        for (const [provider, cfg] of Object.entries(providerSettings.rerankerProvider)) {
          providerConfigs.value.reranker[provider] = {
            apiKey: (cfg.api_key as string) || '',
            model: (cfg.model as string) || '',
            baseUrl: (cfg.base_url as string) || '',
            topK: (cfg.top_k as number) ?? 5
          }
        }
      }

      console.log('[Insight] 已从后端恢复服务商配置缓存')
      // 保存到 localStorage
      saveProviderConfigsToStorage()
    }

    // ===== 解析提示词配置 =====
    const prompts = apiConfig.prompts as Record<string, string> | undefined
    if (prompts) {
      config.value.prompts = prompts
    }

    saveConfigToStorage()

    // 同时将当前激活的服务商配置保存到缓存中
    saveVlmProviderConfig(config.value.vlm.provider)
    saveLlmProviderConfig(config.value.llm.provider)
    saveEmbeddingProviderConfig(config.value.embedding.provider)
    saveRerankerProviderConfig(config.value.reranker.provider)

    console.log('[Insight] 已从 API 加载配置')
  }

  // ============================================================
  // 重置方法
  // ============================================================

  /**
   * 重置分析状态
   */
  function resetAnalysis(): void {
    analysisStatus.value = 'idle'
    progress.value = {
      current: 0,
      total: 0,
      status: 'idle'
    }
    pages.value.clear()
    overview.value = null
    timeline.value = []
    console.log('分析状态已重置')
  }

  /**
   * 重置所有状态
   */
  function reset(): void {
    currentBookId.value = null
    analysisStatus.value = 'idle'
    progress.value = {
      current: 0,
      total: 0,
      status: 'idle'
    }
    analysisMode.value = 'full'
    incrementalAnalysis.value = true
    chapters.value = []
    pages.value.clear()
    overview.value = null
    generatedTemplates.value = []
    timeline.value = []
    qaHistory.value = []
    notes.value = []
    selectedPageNum.value = null
    noteTypeFilter.value = 'all'
    isLoading.value = false
    isStreaming.value = false
    error.value = null
    console.log('漫画分析状态已重置')
  }

  // ============================================================
  // 返回 Store 接口
  // ============================================================

  return {
    // 状态
    currentBookId,
    currentTaskId,
    analysisStatus,
    progress,
    analysisMode,
    incrementalAnalysis,
    chapters,
    pages,
    overview,
    generatedTemplates,
    timeline,
    qaHistory,
    notes,
    selectedPageNum,
    noteTypeFilter,
    isLoading,
    isStreaming,
    error,
    config,

    // 计算属性
    progressPercent,
    isAnalyzing,
    isAnalysisCompleted,
    analyzedPageCount,
    totalPageCount,
    filteredNotes,
    selectedPage,

    // 书籍和分析管理
    setCurrentBook,
    setCurrentTaskId,
    setAnalysisStatus,
    updateProgress,
    setAnalysisMode,
    setIncrementalAnalysis,

    // 章节管理
    setBookTotalPages,
    setAnalyzedPagesCount,
    setChapters,

    // 页面数据管理
    setPageData,
    setPages,
    selectPage,

    // 概览管理
    setOverview,
    setGeneratedTemplates,

    // 时间线管理
    setTimeline,

    // 数据刷新触发器
    dataRefreshKey,
    triggerDataRefresh,

    // 问答管理
    addQAMessage,
    updateLastAssistantMessage,
    clearQAHistory,
    removeLoadingMessages,
    setStreaming,
    setCurrentPage,

    // 笔记管理
    addNote,
    updateNote,
    deleteNote,
    setNoteTypeFilter,
    loadNotesFromAPI,

    // 加载状态
    setLoading,
    setError,

    // 配置管理
    updateVlmConfig,
    updateLlmConfig,
    updateEmbeddingConfig,
    updateRerankerConfig,
    updateBatchConfig,
    updatePrompts,
    setConfig,
    saveConfigToStorage,
    loadConfigFromStorage,
    getConfigForApi,
    setConfigFromApi,

    // 服务商切换（多服务商配置持久化）
    setVlmProvider,
    setLlmProvider,
    setEmbeddingProvider,
    setRerankerProvider,

    // 重置
    resetAnalysis,
    reset
  }
})
