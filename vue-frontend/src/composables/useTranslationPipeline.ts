import { computed, ref, watch, type Ref } from 'vue'

import {
  createChapterRemoveTextJob,
  createChapterTranslationJob,
} from '@/api/v2/translation'
import {
  getPageDocument,
  getPageSummary,
  listChapterPages,
  type V2TranslationBootstrap,
} from '@/api/v2/content'
import type { components } from '@/api/generated/v2'
import {
  NONTERMINAL_JOB_STATUSES,
  type V2Job,
  type V2JobStatus,
} from '@/api/v2/jobs'
import { pageSummaryToImage } from '@/adapters/v2ContentAdapter'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import {
  flushPageDocument,
  hasPendingPageDocument,
  registerPageDocument,
} from '@/services/pageDocumentPersistence'
import { useToast } from '@/utils/toast'
import { pageSelectionToPageIndexes } from '@/utils/pageSelection'
import { parseCompleteTextStyleSettings } from '@/defaults/textStyleDefaults'
import { usePublicUserAccess } from '@/composables/usePublicUserAccess'
type TranslationMode = 'standard' | 'hq' | 'proofread' | 'removeText'

interface PageSelection {
  pages: number[]
}

export interface TranslationProgress {
  current: number
  total: number
  completed: number
  failed: number
  isInProgress: boolean
  label?: string
  percentage?: number
  executionMode: 'sequential' | 'parallel'
  status?: V2JobStatus
  queuePosition?: number | null
  currentStep?: TranslationCurrentStep
  pools: TranslationPoolProgress[]
}

type TranslationCurrentStep = components['schemas']['JobProgressCurrentStep']
type V2JobProgress = components['schemas']['JobProgress']
type V2JobProgressPool = components['schemas']['JobProgressPool']
type V2JobProgressPoolCurrent = components['schemas']['JobProgressPoolCurrent']

export interface TranslationPoolProgress {
  kind: string
  total: number
  completed: number
  failed: number
  skipped: number
  cancelled: number
  waiting: number
  processing: number
  lockWaiting: boolean
  current: components['schemas']['JobProgressPoolCurrent'][]
}

interface TranslationSessionState {
  activeJobId: Ref<string | null>
  lastHandledEventId: number
  progress: Ref<TranslationProgress>
}

const translationSessions = new WeakMap<
  ReturnType<typeof useImageStore>,
  TranslationSessionState
>()

function initialProgress(): TranslationProgress {
  return {
    current: 0,
    total: 0,
    completed: 0,
    failed: 0,
    isInProgress: false,
    label: '',
    percentage: 0,
    executionMode: 'sequential',
    pools: [],
  }
}

function translationSession(
  imageStore: ReturnType<typeof useImageStore>,
): TranslationSessionState {
  const existing = translationSessions.get(imageStore)
  if (existing) return existing
  const created: TranslationSessionState = {
    activeJobId: ref(null),
    lastHandledEventId: 0,
    progress: ref(initialProgress()),
  }
  translationSessions.set(imageStore, created)
  return created
}

function range(start: number, end: number): number[] {
  return Array.from({ length: Math.max(0, end - start) }, (_, index) => start + index)
}

const JOB_STATUSES = new Set<V2JobStatus>([
  ...NONTERMINAL_JOB_STATUSES,
  'cancelled',
  'completed',
  'completed_with_errors',
  'failed',
])

const CHAPTER_CONTENT_JOB_KINDS = new Set<V2Job['kind']>([
  'translation',
  'remove_text',
  'detect',
  'style_apply',
  'text_import',
  'container_import',
  'web_import_commit',
])

function jobStatusLabel(status: V2JobStatus | undefined): string {
  switch (status) {
    case 'queued': return '任务正在后端队列中等待'
    case 'running': return '后端正在处理'
    case 'pausing': return '正在等待当前步骤结束后暂停'
    case 'paused': return '任务已暂停'
    case 'cancelling': return '正在等待当前步骤结束后取消'
    case 'interrupted': return 'Worker 中断，请在任务中心继续'
    case 'completed_with_errors': return '任务完成，但有页面失败'
    case 'completed': return '后端任务已完成'
    case 'cancelled': return '后端任务已取消'
    case 'failed': return '后端任务失败'
    default: return '后端正在处理'
  }
}

function recordValue(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null
}

function nonNegativeInteger(value: unknown): value is number {
  return Number.isInteger(value) && Number(value) >= 0
}

function positiveInteger(value: unknown): value is number {
  return Number.isInteger(value) && Number(value) >= 1
}

function progressPoolCurrent(value: unknown): V2JobProgressPoolCurrent | null {
  const current = recordValue(value)
  if (
    !current
    || typeof current.itemId !== 'string'
    || (current.pageId !== null && typeof current.pageId !== 'string')
    || !positiveInteger(current.itemOrdinal)
    || typeof current.stepId !== 'string'
    || !positiveInteger(current.stepOrdinal)
  ) return null
  return {
    itemId: current.itemId,
    pageId: current.pageId,
    itemOrdinal: current.itemOrdinal,
    stepId: current.stepId,
    stepOrdinal: current.stepOrdinal,
  }
}

function progressPool(value: unknown): V2JobProgressPool | null {
  const pool = recordValue(value)
  if (
    !pool
    || typeof pool.kind !== 'string'
    || !pool.kind
    || !nonNegativeInteger(pool.total)
    || !nonNegativeInteger(pool.completed)
    || !nonNegativeInteger(pool.failed)
    || !nonNegativeInteger(pool.skipped)
    || !nonNegativeInteger(pool.cancelled)
    || !nonNegativeInteger(pool.waiting)
    || !nonNegativeInteger(pool.processing)
    || typeof pool.lockWaiting !== 'boolean'
    || !Array.isArray(pool.current)
  ) return null
  const current = pool.current.map(progressPoolCurrent)
  if (current.some(item => item === null)) return null
  return {
    kind: pool.kind,
    total: pool.total,
    completed: pool.completed,
    failed: pool.failed,
    skipped: pool.skipped,
    cancelled: pool.cancelled,
    waiting: pool.waiting,
    processing: pool.processing,
    lockWaiting: pool.lockWaiting,
    current: current as V2JobProgressPoolCurrent[],
  }
}

function progressCurrentStep(value: unknown): TranslationCurrentStep | null {
  const current = recordValue(value)
  const shared = progressPoolCurrent(value)
  if (!current || !shared || typeof current.kind !== 'string' || !current.kind) return null
  return { kind: current.kind, ...shared }
}

function parseJobProgress(value: unknown): V2JobProgress | null {
  const snapshot = recordValue(value)
  if (
    !snapshot
    || (snapshot.executionMode !== 'sequential' && snapshot.executionMode !== 'parallel')
    || typeof snapshot.jobStatus !== 'string'
    || !JOB_STATUSES.has(snapshot.jobStatus as V2JobStatus)
    || !nonNegativeInteger(snapshot.totalItems)
    || !nonNegativeInteger(snapshot.completedItems)
    || !nonNegativeInteger(snapshot.failedItems)
    || !nonNegativeInteger(snapshot.skippedItems)
    || !nonNegativeInteger(snapshot.cancelledItems)
    || !Array.isArray(snapshot.pools)
  ) return null
  const pools = snapshot.pools.map(progressPool)
  if (pools.some(pool => pool === null)) return null
  const currentStep = snapshot.currentStep === undefined
    ? undefined
    : progressCurrentStep(snapshot.currentStep)
  if (snapshot.currentStep !== undefined && !currentStep) return null
  return {
    executionMode: snapshot.executionMode,
    jobStatus: snapshot.jobStatus as V2JobStatus,
    totalItems: snapshot.totalItems,
    completedItems: snapshot.completedItems,
    failedItems: snapshot.failedItems,
    skippedItems: snapshot.skippedItems,
    cancelledItems: snapshot.cancelledItems,
    pools: pools as V2JobProgressPool[],
    ...(currentStep ? { currentStep } : {}),
  }
}

function applyProgressSnapshot(
  progress: Ref<TranslationProgress>,
  snapshot: V2JobProgress,
  label?: string,
  metadata: {
    queuePosition?: number | null
    status?: V2JobStatus
  } = {},
): void {
  const total = snapshot.totalItems
  const completed = snapshot.completedItems
  const failed = snapshot.failedItems
  const skipped = snapshot.skippedItems
  const cancelled = snapshot.cancelledItems
  const current = completed + failed + skipped + cancelled
  const status = metadata.status ?? snapshot.jobStatus
  progress.value = {
    current,
    total,
    completed,
    failed,
    isInProgress: status ? NONTERMINAL_JOB_STATUSES.has(status) : true,
    label: label ?? jobStatusLabel(status),
    percentage: total > 0 ? current / total * 100 : 0,
    executionMode: snapshot.executionMode,
    status,
    queuePosition: metadata.queuePosition,
    currentStep: snapshot.currentStep,
    pools: snapshot.pools.map(pool => ({
      ...pool,
      current: pool.current.map(currentItem => ({ ...currentItem })),
    })),
  }
}

type TranslationBootstrapJob = V2TranslationBootstrap['activeJobs'][number]

function activeTranslationJob(
  jobs: TranslationBootstrapJob[],
): TranslationBootstrapJob | undefined {
  const priority: Record<V2JobStatus, number> = {
    running: 0,
    pausing: 1,
    cancelling: 2,
    paused: 3,
    interrupted: 4,
    queued: 5,
    completed_with_errors: 6,
    completed: 7,
    failed: 8,
    cancelled: 9,
  }
  return jobs
    .filter(job => (
      (job.kind === 'translation' || job.kind === 'remove_text')
      && NONTERMINAL_JOB_STATUSES.has(job.status)
    ))
    .sort((left, right) => (
      priority[left.status] - priority[right.status]
      || (left.queueRank ?? Number.MAX_SAFE_INTEGER)
        - (right.queueRank ?? Number.MAX_SAFE_INTEGER)
    ))[0]
}

export function restoreTranslationFromBootstrap(
  jobs: TranslationBootstrapJob[],
  imageStore: ReturnType<typeof useImageStore>,
): void {
  const { activeJobId, progress } = translationSession(imageStore)
  const job = activeTranslationJob(jobs)
  if (!job) {
    activeJobId.value = null
    imageStore.setTranslationInProgress(false)
    progress.value = initialProgress()
    return
  }
  activeJobId.value = job.id
  applyProgressSnapshot(
    progress,
    job.progress,
    jobStatusLabel(job.status),
    { queuePosition: job.queueRank, status: job.status },
  )
  imageStore.setTranslationInProgress(true)
  const pageStates = new Map(job.pages.map(page => [page.pageId, page.status]))
  imageStore.images.forEach((image, index) => {
    const status = pageStates.get(image.id)
    if (status === 'pending' || status === 'running') {
      imageStore.setTranslationStatus(index, 'processing')
    } else if (status === 'failed') {
      imageStore.setTranslationStatus(index, 'failed')
    }
  })
}

async function refreshOpenPageDocument(
  pageId: string,
  imageStore: ReturnType<typeof useImageStore>,
  bubbleStore: ReturnType<typeof useBubbleStore>,
  settingsStore: ReturnType<typeof useSettingsStore>,
): Promise<void> {
  const requested = imageStore.currentImage
  if (requested?.id !== pageId) return
  const document = await getPageDocument(pageId)
  if (imageStore.currentImage?.id !== pageId) return
  if (document.pageId !== pageId || document.chapterId !== requested.chapterId) {
    throw new Error(`页面 ${pageId} 的后端文档身份不匹配`)
  }
  const bubbles = registerPageDocument(document)
  const pageTextStyle = parseCompleteTextStyleSettings({
    ...document.pageStyleDefaults,
    ...(document.defaultFontId
      ? { fontFamily: document.defaultFontId }
      : {}),
  })
  imageStore.updateCurrentImage({
    ...pageTextStyle,
    bubbleStates: bubbles,
    documentRevision: document.documentRevision,
    hasUnsavedChanges: false,
  })
  settingsStore.updateTextStyle(pageTextStyle)
  bubbleStore.setBubbles(bubbles, true)
  bubbleStore.saveAsInitial()
}

async function refreshCompletedPage(
  pageId: string,
  imageStore: ReturnType<typeof useImageStore>,
  bubbleStore: ReturnType<typeof useBubbleStore>,
  settingsStore: ReturnType<typeof useSettingsStore>,
): Promise<void> {
  const requested = imageStore.images.find(image => image.id === pageId)
  if (!requested) return
  const summary = await getPageSummary(pageId)
  if (summary.id !== pageId || summary.chapterId !== requested.chapterId) {
    throw new Error(`页面 ${pageId} 的后端摘要身份不匹配`)
  }
  const pageIndex = imageStore.images.findIndex(image => image.id === pageId)
  const existing = imageStore.images[pageIndex]
  if (pageIndex < 0 || !existing) return
  const mapped = pageSummaryToImage(summary)
  imageStore.updateImageByIndex(pageIndex, {
    ...mapped,
    bubbleStates: existing.bubbleStates,
  })
  await refreshOpenPageDocument(pageId, imageStore, bubbleStore, settingsStore)
}

async function refreshCurrentChapter(
  imageStore: ReturnType<typeof useImageStore>,
  bubbleStore: ReturnType<typeof useBubbleStore>,
  settingsStore: ReturnType<typeof useSettingsStore>,
): Promise<void> {
  const chapterId = imageStore.currentImage?.chapterId || imageStore.images[0]?.chapterId
  if (!chapterId) return
  const pageIdBeforeRequest = imageStore.currentImage?.id
  if (pageIdBeforeRequest) await flushPageDocument(pageIdBeforeRequest)
  const result = await listChapterPages(chapterId, { all: true })
  if (result.nextCursor !== null) {
    throw new Error(`章节 ${chapterId} 的全量页面响应不完整`)
  }
  if (result.items.some(page => page.chapterId !== chapterId)) {
    throw new Error(`章节 ${chapterId} 的页面响应包含其他章节数据`)
  }
  const activeChapterId = imageStore.currentImage?.chapterId
    ?? imageStore.images[0]?.chapterId
  if (activeChapterId !== chapterId) return
  const selectedPageId = imageStore.currentImage?.id
  const existingImages = new Map(imageStore.images.map(image => [image.id, image]))
  imageStore.setImages(result.items.map((summary) => {
    const mapped = pageSummaryToImage(summary)
    const existing = existingImages.get(mapped.id)
    if (!existing) return mapped
    return {
      ...existing,
      ...mapped,
      bubbleStates: existing.bubbleStates,
    }
  }))
  if (selectedPageId) {
    const currentIndex = imageStore.images.findIndex(image => image.id === selectedPageId)
    if (currentIndex >= 0) imageStore.setCurrentImageIndex(currentIndex)
  }
  const currentPageId = imageStore.currentImage?.id
  if (!currentPageId) return
  await refreshOpenPageDocument(
    currentPageId,
    imageStore,
    bubbleStore,
    settingsStore,
  )
}

export interface TranslationPipelineOptions {
  beforeCreateJob?: () => Promise<boolean>
  observeProgress?: boolean
}

export function useTranslation(options: TranslationPipelineOptions = {}) {
  const imageStore = useImageStore()
  const bubbleStore = useBubbleStore()
  const settingsStore = useSettingsStore()
  const taskCenterStore = useTaskCenterStore()
  const publicAccess = usePublicUserAccess()
  const toast = useToast()
  const session = translationSession(imageStore)
  const { activeJobId, progress } = session

  const progressPercent = computed(() => progress.value.percentage || 0)

  if (options.observeProgress !== false) watch(
    () => taskCenterStore.latestEvent,
    event => {
      if (!event || event.eventId <= session.lastHandledEventId) return
      session.lastHandledEventId = event.eventId
      const eventProgress = parseJobProgress(event.payload.progress)
      if (
        event.jobId === activeJobId.value
        && eventProgress
      ) {
        applyProgressSnapshot(
          progress,
          eventProgress,
          event.type === 'page_completed' ? '后端正在处理后续页面' : '后端正在处理',
        )
      }
      if (event.type === 'page_completed') {
        const pageId = event.payload.pageId
        if (typeof pageId === 'string') {
          void refreshCompletedPage(
            pageId,
            imageStore,
            bubbleStore,
            settingsStore,
          ).catch((error) => {
            toast.error(
              `刷新已完成页面失败：${error instanceof Error ? error.message : '未知错误'}`,
            )
          })
        }
      }
      if (!['job_finished', 'job_failed', 'job_cancelled'].includes(event.type)) return

      const trackedJobFinished = event.jobId === activeJobId.value
      const eventJob = [...taskCenterStore.queue, ...taskCenterStore.history]
        .find(job => job.jobId === event.jobId)
      const currentChapterId = imageStore.currentImage?.chapterId
        ?? imageStore.images[0]?.chapterId
      const openChapterChanged = Boolean(
        eventJob
        && currentChapterId
        && eventJob.chapterId === currentChapterId
        && CHAPTER_CONTENT_JOB_KINDS.has(eventJob.kind),
      )
      if (trackedJobFinished || openChapterChanged) {
        void refreshCurrentChapter(imageStore, bubbleStore, settingsStore)
          .catch((error) => {
            toast.error(
              `刷新后端翻译结果失败：${error instanceof Error ? error.message : '未知错误'}`,
            )
          })
          .finally(() => {
            if (trackedJobFinished) imageStore.setTranslationInProgress(false)
          })
      }
      if (!trackedJobFinished) return

      const terminalStatus: V2JobStatus = eventJob?.status
        ?? (event.type === 'job_finished'
          ? 'completed'
          : event.type === 'job_cancelled' ? 'cancelled' : 'failed')
      progress.value = {
        ...progress.value,
        isInProgress: false,
        label: jobStatusLabel(terminalStatus),
        status: terminalStatus,
      }
      activeJobId.value = null
    },
  )

  if (options.observeProgress !== false) watch(
    () => [taskCenterStore.queue, taskCenterStore.history] as const,
    ([queue, history]) => {
      const jobId = activeJobId.value
      if (!jobId) return
      const job = [...queue, ...history].find(item => item.jobId === jobId)
      if (!job) return
      applyProgressSnapshot(
        progress,
        job.progress,
        jobStatusLabel(job.status),
        { queuePosition: job.queueRank, status: job.status },
      )
      if (NONTERMINAL_JOB_STATUSES.has(job.status)) return
      activeJobId.value = null
      void refreshCurrentChapter(imageStore, bubbleStore, settingsStore)
        .catch((error) => {
          toast.error(
            `刷新后端翻译结果失败：${error instanceof Error ? error.message : '未知错误'}`,
          )
        })
        .finally(() => imageStore.setTranslationInProgress(false))
    },
    { deep: true },
  )

  async function prepareJobCreation(
    pageIds: string[],
    styleSourcePageId?: string,
  ): Promise<void> {
    if (
      options.beforeCreateJob
      && !(await options.beforeCreateJob())
    ) {
      throw new Error('章节工作态设置写入后端失败，未创建任务')
    }
    const pagesToFlush = new Set(pageIds)
    if (styleSourcePageId) pagesToFlush.add(styleSourcePageId)
    for (const pageId of pagesToFlush) {
      if (!hasPendingPageDocument(pageId)) continue
      await flushPageDocument(pageId)
    }
  }

  async function translatePages(
    pageIndexes: number[],
    mode: TranslationMode,
    pageOptions: { reuseExistingBubbles?: boolean } = {},
  ): Promise<boolean> {
    const uniqueIndexes = [...new Set(pageIndexes)]
    if (uniqueIndexes.length === 0) {
      toast.error('没有指定要处理的页面')
      return false
    }
    const pages = uniqueIndexes.map(index => imageStore.images[index])
    if (pages.some(page => !page)) {
      toast.error('指定页码无效')
      return false
    }
    const chapterId = pages[0]?.chapterId
    if (!chapterId || pages.some(page => page?.chapterId !== chapterId)) {
      toast.error('当前页面尚未写入后端章节')
      return false
    }
    if (imageStore.isTranslationInProgress) {
      toast.info('已有翻译任务正在创建或执行')
      return false
    }

    const pageIds = pages.map(page => page!.id)
    imageStore.setTranslationInProgress(true)
    try {
      const styleSourcePageId = imageStore.currentImage?.id
      if (!styleSourcePageId) {
        throw new Error('没有可用的当前页文字样式，未创建任务')
      }
      await prepareJobCreation(pageIds, styleSourcePageId)
      const committedStyleSource = imageStore.images.find(
        page => page.id === styleSourcePageId,
      )
      if (
        committedStyleSource?.chapterId !== chapterId
        || !Number.isInteger(committedStyleSource.documentRevision)
        || Number(committedStyleSource.documentRevision) < 1
      ) {
        throw new Error('当前页文字样式尚未写入后端，未创建任务')
      }
      const styleSource = {
        pageId: committedStyleSource.id,
        documentRevision: Number(committedStyleSource.documentRevision),
      }
      const executionMode = settingsStore.settings.parallel.enabled
        && publicAccess.parallelAllowed()
        ? 'parallel'
        : 'sequential'
      const batch = mode === 'removeText'
        ? await createChapterRemoveTextJob(
            chapterId,
            pageIds,
            executionMode,
            styleSource,
          )
        : await createChapterTranslationJob(chapterId, pageIds, {
            executionMode,
            mode,
            styleSourcePageId: styleSource.pageId,
            styleSourceDocumentRevision: styleSource.documentRevision,
            ...(pageOptions.reuseExistingBubbles === undefined
              ? {}
              : { reuseExistingBubbles: pageOptions.reuseExistingBubbles }),
          })
      const jobId = batch.jobIds[0]
      if (!jobId) throw new Error('后端没有返回任务')
      activeJobId.value = jobId
      progress.value = {
        current: 0,
        total: pageIds.length,
        completed: 0,
        failed: 0,
        isInProgress: true,
        label: '任务已进入后端队列',
        percentage: 0,
        executionMode,
        status: 'queued',
        pools: [],
      }
      for (const pageId of pageIds) {
        const index = imageStore.images.findIndex(image => image.id === pageId)
        if (index >= 0) imageStore.setTranslationStatus(index, 'processing')
      }
      void taskCenterStore.refresh().catch(() => undefined)
      toast.success('任务已加入后端任务中心，可安全关闭页面')
      return true
    } catch (error) {
      const message = error instanceof Error ? error.message : '创建后端任务失败'
      imageStore.setTranslationInProgress(false)
      toast.error(message)
      return false
    }
  }

  async function translateCurrentImage(): Promise<boolean> {
    return translatePages([imageStore.currentImageIndex], 'standard')
  }

  async function translateAllImages(): Promise<boolean> {
    return translatePages(range(0, imageStore.images.length), 'standard')
  }

  async function translateSelectedImages(selection: PageSelection): Promise<boolean> {
    return translatePages(pageSelectionToPageIndexes(selection.pages), 'standard')
  }

  async function removeTextOnly(): Promise<boolean> {
    return translatePages([imageStore.currentImageIndex], 'removeText')
  }

  async function removeAllTexts(): Promise<boolean> {
    return translatePages(range(0, imageStore.images.length), 'removeText')
  }

  async function removeTextSelection(selection: PageSelection): Promise<boolean> {
    return translatePages(pageSelectionToPageIndexes(selection.pages), 'removeText')
  }

  async function retryFailedImages(): Promise<boolean> {
    const chapterId = imageStore.currentImage?.chapterId || imageStore.images[0]?.chapterId
    if (!chapterId) {
      toast.error('当前页面尚未写入后端章节')
      return false
    }
    if (imageStore.isTranslationInProgress) {
      toast.info('已有翻译任务正在创建或执行')
      return false
    }
    imageStore.setTranslationInProgress(true)
    try {
      await prepareJobCreation(imageStore.images.map(image => image.id))
      const executionMode = settingsStore.settings.parallel.enabled
        && publicAccess.parallelAllowed()
        ? 'parallel'
        : 'sequential'
      const accepted = await taskCenterStore.retryLatestFailed(
        chapterId,
        ['translation'],
        'current',
      )
      if (!accepted) {
        imageStore.setTranslationInProgress(false)
        toast.info('后端没有找到当前章节可重试的部分失败翻译任务')
        return true
      }
      const jobId = accepted.jobIds[0]
      if (!jobId) throw new Error('后端没有返回重试任务')
      const durableFailedPages = imageStore.images
        .filter(image => image.translationStatus === 'failed')
        .map(image => image.id)
      activeJobId.value = jobId
      progress.value = {
        current: 0,
        total: durableFailedPages.length,
        completed: 0,
        failed: 0,
        isInProgress: true,
        label: '失败项重试已进入后端队列',
        percentage: 0,
        executionMode,
        status: 'queued',
        pools: [],
      }
      void taskCenterStore.refresh().catch(() => undefined)
      toast.success('失败项已按当前设置加入后端任务中心')
      return true
    } catch (error) {
      imageStore.setTranslationInProgress(false)
      toast.error(error instanceof Error ? error.message : '创建失败项重试任务失败')
      return false
    }
  }

  async function executeHqTranslation(selection?: PageSelection): Promise<boolean> {
    const indexes = selection
      ? pageSelectionToPageIndexes(selection.pages)
      : range(0, imageStore.images.length)
    return translatePages(indexes, 'hq')
  }

  async function executeProofreading(selection?: PageSelection): Promise<boolean> {
    const indexes = selection
      ? pageSelectionToPageIndexes(selection.pages)
      : range(0, imageStore.images.length)
    return translatePages(indexes, 'proofread')
  }

  async function translateWithCurrentBubbles(): Promise<boolean> {
    if (!imageStore.currentImage || bubbleStore.bubbles.length === 0) {
      toast.error('当前图片没有气泡框，请先检测或手动添加')
      return false
    }
    return translatePages(
      [imageStore.currentImageIndex],
      'standard',
      { reuseExistingBubbles: true },
    )
  }

  return {
    progress,
    progressPercent,
    translateCurrentImage,
    translateAllImages,
    translateSelectedImages,
    removeTextOnly,
    removeAllTexts,
    removeTextSelection,
    retryFailedImages,
    executeHqTranslation,
    executeProofreading,
    translateWithCurrentBubbles,
  }
}
