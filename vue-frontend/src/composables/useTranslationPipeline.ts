import { computed, ref, watch } from 'vue'

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
import type { V2Job, V2JobStatus } from '@/api/v2/jobs'
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

export interface TranslationPoolProgress {
  kind: string
  total: number
  completed: number
  failed: number
  skipped: number
  waiting: number
  processing: number
  lockWaiting: boolean
  current: components['schemas']['JobProgressPoolCurrent'][]
}

interface TranslateResult {
  success: boolean
  completed: number
  failed: number
  errors: string[]
}

const progress = ref<TranslationProgress>({
  current: 0,
  total: 0,
  completed: 0,
  failed: 0,
  isInProgress: false,
  label: '',
  percentage: 0,
  executionMode: 'sequential',
  pools: [],
})
const activeJobId = ref<string | null>(null)
const activePageIds = ref<string[]>([])
let lastHandledEventId = 0

function range(start: number, end: number): number[] {
  return Array.from({ length: Math.max(0, end - start) }, (_, index) => start + index)
}

function numberField(value: unknown): number {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : 0
}

const ACTIVE_JOB_STATUSES = new Set<V2JobStatus>([
  'queued',
  'running',
  'pausing',
  'paused',
  'cancelling',
  'interrupted',
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

function normalizePools(value: unknown): TranslationPoolProgress[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((raw) => {
    if (!raw || typeof raw !== 'object') return []
    const pool = raw as Record<string, unknown>
    const kind = typeof pool.kind === 'string' ? pool.kind : ''
    if (!kind) return []
    return [{
      kind,
      total: numberField(pool.total),
      completed: numberField(pool.completed),
      failed: numberField(pool.failed),
      skipped: numberField(pool.skipped),
      waiting: numberField(pool.waiting),
      processing: numberField(pool.processing),
      lockWaiting: Boolean(pool.lockWaiting),
      current: Array.isArray(pool.current)
        ? pool.current as components['schemas']['JobProgressPoolCurrent'][]
        : [],
    }]
  })
}

function applyProgressSnapshot(
  snapshot: Record<string, unknown>,
  label?: string,
  metadata: {
    queuePosition?: number | null
    status?: V2JobStatus
  } = {},
): void {
  const total = numberField(snapshot.totalItems)
  const completed = numberField(snapshot.completedItems)
  const failed = numberField(snapshot.failedItems)
  const skipped = numberField(snapshot.skippedItems)
  const cancelled = numberField(snapshot.cancelledItems)
  const current = completed + failed + skipped + cancelled
  const snapshotStatus = typeof snapshot.jobStatus === 'string'
    ? snapshot.jobStatus as V2JobStatus
    : undefined
  const status = metadata.status ?? snapshotStatus
  const executionMode = snapshot.executionMode === 'parallel'
    ? 'parallel'
    : 'sequential'
  const currentStep = (
    snapshot.currentStep
    && typeof snapshot.currentStep === 'object'
  )
    ? snapshot.currentStep as TranslationCurrentStep
    : undefined
  progress.value = {
    current,
    total,
    completed,
    failed,
    isInProgress: status ? ACTIVE_JOB_STATUSES.has(status) : true,
    label: label ?? jobStatusLabel(status),
    percentage: total > 0 ? Math.round(current / total * 100) : 0,
    executionMode,
    status,
    queuePosition: metadata.queuePosition,
    currentStep,
    pools: normalizePools(snapshot.pools),
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
      && ACTIVE_JOB_STATUSES.has(job.status)
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
  const job = activeTranslationJob(jobs)
  if (!job) {
    activeJobId.value = null
    activePageIds.value = []
    imageStore.setBatchTranslationInProgress(false)
    progress.value = {
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
    return
  }
  activeJobId.value = job.id
  activePageIds.value = [...job.pageIds]
  applyProgressSnapshot(
    job.progress,
    jobStatusLabel(job.status),
    { queuePosition: job.queueRank, status: job.status },
  )
  imageStore.setBatchTranslationInProgress(job.pageIds.length > 1)
  const targetPages = new Set(job.pageIds)
  imageStore.images.forEach((image, index) => {
    if (targetPages.has(image.id)) imageStore.setTranslationStatus(index, 'processing')
  })
}

async function refreshOpenPageDocument(
  pageId: string,
  imageStore: ReturnType<typeof useImageStore>,
  bubbleStore: ReturnType<typeof useBubbleStore>,
  settingsStore: ReturnType<typeof useSettingsStore>,
): Promise<void> {
  if (imageStore.currentImage?.id !== pageId) return
  const document = await getPageDocument(pageId)
  if (imageStore.currentImage?.id !== pageId) return
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
  if (!imageStore.images.some(image => image.id === pageId)) return
  const summary = await getPageSummary(pageId)
  const pageIndex = imageStore.images.findIndex(image => image.id === pageId)
  const existing = imageStore.images[pageIndex]
  if (pageIndex < 0 || !existing) return
  const mapped = pageSummaryToImage(summary)
  imageStore.updateImageByIndex(pageIndex, {
    ...mapped,
    bubbleStates: existing.bubbleStates,
    errorMessage: mapped.translationFailed ? existing.errorMessage : undefined,
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
  const currentPageId = imageStore.currentImage?.id
  if (currentPageId) await flushPageDocument(currentPageId)
  const result = await listChapterPages(chapterId, { all: true })
  const existingImages = new Map(imageStore.images.map(image => [image.id, image]))
  imageStore.setImages(result.items.map((summary) => {
    const mapped = pageSummaryToImage(summary)
    const existing = existingImages.get(mapped.id)
    if (!existing) return mapped
    return {
      ...existing,
      ...mapped,
      bubbleStates: existing.bubbleStates,
      errorMessage: mapped.translationFailed ? existing.errorMessage : undefined,
    }
  }))
  if (currentPageId) {
    const currentIndex = imageStore.images.findIndex(image => image.id === currentPageId)
    if (currentIndex >= 0) imageStore.setCurrentImageIndex(currentIndex)
  }
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
}

export function useTranslation(options: TranslationPipelineOptions = {}) {
  const imageStore = useImageStore()
  const bubbleStore = useBubbleStore()
  const settingsStore = useSettingsStore()
  const taskCenterStore = useTaskCenterStore()
  const toast = useToast()

  const progressPercent = computed(() => progress.value.percentage || 0)

  watch(
    () => taskCenterStore.latestEvent,
    event => {
      if (!event || event.eventId <= lastHandledEventId) return
      lastHandledEventId = event.eventId
      const eventProgress = event.payload.progress
      if (
        event.jobId === activeJobId.value
        && eventProgress
        && typeof eventProgress === 'object'
      ) {
        applyProgressSnapshot(
          eventProgress as Record<string, unknown>,
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
        void refreshCurrentChapter(imageStore, bubbleStore, settingsStore).catch((error) => {
          toast.error(
            `刷新后端翻译结果失败：${error instanceof Error ? error.message : '未知错误'}`,
          )
        })
      }
      if (!trackedJobFinished) return

      const succeeded = event.type === 'job_finished'
      progress.value = {
        ...progress.value,
        current: progress.value.total,
        isInProgress: false,
        label: succeeded ? '后端任务已完成' : '后端任务未完成',
        percentage: succeeded ? 100 : progress.value.percentage,
      }
      imageStore.setBatchTranslationInProgress(false)
      activeJobId.value = null
      activePageIds.value = []
    },
  )

  watch(
    () => [taskCenterStore.queue, taskCenterStore.history] as const,
    ([queue, history]) => {
      const jobId = activeJobId.value
      if (!jobId) return
      const job = [...queue, ...history].find(item => item.jobId === jobId)
      if (!job) return
      applyProgressSnapshot(
        job.progress,
        jobStatusLabel(job.status),
        { queuePosition: job.queueRank, status: job.status },
      )
      if (ACTIVE_JOB_STATUSES.has(job.status)) return
      imageStore.setBatchTranslationInProgress(false)
      activeJobId.value = null
      activePageIds.value = []
      void refreshCurrentChapter(imageStore, bubbleStore, settingsStore).catch((error) => {
        toast.error(
          `刷新后端翻译结果失败：${error instanceof Error ? error.message : '未知错误'}`,
        )
      })
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
  ): Promise<TranslateResult> {
    const uniqueIndexes = [...new Set(pageIndexes)]
    if (uniqueIndexes.length === 0) {
      toast.error('没有指定要处理的页面')
      return { success: false, completed: 0, failed: 0, errors: ['没有指定页面'] }
    }
    const pages = uniqueIndexes.map(index => imageStore.images[index])
    if (pages.some(page => !page)) {
      toast.error('指定页码无效')
      return { success: false, completed: 0, failed: 0, errors: ['指定页码无效'] }
    }
    const chapterId = pages[0]?.chapterId
    if (!chapterId || pages.some(page => page?.chapterId !== chapterId)) {
      toast.error('当前页面尚未写入后端章节')
      return { success: false, completed: 0, failed: 0, errors: ['页面不属于同一后端章节'] }
    }

    const pageIds = pages.map(page => page!.id)
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
      activePageIds.value = pageIds
      progress.value = {
        current: 0,
        total: pageIds.length,
        completed: 0,
        failed: 0,
        isInProgress: true,
        label: '任务已进入后端队列',
        percentage: 0,
        executionMode: settingsStore.settings.parallel.enabled ? 'parallel' : 'sequential',
        status: 'queued',
        pools: [],
      }
      imageStore.setBatchTranslationInProgress(pageIds.length > 1)
      for (const pageId of pageIds) {
        const index = imageStore.images.findIndex(image => image.id === pageId)
        if (index >= 0) imageStore.setTranslationStatus(index, 'processing')
      }
      await taskCenterStore.refresh()
      toast.success('任务已加入后端任务中心，可安全关闭页面')
      return { success: true, completed: 0, failed: 0, errors: [] }
    } catch (error) {
      const message = error instanceof Error ? error.message : '创建后端任务失败'
      toast.error(message)
      return { success: false, completed: 0, failed: 0, errors: [message] }
    }
  }

  async function translateCurrentImage(): Promise<boolean> {
    return (await translatePages([imageStore.currentImageIndex], 'standard')).success
  }

  async function translateAllImages(): Promise<boolean> {
    return (
      await translatePages(range(0, imageStore.images.length), 'standard')
    ).success
  }

  async function translateSelectedImages(selection: PageSelection): Promise<boolean> {
    return (
      await translatePages(pageSelectionToPageIndexes(selection.pages), 'standard')
    ).success
  }

  async function removeTextOnly(): Promise<boolean> {
    return (await translatePages([imageStore.currentImageIndex], 'removeText')).success
  }

  async function removeAllTexts(): Promise<boolean> {
    return (
      await translatePages(range(0, imageStore.images.length), 'removeText')
    ).success
  }

  async function removeTextSelection(selection: PageSelection): Promise<boolean> {
    return (
      await translatePages(pageSelectionToPageIndexes(selection.pages), 'removeText')
    ).success
  }

  async function retryFailedImages(): Promise<boolean> {
    const chapterId = imageStore.currentImage?.chapterId || imageStore.images[0]?.chapterId
    if (!chapterId) {
      toast.error('当前页面尚未写入后端章节')
      return false
    }
    try {
      await prepareJobCreation(imageStore.images.map(image => image.id))
      const accepted = await taskCenterStore.retryLatestFailed(
        chapterId,
        ['translation'],
        'current',
      )
      if (!accepted) {
        toast.info('后端没有找到当前章节可重试的部分失败翻译任务')
        return true
      }
      const jobId = accepted.jobIds[0]
      if (!jobId) throw new Error('后端没有返回重试任务')
      const durableFailedPages = imageStore.images
        .filter(image => image.translationFailed)
        .map(image => image.id)
      activeJobId.value = jobId
      activePageIds.value = durableFailedPages
      progress.value = {
        current: 0,
        total: durableFailedPages.length,
        completed: 0,
        failed: 0,
        isInProgress: true,
        label: '失败项重试已进入后端队列',
        percentage: 0,
        executionMode: settingsStore.settings.parallel.enabled ? 'parallel' : 'sequential',
        status: 'queued',
        pools: [],
      }
      imageStore.setBatchTranslationInProgress(durableFailedPages.length > 1)
      toast.success('失败项已按当前设置加入后端任务中心')
      return true
    } catch (error) {
      toast.error(error instanceof Error ? error.message : '创建失败项重试任务失败')
      return false
    }
  }

  async function executeHqTranslation(selection?: PageSelection): Promise<boolean> {
    const indexes = selection
      ? pageSelectionToPageIndexes(selection.pages)
      : range(0, imageStore.images.length)
    return (await translatePages(indexes, 'hq')).success
  }

  async function executeProofreading(selection?: PageSelection): Promise<boolean> {
    const indexes = selection
      ? pageSelectionToPageIndexes(selection.pages)
      : range(0, imageStore.images.length)
    return (await translatePages(indexes, 'proofread')).success
  }

  async function translateWithCurrentBubbles(): Promise<boolean> {
    if (!imageStore.currentImage || bubbleStore.bubbles.length === 0) {
      toast.error('当前图片没有气泡框，请先检测或手动添加')
      return false
    }
    return (
      await translatePages(
        [imageStore.currentImageIndex],
        'standard',
        { reuseExistingBubbles: true },
      )
    ).success
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
