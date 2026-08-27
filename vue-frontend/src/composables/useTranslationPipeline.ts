import { computed, getCurrentScope, onScopeDispose, ref, watch, type Ref } from 'vue'

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
    progress: ref(initialProgress()),
  }
  translationSessions.set(imageStore, created)
  return created
}

function range(start: number, end: number): number[] {
  return Array.from({ length: Math.max(0, end - start) }, (_, index) => start + index)
}

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
    case 'paused': return '任务已暂停；当前步骤将在恢复后重做'
    case 'interrupted': return 'Worker 中断，请在任务中心继续'
    case 'completed_with_errors': return '任务完成，但有页面失败'
    case 'completed': return '后端任务已完成'
    case 'cancelled': return '后端任务已取消'
    case 'failed': return '后端任务失败'
    default: return '后端正在处理'
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
    paused: 1,
    interrupted: 2,
    queued: 3,
    completed_with_errors: 4,
    completed: 5,
    failed: 6,
    cancelled: 7,
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

  if (options.observeProgress !== false) {
    const unsubscribe = taskCenterStore.subscribeEvents((event) => {
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
    })
    if (getCurrentScope()) onScopeDispose(unsubscribe)
  }

  let observedChapterId: string | null = null
  let observedTerminalContentJobs = new Set<string>()

  if (options.observeProgress !== false) watch(
    () => [
      taskCenterStore.queue,
      taskCenterStore.history,
      imageStore.currentImage?.chapterId ?? imageStore.images[0]?.chapterId ?? null,
    ] as const,
    ([queue, history, currentChapterId]) => {
      const jobs = [...queue, ...history]
      let shouldRefreshChapter = false
      const jobId = activeJobId.value
      const trackedJob = jobId
        ? jobs.find(item => item.jobId === jobId)
        : undefined
      if (trackedJob) {
        applyProgressSnapshot(
          progress,
          trackedJob.progress,
          jobStatusLabel(trackedJob.status),
          { queuePosition: trackedJob.queueRank, status: trackedJob.status },
        )
        if (!NONTERMINAL_JOB_STATUSES.has(trackedJob.status)) {
          activeJobId.value = null
          imageStore.setTranslationInProgress(false)
          if (trackedJob.chapterId === currentChapterId) {
            shouldRefreshChapter = true
          }
        }
      }

      const currentTerminalContentJobs = new Set<string>()
      if (currentChapterId) {
        for (const job of jobs) {
          if (
            job.chapterId !== currentChapterId
            || !CHAPTER_CONTENT_JOB_KINDS.has(job.kind)
          ) continue
          if (NONTERMINAL_JOB_STATUSES.has(job.status)) continue
          currentTerminalContentJobs.add(job.jobId)
          if (
            observedChapterId === currentChapterId
            && !observedTerminalContentJobs.has(job.jobId)
          ) {
            shouldRefreshChapter = true
          }
        }
      }
      observedChapterId = currentChapterId
      observedTerminalContentJobs = currentTerminalContentJobs

      if (!shouldRefreshChapter) return
      void refreshCurrentChapter(imageStore, bubbleStore, settingsStore)
        .catch((error) => {
          toast.error(
            `刷新后端翻译结果失败：${error instanceof Error ? error.message : '未知错误'}`,
          )
        })
    },
    { deep: true, immediate: true },
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
      taskCenterStore.trackJob(jobId)
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
      taskCenterStore.trackJob(jobId)
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
