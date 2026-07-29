import { computed, ref, watch } from 'vue'

import { createChapterTranslationJob } from '@/api/v2/translation'
import { listChapterPages } from '@/api/v2/content'
import { pageSummaryToImage } from '@/adapters/v2ContentAdapter'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import { hasPendingPageDocument } from '@/services/pageDocumentPersistence'
import { useToast } from '@/utils/toast'
import { pageSelectionToPageIndexes } from '@/utils/pageSelection'
export type TranslationMode = 'standard' | 'hq' | 'proofread' | 'removeText'

export interface PageSelection {
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
}

export interface TranslateResult {
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
})
const activeJobId = ref<string | null>(null)
const activePageIds = ref<string[]>([])
let lastHandledEventId = 0

export function range(start: number, end: number): number[] {
  return Array.from({ length: Math.max(0, end - start) }, (_, index) => start + index)
}

function numberField(value: unknown): number {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : 0
}

function applyProgressSnapshot(snapshot: Record<string, unknown>, label: string): void {
  const total = numberField(snapshot.totalItems)
  const completed = numberField(snapshot.completedItems)
  const failed = numberField(snapshot.failedItems)
  const current = completed + failed
  progress.value = {
    current,
    total,
    completed,
    failed,
    isInProgress: true,
    label,
    percentage: total > 0 ? Math.round(current / total * 100) : 0,
  }
}

async function refreshCurrentChapter(imageStore: ReturnType<typeof useImageStore>): Promise<void> {
  const chapterId = imageStore.currentImage?.chapterId || imageStore.images[0]?.chapterId
  if (!chapterId) return
  const result = await listChapterPages(chapterId, { all: true })
  const summaries = new Map(result.items.map(page => [page.id, page]))
  for (const [index, image] of imageStore.images.entries()) {
    const summary = summaries.get(image.id)
    if (!summary) continue
    const mapped = pageSummaryToImage(summary)
    imageStore.updateImageByIndex(index, {
      chapterId: mapped.chapterId,
      cleanAssetUrl: mapped.cleanAssetUrl,
      documentRevision: mapped.documentRevision,
      height: mapped.height,
      renderedRevision: mapped.renderedRevision,
      sourceAssetUrl: mapped.sourceAssetUrl,
      sourceRevision: mapped.sourceRevision,
      thumbnailSourceUrl: mapped.thumbnailSourceUrl,
      thumbnailTranslatedUrl: mapped.thumbnailTranslatedUrl,
      translatedAssetUrl: mapped.translatedAssetUrl,
      translationFailed: mapped.translationFailed,
      translationStatus: mapped.translationStatus,
      width: mapped.width,
    })
  }
}

export function useTranslation() {
  const imageStore = useImageStore()
  const bubbleStore = useBubbleStore()
  const settingsStore = useSettingsStore()
  const taskCenterStore = useTaskCenterStore()
  const toast = useToast()

  const isTranslating = computed(() => progress.value.isInProgress)
  const isTranslatingSingle = computed(
    () => progress.value.isInProgress && activePageIds.value.length === 1,
  )
  const isHqTranslating = computed(() => progress.value.isInProgress)
  const isProofreading = computed(() => progress.value.isInProgress)
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
      if (!['job_finished', 'job_failed', 'job_cancelled'].includes(event.type)) return

      // Any terminal task may have changed the open chapter. This also covers a task
      // that survived a browser restart and therefore has no local activeJobId.
      void refreshCurrentChapter(imageStore)
      if (event.jobId !== activeJobId.value) return

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

  async function translatePages(
    pageIndexes: number[],
    mode: TranslationMode,
    options: { reuseExistingBubbles?: boolean } = {},
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
    if (pageIds.some(hasPendingPageDocument)) {
      toast.error('页面编辑正在写入后端或写入失败，请稍后重试')
      return {
        success: false,
        completed: 0,
        failed: 0,
        errors: ['页面文档尚未完成后端写入'],
      }
    }
    try {
      const batch = await createChapterTranslationJob(chapterId, pageIds, {
        executionMode: settingsStore.settings.parallel.enabled ? 'parallel' : 'sequential',
        mode: mode === 'removeText' ? 'remove_text' : mode,
        ...(options.reuseExistingBubbles === undefined
          ? {}
          : { reuseExistingBubbles: options.reuseExistingBubbles }),
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

  async function translateImageByIndex(index: number): Promise<boolean> {
    return (await translatePages([index], 'standard')).success
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

  function cancelBatchTranslation(): void {
    if (activeJobId.value) void taskCenterStore.cancel(activeJobId.value)
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
    isTranslatingSingle,
    isHqTranslating,
    isProofreading,
    isTranslating,
    progressPercent,
    translatePages,
    range,
    translateCurrentImage,
    translateImageByIndex,
    translateAllImages,
    translateSelectedImages,
    cancelBatchTranslation,
    removeTextOnly,
    removeAllTexts,
    removeTextSelection,
    retryFailedImages,
    executeHqTranslation,
    executeProofreading,
    translateWithCurrentBubbles,
  }
}
