import { getCurrentInstance, onUnmounted, ref, type Ref } from 'vue'
import { getPageDocument } from '@/api/v2/content'
import {
  runPageOperation,
  type PageOperationKind,
  type V2Operation,
} from '@/api/v2/operations'
import { createChapterDetectJob } from '@/api/v2/translation'
import { useTranslation } from '@/composables/useTranslationPipeline'
import { confirmProductAction } from '@/composables/useProductConfirm'
import {
  queuePageDocumentSave,
  registerPageDocument,
} from '@/services/pageDocumentPersistence'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import type { BubbleState } from '@/types/bubble'
import type { ImageData } from '@/types/image'
import { showToast } from '@/utils/toast'

interface UseEditWorkspaceProcessingActionsOptions {
  images: Ref<ImageData[]>
  currentImage: Ref<ImageData | null | undefined>
  currentImageIndex: Ref<number>
  bubbles: Ref<BubbleState[]>
  selectFirstBubbleIfExists: () => void
}

export function useEditWorkspaceProcessingActions(
  options: UseEditWorkspaceProcessingActionsOptions,
) {
  const bubbleStore = useBubbleStore()
  const imageStore = useImageStore()
  const { translateWithCurrentBubbles: translateWithBubbles } = useTranslation()

  const isProcessing = ref(false)
  const progressText = ref('处理中...')
  const progressCurrent = ref(0)
  const progressTotal = ref(0)
  const isTranslateLoading = ref(false)
  const abortController = new AbortController()
  let completionResetTimer: ReturnType<typeof setTimeout> | null = null

  function clearCompletionResetTimer(): void {
    if (completionResetTimer) {
      clearTimeout(completionResetTimer)
      completionResetTimer = null
    }
  }

  function scheduleCompletionReset(): void {
    clearCompletionResetTimer()
    completionResetTimer = setTimeout(() => {
      completionResetTimer = null
      isProcessing.value = false
    }, 2000)
  }

  if (getCurrentInstance()) {
    onUnmounted(() => {
      abortController.abort()
      clearCompletionResetTimer()
      isProcessing.value = false
    })
  }

  async function persistCurrentDocument(): Promise<ImageData> {
    const image = options.currentImage.value
    if (!image || image.documentRevision === undefined) {
      throw new Error('当前页面尚未完成后端初始化')
    }
    await queuePageDocumentSave(
      image.id,
      image.documentRevision,
      options.bubbles.value,
    )
    const current = options.currentImage.value
    if (!current || current.id !== image.id || current.documentRevision === undefined) {
      throw new Error('当前页面已切换')
    }
    return current
  }

  async function reloadCurrentDocument(
    pageId: string,
    preferredBubbleId?: string,
  ): Promise<void> {
    const document = await getPageDocument(pageId, abortController.signal)
    if (options.currentImage.value?.id !== pageId) return
    const bubbles = registerPageDocument(document)
    imageStore.updateCurrentImage({
      bubbleStates: bubbles,
      documentRevision: document.documentRevision,
      hasUnsavedChanges: false,
    })
    bubbleStore.setBubbles(bubbles, true)
    const preferredIndex = preferredBubbleId
      ? bubbles.findIndex(bubble => bubble.backendBubbleId === preferredBubbleId)
      : -1
    if (preferredIndex >= 0) {
      bubbleStore.selectBubble(preferredIndex)
    } else {
      options.selectFirstBubbleIfExists()
    }
  }

  async function executePageOperation(
    kind: PageOperationKind,
    bubble?: BubbleState,
  ): Promise<V2Operation> {
    const image = await persistCurrentDocument()
    const bubbleId = bubble?.backendBubbleId
    if (bubble && !bubbleId) {
      throw new Error('气泡尚未完成后端持久化')
    }
    const operation = await runPageOperation(
      image.id,
      {
        baseRevision: image.documentRevision!,
        bubbleId,
        kind,
      },
      { signal: abortController.signal },
    )
    await reloadCurrentDocument(image.id, bubbleId)
    return operation
  }

  async function handleReTranslateBubble(index: number): Promise<void> {
    const bubble = options.bubbles.value[index]
    if (!bubble?.originalText) {
      showToast('缺少气泡原文，无法重新翻译', 'warning')
      return
    }
    isTranslateLoading.value = true
    try {
      await executePageOperation('bubble_translate', bubble)
      showToast('重新翻译完成', 'success')
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') return
      showToast(error instanceof Error ? error.message : '重新翻译失败', 'error')
    } finally {
      isTranslateLoading.value = false
    }
  }

  async function autoDetectBubbles(): Promise<void> {
    if (!options.currentImage.value) {
      showToast('没有有效的图片用于检测', 'warning')
      return
    }
    try {
      showToast('检测任务已提交到后端...', 'info')
      const operation = await executePageOperation('page_detect')
      const count = Number(operation.result?.bubbleCount ?? 0)
      showToast(
        count > 0 ? `自动检测到 ${count} 个文本框` : '未检测到文本框',
        count > 0 ? 'success' : 'info',
      )
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') return
      showToast(error instanceof Error ? error.message : '自动检测失败', 'error')
    }
  }

  async function detectAllImages(): Promise<void> {
    if (options.images.value.length <= 1) {
      showToast('至少需要两张图片才能执行批量检测', 'warning')
      return
    }
    const confirmed = await confirmProductAction({
      title: '批量检测文本框',
      message: '此操作将对所有图片进行文本框检测，可能会覆盖已有的检测结果。确定继续吗？',
      confirmText: '加入任务队列',
      cancelText: '取消',
      tone: 'danger',
    })
    if (!confirmed) return

    const image = await persistCurrentDocument().catch(error => {
      showToast(error instanceof Error ? error.message : '保存当前页失败', 'error')
      return null
    })
    if (!image?.chapterId) {
      if (image) showToast('当前章节上下文不存在', 'error')
      return
    }

    isProcessing.value = true
    clearCompletionResetTimer()
    progressText.value = '已加入后端任务队列'
    progressCurrent.value = 0
    progressTotal.value = options.images.value.length
    try {
      await createChapterDetectJob(
        image.chapterId,
        options.images.value.map(item => item.id),
      )
      showToast('批量检测已加入任务中心；关闭浏览器也会继续执行', 'success')
      scheduleCompletionReset()
    } catch (error) {
      showToast(error instanceof Error ? error.message : '批量检测创建失败', 'error')
      isProcessing.value = false
    }
  }

  async function translateWithCurrentBubbles(): Promise<void> {
    if (!options.currentImage.value) {
      showToast('没有有效的图片用于翻译', 'warning')
      return
    }
    if (options.bubbles.value.length === 0) {
      showToast('没有文本框可用于翻译，请先检测或添加文本框', 'warning')
      return
    }
    try {
      await persistCurrentDocument()
      const success = await translateWithBubbles()
      if (success) {
        showToast('翻译任务已加入任务中心', 'success')
        options.selectFirstBubbleIfExists()
      }
    } catch (error) {
      showToast(
        `翻译失败: ${error instanceof Error ? error.message : '未知错误'}`,
        'error',
      )
    }
  }

  return {
    isProcessing,
    progressText,
    progressCurrent,
    progressTotal,
    isTranslateLoading,
    handleReTranslateBubble,
    autoDetectBubbles,
    detectAllImages,
    translateWithCurrentBubbles,
  }
}
