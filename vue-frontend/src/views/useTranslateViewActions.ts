import type { Ref } from 'vue'
import { showToast } from '@/utils/toast'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useTaskCenterStore } from '@/stores/taskCenterStore'
import { useTranslation } from '@/composables/useTranslationPipeline'
import { useTranslateInit } from '@/composables/useTranslateInit'
import { confirmProductAction, type ProductConfirmAction } from '@/composables/useProductConfirm'
import type { WorkflowRunRequest } from '@/types/workflow'
import { clearChapterPages, deletePage, resetQuickWorkspace } from '@/api/v2/content'
import {
  discardPageDocument,
  flushPageDocument,
} from '@/services/pageDocumentPersistence'

type TranslateValidationMode = 'normal' | 'hq' | 'proofread'

interface UseTranslateViewActionsOptions {
  imageStore: ReturnType<typeof useImageStore>
  bubbleStore: ReturnType<typeof useBubbleStore>
  settingsStore: ReturnType<typeof useSettingsStore>
  taskCenterStore: ReturnType<typeof useTaskCenterStore>
  translation: ReturnType<typeof useTranslation>
  translateInit: ReturnType<typeof useTranslateInit>
  validateBeforeTranslation: (mode: TranslateValidationMode) => boolean
  isEditMode: Ref<boolean>
  canUseEditMode?: Readonly<Ref<boolean>>
  confirmAction?: ProductConfirmAction
}

export function useTranslateViewActions(options: UseTranslateViewActionsOptions) {
  const {
    imageStore,
    bubbleStore,
    settingsStore,
    taskCenterStore,
    translation,
    translateInit,
    validateBeforeTranslation,
    isEditMode,
    canUseEditMode = { value: true },
    confirmAction = confirmProductAction,
  } = options

  async function loadChapterSession(): Promise<boolean> {
    try {
      return await translateInit.initializeBookChapterContext()
    } catch (error) {
      showToast(
        `刷新后端章节失败：${error instanceof Error ? error.message : '未知错误'}`,
        'error',
      )
      return false
    }
  }

  async function handleUploadComplete() {
    await loadChapterSession()
  }

  async function translateCurrentImage() {
    if (!imageStore.currentImage) return
    if (!validateBeforeTranslation('normal')) return
    await translation.translateCurrentImage()
  }

  async function translateAllImages() {
    if (!imageStore.hasImages) return
    if (!validateBeforeTranslation('normal')) return
    await translation.translateAllImages()
  }

  async function startHqTranslation() {
    if (!imageStore.hasImages) return
    if (!validateBeforeTranslation('hq')) return
    await translation.executeHqTranslation()
  }

  async function startProofreading() {
    if (!imageStore.hasImages) return
    if (!validateBeforeTranslation('proofread')) return
    await translation.executeProofreading()
  }

  async function removeTextOnly() {
    if (!imageStore.currentImage) return
    await translation.removeTextOnly()
  }

  async function removeAllText() {
    if (!imageStore.hasImages) return
    await translation.removeAllTexts()
  }

  async function translateSelectedImages(pages: number[]) {
    if (!imageStore.hasImages) return
    if (!validateBeforeTranslation('normal')) return
    await translation.translateSelectedImages({ pages })
  }

  async function startHqTranslationSelection(pages: number[]) {
    if (!imageStore.hasImages) return
    if (!validateBeforeTranslation('hq')) return
    await translation.executeHqTranslation({ pages })
  }

  async function startProofreadingSelection(pages: number[]) {
    if (!imageStore.hasImages) return
    if (!validateBeforeTranslation('proofread')) return
    await translation.executeProofreading({ pages })
  }

  async function removeTextSelection(pages: number[]) {
    if (!imageStore.hasImages) return
    await translation.removeTextSelection({ pages })
  }

  async function handleRunWorkflow(payload: WorkflowRunRequest) {
    const selectedPages = payload.pageSelection?.pages

    switch (payload.mode) {
      case 'translate-current':
        await translateCurrentImage()
        return
      case 'translate-batch':
        if (selectedPages?.length) await translateSelectedImages(selectedPages)
        else await translateAllImages()
        return
      case 'hq-batch':
        if (selectedPages?.length) await startHqTranslationSelection(selectedPages)
        else await startHqTranslation()
        return
      case 'proofread-batch':
        if (selectedPages?.length) await startProofreadingSelection(selectedPages)
        else await startProofreading()
        return
      case 'remove-current':
        await removeTextOnly()
        return
      case 'remove-batch':
        if (selectedPages?.length) await removeTextSelection(selectedPages)
        else await removeAllText()
        return
      case 'retry-failed':
        await handleRetryFailed()
        return
      case 'delete-current':
        await deleteCurrentImage()
        return
      case 'clear-all':
        await clearAllImages()
        return
    }
    const exhaustive: never = payload.mode
    return exhaustive
  }

  async function deleteCurrentImage() {
    const target = imageStore.currentImage
    if (!target) return
    const pageId = target.id
    const fileName = target.fileName
    const confirmed = await confirmAction({
      title: '删除当前图片',
      message: `确定要删除当前图片 (${fileName}) 吗？`,
      confirmText: '删除',
      tone: 'danger',
    })
    if (!confirmed) return
    if (imageStore.currentImage?.id !== pageId) {
      showToast('当前图片已切换，未执行删除', 'warning')
      return
    }
    await flushPageDocument(pageId).catch(() => undefined)
    try {
      await deletePage(pageId)
    } catch (error) {
      showToast(
        `删除图片失败：${error instanceof Error ? error.message : '未知错误'}`,
        'error',
      )
      return
    }
    discardPageDocument(pageId)
    imageStore.clearImages()
    bubbleStore.clearBubblesLocal()
    showToast('图片已删除', 'success')
    await loadChapterSession()
  }

  async function clearAllImages() {
    if (!imageStore.hasImages) return
    const bookshelfMode = translateInit.isBookshelfMode.value
    const chapterId = imageStore.currentImage?.chapterId ?? imageStore.images[0]?.chapterId
    if (!chapterId) {
      showToast('当前图片不属于后端章节', 'error')
      return
    }
    const confirmed = await confirmAction({
      title: '清空图片',
      message: '确定要从后端章节中删除所有图片和翻译结果吗？',
      confirmText: '清空',
      tone: 'danger',
    })
    if (!confirmed) return
    const currentChapterId = imageStore.currentImage?.chapterId
      ?? imageStore.images[0]?.chapterId
    if (
      translateInit.isBookshelfMode.value !== bookshelfMode
      || currentChapterId !== chapterId
    ) {
      showToast('当前章节已切换，未执行清空', 'warning')
      return
    }
    const pageIds = imageStore.images.map(image => image.id)
    if (!bookshelfMode && !(await translateInit.flushChapterWorkState())) {
      showToast('章节工作态设置尚未写入后端，无法新建快速工作区', 'error')
      return
    }
    await Promise.allSettled(pageIds.map(pageId => flushPageDocument(pageId)))
    try {
      if (bookshelfMode) await clearChapterPages(chapterId)
      else {
        await resetQuickWorkspace()
        translateInit.forgetReplacedChapter(chapterId)
      }
    } catch (error) {
      showToast(
        `清空图片失败：${error instanceof Error ? error.message : '未知错误'}`,
        'error',
      )
      return
    }
    for (const pageId of pageIds) discardPageDocument(pageId)
    imageStore.clearImages()
    bubbleStore.clearBubblesLocal()
    showToast('所有图片已清除', 'success')
    await loadChapterSession()
  }

  async function goToPrevious() {
    await translateInit.goToPrevious()
  }

  async function goToNext() {
    await translateInit.goToNext()
  }

  async function toggleEditMode() {
    if (isEditMode.value) {
      isEditMode.value = false
      return
    }
    if (!canUseEditMode.value) {
      showToast('管理员已关闭编辑模式', 'info')
      return
    }
    const pageId = imageStore.currentImage?.id
    if (!pageId) return
    if (!(await translateInit.flushChapterWorkState())) {
      showToast('章节工作态设置尚未写入后端，无法进入编辑模式', 'error')
      return
    }
    if (imageStore.currentImage?.id !== pageId) return
    try {
      await flushPageDocument(pageId)
    } catch (error) {
      showToast(
        `当前页写入后端失败：${error instanceof Error ? error.message : '未知错误'}`,
        'error',
      )
      return
    }
    if (imageStore.currentImage?.id !== pageId) return
    isEditMode.value = true
  }

  async function handleRetryFailed() {
    const chapterId = imageStore.currentImage?.chapterId ?? imageStore.images[0]?.chapterId
    if (taskCenterStore.retryableFailedItemCount(chapterId) === 0) {
      showToast('没有失败的图片需要重新翻译', 'info')
      return
    }

    if (!validateBeforeTranslation('normal')) return
    await translation.retryFailedImages()
  }

  function handleKeydown(event: KeyboardEvent) {
    const target = event.target instanceof Element ? event.target : null
    const isInTextInput = Boolean(target?.closest(
      'input, textarea, select, button, [contenteditable]:not([contenteditable="false"]), #bubbleTextEditor',
    ))

    if (isInTextInput || isEditMode.value) return

    if (event.altKey) {
      switch (event.key) {
        case 'ArrowLeft':
          event.preventDefault()
          void goToPrevious()
          break
        case 'ArrowRight':
          event.preventDefault()
          void goToNext()
          break
        case 'ArrowUp':
          event.preventDefault()
          if (!settingsStore.settings.textStyle.autoFontSize) {
            settingsStore.updateTextStyle({ fontSize: settingsStore.settings.textStyle.fontSize + 1 })
          }
          break
        case 'ArrowDown':
          event.preventDefault()
          if (!settingsStore.settings.textStyle.autoFontSize) {
            settingsStore.updateTextStyle({
              fontSize: Math.max(1, settingsStore.settings.textStyle.fontSize - 1),
            })
          }
          break
      }
    }
  }

  async function selectImage(index: number) {
    await translateInit.switchImage(index)
  }

  return {
    clearAllImages,
    deleteCurrentImage,
    goToNext,
    goToPrevious,
    handleKeydown,
    handleRetryFailed,
    handleRunWorkflow,
    handleUploadComplete,
    loadChapterSession,
    selectImage,
    toggleEditMode,
  }
}
