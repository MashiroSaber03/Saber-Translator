import type { ComputedRef, Ref } from 'vue'
import { showToast } from '@/utils/toast'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import { useTranslation } from '@/composables/useTranslationPipeline'
import { useTranslateInit } from '@/composables/useTranslateInit'
import { confirmProductAction, type ProductConfirmAction } from '@/composables/useProductConfirm'
import type { WorkflowRunRequest } from '@/types/workflow'
import { clearChapterPages, deletePage, resetQuickWorkspace } from '@/api/v2/content'
import { flushPageDocument } from '@/services/pageDocumentPersistence'

type TranslateValidationMode = 'normal' | 'hq' | 'proofread'

interface TranslateImageLike {
  fileName?: string
}

interface UseTranslateViewActionsOptions {
  imageStore: ReturnType<typeof useImageStore>
  settingsStore: ReturnType<typeof useSettingsStore>
  translation: ReturnType<typeof useTranslation>
  translateInit: ReturnType<typeof useTranslateInit>
  validateBeforeTranslation: (mode: TranslateValidationMode) => boolean
  currentImage: ComputedRef<TranslateImageLike | null | undefined>
  hasImages: ComputedRef<boolean>
  hasFailedImages: ComputedRef<boolean>
  isEditMode: Ref<boolean>
  confirmAction?: ProductConfirmAction
}

export function useTranslateViewActions(options: UseTranslateViewActionsOptions) {
  const {
    imageStore,
    settingsStore,
    translation,
    translateInit,
    validateBeforeTranslation,
    currentImage,
    hasImages,
    hasFailedImages,
    isEditMode,
    confirmAction = confirmProductAction,
  } = options

  async function loadChapterSession() {
    try {
      await translateInit.initializeBookChapterContext()
    } catch {
      showToast('刷新后端章节失败', 'error')
    }
  }

  async function handleUploadComplete() {
    await translateInit.initializeBookChapterContext()
  }

  async function translateCurrentImage() {
    if (!currentImage.value) return
    if (!validateBeforeTranslation('normal')) return
    await translation.translateCurrentImage()
  }

  async function translateAllImages() {
    if (!hasImages.value) return
    if (!validateBeforeTranslation('normal')) return
    await translation.translateAllImages()
  }

  async function startHqTranslation() {
    if (!hasImages.value) return
    if (!validateBeforeTranslation('hq')) return
    await translation.executeHqTranslation()
  }

  async function startProofreading() {
    if (!hasImages.value) return
    if (!validateBeforeTranslation('proofread')) return
    await translation.executeProofreading()
  }

  async function removeTextOnly() {
    if (!currentImage.value) return
    await translation.removeTextOnly()
  }

  async function removeAllText() {
    if (!hasImages.value) return
    await translation.removeAllTexts()
  }

  async function translateSelectedImages(pages: number[]) {
    if (!hasImages.value) return
    if (!validateBeforeTranslation('normal')) return
    await translation.translateSelectedImages({ pages })
  }

  async function startHqTranslationSelection(pages: number[]) {
    if (!hasImages.value) return
    if (!validateBeforeTranslation('hq')) return
    await translation.executeHqTranslation({ pages })
  }

  async function startProofreadingSelection(pages: number[]) {
    if (!hasImages.value) return
    if (!validateBeforeTranslation('proofread')) return
    await translation.executeProofreading({ pages })
  }

  async function removeTextSelection(pages: number[]) {
    if (!hasImages.value) return
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
      default:
        return
    }
  }

  async function deleteCurrentImage() {
    if (!currentImage.value) return
    const fileName = currentImage.value.fileName || `图片 ${imageStore.currentImageIndex + 1}`
    const confirmed = await confirmAction({
      title: '删除当前图片',
      message: `确定要删除当前图片 (${fileName}) 吗？`,
      confirmText: '删除',
      tone: 'danger',
    })
    if (!confirmed) return
    const pageId = imageStore.currentImage?.id
    if (!pageId) return
    await deletePage(pageId)
    await translateInit.initializeBookChapterContext()
    showToast('图片已删除', 'success')
  }

  async function clearAllImages() {
    if (!hasImages.value) return
    const confirmed = await confirmAction({
      title: '清空图片',
      message: '确定要从后端章节中删除所有图片和翻译结果吗？',
      confirmText: '清空',
      tone: 'danger',
    })
    if (!confirmed) return
    if (!translateInit.isBookshelfMode.value) {
      await resetQuickWorkspace()
    } else {
      const chapterId = imageStore.currentImage?.chapterId ?? imageStore.images[0]?.chapterId
      if (!chapterId) throw new Error('当前图片不属于后端章节')
      await clearChapterPages(chapterId)
    }
    await translateInit.initializeBookChapterContext()
    showToast('所有图片已清除', 'success')
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
    if (!(await translateInit.flushChapterWorkState())) {
      showToast('章节工作态设置尚未写入后端，无法进入编辑模式', 'error')
      return
    }
    const pageId = imageStore.currentImage?.id
    try {
      if (pageId) await flushPageDocument(pageId)
    } catch (error) {
      showToast(
        `当前页写入后端失败：${error instanceof Error ? error.message : '未知错误'}`,
        'error',
      )
      return
    }
    isEditMode.value = true
  }

  async function handleRetryFailed() {
    if (!hasFailedImages.value) {
      showToast('没有失败的图片需要重新翻译', 'info')
      return
    }

    if (!validateBeforeTranslation('normal')) return
    await translation.retryFailedImages()
  }

  function handleKeydown(event: KeyboardEvent) {
    const target = event.target as HTMLElement
    const isInTextInput =
      target instanceof HTMLInputElement ||
      target instanceof HTMLTextAreaElement ||
      target.getAttribute('contenteditable') === 'true' ||
      target.id === 'bubbleTextEditor'

    if (isInTextInput || isEditMode.value) return

    if (event.altKey) {
      switch (event.key) {
        case 'ArrowLeft':
          event.preventDefault()
          goToPrevious()
          break
        case 'ArrowRight':
          event.preventDefault()
          goToNext()
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
              fontSize: Math.max(10, settingsStore.settings.textStyle.fontSize - 1),
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
