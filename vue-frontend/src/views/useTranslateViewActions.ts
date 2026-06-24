import type { ComputedRef, Ref } from 'vue'
import { showToast } from '@/utils/toast'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import { useSessionStore } from '@/stores/sessionStore'
import { useTranslation } from '@/composables/useTranslationPipeline'
import { useTranslateInit } from '@/composables/useTranslateInit'
import type { WorkflowRunRequest } from '@/types/workflow'

type TranslateValidationMode = 'normal' | 'hq' | 'proofread'

interface TranslateImageLike {
  fileName?: string
}

interface UseTranslateViewActionsOptions {
  imageStore: ReturnType<typeof useImageStore>
  settingsStore: ReturnType<typeof useSettingsStore>
  sessionStore: ReturnType<typeof useSessionStore>
  translation: ReturnType<typeof useTranslation>
  translateInit: ReturnType<typeof useTranslateInit>
  validateBeforeTranslation: (mode: TranslateValidationMode) => boolean
  currentImage: ComputedRef<TranslateImageLike | null | undefined>
  hasImages: ComputedRef<boolean>
  hasFailedImages: ComputedRef<boolean>
  currentBookId: ComputedRef<string | undefined>
  currentChapterId: ComputedRef<string | undefined>
  isEditMode: Ref<boolean>
}

export function useTranslateViewActions(options: UseTranslateViewActionsOptions) {
  const {
    imageStore,
    settingsStore,
    sessionStore,
    translation,
    translateInit,
    validateBeforeTranslation,
    currentImage,
    hasImages,
    hasFailedImages,
    currentBookId,
    currentChapterId,
    isEditMode,
  } = options

  async function loadChapterSession() {
    if (!currentBookId.value || !currentChapterId.value) return

    try {
      await translateInit.initializeBookChapterContext()
    } catch {
      showToast('加载章节会话失败', 'error')
    }
  }

  function handleUploadComplete(_count: number) {
    if (imageStore.hasImages) {
      imageStore.sortImagesByFileName()
      translateInit.switchImage(0)
    }
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
        deleteCurrentImage()
        return
      case 'clear-all':
        clearAllImages()
        return
      default:
        return
    }
  }

  function deleteCurrentImage() {
    if (!currentImage.value) return
    const fileName = currentImage.value.fileName || `图片 ${imageStore.currentImageIndex + 1}`
    if (confirm(`确定要删除当前图片 (${fileName}) 吗？`)) {
      imageStore.deleteCurrentImage()
      showToast('图片已删除', 'success')
    }
  }

  function clearAllImages() {
    if (!hasImages.value) return
    if (confirm('确定要清除所有图片吗？这将丢失所有未保存的进度。')) {
      imageStore.clearImages()
      showToast('所有图片已清除', 'success')
    }
  }

  function goToPrevious() {
    translateInit.goToPrevious()
  }

  function goToNext() {
    translateInit.goToNext()
  }

  function toggleEditMode() {
    isEditMode.value = !isEditMode.value
  }

  async function handleRetryFailed() {
    if (!hasFailedImages.value) {
      showToast('没有失败的图片需要重新翻译', 'info')
      return
    }

    if (!validateBeforeTranslation('normal')) return
    await translation.retryFailedImages()
  }

  async function saveCurrentSession() {
    if (!hasImages.value) {
      showToast('没有可保存的内容', 'warning')
      return
    }

    if (!currentBookId.value || !currentChapterId.value) return

    try {
      const success = await sessionStore.saveChapterSession(currentBookId.value, currentChapterId.value)
      showToast(success ? '章节进度已保存' : '保存失败', success ? 'success' : 'error')
    } catch (error) {
      showToast('保存失败: ' + (error instanceof Error ? error.message : '未知错误'), 'error')
    }
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

  function selectImage(index: number) {
    translateInit.switchImage(index)
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
    saveCurrentSession,
    selectImage,
    toggleEditMode,
  }
}
