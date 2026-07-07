import { computed } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useToast } from '@/utils/toast'
import {
  usePipeline,
  getStandardModeConfig,
  getHqModeConfig,
  getProofreadModeConfig,
  getRemoveTextModeConfig,
} from './translation'
import type { PageSelection } from './translation/core/types'
import { pageIndexesToSelection, pageSelectionToPageIndexes } from '@/utils/pageSelection'

export type { TranslationProgress, PageSelection } from './translation/core/types'

export type TranslationMode = 'standard' | 'hq' | 'proofread' | 'removeText'

export interface TranslateResult {
  success: boolean
  completed: number
  failed: number
  errors: string[]
}

export function range(start: number, end: number): number[] {
  const result: number[] = []
  for (let i = start; i < end; i++) {
    result.push(i)
  }
  return result
}

export function useTranslation() {
  const imageStore = useImageStore()
  const bubbleStore = useBubbleStore()
  const toast = useToast()
  const pipeline = usePipeline()

  const progress = pipeline.progress
  const isTranslatingSingle = pipeline.isExecuting
  const isTranslating = pipeline.isTranslating
  const progressPercent = pipeline.progressPercent

  const isHqTranslating = computed(() => pipeline.isExecuting.value)
  const isProofreading = computed(() => pipeline.isExecuting.value)

  async function translatePages(
    pageIndexes: number[],
    mode: TranslationMode
  ): Promise<TranslateResult> {
    if (pageIndexes.length === 0) {
      toast.error('没有指定要翻译的页面')
      return { success: false, completed: 0, failed: 0, errors: ['没有指定要翻译的页面'] }
    }

    const totalImages = imageStore.images.length
    if (totalImages === 0) {
      toast.error('请先上传图片')
      return { success: false, completed: 0, failed: 0, errors: ['请先上传图片'] }
    }

    for (const idx of pageIndexes) {
      if (idx < 0 || idx >= totalImages) {
        toast.error(`无效的页面索引: ${idx}`)
        return { success: false, completed: 0, failed: 0, errors: [`无效的页面索引: ${idx}`] }
      }
    }

    const isSinglePage = pageIndexes.length === 1
    const isAllPages = pageIndexes.length === totalImages

    let config
    if (isSinglePage) {
      const originalIndex = imageStore.currentImageIndex
      const targetIndex = pageIndexes[0]
      if (targetIndex !== undefined) {
        imageStore.setCurrentImageIndex(targetIndex)
      }

      config = getModeConfig(mode, 'current')
      let pipelineResult
      try {
        pipelineResult = await pipeline.execute(config)
      } finally {
        imageStore.setCurrentImageIndex(originalIndex)
      }
      return {
        success: pipelineResult.success,
        completed: pipelineResult.completed,
        failed: pipelineResult.failed,
        errors: pipelineResult.errors ?? [],
      }
    }

    if (isAllPages) {
      config = getModeConfig(mode, 'all')
    } else {
      const pageSelection = { pages: pageIndexesToSelection(pageIndexes) }
      config = getModeConfig(mode, 'selection', pageSelection)
    }

    const pipelineResult = await pipeline.execute(config)
    return {
      success: pipelineResult.success,
      completed: pipelineResult.completed,
      failed: pipelineResult.failed,
      errors: pipelineResult.errors ?? [],
    }
  }

  function getModeConfig(
    mode: TranslationMode,
    scope: 'current' | 'all' | 'selection' | 'failed',
    pageSelection?: PageSelection
  ) {
    const options = pageSelection ? { pageSelection } : undefined
    switch (mode) {
      case 'standard':
        return getStandardModeConfig(scope, options)
      case 'hq':
        return getHqModeConfig(scope, options)
      case 'proofread':
        return getProofreadModeConfig(scope, options)
      case 'removeText':
        return getRemoveTextModeConfig(scope, options)
      default:
        return getStandardModeConfig(scope, options)
    }
  }

  async function translateCurrentImage(): Promise<boolean> {
    const result = await translatePages([imageStore.currentImageIndex], 'standard')
    return result.success
  }

  async function translateImageByIndex(index: number): Promise<boolean> {
    const result = await translatePages([index], 'standard')
    return result.success
  }

  async function translateAllImages(): Promise<boolean> {
    const result = await translatePages(range(0, imageStore.images.length), 'standard')
    return result.success
  }

  async function translateSelectedImages(pageSelection: PageSelection): Promise<boolean> {
    const pageIndexes = pageSelectionToPageIndexes(pageSelection.pages)
    const result = await translatePages(pageIndexes, 'standard')
    return result.success
  }

  function cancelBatchTranslation(): void {
    pipeline.cancel()
  }

  async function removeTextOnly(): Promise<boolean> {
    const result = await translatePages([imageStore.currentImageIndex], 'removeText')
    return result.success
  }

  async function removeAllTexts(): Promise<boolean> {
    const result = await translatePages(range(0, imageStore.images.length), 'removeText')
    return result.success
  }

  async function removeTextSelection(pageSelection: PageSelection): Promise<boolean> {
    const pageIndexes = pageSelectionToPageIndexes(pageSelection.pages)
    const result = await translatePages(pageIndexes, 'removeText')
    return result.success
  }

  async function retryFailedImages(): Promise<boolean> {
    const failedIndices = imageStore.getFailedImageIndices()
    if (failedIndices.length === 0) {
      toast.info('没有失败的图片需要重新翻译')
      return true
    }

    const result = await translatePages(failedIndices, 'standard')
    return result.success
  }

  async function executeHqTranslation(pageSelection?: PageSelection): Promise<boolean> {
    const pageIndexes = pageSelection
      ? pageSelectionToPageIndexes(pageSelection.pages)
      : range(0, imageStore.images.length)
    const result = await translatePages(pageIndexes, 'hq')
    return result.success
  }

  async function executeProofreading(pageSelection?: PageSelection): Promise<boolean> {
    const pageIndexes = pageSelection
      ? pageSelectionToPageIndexes(pageSelection.pages)
      : range(0, imageStore.images.length)
    const result = await translatePages(pageIndexes, 'proofread')
    return result.success
  }

  async function translateWithCurrentBubbles(): Promise<boolean> {
    const currentImage = imageStore.currentImage
    if (!currentImage) {
      toast.error('请先上传图片')
      return false
    }

    const bubbles = bubbleStore.bubbles
    if (!bubbles || bubbles.length === 0) {
      toast.error('当前图片没有气泡框，请先检测或手动添加')
      return false
    }

    imageStore.setManuallyAnnotated(true)
    return translateCurrentImage()
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
