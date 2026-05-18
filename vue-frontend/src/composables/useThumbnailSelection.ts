import { computed, type Ref } from 'vue'

import type { ImageData } from '@/types/image'

export function useThumbnailSelection(images: Ref<ImageData[]>) {
  function getImageGlobalIndex(image: ImageData): number {
    return images.value.findIndex((img) => img.id === image.id)
  }

  function getStatusType(image: ImageData): 'failed' | 'labeled' | 'processing' | null {
    if (image.translationFailed) return 'failed'
    if (image.isManuallyAnnotated) return 'labeled'
    if (image.translationStatus === 'processing') return 'processing'
    return null
  }

  function isTranslated(image: ImageData): boolean {
    return image.translationStatus === 'completed'
  }

  function getThumbnailTitle(image: ImageData): string {
    if (image.translationFailed) return '翻译失败，点击可重试'
    if (image.isManuallyAnnotated) return '包含手动标注'
    if (image.translationStatus === 'completed') return '已完成翻译'
    return image.fileName || ''
  }

  const failedPages = computed(() =>
    images.value
      .map((image, index) => image.translationFailed ? index + 1 : null)
      .filter((page): page is number => page !== null)
  )

  const completedPages = computed(() =>
    images.value
      .map((image, index) => image.translationStatus === 'completed' ? index + 1 : null)
      .filter((page): page is number => page !== null)
  )

  const pendingPages = computed(() =>
    images.value
      .map((image, index) => image.translationStatus !== 'completed' ? index + 1 : null)
      .filter((page): page is number => page !== null)
  )

  const labeledPages = computed(() =>
    images.value
      .map((image, index) => image.isManuallyAnnotated ? index + 1 : null)
      .filter((page): page is number => page !== null)
  )

  return {
    getImageGlobalIndex,
    getStatusType,
    isTranslated,
    getThumbnailTitle,
    failedPages,
    completedPages,
    pendingPages,
    labeledPages,
  }
}
