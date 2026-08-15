import { computed, type Ref } from 'vue'

import type { ImageData } from '@/types/image'

export function useThumbnailSelection(images: Ref<ImageData[]>) {
  const imageIndexes = computed(() => new Map(
    images.value.map((image, index) => [image.id, index]),
  ))

  function getImageGlobalIndex(image: ImageData): number {
    return imageIndexes.value.get(image.id) ?? -1
  }

  function getStatusType(image: ImageData): 'failed' | 'labeled' | 'processing' | null {
    if (image.translationStatus === 'failed') return 'failed'
    if (image.translationStatus === 'processing') return 'processing'
    if (image.isManuallyAnnotated) return 'labeled'
    return null
  }

  function isTranslated(image: ImageData): boolean {
    return image.translationStatus === 'completed'
  }

  function getThumbnailTitle(image: ImageData): string {
    if (image.translationStatus === 'failed') return '翻译失败'
    if (image.translationStatus === 'processing') return '正在处理'
    if (image.isManuallyAnnotated) return '包含手动标注'
    if (image.translationStatus === 'completed') return '已完成翻译'
    return image.fileName || ''
  }

  const failedPages = computed(() =>
    images.value
      .map((image, index) => image.translationStatus === 'failed' ? index + 1 : null)
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
