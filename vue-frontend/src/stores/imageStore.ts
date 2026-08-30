import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import type { BubbleState } from '@/types/bubble'
import type { ImageData, ImageDataLoadInput, ImageDataUpdates, TranslationStatus } from '@/types/image'
import { normalizeImageTextStyleFields } from '@/defaults/textStyleDefaults'

export const useImageStore = defineStore('image', () => {
  const images = ref<ImageData[]>([])
  const currentImageIndex = ref<number>(-1)
  const isTranslationInProgress = ref<boolean>(false)

  const currentImage = computed<ImageData | null>(() => {
    if (currentImageIndex.value >= 0 && currentImageIndex.value < images.value.length) {
      return images.value[currentImageIndex.value] ?? null
    }
    return null
  })

  const imageCount = computed<number>(() => images.value.length)
  const hasImages = computed<boolean>(() => images.value.length > 0)
  const canGoPrevious = computed<boolean>(() => currentImageIndex.value > 0)
  const canGoNext = computed<boolean>(() => currentImageIndex.value < images.value.length - 1)
  function setImages(newImages: ImageDataLoadInput[]): void {
    images.value = newImages.map(img => ({
      ...img,
      ...normalizeImageTextStyleFields(img),
      width: img.width ?? 0,
      height: img.height ?? 0,
    }))

    if (images.value.length > 0) {
      currentImageIndex.value = Math.min(
        Math.max(0, currentImageIndex.value),
        images.value.length - 1,
      )
    } else {
      currentImageIndex.value = -1
    }
  }

  function deleteImage(index: number): boolean {
    if (index < 0 || index >= images.value.length) {
      return false
    }

    images.value.splice(index, 1)

    if (currentImageIndex.value === index) {
      currentImageIndex.value = Math.min(index, images.value.length - 1)
      if (images.value.length === 0) {
        currentImageIndex.value = -1
      }
    } else if (currentImageIndex.value > index) {
      currentImageIndex.value--
    }

    return true
  }

  function deleteCurrentImage(): boolean {
    return deleteImage(currentImageIndex.value)
  }

  function clearImages(): void {
    images.value = []
    currentImageIndex.value = -1
    isTranslationInProgress.value = false
  }

  function setCurrentImageIndex(index: number): void {
    if (index >= -1 && index < images.value.length) {
      currentImageIndex.value = index
    }
  }

  function goToPrevious(): boolean {
    if (canGoPrevious.value) {
      currentImageIndex.value--
      return true
    }
    return false
  }

  function goToNext(): boolean {
    if (canGoNext.value) {
      currentImageIndex.value++
      return true
    }
    return false
  }

  function updateCurrentImage(updates: ImageDataUpdates): void {
    if (currentImage.value) {
      Object.assign(currentImage.value, updates)
    }
  }

  function updateImageByIndex(index: number, updates: ImageDataUpdates): void {
    if (index >= 0 && index < images.value.length) {
      const image = images.value[index]
      if (image) {
        Object.assign(image, updates)
      }
    }
  }

  function updateCurrentBubbleStates(bubbleStates: BubbleState[] | null): void {
    if (currentImage.value) {
      currentImage.value.bubbleStates = bubbleStates
      currentImage.value.hasUnsavedChanges = true
    }
  }

  function setManuallyAnnotated(isManual: boolean): void {
    if (!currentImage.value) return

    currentImage.value.isManuallyAnnotated = isManual
    currentImage.value.hasUnsavedChanges = true
  }

  function updateCurrentImageProperty<K extends keyof ImageData>(
    key: K,
    value: ImageData[K],
  ): void {
    if (currentImage.value) {
      currentImage.value[key] = value
      currentImage.value.hasUnsavedChanges = true
    }
  }

  function updateCurrentImageDimensions(width: number, height: number): void {
    if (currentImage.value) {
      currentImage.value.width = width
      currentImage.value.height = height
    }
  }

  function setTranslationStatus(
    index: number,
    status: TranslationStatus,
  ): void {
    if (index >= 0 && index < images.value.length) {
      const image = images.value[index]
      if (image) {
        image.translationStatus = status
      }
    }
  }

  function setTranslationInProgress(isInProgress: boolean): void {
    isTranslationInProgress.value = isInProgress
  }

  return {
    images,
    currentImageIndex,
    isTranslationInProgress,

    currentImage,
    imageCount,
    hasImages,
    canGoPrevious,
    canGoNext,
    setImages,
    deleteCurrentImage,
    clearImages,
    setCurrentImageIndex,
    goToPrevious,
    goToNext,

    updateCurrentImage,
    updateImageByIndex,
    updateCurrentBubbleStates,
    setManuallyAnnotated,
    updateCurrentImageProperty,
    updateCurrentImageDimensions,

    setTranslationStatus,

    setTranslationInProgress,
  }
})
