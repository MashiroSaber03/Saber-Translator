import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import type { BubbleState } from '@/types/bubble'
import type { ImageData, ImageDataLoadInput, ImageDataUpdates, TranslationStatus } from '@/types/image'
import { useSettingsStore } from '@/stores/settings'
import {
  getImageTextStyleDefaults,
  normalizeImageTextStyleFields,
} from '@/defaults/textStyleDefaults'
import { naturalSortCompare } from '@/utils'
import { applyImageBubbleMirrors } from '@/stores/imageBubbleMirrors'

function generateId(): string {
  return `img_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`
}

function pickDefinedValues<T extends Record<string, unknown>>(value: T): Partial<T> {
  return Object.fromEntries(
    Object.entries(value).filter(([, fieldValue]) => fieldValue !== undefined),
  ) as Partial<T>
}

function getCurrentImageTextStyleSeed(
  preferCurrentTextStyle: boolean,
): ReturnType<typeof getImageTextStyleDefaults> {
  const canonicalDefaults = getImageTextStyleDefaults()
  if (!preferCurrentTextStyle) {
    return canonicalDefaults
  }

  try {
    const settingsStore = useSettingsStore()
    return normalizeImageTextStyleFields(settingsStore.settings.textStyle as Partial<ImageData>)
  } catch {
    return canonicalDefaults
  }
}

export function getImageDimensionsFromDataURL(
  dataURL: string,
): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => {
      resolve({ width: img.naturalWidth, height: img.naturalHeight })
    }
    img.onerror = () => {
      reject(new Error('Failed to load image'))
    }
    img.src = dataURL
  })
}

function createDefaultImageData(
  fileName: string,
  originalDataURL: string,
  overrides?: Partial<ImageData>,
  options?: { preferCurrentTextStyle?: boolean },
): ImageData {
  const imageTextStyle = getCurrentImageTextStyleSeed(options?.preferCurrentTextStyle ?? true)
  return {
    id: generateId(),
    fileName,
    width: 0,
    height: 0,
    originalDataURL,
    translatedDataURL: null,
    cleanImageData: null,
    bubbleStates: null,
    translationStatus: 'pending',
    translationFailed: false,
    ...imageTextStyle,
    hasUnsavedChanges: false,
    ...overrides,
  }
}

export const useImageStore = defineStore('image', () => {
  const images = ref<ImageData[]>([])
  const currentImageIndex = ref<number>(-1)
  const isBatchTranslationInProgress = ref<boolean>(false)

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
  const failedImageCount = computed<number>(
    () => images.value.filter(img => img.translationFailed).length,
  )
  const completedImageCount = computed<number>(
    () => images.value.filter(img => img.translationStatus === 'completed').length,
  )
  const pendingImageCount = computed<number>(
    () => images.value.filter(img => img.translationStatus === 'pending').length,
  )

  function addImage(
    fileName: string,
    originalDataURL: string,
    overrides?: Partial<ImageData>,
  ): ImageData {
    const newImage = createDefaultImageData(fileName, originalDataURL, overrides, {
      preferCurrentTextStyle: true,
    })
    images.value.push(newImage)

    if (images.value.length === 1) {
      currentImageIndex.value = 0
    }

    return newImage
  }

  function addImages(
    imageList: Array<{
      fileName: string
      originalDataURL: string
      overrides?: Partial<ImageData>
    }>,
  ): ImageData[] {
    const newImages = imageList.map(({ fileName, originalDataURL, overrides }) =>
      createDefaultImageData(fileName, originalDataURL, overrides),
    )

    const wasEmpty = images.value.length === 0
    images.value.push(...newImages)

    if (wasEmpty && images.value.length > 0) {
      currentImageIndex.value = 0
    }

    return newImages
  }

  function setImages(newImages: ImageDataLoadInput[]): void {
    images.value = newImages.map(img => {
      const normalizedTextStyle = normalizeImageTextStyleFields(img)
      return createDefaultImageData(
        img.fileName,
        img.originalDataURL,
        {
          ...pickDefinedValues(img as Record<string, unknown>),
          ...normalizedTextStyle,
          width: img.width || 0,
          height: img.height || 0,
          hasUnsavedChanges: img.hasUnsavedChanges || false,
        } as Partial<ImageData>,
        { preferCurrentTextStyle: false },
      )
    })

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
    isBatchTranslationInProgress.value = false
  }

  function sortImagesByFileName(): void {
    const currentImageId = currentImage.value?.id || null

    images.value.sort((a, b) => {
      const pathA = a.relativePath || a.fileName
      const pathB = b.relativePath || b.fileName
      return naturalSortCompare(pathA, pathB)
    })

    if (currentImageId) {
      const newIndex = images.value.findIndex(img => img.id === currentImageId)
      if (newIndex >= 0 && newIndex !== currentImageIndex.value) {
        currentImageIndex.value = newIndex
      }
    }
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
      if (updates.bubbleStates !== undefined) {
        applyImageBubbleMirrors(currentImage.value, updates.bubbleStates)
      }
    }
  }

  function updateImageByIndex(index: number, updates: ImageDataUpdates): void {
    if (index >= 0 && index < images.value.length) {
      const image = images.value[index]
      if (image) {
        Object.assign(image, updates)
        if (updates.bubbleStates !== undefined) {
          applyImageBubbleMirrors(image, updates.bubbleStates)
        }
      }
    }
  }

  function updateCurrentBubbleStates(bubbleStates: BubbleState[] | null): void {
    if (currentImage.value) {
      applyImageBubbleMirrors(currentImage.value, bubbleStates)
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

  function updateCurrentTranslationResult(
    translatedDataURL: string,
    cleanImageData?: string,
  ): void {
    if (currentImage.value) {
      currentImage.value.translatedDataURL = translatedDataURL
      if (cleanImageData) {
        currentImage.value.cleanImageData = cleanImageData
      }
      currentImage.value.translationStatus = 'completed'
      currentImage.value.translationFailed = false
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
    errorMessage?: string,
  ): void {
    if (index >= 0 && index < images.value.length) {
      const image = images.value[index]
      if (image) {
        image.translationStatus = status
        image.translationFailed = status === 'failed'
        if (status === 'failed') {
          image.errorMessage = errorMessage
        } else {
          image.errorMessage = undefined
        }
      }
    }
  }

  function markCurrentAsFailed(errorMessage: string): void {
    if (currentImage.value) {
      currentImage.value.translationStatus = 'failed'
      currentImage.value.translationFailed = true
      currentImage.value.errorMessage = errorMessage
    }
  }

  function resetAllTranslationStatus(): void {
    images.value.forEach(img => {
      img.translationStatus = 'pending'
      img.translationFailed = false
      img.errorMessage = undefined
    })
  }

  function setBatchTranslationInProgress(isInProgress: boolean): void {
    isBatchTranslationInProgress.value = isInProgress
  }

  function getFailedImageIndices(): number[] {
    return images.value
      .map((img, index) => (img.translationFailed ? index : -1))
      .filter(index => index !== -1)
  }

  return {
    images,
    currentImageIndex,
    isBatchTranslationInProgress,

    currentImage,
    imageCount,
    hasImages,
    canGoPrevious,
    canGoNext,
    failedImageCount,
    completedImageCount,
    pendingImageCount,

    addImage,
    addImages,
    setImages,
    deleteImage,
    deleteCurrentImage,
    clearImages,
    sortImagesByFileName,

    setCurrentImageIndex,
    goToPrevious,
    goToNext,

    updateCurrentImage,
    updateImageByIndex,
    updateCurrentBubbleStates,
    setManuallyAnnotated,
    updateCurrentImageProperty,
    updateCurrentTranslationResult,
    updateCurrentImageDimensions,

    setTranslationStatus,
    markCurrentAsFailed,
    resetAllTranslationStatus,

    setBatchTranslationInProgress,
    getFailedImageIndices,
  }
})
