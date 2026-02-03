/**
 * 图片状态管理 Store
 * 管理翻译页面中的图片数据、当前索引、翻译状态等
 */

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { ImageData, TranslationStatus, ImageDataUpdates } from '@/types/image'
import type { BubbleState } from '@/types/bubble'
import {
  DEFAULT_FONT_FAMILY,
  DEFAULT_FILL_COLOR,
  DEFAULT_STROKE_ENABLED,
  DEFAULT_STROKE_COLOR,
  DEFAULT_STROKE_WIDTH
} from '@/constants'

/**
 * 生成唯一 ID
 */
function generateId(): string {
  return `img_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`
}

/**
 * 从 dataURL 异步获取图片尺寸
 * 用于在添加图片时计算真实尺寸
 */
export function getImageDimensionsFromDataURL(dataURL: string): Promise<{ width: number; height: number }> {
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

/**
 * 创建默认图片数据
 */
function createDefaultImageData(
  fileName: string,
  originalDataURL: string,
  overrides?: Partial<ImageData>
): ImageData {
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
    fontSize: 25,
    autoFontSize: false,
    fontFamily: DEFAULT_FONT_FAMILY,
    layoutDirection: 'auto',
    textColor: '#000000',
    fillColor: DEFAULT_FILL_COLOR,
    inpaintMethod: 'solid',
    strokeEnabled: DEFAULT_STROKE_ENABLED,
    strokeColor: DEFAULT_STROKE_COLOR,
    strokeWidth: DEFAULT_STROKE_WIDTH,
    hasUnsavedChanges: false,
    ...overrides
  }
}


export const useImageStore = defineStore('image', () => {
  // ============================================================
  // 状态定义
  // ============================================================

  /** 图片数据数组 */
  const images = ref<ImageData[]>([])

  /** 当前图片索引 */
  const currentImageIndex = ref<number>(-1)

  /** 批量翻译是否正在进行 */
  const isBatchTranslationInProgress = ref<boolean>(false)

  // ============================================================
  // 计算属性
  // ============================================================

  /** 当前图片对象 */
  const currentImage = computed<ImageData | null>(() => {
    if (currentImageIndex.value >= 0 && currentImageIndex.value < images.value.length) {
      return images.value[currentImageIndex.value] ?? null
    }
    return null
  })

  /** 图片总数 */
  const imageCount = computed<number>(() => images.value.length)

  /** 是否有图片 */
  const hasImages = computed<boolean>(() => images.value.length > 0)

  /** 是否可以切换到上一张 */
  const canGoPrevious = computed<boolean>(() => currentImageIndex.value > 0)

  /** 是否可以切换到下一张 */
  const canGoNext = computed<boolean>(
    () => currentImageIndex.value < images.value.length - 1
  )

  /** 翻译失败的图片数量 */
  const failedImageCount = computed<number>(
    () => images.value.filter((img) => img.translationFailed).length
  )

  /** 已完成翻译的图片数量 */
  const completedImageCount = computed<number>(
    () => images.value.filter((img) => img.translationStatus === 'completed').length
  )

  /** 待处理的图片数量 */
  const pendingImageCount = computed<number>(
    () => images.value.filter((img) => img.translationStatus === 'pending').length
  )

  // ============================================================
  // 图片管理方法
  // ============================================================

  /**
   * 添加图片
   * @param fileName - 文件名
   * @param originalDataURL - 原始图片 Base64 数据
   * @param overrides - 可选的覆盖属性
   * @returns 新添加的图片数据
   */
  function addImage(
    fileName: string,
    originalDataURL: string,
    overrides?: Partial<ImageData>
  ): ImageData {
    const newImage = createDefaultImageData(fileName, originalDataURL, overrides)
    images.value.push(newImage)

    // 如果是第一张图片，自动设置为当前图片
    if (images.value.length === 1) {
      currentImageIndex.value = 0
    }

    console.log(`图片已添加: ${fileName}，当前共 ${images.value.length} 张图片`)
    return newImage
  }

  /**
   * 批量添加图片
   * @param imageList - 图片列表 [{fileName, originalDataURL, overrides?}]
   * @returns 新添加的图片数据数组
   */
  function addImages(
    imageList: Array<{
      fileName: string
      originalDataURL: string
      overrides?: Partial<ImageData>
    }>
  ): ImageData[] {
    const newImages = imageList.map(({ fileName, originalDataURL, overrides }) =>
      createDefaultImageData(fileName, originalDataURL, overrides)
    )

    const wasEmpty = images.value.length === 0
    images.value.push(...newImages)

    // 如果之前没有图片，自动设置第一张为当前图片
    if (wasEmpty && images.value.length > 0) {
      currentImageIndex.value = 0
    }

    console.log(`批量添加了 ${newImages.length} 张图片，当前共 ${images.value.length} 张`)
    return newImages
  }

  /**
   * 设置图片数组（用于加载会话）
   * @param newImages - 新的图片数组
   */
  function setImages(newImages: ImageData[]): void {
    images.value = newImages.map((img) => ({
      ...img,
      // 确保 width 和 height 有默认值（兼容旧会话数据）
      width: img.width || 0,
      height: img.height || 0,
      strokeEnabled: img.strokeEnabled ?? DEFAULT_STROKE_ENABLED,
      strokeColor: img.strokeColor || DEFAULT_STROKE_COLOR,
      strokeWidth: img.strokeWidth ?? DEFAULT_STROKE_WIDTH,
      hasUnsavedChanges: img.hasUnsavedChanges || false
    }))

    // 重置当前索引
    if (images.value.length > 0) {
      currentImageIndex.value = Math.min(
        Math.max(0, currentImageIndex.value),
        images.value.length - 1
      )
    } else {
      currentImageIndex.value = -1
    }

    console.log(`图片数组已设置，共 ${images.value.length} 张图片`)
  }

  /**
   * 删除指定索引的图片
   * @param index - 要删除的图片索引
   * @returns 是否删除成功
   */
  function deleteImage(index: number): boolean {
    if (index < 0 || index >= images.value.length) {
      console.warn(`删除失败: 无效的索引 ${index}`)
      return false
    }

    images.value.splice(index, 1)

    // 调整当前索引
    if (currentImageIndex.value === index) {
      currentImageIndex.value = Math.min(index, images.value.length - 1)
      if (images.value.length === 0) {
        currentImageIndex.value = -1
      }
    } else if (currentImageIndex.value > index) {
      currentImageIndex.value--
    }

    console.log(`图片已删除，当前共 ${images.value.length} 张图片`)
    return true
  }

  /**
   * 删除当前图片
   * @returns 是否删除成功
   */
  function deleteCurrentImage(): boolean {
    return deleteImage(currentImageIndex.value)
  }

  /**
   * 清除所有图片
   */
  function clearImages(): void {
    images.value = []
    currentImageIndex.value = -1
    isBatchTranslationInProgress.value = false
    console.log('所有图片已清除')
  }

  /**
   * 按文件路径/文件名对所有图片进行自然排序
   * 
   * 【多文件夹支持】优先使用 relativePath（完整路径）进行排序
   * 这样可以正确处理多文件夹导入的场景，例如：
   * - 第0章/01.jpg, 第0章/02.jpg, 第1章/01.jpg, 第1章/02.jpg...
   * 
   * 如果没有 relativePath（例如直接拖拽单个图片），则使用 fileName
   * 
   * 【重要修复】排序后更新 currentImageIndex，使其仍然指向排序前的那张图片
   * 这避免了排序后 currentImageIndex 指向错误图片，导致气泡状态保存错乱的问题
   */
  function sortImagesByFileName(): void {
    // 记录排序前当前图片的 id（用于排序后找回）
    const currentImageId = currentImage.value?.id || null

    // 执行排序：优先使用 relativePath，没有则使用 fileName
    images.value.sort((a, b) => {
      const pathA = a.relativePath || a.fileName
      const pathB = b.relativePath || b.fileName
      return pathA.localeCompare(pathB, undefined, { numeric: true, sensitivity: 'base' })
    })

    // 【关键修复】如果有当前图片，找到它在排序后的新位置
    if (currentImageId) {
      const newIndex = images.value.findIndex(img => img.id === currentImageId)
      if (newIndex >= 0 && newIndex !== currentImageIndex.value) {
        console.log(`图片排序: currentImageIndex 从 ${currentImageIndex.value} 更新为 ${newIndex}`)
        currentImageIndex.value = newIndex
      }
    }

    console.log('图片已按文件名排序')
  }

  // ============================================================
  // 图片切换方法
  // ============================================================

  /**
   * 设置当前图片索引
   * @param index - 新的索引
   */
  function setCurrentImageIndex(index: number): void {
    if (index >= -1 && index < images.value.length) {
      currentImageIndex.value = index
      console.log(`当前图片索引已设置为 ${index}`)
    } else {
      console.warn(`设置索引失败: 无效的索引 ${index}`)
    }
  }

  /**
   * 切换到上一张图片
   * @returns 是否切换成功
   */
  function goToPrevious(): boolean {
    if (canGoPrevious.value) {
      currentImageIndex.value--
      return true
    }
    return false
  }

  /**
   * 切换到下一张图片
   * @returns 是否切换成功
   */
  function goToNext(): boolean {
    if (canGoNext.value) {
      currentImageIndex.value++
      return true
    }
    return false
  }

  // ============================================================
  // 图片属性更新方法
  // ============================================================

  /**
   * 更新当前图片的属性
   * @param updates - 要更新的属性
   */
  function updateCurrentImage(updates: ImageDataUpdates): void {
    if (currentImage.value) {
      Object.assign(currentImage.value, updates)
      console.log('当前图片属性已更新:', Object.keys(updates))
    } else {
      console.warn('更新失败: 当前没有选中的图片')
    }
  }

  /**
   * 更新指定索引图片的属性
   * @param index - 图片索引
   * @param updates - 要更新的属性
   */
  function updateImageByIndex(index: number, updates: ImageDataUpdates): void {
    if (index >= 0 && index < images.value.length) {
      const image = images.value[index]
      if (image) {
        Object.assign(image, updates)
        console.log(`图片 ${index} 属性已更新:`, Object.keys(updates))
      }
    } else {
      console.warn(`更新失败: 无效的索引 ${index}`)
    }
  }

  /**
   * 更新当前图片的气泡状态
   * @param bubbleStates - 新的气泡状态数组
   */
  function updateCurrentBubbleStates(bubbleStates: BubbleState[] | null): void {
    if (currentImage.value) {
      currentImage.value.bubbleStates = bubbleStates
      currentImage.value.hasUnsavedChanges = true
      console.log(`当前图片气泡状态已更新，共 ${bubbleStates?.length ?? 0} 个气泡`)
    }
  }

  /**
   * 设置当前图片的手动标注标记
   * @param isManual - 是否经过手动标注
   */
  function setManuallyAnnotated(isManual: boolean): void {
    if (!currentImage.value) return

    currentImage.value.isManuallyAnnotated = isManual
    currentImage.value.hasUnsavedChanges = true
    console.log(`手动标注标记已设置为: ${isManual}`)
  }

  /**
   * 更新当前图片的单个属性
   * 迁移自 main.js 的 state.updateCurrentImageProperty
   * @param key - 属性名
   * @param value - 属性值
   */
  function updateCurrentImageProperty<K extends keyof ImageData>(
    key: K,
    value: ImageData[K]
  ): void {
    if (currentImage.value) {
      currentImage.value[key] = value
      currentImage.value.hasUnsavedChanges = true
      console.log(`当前图片属性 ${String(key)} 已更新`)
    } else {
      console.warn(`更新失败: 当前没有选中的图片`)
    }
  }

  /**
   * 更新当前图片的翻译结果
   * @param translatedDataURL - 翻译后的图片数据
   * @param cleanImageData - 干净背景图片数据（可选）
   */
  function updateCurrentTranslationResult(
    translatedDataURL: string,
    cleanImageData?: string
  ): void {
    if (currentImage.value) {
      currentImage.value.translatedDataURL = translatedDataURL
      if (cleanImageData) {
        currentImage.value.cleanImageData = cleanImageData
      }
      currentImage.value.translationStatus = 'completed'
      currentImage.value.translationFailed = false
      currentImage.value.hasUnsavedChanges = true
      console.log('当前图片翻译结果已更新')
    }
  }

  /**
   * 更新当前图片的尺寸
   * @param width - 图片宽度
   * @param height - 图片高度
   */
  function updateCurrentImageDimensions(width: number, height: number): void {
    if (currentImage.value) {
      currentImage.value.width = width
      currentImage.value.height = height
      console.log(`当前图片尺寸已更新: ${width} x ${height}`)
    }
  }

  // ============================================================
  // 翻译状态管理方法
  // ============================================================

  /**
   * 设置图片的翻译状态
   * @param index - 图片索引
   * @param status - 翻译状态
   * @param errorMessage - 错误信息（可选）
   */
  function setTranslationStatus(
    index: number,
    status: TranslationStatus,
    errorMessage?: string
  ): void {
    if (index >= 0 && index < images.value.length) {
      const image = images.value[index]
      if (image) {
        image.translationStatus = status
        image.translationFailed = status === 'failed'
        if (errorMessage) {
          image.errorMessage = errorMessage
        }
        console.log(`图片 ${index} 翻译状态: ${status}`)
      }
    }
  }

  /**
   * 标记当前图片翻译失败
   * @param errorMessage - 错误信息
   */
  function markCurrentAsFailed(errorMessage: string): void {
    if (currentImage.value) {
      currentImage.value.translationStatus = 'failed'
      currentImage.value.translationFailed = true
      currentImage.value.errorMessage = errorMessage
      console.log(`当前图片翻译失败: ${errorMessage}`)
    }
  }

  /**
   * 重置所有图片的翻译状态为待处理
   */
  function resetAllTranslationStatus(): void {
    images.value.forEach((img) => {
      img.translationStatus = 'pending'
      img.translationFailed = false
      img.errorMessage = undefined
    })
    console.log('所有图片翻译状态已重置')
  }

  // ============================================================
  // 批量翻译状态管理方法
  // ============================================================

  /**
   * 设置批量翻译进行状态
   * @param isInProgress - 是否正在进行
   */
  function setBatchTranslationInProgress(isInProgress: boolean): void {
    isBatchTranslationInProgress.value = isInProgress
    console.log(`批量翻译状态: ${isInProgress ? '进行中' : '已完成'}`)
  }

  /**
   * 获取所有失败的图片索引
   * @returns 失败图片的索引数组
   */
  function getFailedImageIndices(): number[] {
    return images.value
      .map((img, index) => (img.translationFailed ? index : -1))
      .filter((index) => index !== -1)
  }

  // ============================================================
  // 返回 Store
  // ============================================================

  return {
    // 状态
    images,
    currentImageIndex,
    isBatchTranslationInProgress,

    // 计算属性
    currentImage,
    imageCount,
    hasImages,
    canGoPrevious,
    canGoNext,
    failedImageCount,
    completedImageCount,
    pendingImageCount,

    // 图片管理方法
    addImage,
    addImages,
    setImages,
    deleteImage,
    deleteCurrentImage,
    clearImages,
    sortImagesByFileName,

    // 图片切换方法
    setCurrentImageIndex,
    goToPrevious,
    goToNext,

    // 图片属性更新方法
    updateCurrentImage,
    updateImageByIndex,
    updateCurrentBubbleStates,
    setManuallyAnnotated,
    updateCurrentImageProperty,
    updateCurrentTranslationResult,
    updateCurrentImageDimensions,

    // 翻译状态管理方法
    setTranslationStatus,
    markCurrentAsFailed,
    resetAllTranslationStatus,

    // 批量翻译状态管理方法
    setBatchTranslationInProgress,
    getFailedImageIndices
  }
})
