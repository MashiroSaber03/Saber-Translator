/**
 * 翻译功能组合式函数
 * 提供单张翻译、批量翻译、仅消除文字等核心功能
 * 
 * 拆分后的模块：
 * - translation/useHqTranslation.ts: 高质量翻译
 * - translation/useProofreading.ts: AI校对
 * - translation/types.ts: 类型定义
 * - translation/utils.ts: 工具函数
 */

import { ref, computed } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useValidation } from './useValidation'
import { useToast } from '@/utils/toast'
import { createRateLimiter, type RateLimiter } from '@/utils/rateLimiter'
import {
  translateImage as translateImageApi,
  type TranslateImageParams
} from '@/api/translate'
// BubbleState 类型通过 bubbleStore.setBubbles 间接使用

// 从拆分模块导入
import type { TranslationProgress, TranslationOptions } from './translation/types'
import {
  extractExistingBubbleData,
  buildTranslateParams,
  createBubbleStatesFromApiResponse
} from './translation/utils'
import { useHqTranslation } from './translation/useHqTranslation'
import { useProofreading } from './translation/useProofreading'

// 重新导出类型供外部使用
export type { TranslationProgress, TranslationOptions } from './translation/types'

// ============================================================
// 组合式函数
// ============================================================

/**
 * 翻译功能组合式函数
 */
export function useTranslation() {
  const imageStore = useImageStore()
  const settingsStore = useSettingsStore()
  const bubbleStore = useBubbleStore()
  const validation = useValidation()
  const toast = useToast()

  // 导入拆分的子模块
  const hqTranslation = useHqTranslation()
  const proofreading = useProofreading()

  // ============================================================
  // 状态
  // ============================================================

  /** 翻译限速器 */
  const translationLimiter = ref<RateLimiter | null>(null)

  /** 当前翻译进度 */
  const progress = ref<TranslationProgress>({
    current: 0,
    total: 0,
    completed: 0,
    failed: 0,
    isInProgress: false
  })

  /** 是否正在翻译单张图片 */
  const isTranslatingSingle = ref(false)

  // ============================================================
  // 计算属性
  // ============================================================

  /** 是否正在进行任何翻译操作 */
  const isTranslating = computed(() => {
    return (
      isTranslatingSingle.value ||
      imageStore.isBatchTranslationInProgress ||
      hqTranslation.isHqTranslating.value ||
      proofreading.isProofreading.value
    )
  })

  /** 翻译进度百分比 */
  const progressPercent = computed(() => {
    if (progress.value.total === 0) return 0
    return Math.round((progress.value.completed / progress.value.total) * 100)
  })

  // ============================================================
  // 工具函数
  // ============================================================

  /**
   * 初始化或更新限速器
   */
  function initRateLimiter(): void {
    const rpm = settingsStore.settings.translation.rpmLimit
    if (!translationLimiter.value) {
      translationLimiter.value = createRateLimiter(rpm)
    } else {
      translationLimiter.value.setRpm(rpm)
    }
  }

  /**
   * 处理翻译响应，更新图片状态
   */
  function handleTranslateResponse(
    index: number,
    response: any,
    _options: TranslationOptions = {}
  ): void {
    // 后端成功时返回 translated_image，失败时返回 error
    if (response.error || !response.translated_image) {
      imageStore.setTranslationStatus(index, 'failed', response.error || '翻译失败')
      return
    }

    // 创建气泡状态
    const bubbleStates = createBubbleStatesFromApiResponse(response)

    // 更新图片数据
    const translatedDataURL = response.translated_image
      ? `data:image/png;base64,${response.translated_image}`
      : null

    // 【重要】保存用户选择的排版方向（包括 'auto'），以便切换图片时恢复
    const userLayoutDirection = settingsStore.settings.textStyle.layoutDirection

    imageStore.updateImageByIndex(index, {
      translatedDataURL,
      cleanImageData: response.clean_image || null,
      bubbleStates,
      // 保存兼容数据（与原版一致）
      bubbleCoords: response.bubble_coords,
      bubbleAngles: response.bubble_angles || [],
      originalTexts: response.original_texts,
      textboxTexts: response.textbox_texts || [],
      bubbleTexts: bubbleStates?.map(s => s.translatedText || '') || [],
      userLayoutDirection,
      translationStatus: 'completed',
      translationFailed: false,
      hasUnsavedChanges: true
    })

    // 如果是当前图片，同步更新 bubbleStore
    if (index === imageStore.currentImageIndex && bubbleStates) {
      bubbleStore.setBubbles(bubbleStates)
    }
  }

  // ============================================================
  // 单张翻译
  // ============================================================

  /**
   * 翻译当前图片
   */
  async function translateCurrentImage(options: TranslationOptions = {}): Promise<boolean> {
    // 验证配置
    if (options.removeTextOnly) {
      if (!validation.validateBeforeTranslation('ocr')) {
        return false
      }
    } else {
      if (!validation.validateBeforeTranslation('normal')) {
        return false
      }
    }

    const currentImage = imageStore.currentImage
    if (!currentImage) {
      toast.error('请先上传图片')
      return false
    }

    isTranslatingSingle.value = true
    imageStore.setTranslationStatus(imageStore.currentImageIndex, 'processing')

    const translatingMsgId = toast.showGeneralMessage('翻译中...', 'info', false, 0)

    try {
      initRateLimiter()
      if (translationLimiter.value) {
        await translationLimiter.value.acquire()
      }

      // 提取已有坐标
      const translationOptions = { ...options }

      if (!translationOptions.existingBubbleCoords && !translationOptions.existingBubbleStates) {
        const existingData = extractExistingBubbleData(currentImage)
        if (existingData) {
          if (!existingData.isEmpty) {
            translationOptions.existingBubbleCoords = existingData.coords
            translationOptions.existingBubbleAngles = existingData.angles
            translationOptions.useExistingBubbles = true
            const sourceLabel = existingData.isManual ? '手动标注框' : '已有的文本框'
            console.log(`翻译当前图片: 使用 ${existingData.coords.length} 个${sourceLabel}`)
            toast.info(existingData.isManual
              ? '检测到手动标注框，将优先使用...'
              : '使用已有的文本框进行翻译...', 3000)
          } else {
            console.log('翻译当前图片: 文本框已被用户清空，将跳过翻译')
            toast.info('该图片无文本框，跳过翻译', 3000)
            translationOptions.existingBubbleCoords = []
            translationOptions.existingBubbleAngles = []
            translationOptions.useExistingBubbles = true
          }
        }
      }

      const params = buildTranslateParams(currentImage.originalDataURL, translationOptions)
      const response = await translateImageApi(params as unknown as TranslateImageParams)

      toast.clearGeneralMessageById(translatingMsgId)

      handleTranslateResponse(imageStore.currentImageIndex, response, options)

      if (response.translated_image && !response.error) {
        toast.success('翻译成功！')
        return true
      } else {
        toast.error(response.error || '翻译失败')
        return false
      }
    } catch (error) {
      toast.clearGeneralMessageById(translatingMsgId)
      const errorMessage = error instanceof Error ? error.message : '翻译请求失败'
      imageStore.markCurrentAsFailed(errorMessage)
      toast.error(errorMessage)
      return false
    } finally {
      isTranslatingSingle.value = false
    }
  }

  /**
   * 翻译指定索引的图片
   */
  async function translateImageByIndex(
    index: number,
    options: TranslationOptions = {}
  ): Promise<boolean> {
    const image = imageStore.images[index]
    if (!image) {
      console.error(`图片索引 ${index} 不存在`)
      return false
    }

    imageStore.setTranslationStatus(index, 'processing')

    try {
      if (translationLimiter.value) {
        await translationLimiter.value.acquire()
      }

      const translationOptions = { ...options }

      if (!translationOptions.existingBubbleCoords && !translationOptions.existingBubbleStates) {
        const existingData = extractExistingBubbleData(image)
        if (existingData) {
          if (!existingData.isEmpty) {
            translationOptions.existingBubbleCoords = existingData.coords
            translationOptions.existingBubbleAngles = existingData.angles
            translationOptions.useExistingBubbles = true
            const sourceLabel = existingData.isManual ? '手动标注框' : '已有的文本框'
            console.log(`批量翻译图片 ${index}: 使用 ${existingData.coords.length} 个${sourceLabel}`)
          } else {
            console.log(`批量翻译图片 ${index}: 文本框已被用户清空，跳过此图片的翻译`)
            translationOptions.existingBubbleCoords = []
            translationOptions.existingBubbleAngles = []
            translationOptions.useExistingBubbles = true
          }
        }
      }

      const params = buildTranslateParams(image.originalDataURL, translationOptions)
      const response = await translateImageApi(params as unknown as TranslateImageParams)

      handleTranslateResponse(index, response, options)
      return !!response.translated_image && !response.error
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '翻译请求失败'
      imageStore.setTranslationStatus(index, 'failed', errorMessage)
      return false
    }
  }

  // ============================================================
  // 批量翻译
  // ============================================================

  /**
   * 翻译所有图片
   */
  async function translateAllImages(options: TranslationOptions = {}): Promise<boolean> {
    if (options.removeTextOnly) {
      if (!validation.validateBeforeTranslation('ocr')) {
        return false
      }
    } else {
      if (!validation.validateBeforeTranslation('normal')) {
        return false
      }
    }

    const images = imageStore.images
    if (images.length === 0) {
      toast.error('请先上传图片')
      return false
    }

    const isRemoveTextMode = options.removeTextOnly === true

    progress.value = {
      current: 0,
      total: images.length,
      completed: 0,
      failed: 0,
      isInProgress: true,
      label: isRemoveTextMode ? `消除文字: 0/${images.length}` : `0/${images.length}`,
      percentage: 0
    }

    imageStore.setBatchTranslationInProgress(true)
    initRateLimiter()

    toast.info(isRemoveTextMode ? `开始消除 ${images.length} 张图片文字...` : `开始批量翻译 ${images.length} 张图片...`)

    let allSuccess = true

    try {
      for (let i = 0; i < images.length; i++) {
        // 检查是否已取消
        if (!imageStore.isBatchTranslationInProgress) {
          console.log('批量翻译已取消')
          break
        }

        progress.value.current = i + 1
        progress.value.label = isRemoveTextMode
          ? `消除文字: ${i + 1}/${images.length}`
          : `${i + 1}/${images.length}`
        progress.value.percentage = Math.round(((i + 1) / images.length) * 100)

        const success = await translateImageByIndex(i, options)

        if (success) {
          progress.value.completed++
        } else {
          progress.value.failed++
          allSuccess = false
        }
      }

      if (progress.value.failed > 0) {
        toast.warning(isRemoveTextMode
          ? `消除文字完成，${progress.value.failed} 张图片失败`
          : `翻译完成，${progress.value.failed} 张图片失败`)
      } else {
        toast.success(isRemoveTextMode ? '所有图片文字消除完成' : '所有图片翻译完成')
      }

      return allSuccess
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '批量翻译失败'
      toast.error(errorMessage)
      return false
    } finally {
      imageStore.setBatchTranslationInProgress(false)

      // 刷新当前显示的图片
      const currentIndex = imageStore.currentImageIndex
      if (currentIndex >= 0 && currentIndex < imageStore.images.length) {
        const currentImage = imageStore.images[currentIndex]
        if (currentImage) {
          if (currentImage.bubbleStates && currentImage.bubbleStates.length > 0) {
            bubbleStore.setBubbles(currentImage.bubbleStates)
          }
          imageStore.setCurrentImageIndex(currentIndex)
        }
      }

      setTimeout(() => {
        progress.value.isInProgress = false
      }, 1000)
    }
  }

  /**
   * 取消批量翻译
   */
  function cancelBatchTranslation(): void {
    if (imageStore.isBatchTranslationInProgress) {
      imageStore.setBatchTranslationInProgress(false)
      progress.value.isInProgress = false
      toast.info('批量翻译已取消')
    }
  }

  // ============================================================
  // 仅消除文字
  // ============================================================

  /**
   * 仅消除当前图片文字（不翻译）
   */
  async function removeTextOnly(): Promise<boolean> {
    return translateCurrentImage({ removeTextOnly: true })
  }

  /**
   * 消除所有图片文字（不翻译）
   */
  async function removeAllTexts(): Promise<boolean> {
    return translateAllImages({ removeTextOnly: true })
  }

  // ============================================================
  // 重新翻译失败图片
  // ============================================================

  /**
   * 重新翻译所有失败的图片
   */
  async function retryFailedImages(options: TranslationOptions = {}): Promise<boolean> {
    if (!validation.validateBeforeTranslation('normal')) {
      return false
    }

    const failedIndices = imageStore.getFailedImageIndices()
    if (failedIndices.length === 0) {
      toast.info('没有失败的图片需要重新翻译')
      return true
    }

    progress.value = {
      current: 0,
      total: failedIndices.length,
      completed: 0,
      failed: 0,
      isInProgress: true
    }

    imageStore.setBatchTranslationInProgress(true)
    initRateLimiter()

    let allSuccess = true

    try {
      for (let i = 0; i < failedIndices.length; i++) {
        if (!imageStore.isBatchTranslationInProgress) {
          break
        }

        progress.value.current = i + 1
        const imageIndex = failedIndices[i]

        if (imageIndex === undefined) continue

        const success = await translateImageByIndex(imageIndex, options)

        if (success) {
          progress.value.completed++
        } else {
          progress.value.failed++
          allSuccess = false
        }
      }

      if (progress.value.failed > 0) {
        toast.warning(`重试完成，仍有 ${progress.value.failed} 张图片失败`)
      } else {
        toast.success('所有失败图片重新翻译完成')
      }

      return allSuccess
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '重试翻译失败'
      toast.error(errorMessage)
      return false
    } finally {
      imageStore.setBatchTranslationInProgress(false)
      setTimeout(() => {
        progress.value.isInProgress = false
      }, 1000)
    }
  }

  // ============================================================
  // 使用已有气泡框翻译
  // ============================================================

  /**
   * 使用当前手动标注的气泡框进行翻译
   */
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

    return translateCurrentImage({
      useExistingBubbles: true,
      existingBubbleStates: bubbles
    })
  }

  // ============================================================
  // 返回
  // ============================================================

  return {
    // 状态
    progress,
    isTranslatingSingle,
    isHqTranslating: hqTranslation.isHqTranslating,
    isProofreading: proofreading.isProofreading,

    // 计算属性
    isTranslating,
    progressPercent,

    // 单张翻译
    translateCurrentImage,
    translateImageByIndex,

    // 批量翻译
    translateAllImages,
    cancelBatchTranslation,

    // 仅消除文字
    removeTextOnly,
    removeAllTexts,

    // 重新翻译失败图片
    retryFailedImages,

    // 高质量翻译（从子模块导入）
    executeHqTranslation: hqTranslation.executeHqTranslation,

    // AI 校对（从子模块导入）
    executeProofreading: proofreading.executeProofreading,

    // 使用已有气泡框翻译
    translateWithCurrentBubbles
  }
}
