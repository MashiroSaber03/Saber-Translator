/**
 * 翻译功能组合式函数
 * 提供单张翻译、批量翻译、高质量翻译、AI校对等功能
 * 
 * 功能：
 * - 单张翻译（调用 /api/translate_image）
 * - 批量翻译（支持暂停/继续）
 * - 仅消除文字功能
 * - 高质量翻译模式（批量上下文翻译）
 * - AI校对功能（多轮校对）
 * - 重新翻译失败图片
 * - 翻译状态管理
 * - RPM限速器实现
 */

import { ref, computed } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useValidation } from './useValidation'
import { useToast } from '@/utils/toast'
import { createRateLimiter, type RateLimiter } from '@/utils/rateLimiter'
import { createBubbleStatesFromResponse } from '@/utils/bubbleFactory'
import {
  translateImage as translateImageApi,
  hqTranslateBatch as hqTranslateBatchApi,
  type TranslateImageParams,
  type HqTranslateParams
} from '@/api/translate'
import type { BubbleState, BubbleCoords, TextDirection } from '@/types/bubble'
import type { TranslateImageResponse } from '@/types/api'
import type { ProofreadingRound } from '@/types/settings'
import type { ImageData as AppImageData } from '@/types/image'

// ============================================================
// 类型定义
// ============================================================

/** 翻译进度信息 */
export interface TranslationProgress {
  /** 当前处理的图片索引 */
  current: number
  /** 总图片数 */
  total: number
  /** 已完成数量 */
  completed: number
  /** 失败数量 */
  failed: number
  /** 是否正在进行 */
  isInProgress: boolean
  /** 是否暂停 */
  isPaused: boolean
  /** 自定义进度标签（复刻原版） */
  label?: string
  /** 进度百分比（0-100，用于精确控制进度条） */
  percentage?: number
}

/** 翻译选项 */
export interface TranslationOptions {
  /** 仅消除文字，不翻译 */
  removeTextOnly?: boolean
  /** 使用已有气泡框 */
  useExistingBubbles?: boolean
  /**
   * 已有气泡状态数组（推荐使用）
   * 从中自动提取坐标和角度，避免遗漏参数
   */
  existingBubbleStates?: BubbleState[]
  /** 已有气泡坐标（兼容旧接口，优先使用 existingBubbleStates） */
  existingBubbleCoords?: BubbleCoords[]
  /** 已有气泡角度（兼容旧接口，优先使用 existingBubbleStates） */
  existingBubbleAngles?: number[]
}


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
    isInProgress: false,
    isPaused: false
  })

  /** 是否正在翻译单张图片 */
  const isTranslatingSingle = ref(false)

  /** 是否正在进行高质量翻译 */
  const isHqTranslating = ref(false)

  /** 是否正在进行AI校对 */
  const isProofreading = ref(false)

  // ============================================================
  // 计算属性
  // ============================================================

  /** 是否正在进行任何翻译操作 */
  const isTranslating = computed(() => {
    return (
      isTranslatingSingle.value ||
      imageStore.isBatchTranslationInProgress ||
      isHqTranslating.value ||
      isProofreading.value
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
   * 从 DataURL 中提取纯 Base64 数据
   * @param dataUrl - 完整的 DataURL (data:image/...;base64,...)
   * @returns 纯 Base64 数据
   */
  function extractBase64FromDataUrl(dataUrl: string): string {
    if (dataUrl.includes(',')) {
      return dataUrl.split(',')[1] || dataUrl
    }
    return dataUrl
  }

  /**
   * 从图片数据中提取已有的气泡坐标和角度
   * 
   * 简化逻辑：
   * - isManuallyAnnotated = true → 用户手动操作过（使用 bubbleStates，跳过自动检测）
   * - isManuallyAnnotated = false/undefined → 未手动操作（可能有自动检测结果或需要检测）
   * 
   * @param image - 图片数据
   * @returns { coords, angles, isEmpty, isManual } 或 null（表示需要自动检测）
   */
  function extractExistingBubbleData(image: AppImageData): {
    coords: BubbleCoords[]
    angles: number[] | undefined
    isEmpty: boolean
    isManual: boolean
  } | null {
    // 1. 如果用户手动操作过，优先使用 bubbleStates（包括空数组）
    if (image.isManuallyAnnotated) {
      const bubbles = image.bubbleStates || []
      return {
        coords: bubbles.map(s => s.coords),
        angles: bubbles.map(s => s.rotationAngle || 0),
        isEmpty: bubbles.length === 0,
        isManual: true
      }
    }

    // 2. 如果有 bubbleStates（自动检测结果）
    if (Array.isArray(image.bubbleStates) && image.bubbleStates.length > 0) {
      return {
        coords: image.bubbleStates.map(s => s.coords),
        angles: image.bubbleStates.map(s => s.rotationAngle || 0),
        isEmpty: false,
        isManual: false
      }
    }

    // 3. 兼容旧数据：回退到 bubbleCoords
    if (Array.isArray(image.bubbleCoords) && image.bubbleCoords.length > 0) {
      return {
        coords: image.bubbleCoords,
        angles: image.bubbleAngles,
        isEmpty: false,
        isManual: false
      }
    }

    // 4. 没有任何已有数据，需要自动检测
    return null
  }

  /**
   * 构建翻译请求参数
   * 注意：参数名称需要与后端 API 期望的名称一致
   * @param imageData - 图片 Base64 数据（可能包含 DataURL 前缀）
   * @param options - 翻译选项
   * @returns 翻译请求参数
   */
  function buildTranslateParams(
    imageData: string,
    options: TranslationOptions = {}
  ): Record<string, unknown> {
    const { settings } = settingsStore
    const { textStyle, translation, baiduOcr, aiVisionOcr, boxExpand, preciseMask } = settings

    // 使用后端期望的参数名称（驼峰命名）
    const params: Record<string, unknown> = {
      // 图片数据（去除 DataURL 前缀，后端期望纯 Base64）
      image: extractBase64FromDataUrl(imageData),

      // OCR 设置
      ocr_engine: settings.ocrEngine,
      source_language: settings.sourceLanguage,

      // 翻译服务设置（后端使用 model_provider, model_name, api_key）
      model_provider: translation.provider,
      api_key: translation.apiKey,
      model_name: translation.modelName,
      custom_base_url: translation.customBaseUrl,
      target_language: settings.targetLanguage,
      prompt_content: settings.translatePrompt,
      textbox_prompt_content: settings.textboxPrompt,
      use_textbox_prompt: settings.useTextboxPrompt,
      // 与原版保持一致的参数名称
      rpm_limit_translation: translation.rpmLimit,
      max_retries: translation.maxRetries,
      use_json_format_translation: translation.isJsonMode,

      // 文字检测设置（后端使用 detector_type）
      detector_type: settings.textDetector,
      box_expand_ratio: boxExpand.ratio,
      box_expand_top: boxExpand.top,
      box_expand_bottom: boxExpand.bottom,
      box_expand_left: boxExpand.left,
      box_expand_right: boxExpand.right,

      // 精确文字掩膜设置（后端使用驼峰命名：usePreciseMask, maskDilateSize, maskBoxExpandRatio）
      usePreciseMask: preciseMask.enabled,
      maskDilateSize: preciseMask.dilateSize,
      maskBoxExpandRatio: preciseMask.boxExpandRatio,

      // 文字样式设置（后端使用驼峰命名）
      // 【修复Vue版逻辑】传递实际字号数字而非'auto'字符串（避免后端警告）
      // 后端通过autoFontSize标记启用自动计算，会忽略fontSize数值
      fontSize: textStyle.fontSize,  // 始终传递数字
      autoFontSize: textStyle.autoFontSize,
      fontFamily: textStyle.fontFamily,
      // 处理排版方向：如果是 "auto" 则启用自动排版，传递 'vertical' 作为默认值
      // 与原版 main.js 保持一致的处理逻辑
      textDirection: textStyle.layoutDirection === 'auto' ? 'vertical' : textStyle.layoutDirection,
      autoTextDirection: textStyle.layoutDirection === 'auto',  // 自动排版开关
      textColor: textStyle.textColor,
      fillColor: textStyle.fillColor,
      strokeEnabled: textStyle.strokeEnabled,
      strokeColor: textStyle.strokeColor,
      strokeWidth: textStyle.strokeWidth,

      // 修复方式（与原版 ui.getRepairSettings() 保持一致）
      // inpaintMethod: 'solid' | 'lama_mpe' | 'litelama'
      use_inpainting: false,  // MI-GAN (保留兼容，已废弃)
      use_lama: textStyle.inpaintMethod === 'lama_mpe' || textStyle.inpaintMethod === 'litelama',
      lamaModel: textStyle.inpaintMethod === 'litelama' ? 'litelama' : 'lama_mpe',

      // 调试选项（后端使用 camelCase: showDetectionDebug）
      showDetectionDebug: settings.showDetectionDebug,

      // 特殊模式
      remove_only: options.removeTextOnly,
      skip_translation: options.removeTextOnly  // 仅消除文字时跳过翻译
    }

    // 【优化】从 existingBubbleStates 自动提取坐标和角度（避免遗漏参数）
    // 优先使用 existingBubbleStates，回退到旧的分离参数
    // 【修复】确保坐标是整数（手动绘制的气泡可能产生浮点数坐标，后端 OCR 需要整数）
    const normalizeCoords = (coords: BubbleCoords[]): BubbleCoords[] => {
      return coords.map(c => [
        Math.round(c[0]),
        Math.round(c[1]),
        Math.round(c[2]),
        Math.round(c[3])
      ] as BubbleCoords)
    }

    if (options.existingBubbleStates && options.existingBubbleStates.length > 0) {
      params.bubble_coords = normalizeCoords(options.existingBubbleStates.map((s) => s.coords))
      params.bubble_angles = options.existingBubbleStates.map((s) => s.rotationAngle || 0)
    } else if (options.existingBubbleCoords) {
      params.bubble_coords = normalizeCoords(options.existingBubbleCoords)
      params.bubble_angles = options.existingBubbleAngles
    }

    // 百度 OCR 设置（后端使用 baidu_api_key, baidu_secret_key）
    if (settings.ocrEngine === 'baidu_ocr') {
      params.baidu_api_key = baiduOcr.apiKey
      params.baidu_secret_key = baiduOcr.secretKey
      params.baidu_version = baiduOcr.version
      params.baidu_ocr_language = baiduOcr.sourceLanguage
    }

    // AI 视觉 OCR 设置
    if (settings.ocrEngine === 'ai_vision') {
      params.ai_vision_provider = aiVisionOcr.provider
      params.ai_vision_api_key = aiVisionOcr.apiKey
      params.ai_vision_model_name = aiVisionOcr.modelName
      params.ai_vision_ocr_prompt = aiVisionOcr.prompt
      params.rpm_limit_ai_vision_ocr = aiVisionOcr.rpmLimit
      params.custom_ai_vision_base_url = aiVisionOcr.customBaseUrl
      params.use_json_format_ai_vision_ocr = aiVisionOcr.isJsonMode
    }

    return params
  }


  /**
   * 处理翻译响应，更新图片状态
   * @param index - 图片索引
   * @param response - 翻译响应
   * @param _options - 翻译选项（保留用于未来扩展）
   */
  function handleTranslateResponse(
    index: number,
    response: TranslateImageResponse,
    _options: TranslationOptions = {}
  ): void {
    // 后端成功时返回 translated_image，失败时返回 error
    // 注意：后端成功响应不包含 success 字段
    if (response.error || !response.translated_image) {
      imageStore.setTranslationStatus(index, 'failed', response.error || '翻译失败')
      return
    }

    const { settings } = settingsStore
    const { textStyle } = settings

    // 创建气泡状态
    let bubbleStates: BubbleState[] | null = null
    if (response.bubble_coords && response.bubble_coords.length > 0) {
      // 构建 API 响应对象
      // 【复刻原版】包含 auto_directions 字段，用于后端基于文本行分析的排版方向
      const apiResponse = {
        bubble_coords: response.bubble_coords,
        bubble_states: response.bubble_states,
        original_texts: response.original_texts,
        bubble_texts: response.bubble_texts,
        textbox_texts: response.textbox_texts,
        bubble_angles: response.bubble_angles,
        auto_directions: response.auto_directions  // 后端基于文本行分析的排版方向
      }

      // 构建全局默认设置
      const globalDefaults = {
        fontSize: textStyle.fontSize,
        fontFamily: textStyle.fontFamily,
        textDirection: textStyle.layoutDirection === 'auto' ? 'vertical' as const : textStyle.layoutDirection,
        textColor: textStyle.textColor,
        fillColor: textStyle.fillColor,
        strokeEnabled: textStyle.strokeEnabled,
        strokeColor: textStyle.strokeColor,
        strokeWidth: textStyle.strokeWidth,
        inpaintMethod: textStyle.inpaintMethod
      }

      bubbleStates = createBubbleStatesFromResponse(apiResponse, globalDefaults)
    }

    // 更新图片数据
    // 后端返回的是纯 Base64 数据，需要添加 DataURL 前缀
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
   * @param options - 翻译选项
   * @returns 是否成功
   */
  async function translateCurrentImage(options: TranslationOptions = {}): Promise<boolean> {
    // 验证配置：仅消除文字模式只校验 OCR，普通翻译校验全部
    if (options.removeTextOnly) {
      // 仅消除文字模式：只校验 OCR 配置（与原版一致）
      if (!validation.validateBeforeTranslation('ocr')) {
        return false
      }
    } else {
      // 普通翻译模式：校验翻译服务配置
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

    // 显示翻译中提示（与原版一致，duration=0 表示持续显示直到手动清除）
    const translatingMsgId = toast.showGeneralMessage('翻译中...', 'info', false, 0)

    try {
      initRateLimiter()
      if (translationLimiter.value) {
        await translationLimiter.value.acquire()
      }

      // --- 关键逻辑：使用统一的函数提取已有坐标（与原版 main.js 一致）---
      const translationOptions = { ...options }

      // 只在未显式传入 existingBubbleCoords 时才从图片数据中提取
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
            // 用户清空了文本框，显示消息但仍传递空数组
            console.log('翻译当前图片: 文本框已被用户清空，将跳过翻译')
            toast.info('该图片无文本框，跳过翻译', 3000)
            translationOptions.existingBubbleCoords = []
            translationOptions.existingBubbleAngles = []
            translationOptions.useExistingBubbles = true
          }
        }
      }
      // ------------------------------------------

      const params = buildTranslateParams(currentImage.originalDataURL, translationOptions)
      const response = await translateImageApi(params as unknown as TranslateImageParams)

      // 清除翻译中提示
      toast.clearGeneralMessageById(translatingMsgId)

      handleTranslateResponse(imageStore.currentImageIndex, response, options)

      // 后端成功时返回 translated_image，失败时返回 error
      if (response.translated_image && !response.error) {
        toast.success('翻译成功！')
        return true
      } else {
        toast.error(response.error || '翻译失败')
        return false
      }
    } catch (error) {
      // 清除翻译中提示
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
   * @param index - 图片索引
   * @param options - 翻译选项
   * @returns 是否成功
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

      // --- 使用统一的函数提取已有坐标（与原版批量翻译一致）---
      const translationOptions = { ...options }

      // 只在未显式传入 existingBubbleCoords 时才从图片数据中提取
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
            // 用户清空了文本框
            console.log(`批量翻译图片 ${index}: 文本框已被用户清空，跳过此图片的翻译`)
            translationOptions.existingBubbleCoords = []
            translationOptions.existingBubbleAngles = []
            translationOptions.useExistingBubbles = true
          }
        }
      }
      // ------------------------------------------

      const params = buildTranslateParams(image.originalDataURL, translationOptions)
      const response = await translateImageApi(params as unknown as TranslateImageParams)

      handleTranslateResponse(index, response, options)
      // 后端成功时返回 translated_image，失败时返回 error
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
   * @param options - 翻译选项
   * @returns 是否全部成功
   */
  async function translateAllImages(options: TranslationOptions = {}): Promise<boolean> {
    // 验证配置：仅消除文字模式只校验 OCR，普通翻译校验全部
    if (options.removeTextOnly) {
      // 仅消除文字模式：只校验 OCR 配置（与原版一致）
      if (!validation.validateBeforeTranslation('ocr')) {
        return false
      }
    } else {
      // 普通翻译模式：校验翻译服务配置
      if (!validation.validateBeforeTranslation('normal')) {
        return false
      }
    }

    const images = imageStore.images
    if (images.length === 0) {
      toast.error('请先上传图片')
      return false
    }

    // 判断是否为消除文字模式 - 复刻原版
    const isRemoveTextMode = options.removeTextOnly === true

    // 初始化进度 - 复刻原版
    progress.value = {
      current: 0,
      total: images.length,
      completed: 0,
      failed: 0,
      isInProgress: true,
      isPaused: false,
      label: isRemoveTextMode ? `消除文字: 0/${images.length}` : `0/${images.length}`,
      percentage: 0
    }

    imageStore.setBatchTranslationInProgress(true)
    initRateLimiter()

    // 显示批量翻译开始提示 - 复刻原版
    toast.info(isRemoveTextMode ? `开始消除 ${images.length} 张图片文字...` : `开始批量翻译 ${images.length} 张图片...`)

    let allSuccess = true

    try {
      for (let i = 0; i < images.length; i++) {
        // 检查是否暂停
        if (imageStore.isBatchTranslationPaused) {
          // 等待继续
          await new Promise<void>((resolve) => {
            imageStore.setBatchTranslationResumeCallback(resolve)
          })
        }

        // 检查是否已取消（批量翻译不再进行中）
        if (!imageStore.isBatchTranslationInProgress) {
          console.log('批量翻译已取消')
          break
        }

        progress.value.current = i + 1
        // 更新进度标签和百分比 - 复刻原版
        progress.value.label = isRemoveTextMode
          ? `消除文字: ${i + 1}/${images.length}`
          : `${i + 1}/${images.length}`
        progress.value.percentage = Math.round(((i + 1) / images.length) * 100)

        // 【复刻原版】不跳过已完成的图片，点击"翻译所有图片"应重新翻译所有内容
        // 原版 main.js 中是没有 check completion logic 的，每次都是重头开始翻译

        // 翻译图片
        const success = await translateImageByIndex(i, options)

        if (success) {
          progress.value.completed++
        } else {
          progress.value.failed++
          allSuccess = false
        }
      }

      // 完成 - 复刻原版
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

      // 【复刻原版】批量翻译完成后，刷新当前显示的图片
      // 参考原版 main.js 行 1244-1249：重新显示当前图片，触发 UI 刷新显示最新渲染结果
      const currentIndex = imageStore.currentImageIndex
      if (currentIndex >= 0 && currentIndex < imageStore.images.length) {
        // 刷新当前图片的 bubbleStore 和 imageStore
        const currentImage = imageStore.images[currentIndex]
        if (currentImage) {
          // 更新 bubbleStore 以刷新气泡显示
          if (currentImage.bubbleStates && currentImage.bubbleStates.length > 0) {
            bubbleStore.setBubbles(currentImage.bubbleStates)
          }
          // 触发 imageStore 的响应式更新（通过重新设置当前索引）
          imageStore.setCurrentImageIndex(currentIndex)
        }
      }

      // 【复刻原版】完成后延迟1秒再隐藏进度条，让用户看清完成状态
      // 参考原版 ui.js 中的 setTimeout(() => translationProgressBar.hide(), 1000)
      setTimeout(() => {
        progress.value.isInProgress = false
      }, 1000)
    }
  }

  /**
   * 暂停批量翻译
   */
  function pauseBatchTranslation(): void {
    if (imageStore.isBatchTranslationInProgress && !imageStore.isBatchTranslationPaused) {
      imageStore.setBatchTranslationPaused(true)
      progress.value.isPaused = true
      toast.info('批量翻译已暂停')
    }
  }

  /**
   * 继续批量翻译
   */
  function resumeBatchTranslation(): void {
    if (imageStore.isBatchTranslationInProgress && imageStore.isBatchTranslationPaused) {
      imageStore.resumeBatchTranslation()
      progress.value.isPaused = false
      toast.info('批量翻译继续中')
    }
  }

  /**
   * 取消批量翻译
   */
  function cancelBatchTranslation(): void {
    if (imageStore.isBatchTranslationInProgress) {
      // 如果暂停中，先恢复以便退出等待
      if (imageStore.isBatchTranslationPaused) {
        imageStore.resumeBatchTranslation()
      }
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
   * @returns 是否成功
   */
  async function removeTextOnly(): Promise<boolean> {
    return translateCurrentImage({ removeTextOnly: true })
  }

  /**
   * 消除所有图片文字（不翻译）
   * @returns 是否全部成功
   */
  async function removeAllTexts(): Promise<boolean> {
    return translateAllImages({ removeTextOnly: true })
  }

  // ============================================================
  // 重新翻译失败图片
  // ============================================================

  /**
   * 重新翻译所有失败的图片
   * @param options - 翻译选项
   * @returns 是否全部成功
   */
  async function retryFailedImages(options: TranslationOptions = {}): Promise<boolean> {
    // 验证配置
    if (!validation.validateBeforeTranslation('normal')) {
      return false
    }

    const failedIndices = imageStore.getFailedImageIndices()
    if (failedIndices.length === 0) {
      toast.info('没有失败的图片需要重新翻译')
      return true
    }

    // 初始化进度
    progress.value = {
      current: 0,
      total: failedIndices.length,
      completed: 0,
      failed: 0,
      isInProgress: true,
      isPaused: false
    }

    imageStore.setBatchTranslationInProgress(true)
    initRateLimiter()

    let allSuccess = true

    try {
      for (let i = 0; i < failedIndices.length; i++) {
        // 检查是否暂停
        if (imageStore.isBatchTranslationPaused) {
          await new Promise<void>((resolve) => {
            imageStore.setBatchTranslationResumeCallback(resolve)
          })
        }

        // 检查是否已取消
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
      // 【复刻原版】完成后延迟1秒再隐藏进度条
      setTimeout(() => {
        progress.value.isInProgress = false
      }, 1000)
    }
  }


  // ============================================================
  // 高质量翻译（完全复刻原版 high_quality_translation.js）
  // ============================================================

  /** 保存翻译前的文本样式设置（复刻原版） */
  interface SavedTextStyles {
    fontFamily: string
    fontSize: number
    autoFontSize: boolean
    textDirection: string
    autoTextDirection: boolean
    fillColor: string
    textColor: string
    rotationAngle: number
    strokeEnabled: boolean
    strokeColor: string
    strokeWidth: number
  }

  /** 高质量翻译的JSON数据格式（复刻原版 exportTextToJson） */
  interface HqJsonData {
    imageIndex: number
    bubbles: Array<{
      bubbleIndex: number
      original: string
      translated: string
      textDirection: string
    }>
  }

  /** 所有批次结果（复刻原版） */
  let allBatchResults: HqJsonData[][] = []

  /** 保存的文本样式（复刻原版） */
  let savedTextStyles: SavedTextStyles | null = null

  /**
   * 生成会话ID（复刻原版）
   */
  function generateSessionId(): string {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substring(2, 9)
  }

  /**
   * 导出文本为JSON（完全复刻原版 exportTextToJson）
   */
  function exportTextToJson(): HqJsonData[] | null {
    const allImages = imageStore.images
    if (allImages.length === 0) return null

    const exportData: HqJsonData[] = []

    for (let imageIndex = 0; imageIndex < allImages.length; imageIndex++) {
      const image = allImages[imageIndex]
      if (!image) continue

      const originalTexts = image.originalTexts || []
      const imageTextData: HqJsonData = {
        imageIndex: imageIndex,
        bubbles: []
      }

      for (let bubbleIndex = 0; bubbleIndex < originalTexts.length; bubbleIndex++) {
        const original = originalTexts[bubbleIndex] || ''

        // 获取气泡的排版方向（复刻原版逻辑）
        let textDirection = 'vertical'
        const bubbleState = image.bubbleStates?.[bubbleIndex]
        if (bubbleState && bubbleState.textDirection) {
          const bubbleDir = bubbleState.textDirection
          textDirection = (bubbleDir === 'auto') ? 'vertical' : bubbleDir
        } else if (image.userLayoutDirection && image.userLayoutDirection !== 'auto') {
          textDirection = image.userLayoutDirection
        }

        imageTextData.bubbles.push({
          bubbleIndex: bubbleIndex,
          original: original,
          translated: '',
          textDirection: textDirection
        })
      }

      exportData.push(imageTextData)
    }

    return exportData
  }

  /**
   * 收集所有图片的Base64数据（复刻原版 collectAllImageBase64）
   */
  function collectAllImageBase64(): string[] {
    return imageStore.images.map(image => {
      const dataUrl = image.originalDataURL
      if (dataUrl.includes(',')) {
        return dataUrl.split(',')[1] || ''
      }
      return dataUrl
    })
  }

  /**
   * 为特定批次过滤JSON数据（复刻原版 filterJsonForBatch）
   */
  function filterJsonForBatch(jsonData: HqJsonData[], startIdx: number, endIdx: number): HqJsonData[] {
    return jsonData.filter(item => item.imageIndex >= startIdx && item.imageIndex < endIdx)
  }

  /**
   * 合并所有批次的JSON结果（复刻原版 mergeJsonResults）
   */
  function mergeJsonResults(batchResults: HqJsonData[][]): HqJsonData[] {
    if (!batchResults || batchResults.length === 0) {
      return []
    }

    const mergedResult: HqJsonData[] = []

    for (const batchResult of batchResults) {
      if (!batchResult) {
        console.warn('mergeJsonResults: 跳过空的批次结果')
        continue
      }

      const batchArray = Array.isArray(batchResult) ? batchResult : [batchResult]

      for (const imageData of batchArray) {
        if (imageData && typeof imageData === 'object' && 'imageIndex' in imageData) {
          mergedResult.push(imageData)
        } else {
          console.warn('mergeJsonResults: 跳过无效的图片数据', imageData)
        }
      }
    }

    mergedResult.sort((a, b) => a.imageIndex - b.imageIndex)
    return mergedResult
  }

  /**
   * 调用AI进行翻译（复刻原版 callAiForTranslation）
   */
  async function callAiForTranslation(
    imageBase64Array: string[],
    jsonData: HqJsonData[],
    _sessionId: string
  ): Promise<HqJsonData[] | null> {
    const { hqTranslation } = settingsStore.settings
    const jsonString = JSON.stringify(jsonData, null, 2)

    // 构建消息（复刻原版格式）
    type MessageContent = { type: 'text'; text: string } | { type: 'image_url'; image_url: { url: string } }
    const userContent: MessageContent[] = [
      {
        type: 'text',
        text: hqTranslation.prompt + '\n\n以下是JSON数据:\n```json\n' + jsonString + '\n```'
      }
    ]

    // 添加图片到消息中
    for (const imgBase64 of imageBase64Array) {
      userContent.push({
        type: 'image_url',
        image_url: {
          url: `data:image/png;base64,${imgBase64}`
        }
      })
    }

    const messages: HqTranslateParams['messages'] = [
      {
        role: 'system',
        content: '你是一个专业的漫画翻译助手，能够根据漫画图像内容和上下文提供高质量的翻译。'
      },
      {
        role: 'user',
        content: userContent
      }
    ]

    // 构建请求参数（复刻原版格式）
    const params: HqTranslateParams = {
      provider: hqTranslation.provider,
      api_key: hqTranslation.apiKey,
      model_name: hqTranslation.modelName,
      custom_base_url: hqTranslation.customBaseUrl,
      messages: messages,
      low_reasoning: hqTranslation.lowReasoning,
      force_json_output: hqTranslation.forceJsonOutput,
      no_thinking_method: hqTranslation.noThinkingMethod,
      use_stream: hqTranslation.useStream
    }

    try {
      console.log(`高质量翻译: 通过后端代理调用 ${hqTranslation.provider} API...`)

      const response = await hqTranslateBatchApi(params)

      if (!response.success) {
        throw new Error(response.error || 'API 调用失败')
      }

      // 优先使用后端已解析的 results（后端会尝试解析 LLM 返回的 JSON）
      // 格式为: [{ imageIndex, bubbles: [{ bubbleIndex, original, translated, textDirection }] }]
      if (response.results && response.results.length > 0) {
        const firstItem = response.results[0]
        // 验证结构正确性
        if (firstItem && 'imageIndex' in firstItem && 'bubbles' in firstItem) {
          return response.results as unknown as HqJsonData[]
        }
      }

      // 如果 results 不存在或格式不对，使用 content（复刻原版逻辑）
      const content = (response as any).content
      if (content) {
        if (hqTranslation.forceJsonOutput) {
          try {
            return JSON.parse(content)
          } catch (e) {
            console.error('解析AI强制JSON返回的内容失败:', e)
            throw new Error('解析AI返回的JSON结果失败')
          }
        } else {
          // 从 markdown 代码块中提取 JSON
          const jsonMatch = content.match(/```json\s*([\s\S]*?)\s*```/)
          if (jsonMatch && jsonMatch[1]) {
            try {
              return JSON.parse(jsonMatch[1])
            } catch (e) {
              console.error('解析AI返回的JSON失败:', e)
              throw new Error('解析AI返回的翻译结果失败')
            }
          }
        }
      }

      return null
    } catch (error) {
      console.error('调用AI翻译API失败:', error)
      throw error
    }
  }

  /**
   * 执行高质量翻译（完全复刻原版 startHqTranslation）
   */
  async function executeHqTranslation(): Promise<boolean> {
    // 检查是否有图片
    if (imageStore.images.length === 0) {
      toast.warning('请先添加图片')
      return false
    }

    // 检查是否正在进行其他批量操作
    if (imageStore.isBatchTranslationInProgress) {
      toast.warning('请等待当前批量操作完成')
      return false
    }

    // 验证高质量翻译配置
    if (!validation.validateBeforeTranslation('hq')) {
      return false
    }

    const { hqTranslation, textStyle } = settingsStore.settings

    // 保存当前选择的所有文本样式设置（复刻原版）
    const layoutDirectionValue = textStyle.layoutDirection
    savedTextStyles = {
      fontFamily: textStyle.fontFamily,
      fontSize: textStyle.fontSize,
      autoFontSize: textStyle.autoFontSize,
      autoTextDirection: layoutDirectionValue === 'auto',
      textDirection: layoutDirectionValue === 'auto' ? 'vertical' : layoutDirectionValue,
      fillColor: textStyle.fillColor,
      textColor: textStyle.textColor,
      rotationAngle: 0,
      strokeEnabled: textStyle.strokeEnabled,
      strokeColor: textStyle.strokeColor,
      strokeWidth: textStyle.strokeWidth
    }
    console.log('高质量翻译前保存的文本样式设置:', savedTextStyles)

    // 立即显示进度条
    progress.value = {
      current: 0,
      total: imageStore.images.length,
      completed: 0,
      failed: 0,
      isInProgress: true,
      isPaused: false,
      label: '准备翻译...',
      percentage: 0
    }

    toast.info('步骤1/4: 消除所有图片文字...')

    // 设置翻译状态
    isHqTranslating.value = true
    imageStore.setBatchTranslationInProgress(true)

    try {
      // 1. 消除所有图片文字
      await removeAllTextsForHq()

      // 2. 导出文本为JSON
      toast.info('步骤2/4: 导出文本数据...')
      progress.value.label = '导出文本数据...'
      progress.value.percentage = 25

      const currentJsonData = exportTextToJson()
      if (!currentJsonData) {
        throw new Error('导出文本失败')
      }

      // 3. 收集所有图片的Base64数据
      toast.info('步骤3/4: 准备图片数据...')
      progress.value.label = '准备图片数据...'
      progress.value.percentage = 40

      const allImageBase64 = collectAllImageBase64()

      // 4. 分批发送给AI翻译
      toast.info('步骤4/4: 发送到AI进行翻译...')
      progress.value.label = '开始发送到AI...'
      progress.value.percentage = 50

      const batchSize = hqTranslation.batchSize || 3
      const sessionResetFrequency = hqTranslation.sessionReset || 5
      const maxRetries = hqTranslation.maxRetries ?? 2

      // 重置批次结果
      allBatchResults = []

      const totalImages = allImageBase64.length
      const totalBatches = Math.ceil(totalImages / batchSize)

      // 创建限速器
      const rpmLimit = hqTranslation.rpmLimit || 0
      const rateLimiter = rpmLimit > 0 ? createRateLimiter(rpmLimit) : null

      // 跟踪批次计数
      let batchCount = 0
      let sessionId = generateSessionId()

      for (let batchIndex = 0; batchIndex < totalBatches; batchIndex++) {
        // 更新进度（复刻原版：50% - 90% 区间）
        progress.value.label = `${batchIndex + 1}/${totalBatches}`
        progress.value.percentage = Math.round(50 + (batchIndex / totalBatches) * 40)

        // 检查是否需要重置会话
        if (batchCount >= sessionResetFrequency) {
          console.log('重置会话上下文')
          sessionId = generateSessionId()
          batchCount = 0
        }

        // 准备这一批次的图片和JSON数据
        const startIdx = batchIndex * batchSize
        const endIdx = Math.min(startIdx + batchSize, totalImages)
        const batchImages = allImageBase64.slice(startIdx, endIdx)
        const batchJsonData = filterJsonForBatch(currentJsonData, startIdx, endIdx)

        // 重试逻辑
        let retryCount = 0
        let success = false

        while (retryCount <= maxRetries && !success) {
          try {
            // 等待速率限制
            if (rateLimiter) {
              await rateLimiter.acquire()
            }

            // 发送批次到AI
            const result = await callAiForTranslation(batchImages, batchJsonData, sessionId)

            if (result) {
              allBatchResults.push(result)
              success = true
            } else {
              // 如果返回 null，也应该视为失败并增加重试计数
              retryCount++
              if (retryCount > maxRetries) {
                break
              }
              await new Promise(r => setTimeout(r, 1000))
              continue
            }

            batchCount++
          } catch (error) {
            retryCount++
            if (retryCount <= maxRetries) {
              console.log(`批次 ${batchIndex + 1} 翻译失败，第 ${retryCount}/${maxRetries} 次重试...`)
              toast.warning(`批次 ${batchIndex + 1} 失败，正在重试 (${retryCount}/${maxRetries})...`)
              await new Promise(r => setTimeout(r, 1000))
            } else {
              console.error(`批次 ${batchIndex + 1} 翻译最终失败:`, error)
              toast.error(`批次 ${batchIndex + 1} 翻译失败: ${error instanceof Error ? error.message : '未知错误'}`)
            }
          }
        }
      }

      // 5. 解析合并的JSON结果并导入
      toast.info('翻译完成，正在导入翻译结果...')
      progress.value.label = '导入翻译结果...'
      progress.value.percentage = 90

      await importTranslationResult(mergeJsonResults(allBatchResults))

      // 完成
      progress.value.label = '翻译完成！'
      progress.value.percentage = 100
      toast.success('高质量翻译完成！')
      return true
    } catch (error) {
      console.error('高质量翻译过程出错:', error)
      toast.error(`翻译失败: ${error instanceof Error ? error.message : '未知错误'}`)
      return false
    } finally {
      isHqTranslating.value = false
      imageStore.setBatchTranslationInProgress(false)
      setTimeout(() => {
        progress.value.isInProgress = false
      }, 1000)
    }
  }

  /**
   * 消除所有图片文字并获取原文（复刻原版 removeAllImagesText）
   */
  async function removeAllTextsForHq(): Promise<void> {
    const totalImages = imageStore.images.length
    let failCount = 0

    progress.value.label = `消除文字: 0/${totalImages}`
    progress.value.percentage = 0

    for (let currentIndex = 0; currentIndex < totalImages; currentIndex++) {
      const progressPercent = Math.floor((currentIndex / totalImages) * 25)
      progress.value.label = `消除文字: ${currentIndex + 1}/${totalImages}`
      progress.value.percentage = progressPercent

      // 【修复】设置处理中状态，显示缩略图指示器（复刻原版 ui.showTranslatingIndicator）
      imageStore.setTranslationStatus(currentIndex, 'processing')

      const image = imageStore.images[currentIndex]
      if (!image || !image.originalDataURL) {
        // 跳过时重置状态，避免指示器残留
        imageStore.setTranslationStatus(currentIndex, 'pending')
        continue
      }

      // --- 使用统一的函数提取已有坐标（复刻原版 high_quality_translation.js）---
      const existingData = extractExistingBubbleData(image)

      // 如果用户清空了文本框，跳过
      if (existingData?.isEmpty) {
        console.log(`高质量翻译[${currentIndex}]: 文本框已被用户清空，跳过此图片`)
        // 跳过时重置状态，避免指示器残留
        imageStore.setTranslationStatus(currentIndex, 'pending')
        continue
      }

      // 日志
      if (existingData) {
        const sourceLabel = existingData.isManual ? '手动标注框' : '已有文本框'
        console.log(`高质量翻译[${currentIndex}]: 使用${sourceLabel} ${existingData.coords.length} 个`)
      }

      try {
        const params = buildTranslateParams(image.originalDataURL, {
          removeTextOnly: true,
          existingBubbleCoords: existingData?.coords,
          existingBubbleAngles: existingData?.angles,
          useExistingBubbles: !!existingData
        })

        const response = await translateImageApi(params as any)

        if (response.translated_image) {
          // 使用统一的 bubbleStates 保存所有设置
          const savedDir = savedTextStyles?.textDirection
          const textDir = (savedDir === 'vertical' || savedDir === 'horizontal' || savedDir === 'auto')
            ? savedDir
            : 'vertical'
          const bubbleStates = response.bubble_coords
            ? createBubbleStatesFromResponse(response, {
              fontSize: savedTextStyles?.fontSize || settingsStore.settings.textStyle.fontSize,
              fontFamily: savedTextStyles?.fontFamily || settingsStore.settings.textStyle.fontFamily,
              textDirection: textDir as TextDirection,
              textColor: savedTextStyles?.textColor || settingsStore.settings.textStyle.textColor,
              fillColor: savedTextStyles?.fillColor || settingsStore.settings.textStyle.fillColor,
              strokeEnabled: savedTextStyles?.strokeEnabled ?? settingsStore.settings.textStyle.strokeEnabled,
              strokeColor: savedTextStyles?.strokeColor || settingsStore.settings.textStyle.strokeColor,
              strokeWidth: savedTextStyles?.strokeWidth ?? settingsStore.settings.textStyle.strokeWidth,
              inpaintMethod: settingsStore.settings.textStyle.inpaintMethod
            })
            : []

          imageStore.updateImageByIndex(currentIndex, {
            translatedDataURL: `data:image/png;base64,${response.translated_image}`,
            cleanImageData: response.clean_image || null,
            bubbleCoords: response.bubble_coords || [],
            bubbleAngles: response.bubble_angles || [],
            originalTexts: response.original_texts || [],
            textboxTexts: response.textbox_texts || [],
            bubbleStates,
            bubbleTexts: bubbleStates.map(s => s.translatedText || ''),
            translationFailed: false,
            translationStatus: 'completed',
            showOriginal: false,  // 【修复】复刻原版：确保显示翻译结果而非原图
            hasUnsavedChanges: true
          })

          console.log(`高质量翻译-消除文字[${currentIndex + 1}/${totalImages}]: 处理完成`)
        } else {
          failCount++
          imageStore.updateImageByIndex(currentIndex, { translationFailed: true, translationStatus: 'failed' })
        }
      } catch (error) {
        console.error(`图片 ${currentIndex} 消除文字失败:`, error)
        failCount++
        imageStore.updateImageByIndex(currentIndex, { translationFailed: true, translationStatus: 'failed' })
      }
    }

    progress.value.label = '消除文字完成'
    progress.value.percentage = 25

    if (failCount > 0) {
      throw new Error(`消除文字完成，但有 ${failCount} 张图片失败`)
    }
  }

  /**
   * 导入翻译结果（复刻原版 importTranslationResult）
   */
  async function importTranslationResult(importedData: HqJsonData[]): Promise<void> {
    if (!importedData || importedData.length === 0) {
      throw new Error('没有有效的翻译数据可导入')
    }

    const images = imageStore.images
    const originalImageIndex = imageStore.currentImageIndex

    // 获取当前的全局设置作为默认值
    const currentFontSize = savedTextStyles?.fontSize || settingsStore.settings.textStyle.fontSize
    const currentAutoFontSize = savedTextStyles?.autoFontSize ?? settingsStore.settings.textStyle.autoFontSize
    const currentFontFamily = savedTextStyles?.fontFamily || settingsStore.settings.textStyle.fontFamily
    const rawTextDirection = savedTextStyles?.textDirection || settingsStore.settings.textStyle.layoutDirection
    const currentTextDirection = (rawTextDirection === 'auto') ? 'vertical' : rawTextDirection
    const currentTextColor = savedTextStyles?.textColor || settingsStore.settings.textStyle.textColor
    const currentFillColor = savedTextStyles?.fillColor || settingsStore.settings.textStyle.fillColor
    const currentStrokeEnabled = savedTextStyles?.strokeEnabled ?? settingsStore.settings.textStyle.strokeEnabled
    const currentStrokeColor = savedTextStyles?.strokeColor || settingsStore.settings.textStyle.strokeColor
    const currentStrokeWidth = savedTextStyles?.strokeWidth ?? settingsStore.settings.textStyle.strokeWidth

    progress.value.label = '更新图片数据...'
    progress.value.percentage = 90

    const totalImages = importedData.length
    let processedImages = 0

    for (const imageData of importedData) {
      processedImages++
      progress.value.label = `处理图片 ${processedImages}/${totalImages}`
      progress.value.percentage = 90 + (processedImages / totalImages * 5)

      const imageIndex = imageData.imageIndex
      if (imageIndex < 0 || imageIndex >= images.length) {
        console.warn(`跳过无效的图片索引: ${imageIndex}`)
        continue
      }

      const image = images[imageIndex]
      if (!image) continue

      let imageUpdated = false
      const bubbleTexts = image.bubbleTexts || []
      const bubbleCoords = image.bubbleCoords || []

      for (const bubbleData of imageData.bubbles || []) {
        const bubbleIndex = bubbleData.bubbleIndex
        if (bubbleIndex < 0 || bubbleIndex >= bubbleCoords.length) {
          console.warn(`图片 ${imageIndex}: 跳过无效的气泡索引 ${bubbleIndex}`)
          continue
        }

        const translatedText = bubbleData.translated
        let textDirection = bubbleData.textDirection
        if (textDirection === 'auto') {
          textDirection = currentTextDirection
        }

        bubbleTexts[bubbleIndex] = translatedText
        const effectiveTextDirection: TextDirection = (textDirection === 'vertical' || textDirection === 'horizontal')
          ? textDirection
          : (currentTextDirection as TextDirection)

        // 更新 bubbleStates
        if (!image.bubbleStates || !Array.isArray(image.bubbleStates) || image.bubbleStates.length !== bubbleCoords.length) {
          // 创建新的气泡设置
          const detectedAngles = image.bubbleAngles || []
          const newSettings: BubbleState[] = []
          for (let i = 0; i < bubbleCoords.length; i++) {
            const bubbleTextDirection: TextDirection = (i === bubbleIndex) ? effectiveTextDirection : (currentTextDirection as TextDirection)
            const coords = bubbleCoords[i]
            // 【复刻原版】优先使用已有 bubbleStates 中的 autoTextDirection（后端基于文本行分析的结果）
            // 只有在没有已有值时才降级为宽高比判断
            let autoDir: TextDirection = bubbleTextDirection
            const existingState = image.bubbleStates?.[i]
            if (existingState?.autoTextDirection && existingState.autoTextDirection !== 'auto') {
              autoDir = existingState.autoTextDirection
            } else if (coords && coords.length >= 4) {
              const [x1, y1, x2, y2] = coords
              autoDir = (y2 - y1) > (x2 - x1) ? 'vertical' : 'horizontal'
            }
            newSettings.push({
              translatedText: bubbleTexts[i] || '',
              originalText: image.originalTexts?.[i] || '',
              textboxText: '',  // 【修复】添加缺失字段
              coords: coords as [number, number, number, number],
              polygon: [],  // 【修复】添加缺失字段
              fontSize: currentFontSize,
              fontFamily: currentFontFamily,
              textDirection: bubbleTextDirection,
              autoTextDirection: autoDir,
              position: { x: 0, y: 0 },
              textColor: currentTextColor,
              rotationAngle: detectedAngles[i] || 0,
              fillColor: currentFillColor,
              strokeEnabled: currentStrokeEnabled,
              strokeColor: currentStrokeColor,
              strokeWidth: currentStrokeWidth,
              inpaintMethod: settingsStore.settings.textStyle.inpaintMethod  // 【修复】添加缺失字段
            })
          }
          image.bubbleStates = newSettings
        } else if (image.bubbleStates[bubbleIndex]) {
          // 更新现有的 bubbleState
          image.bubbleStates[bubbleIndex].translatedText = translatedText
          if (textDirection && textDirection !== 'auto') {
            image.bubbleStates[bubbleIndex].textDirection = effectiveTextDirection
          }
        } else {
          // 【修复】第三分支：创建新的 bubbleState（当 bubbleStates[bubbleIndex] 不存在时）
          const imgAngles = image.bubbleAngles || []
          const bubbleDetectedAngle = imgAngles[bubbleIndex] || 0
          const coords = bubbleCoords[bubbleIndex]
          // 计算自动排版方向
          let autoDir: TextDirection = effectiveTextDirection
          if (coords && coords.length >= 4) {
            const [x1, y1, x2, y2] = coords
            autoDir = (y2 - y1) > (x2 - x1) ? 'vertical' : 'horizontal'
          }
          image.bubbleStates[bubbleIndex] = {
            translatedText: translatedText,
            originalText: image.originalTexts?.[bubbleIndex] || '',
            textboxText: '',
            coords: coords as [number, number, number, number],
            polygon: [],
            fontSize: currentFontSize,
            fontFamily: currentFontFamily,
            textDirection: effectiveTextDirection,
            autoTextDirection: autoDir,
            position: { x: 0, y: 0 },
            textColor: currentTextColor,
            rotationAngle: bubbleDetectedAngle,
            fillColor: currentFillColor,
            strokeEnabled: currentStrokeEnabled,
            strokeColor: currentStrokeColor,
            strokeWidth: currentStrokeWidth,
            inpaintMethod: settingsStore.settings.textStyle.inpaintMethod
          }
        }

        imageUpdated = true
      }

      if (imageUpdated && image.bubbleStates) {
        // 同步 bubbleTexts
        const newBubbleTexts = image.bubbleStates.map(bs => bs.translatedText || '')

        // 重新渲染图片
        if (image.cleanImageData) {
          try {
            const { apiClient } = await import('@/api/client')
            const bubbleStatesForApi = image.bubbleStates.map(bs => ({
              translatedText: bs.translatedText || '',
              coords: bs.coords,
              fontSize: bs.fontSize || currentFontSize,
              fontFamily: bs.fontFamily || currentFontFamily,
              textDirection: bs.textDirection || currentTextDirection,
              textColor: bs.textColor || currentTextColor,
              rotationAngle: bs.rotationAngle || 0,
              position: bs.position || { x: 0, y: 0 },
              strokeEnabled: bs.strokeEnabled ?? currentStrokeEnabled,
              strokeColor: bs.strokeColor || currentStrokeColor,
              strokeWidth: bs.strokeWidth ?? currentStrokeWidth
            }))

            let cleanImageBase64 = image.cleanImageData
            if (cleanImageBase64.includes('base64,')) {
              cleanImageBase64 = cleanImageBase64.split('base64,')[1] || ''
            }

            const renderResponse = await apiClient.post<{ rendered_image?: string; error?: string }>(
              '/api/re_render_image',
              {
                clean_image: cleanImageBase64,
                bubble_texts: newBubbleTexts,
                bubble_coords: bubbleCoords,
                // 【修复】当启用自动字号时，传递 'auto' 字符串而不是数值，避免后端警告
                fontSize: currentAutoFontSize ? 'auto' : currentFontSize,
                autoFontSize: currentAutoFontSize,
                fontFamily: currentFontFamily,
                textDirection: currentTextDirection,
                textColor: currentTextColor,
                bubble_states: bubbleStatesForApi,
                use_individual_styles: true,
                use_inpainting: false,
                use_lama: false,
                fillColor: null,
                is_font_style_change: false,
                strokeEnabled: currentStrokeEnabled,
                strokeColor: currentStrokeColor,
                strokeWidth: currentStrokeWidth
              }
            )

            if (renderResponse.rendered_image) {
              imageStore.updateImageByIndex(imageIndex, {
                translatedDataURL: `data:image/png;base64,${renderResponse.rendered_image}`,
                bubbleStates: image.bubbleStates,
                bubbleTexts: newBubbleTexts,
                hasUnsavedChanges: true
              })
              console.log(`已完成图片 ${imageIndex} 的渲染`)
            }
          } catch (renderError) {
            console.error(`重新渲染图片 ${imageIndex} 失败:`, renderError)
          }
        } else {
          imageStore.updateImageByIndex(imageIndex, {
            bubbleStates: image.bubbleStates,
            bubbleTexts: newBubbleTexts,
            hasUnsavedChanges: true
          })
        }
      }
    }

    // 回到最初的图片
    if (originalImageIndex >= 0 && originalImageIndex < images.length) {
      imageStore.setCurrentImageIndex(originalImageIndex)
    }
  }


  // ============================================================
  // AI 校对（完全复刻原版 ai_proofreading.js）
  // ============================================================

  /** 校对JSON数据格式（复刻原版） */
  interface ProofreadingJsonData {
    imageIndex: number
    bubbles: Array<{
      bubbleIndex: number
      original: string
      translated: string
      textDirection: string
    }>
  }

  /** 校对批次结果 */
  let proofreadingBatchResults: ProofreadingJsonData[][] = []

  /**
   * 导出校对文本为JSON（复刻原版 exportTextToJson，但导出已翻译文本）
   */
  function exportProofreadingTextToJson(): ProofreadingJsonData[] | null {
    const allImages = imageStore.images
    if (allImages.length === 0) return null

    const exportData: ProofreadingJsonData[] = []

    for (let imageIndex = 0; imageIndex < allImages.length; imageIndex++) {
      const image = allImages[imageIndex]
      if (!image) continue

      const originalTexts = image.originalTexts || []
      const translatedTexts = image.bubbleTexts || []

      const imageTextData: ProofreadingJsonData = {
        imageIndex: imageIndex,
        bubbles: []
      }

      for (let bubbleIndex = 0; bubbleIndex < originalTexts.length; bubbleIndex++) {
        const original = originalTexts[bubbleIndex] || ''
        const translated = (bubbleIndex < translatedTexts.length ? translatedTexts[bubbleIndex] : '') || ''

        // 获取气泡的排版方向（复刻原版逻辑）
        let textDirection = 'vertical'
        const bubbleState = image.bubbleStates?.[bubbleIndex]
        if (bubbleState && bubbleState.textDirection) {
          const bubbleDir = bubbleState.textDirection
          textDirection = (bubbleDir === 'auto') ? 'vertical' : bubbleDir
        } else if (image.userLayoutDirection && image.userLayoutDirection !== 'auto') {
          textDirection = image.userLayoutDirection
        }

        imageTextData.bubbles.push({
          bubbleIndex: bubbleIndex,
          original: original,
          translated: translated,
          textDirection: textDirection
        })
      }

      exportData.push(imageTextData)
    }

    return exportData
  }

  /**
   * 为校对过滤特定批次的JSON数据（复刻原版 filterJsonForBatch）
   */
  function filterProofreadingJsonForBatch(jsonData: ProofreadingJsonData[], startIdx: number, endIdx: number): ProofreadingJsonData[] {
    return jsonData.filter(item => item.imageIndex >= startIdx && item.imageIndex < endIdx)
  }

  /**
   * 合并校对批次结果（复刻原版 mergeJsonResults）
   */
  function mergeProofreadingResults(batchResults: ProofreadingJsonData[][]): ProofreadingJsonData[] {
    if (!batchResults || batchResults.length === 0) {
      return []
    }

    const mergedResult: ProofreadingJsonData[] = []

    for (const batchResult of batchResults) {
      if (!batchResult) continue

      const batchArray = Array.isArray(batchResult) ? batchResult : [batchResult]

      for (const imageData of batchArray) {
        if (imageData && typeof imageData === 'object' && 'imageIndex' in imageData) {
          mergedResult.push(imageData)
        }
      }
    }

    mergedResult.sort((a, b) => a.imageIndex - b.imageIndex)
    return mergedResult
  }

  /**
   * 调用AI进行校对（复刻原版 callAiForProofreading）
   */
  async function callAiForProofreading(
    imageBase64Array: string[],
    jsonData: ProofreadingJsonData[],
    round: ProofreadingRound,
    _sessionId: string
  ): Promise<ProofreadingJsonData[] | null> {
    const jsonString = JSON.stringify(jsonData, null, 2)

    // 构建消息（复刻原版格式）
    type MessageContent = { type: 'text'; text: string } | { type: 'image_url'; image_url: { url: string } }
    const userContent: MessageContent[] = [
      {
        type: 'text',
        // 复刻原版：附加说明校对translated字段
        text: round.prompt + '\n\n以下是JSON数据，包含原文和已有译文:\n```json\n' + jsonString + '\n```\n请在保持JSON格式的情况下，校对每个bubble的translated字段，使翻译更加准确、自然、符合语境。'
      }
    ]

    // 添加图片到消息中
    for (const imgBase64 of imageBase64Array) {
      userContent.push({
        type: 'image_url',
        image_url: {
          url: `data:image/png;base64,${imgBase64}`
        }
      })
    }

    const params: HqTranslateParams = {
      provider: round.provider,
      api_key: round.apiKey,
      model_name: round.modelName,
      custom_base_url: round.customBaseUrl,
      messages: [
        {
          role: 'system',
          content: '你是一个专业的漫画翻译校对助手，能够根据漫画图像内容和上下文对已有翻译进行校对和润色。'
        },
        {
          role: 'user',
          content: userContent
        }
      ],
      low_reasoning: round.lowReasoning,
      no_thinking_method: round.noThinkingMethod,
      force_json_output: round.forceJsonOutput
    }

    try {
      console.log(`AI校对: 通过后端代理调用 ${round.provider} API...`)
      const response = await hqTranslateBatchApi(params)

      if (!response.success) {
        throw new Error(response.error || 'API 调用失败')
      }

      // 优先使用后端已解析的 results（与 callAiForTranslation 保持一致）
      if (response.results && response.results.length > 0) {
        const firstItem = response.results[0]
        // 验证结构正确性（ProofreadingJsonData 与 HqJsonData 结构相同）
        if (firstItem && 'imageIndex' in firstItem && 'bubbles' in firstItem) {
          return response.results as unknown as ProofreadingJsonData[]
        }
      }

      // 如果 results 不存在或格式不对，使用 content
      const content = (response as any).content
      if (content) {
        if (round.forceJsonOutput) {
          try {
            return JSON.parse(content)
          } catch (e) {
            console.error('解析AI强制JSON返回的内容失败:', e)
            throw new Error('解析AI返回的JSON结果失败')
          }
        } else {
          // 从 markdown 代码块中提取 JSON
          const jsonMatch = content.match(/```json\s*([\s\S]*?)\s*```/)
          if (jsonMatch && jsonMatch[1]) {
            try {
              return JSON.parse(jsonMatch[1])
            } catch (e) {
              console.error('解析AI返回的JSON失败:', e)
              throw new Error('解析AI返回的校对结果失败')
            }
          }
          // 尝试直接解析
          try {
            return JSON.parse(content)
          } catch (e) {
            console.error('直接解析AI返回内容失败:', e)
            throw new Error('无法解析AI返回的校对结果')
          }
        }
      }

      return null
    } catch (error) {
      console.error('调用AI校对API失败:', error)
      throw error
    }
  }

  /**
   * 导入校对结果（复刻原版 importProofreadingResult）
   */
  async function importProofreadingResult(importedData: ProofreadingJsonData[]): Promise<void> {
    if (!importedData || importedData.length === 0) {
      console.warn('没有有效的校对数据可导入')
      toast.warning('没有有效的校对结果可导入')
      return
    }

    const images = imageStore.images
    const originalImageIndex = imageStore.currentImageIndex

    // 获取当前的全局设置作为默认值
    const { textStyle } = settingsStore.settings
    const currentFontSize = textStyle.fontSize
    const currentFontFamily = textStyle.fontFamily
    const rawTextDirection = textStyle.layoutDirection
    const currentTextDirection = (rawTextDirection === 'auto') ? 'vertical' : rawTextDirection
    const currentTextColor = textStyle.textColor
    const currentFillColor = textStyle.fillColor
    const currentStrokeEnabled = textStyle.strokeEnabled
    const currentStrokeColor = textStyle.strokeColor
    const currentStrokeWidth = textStyle.strokeWidth

    for (const imageData of importedData) {

      const imageIndex = imageData.imageIndex
      if (imageIndex < 0 || imageIndex >= images.length) {
        console.warn(`跳过无效的图片索引: ${imageIndex}`)
        continue
      }

      const image = images[imageIndex]
      if (!image) continue

      if (!imageData.bubbles || !Array.isArray(imageData.bubbles) || imageData.bubbles.length === 0) {
        console.warn(`图片 ${imageIndex}: 没有有效的气泡数据`)
        continue
      }

      let imageUpdated = false
      const bubbleTexts = image.bubbleTexts || []
      const bubbleCoords = image.bubbleCoords || []

      for (const bubble of imageData.bubbles) {
        const bubbleIndex = bubble.bubbleIndex
        const proofreadText = bubble.translated || ''
        let textDirection = bubble.textDirection
        if (!textDirection || textDirection === 'auto') {
          textDirection = currentTextDirection
        }

        if (bubbleIndex < 0 || bubbleIndex >= bubbleCoords.length) {
          console.warn(`图片 ${imageIndex}: 跳过无效的气泡索引 ${bubbleIndex}`)
          continue
        }

        // 确保bubbleTexts数组足够长
        while (bubbleTexts.length <= bubbleIndex) {
          bubbleTexts.push('')
        }

        bubbleTexts[bubbleIndex] = proofreadText

        // 更新 bubbleStates
        if (textDirection && textDirection !== 'auto') {
          const effectiveDir: TextDirection = (textDirection === 'vertical' || textDirection === 'horizontal')
            ? textDirection
            : (currentTextDirection as TextDirection)

          if (!image.bubbleStates || !Array.isArray(image.bubbleStates) || image.bubbleStates.length !== bubbleCoords.length) {
            const detectedAngles = image.bubbleAngles || []
            const newSettings: BubbleState[] = []
            for (let i = 0; i < bubbleCoords.length; i++) {
              const bubbleTextDirection: TextDirection = (i === bubbleIndex) ? effectiveDir : (currentTextDirection as TextDirection)
              const coords = bubbleCoords[i]
              // 【复刻原版】优先使用已有 bubbleStates 中的 autoTextDirection（后端基于文本行分析的结果）
              // 只有在没有已有值时才降级为宽高比判断
              let autoDir: TextDirection = bubbleTextDirection
              const existingState = image.bubbleStates?.[i]
              if (existingState?.autoTextDirection && existingState.autoTextDirection !== 'auto') {
                autoDir = existingState.autoTextDirection
              } else if (coords && coords.length >= 4) {
                const [x1, y1, x2, y2] = coords
                autoDir = (y2 - y1) > (x2 - x1) ? 'vertical' : 'horizontal'
              }
              newSettings.push({
                translatedText: bubbleTexts[i] || '',
                originalText: image.originalTexts?.[i] || '',
                coords: coords as [number, number, number, number],
                fontSize: currentFontSize,
                fontFamily: currentFontFamily,
                textDirection: bubbleTextDirection,
                autoTextDirection: autoDir,
                position: { x: 0, y: 0 },
                textColor: currentTextColor,
                rotationAngle: detectedAngles[i] || 0,
                fillColor: currentFillColor,
                strokeEnabled: currentStrokeEnabled,
                strokeColor: currentStrokeColor,
                strokeWidth: currentStrokeWidth
              } as BubbleState)
            }
            image.bubbleStates = newSettings
          } else if (image.bubbleStates[bubbleIndex]) {
            image.bubbleStates[bubbleIndex].translatedText = proofreadText
            image.bubbleStates[bubbleIndex].textDirection = effectiveDir
          }
        }

        imageUpdated = true
      }

      if (imageUpdated && image.bubbleStates) {
        const newBubbleTexts = image.bubbleStates.map(bs => bs.translatedText || '')

        // 重新渲染图片（复刻原版）
        if (image.cleanImageData) {
          try {
            const { apiClient } = await import('@/api/client')
            const bubbleStatesForApi = image.bubbleStates.map(bs => ({
              translatedText: bs.translatedText || '',
              coords: bs.coords,
              fontSize: bs.fontSize || currentFontSize,
              fontFamily: bs.fontFamily || currentFontFamily,
              textDirection: bs.textDirection || currentTextDirection,
              textColor: bs.textColor || currentTextColor,
              rotationAngle: bs.rotationAngle || 0,
              position: bs.position || { x: 0, y: 0 },
              strokeEnabled: bs.strokeEnabled ?? currentStrokeEnabled,
              strokeColor: bs.strokeColor || currentStrokeColor,
              strokeWidth: bs.strokeWidth ?? currentStrokeWidth
            }))

            let cleanImageBase64 = image.cleanImageData
            if (cleanImageBase64.includes('base64,')) {
              cleanImageBase64 = cleanImageBase64.split('base64,')[1] || ''
            }

            const renderResponse = await apiClient.post<{ rendered_image?: string; error?: string }>(
              '/api/re_render_image',
              {
                clean_image: cleanImageBase64,
                bubble_texts: newBubbleTexts,
                bubble_coords: bubbleCoords,
                fontSize: currentFontSize,
                fontFamily: currentFontFamily,
                textDirection: currentTextDirection,
                textColor: currentTextColor,
                bubble_states: bubbleStatesForApi,
                use_individual_styles: true,
                use_inpainting: false,
                use_lama: false,
                fillColor: null,
                is_font_style_change: false,
                strokeEnabled: currentStrokeEnabled,
                strokeColor: currentStrokeColor,
                strokeWidth: currentStrokeWidth
              }
            )

            if (renderResponse.rendered_image) {
              imageStore.updateImageByIndex(imageIndex, {
                translatedDataURL: `data:image/png;base64,${renderResponse.rendered_image}`,
                bubbleStates: image.bubbleStates,
                bubbleTexts: newBubbleTexts,
                hasUnsavedChanges: true
              })
              console.log(`已完成图片 ${imageIndex} 的校对渲染`)
            }
          } catch (renderError) {
            console.error(`重新渲染图片 ${imageIndex} 失败:`, renderError)
          }
        } else {
          imageStore.updateImageByIndex(imageIndex, {
            bubbleStates: image.bubbleStates,
            bubbleTexts: newBubbleTexts,
            hasUnsavedChanges: true
          })
        }
      }
    }

    // 回到最初的图片
    if (originalImageIndex >= 0 && originalImageIndex < images.length) {
      imageStore.setCurrentImageIndex(originalImageIndex)
    }
  }

  /**
   * 执行 AI 校对（完全复刻原版 startProofreading）
   */
  async function executeProofreading(): Promise<boolean> {
    // 验证配置
    if (!validation.validateBeforeTranslation('proofread')) {
      return false
    }

    const { proofreading } = settingsStore.settings
    if (!proofreading.enabled || proofreading.rounds.length === 0) {
      toast.error('请先启用 AI 校对并添加校对轮次')
      return false
    }

    const images = imageStore.images
    if (images.length === 0) {
      toast.error('请先上传图片')
      return false
    }

    // 检查是否有已翻译的图片（复刻原版）
    const hasTranslatedImages = images.some(
      (img) => img.bubbleTexts && img.bubbleTexts.length > 0
    )

    if (!hasTranslatedImages) {
      toast.error('请先进行翻译以获取译文')
      return false
    }

    isProofreading.value = true
    imageStore.setBatchTranslationInProgress(true)
    const totalRounds = proofreading.rounds.length

    // 初始化进度 - 复刻原版
    progress.value = {
      current: 0,
      total: totalRounds,
      completed: 0,
      failed: 0,
      isInProgress: true,
      isPaused: false,
      label: '准备校对...',
      percentage: 0
    }

    toast.info(`开始校对，共 ${totalRounds} 轮`)

    try {
      // 遍历每个校对轮次（复刻原版主校对循环）
      for (let roundIndex = 0; roundIndex < totalRounds; roundIndex++) {
        const round = proofreading.rounds[roundIndex]
        if (!round) continue

        const roundName = round.name || `轮次${roundIndex + 1}`
        const roundBasePercent = (roundIndex / totalRounds) * 100
        const roundPercent = (1 / totalRounds) * 100

        // 显示轮次信息 - 复刻原版
        toast.info(`校对第 ${roundIndex + 1}/${totalRounds} 轮: ${roundName}`)
        progress.value.label = `轮次 ${roundIndex + 1}/${totalRounds}`
        progress.value.percentage = Math.round(roundBasePercent)

        // 步骤1: 导出文本数据 - 复刻原版
        toast.info(`轮次 ${roundIndex + 1}/${totalRounds}: 导出文本数据...`)
        progress.value.label = '导出文本...'
        progress.value.percentage = Math.round(roundBasePercent + roundPercent * 0.2)

        const currentJsonData = exportProofreadingTextToJson()
        if (!currentJsonData) {
          throw new Error('导出文本失败')
        }

        // 步骤2: 收集所有图片的Base64数据 - 复刻原版
        toast.info(`轮次 ${roundIndex + 1}/${totalRounds}: 准备图片数据...`)
        progress.value.label = '准备图片数据...'
        progress.value.percentage = Math.round(roundBasePercent + roundPercent * 0.4)

        const allImageBase64 = collectAllImageBase64()

        // 步骤3: 分批发送给AI校对 - 复刻原版
        toast.info(`轮次 ${roundIndex + 1}/${totalRounds}: 发送到AI进行校对...`)
        progress.value.label = '开始发送到AI...'
        progress.value.percentage = Math.round(roundBasePercent + roundPercent * 0.5)

        const batchSize = round.batchSize || 3
        const sessionResetFrequency = round.sessionReset || 20
        const maxRetries = proofreading.maxRetries ?? 2

        // 重置批次结果
        proofreadingBatchResults = []

        const totalImages = allImageBase64.length
        const totalBatches = Math.ceil(totalImages / batchSize)

        // 创建限速器
        const rpmLimit = round.rpmLimit || 7
        const rateLimiter = rpmLimit > 0 ? createRateLimiter(rpmLimit) : null

        // 跟踪批次计数（复刻原版）
        let batchCount = 0
        let sessionId = generateSessionId()
        let successCount = 0

        for (let batchIndex = 0; batchIndex < totalBatches; batchIndex++) {
          // 更新批次进度 - 复刻原版
          const batchProgress = roundBasePercent + roundPercent * 0.5 + roundPercent * 0.4 * (batchIndex / totalBatches)
          progress.value.label = `轮次 ${roundIndex + 1}/${totalRounds}: ${batchIndex + 1}/${totalBatches}`
          progress.value.percentage = Math.round(batchProgress)

          // 检查是否需要重置会话（复刻原版）
          if (batchCount >= sessionResetFrequency) {
            console.log('重置会话上下文')
            sessionId = generateSessionId()
            batchCount = 0
          }

          // 准备这一批次的图片和JSON数据
          const startIdx = batchIndex * batchSize
          const endIdx = Math.min(startIdx + batchSize, totalImages)
          const batchImages = allImageBase64.slice(startIdx, endIdx)
          const batchJsonData = filterProofreadingJsonForBatch(currentJsonData, startIdx, endIdx)

          // 重试逻辑（复刻原版）
          let retryCount = 0
          let success = false

          while (retryCount <= maxRetries && !success) {
            try {
              // 等待速率限制
              if (rateLimiter) {
                await rateLimiter.acquire()
              }

              // 发送批次到AI
              const result = await callAiForProofreading(batchImages, batchJsonData, round, sessionId)

              if (result) {
                proofreadingBatchResults.push(result)
                successCount++
                success = true
              }

              batchCount++
            } catch (error) {
              retryCount++
              if (retryCount <= maxRetries) {
                console.log(`轮次 ${roundIndex + 1}, 批次 ${batchIndex + 1} 校对失败，第 ${retryCount}/${maxRetries} 次重试...`)
                toast.warning(`轮次 ${roundIndex + 1}, 批次 ${batchIndex + 1} 失败，正在重试 (${retryCount}/${maxRetries})...`)
                await new Promise(r => setTimeout(r, 1000))
              } else {
                console.error(`轮次 ${roundIndex + 1}, 批次 ${batchIndex + 1} 校对最终失败:`, error)
                toast.error(`轮次 ${roundIndex + 1}, 批次 ${batchIndex + 1} 校对失败: ${error instanceof Error ? error.message : '未知错误'}`)
              }
            }
          }
        }

        // 如果所有批次都失败，抛出错误（复刻原版）
        if (successCount === 0) {
          throw new Error(`轮次 ${roundIndex + 1} 校对完全失败，请检查API设置或校对提示词`)
        }

        // 步骤4: 解析合并的JSON结果并导入 - 复刻原版
        toast.info(`轮次 ${roundIndex + 1}/${totalRounds}: 导入校对结果...`)
        progress.value.label = '导入校对结果...'
        progress.value.percentage = Math.round(roundBasePercent + roundPercent * 0.9)

        await importProofreadingResult(mergeProofreadingResults(proofreadingBatchResults))

        progress.value.completed++
      }

      // 完成 - 复刻原版
      progress.value.label = '校对完成！'
      progress.value.percentage = 100
      toast.success(`AI校对完成，共 ${totalRounds} 轮`)
      return true
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'AI 校对失败'
      toast.error(`校对失败: ${errorMessage}`)

      // 重置进度条 - 复刻原版
      progress.value.label = '校对已取消'
      progress.value.percentage = 0
      return false
    } finally {
      isProofreading.value = false
      imageStore.setBatchTranslationInProgress(false)
      // 延迟隐藏进度条
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
   * 跳过检测步骤，直接使用已有坐标
   * @returns 是否成功
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

    // 【优化】直接传递完整的 BubbleState 数组，由 buildTranslateParams 统一提取坐标和角度
    // 避免分别提取时遗漏参数
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
    isHqTranslating,
    isProofreading,

    // 计算属性
    isTranslating,
    progressPercent,

    // 单张翻译
    translateCurrentImage,
    translateImageByIndex,

    // 批量翻译
    translateAllImages,
    pauseBatchTranslation,
    resumeBatchTranslation,
    cancelBatchTranslation,

    // 仅消除文字
    removeTextOnly,
    removeAllTexts,

    // 重新翻译失败图片
    retryFailedImages,

    // 高质量翻译
    executeHqTranslation,

    // AI 校对
    executeProofreading,

    // 使用已有气泡框翻译
    translateWithCurrentBubbles
  }
}

