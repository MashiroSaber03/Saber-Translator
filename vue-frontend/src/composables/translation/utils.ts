/**
 * 翻译功能共享工具函数
 */

import { useSettingsStore } from '@/stores/settingsStore'
import { createBubbleStatesFromResponse } from '@/utils/bubbleFactory'
import type { BubbleState, BubbleCoords } from '@/types/bubble'
import type { TranslateImageResponse } from '@/types/api'
import type { ImageData as AppImageData } from '@/types/image'
import type { ExistingBubbleData, TranslationOptions } from './core/types'

// ============================================================
// 工具函数
// ============================================================

/**
 * 从 DataURL 中提取纯 Base64 数据
 * @param dataUrl - 完整的 DataURL (data:image/...;base64,...)
 * @returns 纯 Base64 数据
 */
export function extractBase64FromDataUrl(dataUrl: string): string {
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
export function extractExistingBubbleData(image: AppImageData): ExistingBubbleData | null {
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
export function buildTranslateParams(
  imageData: string,
  options: TranslationOptions = {}
): Record<string, unknown> {
  const settingsStore = useSettingsStore()
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

    // 智能颜色识别设置（仅文字颜色）
    useAutoTextColor: textStyle.useAutoTextColor ?? true,

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
 * 处理翻译响应，创建气泡状态
 * @param response - 翻译响应
 * @returns 气泡状态数组或 null
 */
export function createBubbleStatesFromApiResponse(
  response: TranslateImageResponse
): BubbleState[] | null {
  const settingsStore = useSettingsStore()
  const { textStyle } = settingsStore.settings

  if (!response.bubble_coords || response.bubble_coords.length === 0) {
    return null
  }

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

  return createBubbleStatesFromResponse(apiResponse, globalDefaults)
}

// ============================================================
// 高质量翻译工具函数
// ============================================================

/**
 * 生成会话ID（复刻原版）
 */
export function generateSessionId(): string {
  return 'session_' + Date.now() + '_' + Math.random().toString(36).substring(2, 9)
}

/**
 * 为特定批次过滤JSON数据（复刻原版 filterJsonForBatch）
 */
export function filterJsonForBatch<T extends { imageIndex: number }>(
  jsonData: T[],
  startIdx: number,
  endIdx: number
): T[] {
  return jsonData.filter(item => item.imageIndex >= startIdx && item.imageIndex < endIdx)
}

/**
 * 合并所有批次的JSON结果（复刻原版 mergeJsonResults）
 */
export function mergeJsonResults<T extends { imageIndex: number }>(
  batchResults: T[][]
): T[] {
  if (!batchResults || batchResults.length === 0) {
    return []
  }

  const mergedResult: T[] = []

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
