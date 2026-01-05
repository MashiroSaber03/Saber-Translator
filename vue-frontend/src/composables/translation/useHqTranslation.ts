/**
 * 高质量翻译组合式函数
 * 提供基于多模态AI的批量上下文翻译功能
 * 
 * 完全复刻原版 high_quality_translation.js
 */

import { ref } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { useValidation } from '../useValidation'
import { useToast } from '@/utils/toast'
import { createRateLimiter } from '@/utils/rateLimiter'
import { createBubbleStatesFromResponse } from '@/utils/bubbleFactory'
import {
  translateImage as translateImageApi,
  hqTranslateBatch as hqTranslateBatchApi,
  type TranslateImageParams,
  type HqTranslateParams
} from '@/api/translate'
import type { BubbleState, TextDirection } from '@/types/bubble'
import type {
  TranslationProgress,
  SavedTextStyles,
  HqJsonData
} from './types'
import {
  extractExistingBubbleData,
  buildTranslateParams,
  generateSessionId,
  filterJsonForBatch,
  mergeJsonResults
} from './utils'

// ============================================================
// 组合式函数
// ============================================================

/**
 * 高质量翻译组合式函数
 */
export function useHqTranslation() {
  const imageStore = useImageStore()
  const settingsStore = useSettingsStore()
  const validation = useValidation()
  const toast = useToast()

  // ============================================================
  // 状态
  // ============================================================

  /** 是否正在进行高质量翻译 */
  const isHqTranslating = ref(false)

  /** 当前翻译进度 */
  const progress = ref<TranslationProgress>({
    current: 0,
    total: 0,
    completed: 0,
    failed: 0,
    isInProgress: false
  })

  /** 所有批次结果（复刻原版） */
  let allBatchResults: HqJsonData[][] = []

  /** 保存的文本样式（复刻原版） */
  let savedTextStyles: SavedTextStyles | null = null

  // ============================================================
  // 内部辅助函数
  // ============================================================

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

        const response = await translateImageApi(params as unknown as TranslateImageParams)

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
  // 主函数
  // ============================================================

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

  // ============================================================
  // 返回
  // ============================================================

  return {
    // 状态
    isHqTranslating,
    progress,

    // 主函数
    executeHqTranslation
  }
}
