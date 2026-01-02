/**
 * AI校对组合式函数
 * 提供基于多模态AI的翻译校对功能
 * 
 * 完全复刻原版 ai_proofreading.js
 */

import { ref } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { useValidation } from '../useValidation'
import { useToast } from '@/utils/toast'
import { createRateLimiter } from '@/utils/rateLimiter'
import {
  hqTranslateBatch as hqTranslateBatchApi,
  type HqTranslateParams
} from '@/api/translate'
import type { BubbleState, TextDirection } from '@/types/bubble'
import type { ProofreadingRound } from '@/types/settings'
import type {
  TranslationProgress,
  ProofreadingJsonData
} from './types'
import {
  generateSessionId,
  filterJsonForBatch,
  mergeJsonResults
} from './utils'

// ============================================================
// 组合式函数
// ============================================================

/**
 * AI校对组合式函数
 */
export function useProofreading() {
  const imageStore = useImageStore()
  const settingsStore = useSettingsStore()
  const validation = useValidation()
  const toast = useToast()

  // ============================================================
  // 状态
  // ============================================================

  /** 是否正在进行AI校对 */
  const isProofreading = ref(false)

  /** 当前校对进度 */
  const progress = ref<TranslationProgress>({
    current: 0,
    total: 0,
    completed: 0,
    failed: 0,
    isInProgress: false,
    isPaused: false
  })

  /** 校对批次结果 */
  let proofreadingBatchResults: ProofreadingJsonData[][] = []

  // ============================================================
  // 内部辅助函数
  // ============================================================

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

  // ============================================================
  // 主函数
  // ============================================================

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
          const batchJsonData = filterJsonForBatch(currentJsonData, startIdx, endIdx)

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

        await importProofreadingResult(mergeJsonResults(proofreadingBatchResults))

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
  // 返回
  // ============================================================

  return {
    // 状态
    isProofreading,
    progress,

    // 主函数
    executeProofreading
  }
}
