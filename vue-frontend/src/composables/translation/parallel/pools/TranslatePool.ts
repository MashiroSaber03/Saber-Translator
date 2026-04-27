/**
 * 翻译池
 * 
 * 根据翻译模式采用不同处理策略：
 * - standard: 逐页调用翻译API
 * - hq: 批量收集后调用多模态AI翻译
 * - proofread: 批量收集后调用AI校对
 * - removeText: 跳过翻译
 */

import { TaskPool } from '../TaskPool'
import type { PipelineTask, ParallelTranslationMode, TranslationJsonData } from '../types'
import type { ParallelProgressTracker } from '../ParallelProgressTracker'
import { parallelTranslate } from '@/api/parallelTranslate'
import { hqTranslateBatch, translateSingleText } from '@/api/translate'
import { useSettingsStore } from '@/stores/settingsStore'

export class TranslatePool extends TaskPool {
  private mode: ParallelTranslationMode = 'standard'
  private batchBuffer: PipelineTask[] = []
  private totalTasks = 0
  private processedCount = 0

  constructor(
    nextPool: TaskPool | null,
    progressTracker: ParallelProgressTracker,
    onTaskComplete?: (task: PipelineTask) => void
  ) {
    // 翻译池不需要深度学习锁
    super('翻译', '🌐', nextPool, null, progressTracker, onTaskComplete)
  }

  /**
   * 设置翻译模式
   */
  setMode(mode: ParallelTranslationMode, totalTasks: number, nextPool: TaskPool | null): void {
    this.mode = mode
    this.totalTasks = totalTasks
    this.batchBuffer = []
    this.processedCount = 0
    this.nextPool = nextPool
  }

  protected async process(task: PipelineTask): Promise<PipelineTask> {
    switch (this.mode) {
      case 'standard':
        return this.handleStandardTranslate(task)
      case 'hq':
        return this.handleHqTranslate(task)
      case 'proofread':
        return this.handleProofread(task)
      default:
        return task
    }
  }

  // ==================== 普通翻译 ====================
  private async handleStandardTranslate(task: PipelineTask): Promise<PipelineTask> {
    const settingsStore = useSettingsStore()
    const settings = settingsStore.settings

    if (!task.ocrResult || task.ocrResult.originalTexts.length === 0) {
      task.translateResult = { translatedTexts: [], textboxTexts: [] }
      task.status = 'processing'
      return task
    }

    const translationMode = settings.translation.translationMode || 'batch'
    const originalTexts = task.ocrResult.originalTexts

    if (translationMode === 'single') {
      // ==================== 逐气泡翻译模式 ====================
      console.log(`[并行翻译池] 使用逐气泡翻译模式，图片 ${task.imageIndex}，共 ${originalTexts.length} 个气泡`)

      const translatedTexts: string[] = []
      const textboxTexts: string[] = []

      for (let i = 0; i < originalTexts.length; i++) {
        const originalText = originalTexts[i]

        // 跳过空文本
        if (!originalText || originalText.trim() === '') {
          translatedTexts.push('')
          if (settings.useTextboxPrompt) {
            textboxTexts.push('')
          }
          continue
        }

        try {
          // 调用单文本翻译API
          // 固定使用逐气泡翻译的提示词，避免使用批量翻译提示词导致语义不匹配
          const promptContent = settings.translation.isJsonMode
            ? settings.translation.singleJsonPrompt
            : settings.translation.singleNormalPrompt

          const response = await translateSingleText({
            original_text: originalText,
            model_provider: settings.translation.provider,
            model_name: settings.translation.modelName,
            api_key: settings.translation.apiKey,
            custom_base_url: settings.translation.customBaseUrl,
            target_language: settings.targetLanguage,
            prompt_content: promptContent,  // 使用逐气泡翻译的提示词
            use_json_format: settings.translation.isJsonMode,  // 传递 JSON 模式设置
            rpm_limit_translation: settings.translation.rpmLimit,
            max_retries: settings.translation.maxRetries
          })

          if (response.success && response.data) {
            translatedTexts.push(response.data.translated_text || '')
          } else {
            console.warn(`[并行翻译池] 气泡 ${i + 1} 翻译失败: ${response.error}`)
            translatedTexts.push(`【翻译失败】请检查终端中的错误日志`)
          }

          // 如果启用了文本框提示词，需要再翻译一次
          if (settings.useTextboxPrompt && settings.textboxPrompt) {
            const textboxResponse = await translateSingleText({
              original_text: originalText,
              model_provider: settings.translation.provider,
              model_name: settings.translation.modelName,
              api_key: settings.translation.apiKey,
              custom_base_url: settings.translation.customBaseUrl,
              target_language: settings.targetLanguage,
              prompt_content: settings.textboxPrompt,
              rpm_limit_translation: settings.translation.rpmLimit,
              max_retries: settings.translation.maxRetries
            })

            if (textboxResponse.success && textboxResponse.data) {
              textboxTexts.push(textboxResponse.data.translated_text || '')
            } else {
              textboxTexts.push('')
            }
          }
        } catch (error) {
          console.error(`[并行翻译池] 气泡 ${i + 1} 翻译出错:`, error)
          translatedTexts.push(`【翻译失败】请检查终端中的错误日志`)
          if (settings.useTextboxPrompt) {
            textboxTexts.push('')
          }
        }
      }

      task.translateResult = { translatedTexts, textboxTexts }
    } else {
      // ==================== 整页批量翻译模式 ====================
      console.log(`[并行翻译池] 使用整页批量翻译模式，图片 ${task.imageIndex}，共 ${originalTexts.length} 个气泡`)

      const response = await parallelTranslate({
        original_texts: originalTexts,
        target_language: settings.targetLanguage,
        source_language: settings.sourceLanguage,
        model_provider: settings.translation.provider,
        model_name: settings.translation.modelName,
        api_key: settings.translation.apiKey,
        custom_base_url: settings.translation.customBaseUrl,
        prompt_content: settings.translatePrompt,
        textbox_prompt_content: settings.textboxPrompt,
        use_textbox_prompt: settings.useTextboxPrompt,
        rpm_limit: settings.translation.rpmLimit,
        max_retries: settings.translation.maxRetries,
        use_json_format: settings.translation.isJsonMode
      })

      if (!response.success) {
        throw new Error(response.error || '翻译失败')
      }

      task.translateResult = {
        translatedTexts: response.translated_texts || [],
        textboxTexts: response.textbox_texts || []
      }
    }

    task.status = 'processing'
    return task
  }

  // ==================== 高质量翻译 ====================
  private async handleHqTranslate(task: PipelineTask): Promise<PipelineTask> {
    this.batchBuffer.push(task)
    this.processedCount++

    const settingsStore = useSettingsStore()
    const { hqTranslation } = settingsStore.settings
    const batchSize = hqTranslation.batchSize || 3
    const isLastBatch = this.processedCount >= this.totalTasks
    const batchReady = this.batchBuffer.length >= batchSize || isLastBatch

    if (!batchReady) {
      // 标记为缓冲状态，阻止TaskPool自动传递到下一个池子
      task.status = 'buffered'
      return task
    }

    // 凑够批次，开始批量处理
    const batch = [...this.batchBuffer]
    this.batchBuffer = []

    // 1. 收集 JSON 数据
    const jsonData: TranslationJsonData[] = batch.map(t => ({
      imageIndex: t.imageIndex,
      bubbles: (t.ocrResult?.originalTexts || []).map((text, idx) => ({
        bubbleIndex: idx,
        original: text,
        translated: '',
        textDirection: t.detectionResult?.autoDirections[idx] || 'vertical'
      }))
    }))

    // 2. 收集图片 Base64
    const imageBase64Array = batch.map(t => this.extractBase64(t.imageData.originalDataURL))

    // 3. 调用多模态 AI API - 使用新接口，传数据而不是消息
    const response = await hqTranslateBatch({
      provider: hqTranslation.provider,
      api_key: hqTranslation.apiKey,
      model_name: hqTranslation.modelName,
      custom_base_url: hqTranslation.customBaseUrl,
      // 新接口：传数据，后端构建消息
      jsonData,
      imageBase64Array,
      prompt: hqTranslation.prompt,
      systemPrompt: '你是一个专业的漫画翻译助手，能够根据漫画图像内容和上下文提供高质量的翻译。',
      isProofreading: false,
      enableDebugLogs: settingsStore.settings.enableVerboseLogs,  // 使用全局的详细日志开关
      // 其他参数
      low_reasoning: hqTranslation.lowReasoning,
      force_json_output: hqTranslation.forceJsonOutput,
      no_thinking_method: hqTranslation.noThinkingMethod,
      use_stream: hqTranslation.useStream,
      max_retries: hqTranslation.maxRetries ?? 2
    })

    // 4. 解析结果
    const translatedData = this.parseHqResponse(response, hqTranslation.forceJsonOutput)

    // 5. 填充结果到各任务，并批量传递给下一个池子
    for (const t of batch) {
      const taskData = translatedData?.find(d => d.imageIndex === t.imageIndex)
      if (taskData) {
        t.translateResult = {
          translatedTexts: taskData.bubbles.map(b => b.translated),
          textboxTexts: []
        }
      } else {
        t.translateResult = { translatedTexts: [], textboxTexts: [] }
      }
      t.status = 'processing'

      if (this.nextPool) {
        this.nextPool.enqueue(t)
      }
    }

    // 批次中的任务已在上面的 for 循环中手动 enqueue，
    // 返回 'buffered' 状态阻止 TaskPool 再次自动 enqueue，
    // 避免任务被重复处理导致进度计数错误
    task.status = 'buffered'
    return task
  }

  // ==================== AI 校对 ====================
  private async handleProofread(task: PipelineTask): Promise<PipelineTask> {
    this.batchBuffer.push(task)
    this.processedCount++

    const settingsStore = useSettingsStore()
    const { proofreading, useTextboxPrompt } = settingsStore.settings
    const batchSize = proofreading.rounds[0]?.batchSize || 3
    const isLastBatch = this.processedCount >= this.totalTasks
    const batchReady = this.batchBuffer.length >= batchSize || isLastBatch

    if (!batchReady) {
      // 标记为缓冲状态，阻止TaskPool自动传递到下一个池子
      task.status = 'buffered'
      return task
    }

    const batch = [...this.batchBuffer]
    this.batchBuffer = []

    // 1. 收集 JSON 数据（包含已有译文）
    const jsonData: TranslationJsonData[] = batch.map(t => ({
      imageIndex: t.imageIndex,
      bubbles: (t.imageData.bubbleStates || []).map((state, idx) => ({
        bubbleIndex: idx,
        original: state.originalText || '',
        translated: useTextboxPrompt
          ? (state.textboxText || state.translatedText || '')
          : (state.translatedText || ''),
        // 【简化设计】直接使用 textDirection，它已经是具体方向值
        textDirection: (state.textDirection === 'vertical' || state.textDirection === 'horizontal')
          ? state.textDirection
          : (state.autoTextDirection === 'vertical' || state.autoTextDirection === 'horizontal')
            ? state.autoTextDirection
            : 'vertical'
      }))
    }))

    // 2. 收集图片（优先使用翻译后的图片）
    const imageBase64Array = batch.map(t => {
      const dataUrl = t.imageData.translatedDataURL || t.imageData.originalDataURL
      return this.extractBase64(dataUrl)
    })

    // 3. 遍历所有校对轮次
    let currentData = jsonData
    for (const round of proofreading.rounds) {
      // 使用新接口，不再手动构建messages

      const response = await hqTranslateBatch({
        provider: round.provider,
        api_key: round.apiKey,
        model_name: round.modelName,
        custom_base_url: round.customBaseUrl,
        // 使用新接口：传数据，后端构建消息
        jsonData: currentData as any[],
        imageBase64Array,
        prompt: round.prompt,
        systemPrompt: '你是一个专业的漫画翻译校对助手，能够根据漫画图像内容检查和修正翻译。',
        isProofreading: true,
        enableDebugLogs: settingsStore.settings.enableVerboseLogs,  // 使用全局的详细日志开关
        // 其他参数
        low_reasoning: round.lowReasoning,
        force_json_output: round.forceJsonOutput,
        no_thinking_method: round.noThinkingMethod,
        use_stream: round.useStream ?? true,
        max_retries: round.maxRetries ?? proofreading.maxRetries ?? 2
      })

      const parsedResult = this.parseHqResponse(response, round.forceJsonOutput)
      if (parsedResult) {
        currentData = parsedResult
      }
    }

    // 4. 填充校对结果并传递给渲染池
    for (const t of batch) {
      const taskData = currentData.find(d => d.imageIndex === t.imageIndex)
      if (taskData) {
        t.translateResult = {
          translatedTexts: taskData.bubbles.map(b => b.translated),
          textboxTexts: []
        }
      } else {
        t.translateResult = { translatedTexts: [], textboxTexts: [] }
      }
      t.status = 'processing'

      if (this.nextPool) {
        this.nextPool.enqueue(t)
      }
    }

    // 批次中的任务已在上面的 for 循环中手动 enqueue，
    // 返回 'buffered' 状态阻止 TaskPool 再次自动 enqueue，
    // 避免任务被重复处理导致进度计数错误
    task.status = 'buffered'
    return task
  }

  // ==================== 辅助方法 ====================
  private extractBase64(dataUrl: string): string {
    if (dataUrl.includes('base64,')) {
      return dataUrl.split('base64,')[1] || ''
    }
    return dataUrl
  }

  private parseHqResponse(
    response: { success: boolean; results?: any[]; content?: string; error?: string },
    forceJsonOutput: boolean
  ): TranslationJsonData[] | null {
    if (!response.success) {
      console.error('API调用失败:', response.error)
      return null
    }

    // 优先使用后端已解析的 results
    if (response.results && response.results.length > 0) {
      const firstItem = response.results[0]
      if (firstItem && 'imageIndex' in firstItem && 'bubbles' in firstItem) {
        return response.results as TranslationJsonData[]
      }
    }

    // 尝试从 content 解析
    const content = (response as { content?: string }).content
    if (content) {
      let parsed: any = null
      if (forceJsonOutput) {
        try {
          parsed = JSON.parse(content)
        } catch (e) {
          console.error('解析AI强制JSON返回的内容失败:', e)
          return null
        }
      } else {
        const jsonMatch = content.match(/```json\s*([\s\S]*?)\s*```/)
        if (jsonMatch?.[1]) {
          try {
            parsed = JSON.parse(jsonMatch[1])
          } catch (e) {
            console.error('解析AI返回的JSON失败:', e)
            return null
          }
        }
      }

      // 兼容单张图片格式：{imageIndex, bubbles} -> [{imageIndex, bubbles}]
      if (parsed) {
        if (Array.isArray(parsed)) {
          return parsed as TranslationJsonData[]
        } else if (typeof parsed === 'object' && 'imageIndex' in parsed && 'bubbles' in parsed) {
          console.log('[TranslatePool.parseHqResponse] 检测到单张图片格式，自动包装为数组')
          return [parsed] as TranslationJsonData[]
        }
      }
    }

    return null
  }
}
