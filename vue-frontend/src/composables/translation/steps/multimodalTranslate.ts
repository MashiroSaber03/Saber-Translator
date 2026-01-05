/**
 * 多模态翻译执行器
 * 
 * 负责使用多模态 AI（如 GPT-4V、Gemini）进行批量翻译
 * 用于高质量翻译模式
 */

import { useSettingsStore } from '@/stores/settingsStore'
import { useImageStore } from '@/stores/imageStore'
import {
    hqTranslateBatch as hqTranslateBatchApi,
    type HqTranslateParams
} from '@/api/translate'
import { extractBase64FromDataUrl, generateSessionId, filterJsonForBatch, mergeJsonResults } from '../utils'
import { createRateLimiter } from '@/utils/rateLimiter'
import type {
    TranslationJsonData,
    TranslateStepResult,
    BatchExecutionContext,
    ProgressReporter
} from '../core/types'

/**
 * 导出所有图片的原文为 JSON
 */
export function exportTextsToJson(): TranslationJsonData[] | null {
    const imageStore = useImageStore()
    const allImages = imageStore.images

    if (allImages.length === 0) return null

    const exportData: TranslationJsonData[] = []

    for (let imageIndex = 0; imageIndex < allImages.length; imageIndex++) {
        const image = allImages[imageIndex]
        if (!image) continue

        const originalTexts = image.originalTexts || []
        const imageTextData: TranslationJsonData = {
            imageIndex: imageIndex,
            bubbles: []
        }

        for (let bubbleIndex = 0; bubbleIndex < originalTexts.length; bubbleIndex++) {
            const original = originalTexts[bubbleIndex] || ''

            // 获取气泡的排版方向
            let textDirection = 'vertical'
            const bubbleState = image.bubbleStates?.[bubbleIndex]
            if (bubbleState?.textDirection && bubbleState.textDirection !== 'auto') {
                textDirection = bubbleState.textDirection
            } else if (bubbleState?.autoTextDirection && bubbleState.autoTextDirection !== 'auto') {
                textDirection = bubbleState.autoTextDirection
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
 * 收集所有图片的 Base64 数据
 */
export function collectAllImageBase64(): string[] {
    const imageStore = useImageStore()
    return imageStore.images.map(image => {
        return extractBase64FromDataUrl(image.originalDataURL)
    })
}

/**
 * 调用多模态 AI 进行翻译
 */
export async function callMultimodalAI(
    imageBase64Array: string[],
    jsonData: TranslationJsonData[],
    _sessionId: string
): Promise<TranslationJsonData[] | null> {
    const settingsStore = useSettingsStore()
    const { hqTranslation } = settingsStore.settings
    const jsonString = JSON.stringify(jsonData, null, 2)

    // 构建消息
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

    // 构建请求参数
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
        console.log(`多模态翻译: 通过后端代理调用 ${hqTranslation.provider} API...`)

        const response = await hqTranslateBatchApi(params)

        if (!response.success) {
            throw new Error(response.error || 'API 调用失败')
        }

        // 优先使用后端已解析的 results
        if (response.results && response.results.length > 0) {
            const firstItem = response.results[0]
            if (firstItem && 'imageIndex' in firstItem && 'bubbles' in firstItem) {
                return response.results as unknown as TranslationJsonData[]
            }
        }

        // 如果 results 不存在，使用 content
        const content = (response as { content?: string }).content
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
                if (jsonMatch?.[1]) {
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
        console.error('调用多模态AI翻译失败:', error)
        throw error
    }
}

/**
 * 执行多模态批量翻译
 */
export async function executeMultimodalTranslation(
    context: BatchExecutionContext
): Promise<TranslateStepResult> {
    const settingsStore = useSettingsStore()
    const { hqTranslation } = settingsStore.settings
    const { progress } = context

    // 导出文本数据
    const jsonData = exportTextsToJson()
    if (!jsonData) {
        return { success: false, error: '导出文本失败' }
    }

    // 收集图片数据
    const allImageBase64 = collectAllImageBase64()

    // 分批配置
    const batchSize = context.config.batchOptions?.batchSize || hqTranslation.batchSize || 3
    const maxRetries = context.config.batchOptions?.maxRetries ?? hqTranslation.maxRetries ?? 2
    const rpmLimit = context.config.batchOptions?.rpmLimit || hqTranslation.rpmLimit || 0
    const sessionResetFrequency = context.config.batchOptions?.sessionResetFrequency || hqTranslation.sessionReset || 5

    const totalImages = allImageBase64.length
    const totalBatches = Math.ceil(totalImages / batchSize)

    // 创建限速器
    const rateLimiter = rpmLimit > 0 ? createRateLimiter(rpmLimit) : null

    const allBatchResults: TranslationJsonData[][] = []
    let batchCount = 0
    let sessionId = generateSessionId()

    for (let batchIndex = 0; batchIndex < totalBatches; batchIndex++) {
        // 更新进度
        const percentage = Math.round(50 + (batchIndex / totalBatches) * 40)
        progress.setPercentage(percentage, `翻译批次 ${batchIndex + 1}/${totalBatches}`)

        // 检查是否需要重置会话
        if (batchCount >= sessionResetFrequency) {
            console.log('重置会话上下文')
            sessionId = generateSessionId()
            batchCount = 0
        }

        // 准备批次数据
        const startIdx = batchIndex * batchSize
        const endIdx = Math.min(startIdx + batchSize, totalImages)
        const batchImages = allImageBase64.slice(startIdx, endIdx)
        const batchJsonData = filterJsonForBatch(jsonData, startIdx, endIdx)

        // 重试逻辑
        let retryCount = 0
        let success = false

        while (retryCount <= maxRetries && !success) {
            try {
                if (rateLimiter) {
                    await rateLimiter.acquire()
                }

                const result = await callMultimodalAI(batchImages, batchJsonData, sessionId)

                if (result) {
                    allBatchResults.push(result)
                    success = true
                    batchCount++
                } else {
                    retryCount++
                    if (retryCount > maxRetries) break
                    await new Promise(r => setTimeout(r, 1000))
                }
            } catch (error) {
                retryCount++
                if (retryCount <= maxRetries) {
                    console.log(`批次 ${batchIndex + 1} 翻译失败，第 ${retryCount}/${maxRetries} 次重试...`)
                    await new Promise(r => setTimeout(r, 1000))
                } else {
                    console.error(`批次 ${batchIndex + 1} 翻译最终失败:`, error)
                }
            }
        }
    }

    // 合并所有批次结果
    const mergedResults = mergeJsonResults(allBatchResults)

    return {
        success: mergedResults.length > 0,
        translationData: mergedResults
    }
}
