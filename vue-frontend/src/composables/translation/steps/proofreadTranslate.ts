/**
 * AI 校对执行器
 * 
 * 负责使用多模态 AI 对已翻译的文本进行校对
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
    BatchExecutionContext
} from '../core/types'
import type { ProofreadingRound } from '@/types/settings'

/**
 * 导出已翻译文本为 JSON（用于校对）
 */
export function exportProofreadingTextsToJson(): TranslationJsonData[] | null {
    const imageStore = useImageStore()
    const settingsStore = useSettingsStore()
    const allImages = imageStore.images
    const useTextboxPrompt = settingsStore.settings.useTextboxPrompt

    if (allImages.length === 0) return null

    const exportData: TranslationJsonData[] = []

    for (let imageIndex = 0; imageIndex < allImages.length; imageIndex++) {
        const image = allImages[imageIndex]
        if (!image) continue

        const bubbleStates = image.bubbleStates || []
        const imageTextData: TranslationJsonData = {
            imageIndex: imageIndex,
            bubbles: []
        }

        for (let bubbleIndex = 0; bubbleIndex < bubbleStates.length; bubbleIndex++) {
            const state = bubbleStates[bubbleIndex]
            if (!state) continue

            const original = state.originalText || ''
            // 校对模式：导出已翻译的文本
            const translated = useTextboxPrompt
                ? (state.textboxText || state.translatedText || '')
                : (state.translatedText || '')

            let textDirection = 'vertical'
            if (state.textDirection && state.textDirection !== 'auto') {
                textDirection = state.textDirection
            } else if (state.autoTextDirection && state.autoTextDirection !== 'auto') {
                textDirection = state.autoTextDirection
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
 * 收集所有图片的 Base64 数据（用于校对）
 */
export function collectProofreadingImageBase64(): string[] {
    const imageStore = useImageStore()
    return imageStore.images.map(image => {
        // 校对时优先使用翻译后的图片
        if (image.translatedDataURL) {
            return extractBase64FromDataUrl(image.translatedDataURL)
        }
        return extractBase64FromDataUrl(image.originalDataURL)
    })
}

/**
 * 调用 AI 进行校对
 */
export async function callProofreadingAI(
    imageBase64Array: string[],
    jsonData: TranslationJsonData[],
    roundConfig: ProofreadingRound,
    _sessionId: string
): Promise<TranslationJsonData[] | null> {
    const jsonString = JSON.stringify(jsonData, null, 2)

    // 使用轮次配置的提示词
    const promptContent = roundConfig.prompt

    // 构建消息
    type MessageContent = { type: 'text'; text: string } | { type: 'image_url'; image_url: { url: string } }
    const userContent: MessageContent[] = [
        {
            type: 'text',
            text: promptContent + '\n\n以下是JSON数据:\n```json\n' + jsonString + '\n```'
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
            content: '你是一个专业的漫画翻译校对助手，能够根据漫画图像内容检查和修正翻译。'
        },
        {
            role: 'user',
            content: userContent
        }
    ]

    // 构建请求参数（使用轮次配置）
    const params: HqTranslateParams = {
        provider: roundConfig.provider,
        api_key: roundConfig.apiKey,
        model_name: roundConfig.modelName,
        custom_base_url: roundConfig.customBaseUrl,
        messages: messages,
        low_reasoning: roundConfig.lowReasoning,
        force_json_output: roundConfig.forceJsonOutput,
        no_thinking_method: roundConfig.noThinkingMethod,
        use_stream: false  // 校对不使用流式
    }

    try {
        console.log(`AI校对(${roundConfig.name}): 通过后端代理调用 ${roundConfig.provider} API...`)

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
            if (roundConfig.forceJsonOutput) {
                try {
                    return JSON.parse(content)
                } catch (e) {
                    console.error('解析AI校对JSON返回的内容失败:', e)
                    throw new Error('解析AI返回的JSON结果失败')
                }
            } else {
                const jsonMatch = content.match(/```json\s*([\s\S]*?)\s*```/)
                if (jsonMatch?.[1]) {
                    try {
                        return JSON.parse(jsonMatch[1])
                    } catch (e) {
                        console.error('解析AI校对返回的JSON失败:', e)
                        throw new Error('解析AI返回的校对结果失败')
                    }
                }
            }
        }

        return null
    } catch (error) {
        console.error('调用AI校对失败:', error)
        throw error
    }
}

/**
 * 执行 AI 校对
 */
export async function executeProofreadingTranslation(
    context: BatchExecutionContext
): Promise<TranslateStepResult> {
    const settingsStore = useSettingsStore()
    const { proofreading } = settingsStore.settings
    const { progress } = context

    // 检查是否有配置的轮次
    if (!proofreading.rounds || proofreading.rounds.length === 0) {
        return { success: false, error: '请先配置校对轮次' }
    }

    // 导出已翻译文本数据
    const jsonData = exportProofreadingTextsToJson()
    if (!jsonData) {
        return { success: false, error: '导出校对文本失败' }
    }

    // 收集图片数据
    const allImageBase64 = collectProofreadingImageBase64()

    const totalImages = allImageBase64.length
    const allRoundResults: TranslationJsonData[][] = []
    const maxRetries = proofreading.maxRetries ?? 2

    // 遍历所有校对轮次
    for (let roundIdx = 0; roundIdx < proofreading.rounds.length; roundIdx++) {
        const roundConfig = proofreading.rounds[roundIdx]
        if (!roundConfig) continue

        const roundLabel = roundConfig.name || `第${roundIdx + 1}轮校对`
        const batchSize = roundConfig.batchSize || 3
        const rpmLimit = roundConfig.rpmLimit || 0
        const sessionResetFrequency = roundConfig.sessionReset || 5

        const totalBatches = Math.ceil(totalImages / batchSize)

        // 创建限速器
        const rateLimiter = rpmLimit > 0 ? createRateLimiter(rpmLimit) : null

        // 使用上一轮的结果作为本轮输入（如果有）
        const currentJsonData = roundIdx === 0 ? jsonData : mergeJsonResults(allRoundResults)

        let batchCount = 0
        let sessionId = generateSessionId()

        for (let batchIndex = 0; batchIndex < totalBatches; batchIndex++) {
            // 更新进度
            const roundBasePercentage = (roundIdx / proofreading.rounds.length) * 90
            const batchPercentage = (batchIndex / totalBatches) * (90 / proofreading.rounds.length)
            const percentage = Math.round(5 + roundBasePercentage + batchPercentage)
            progress.setPercentage(percentage, `${roundLabel} ${batchIndex + 1}/${totalBatches}`)

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
            const batchJsonData = filterJsonForBatch(currentJsonData, startIdx, endIdx)

            // 重试逻辑
            let retryCount = 0
            let success = false

            while (retryCount <= maxRetries && !success) {
                try {
                    if (rateLimiter) {
                        await rateLimiter.acquire()
                    }

                    const result = await callProofreadingAI(batchImages, batchJsonData, roundConfig, sessionId)

                    if (result) {
                        allRoundResults.push(result)
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
                        console.log(`${roundLabel}批次 ${batchIndex + 1} 失败，第 ${retryCount}/${maxRetries} 次重试...`)
                        await new Promise(r => setTimeout(r, 1000))
                    } else {
                        console.error(`${roundLabel}批次 ${batchIndex + 1} 最终失败:`, error)
                    }
                }
            }
        }
    }

    // 合并所有轮次结果
    const mergedResults = mergeJsonResults(allRoundResults)

    return {
        success: mergedResults.length > 0,
        translationData: mergedResults
    }
}
