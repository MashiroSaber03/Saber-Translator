/**
 * AI翻译步骤（高质量翻译 & 校对）
 * 
 * 支持两种模式：
 * - 高质量翻译（hq）：使用多模态AI根据图像和OCR结果翻译
 * - AI校对（proofread）：使用AI检查和修正已有译文
 * 
 * 特点：
 * - 批量处理多张图片
 * - 支持多轮校对
 * - 使用图像+文本上下文提升翻译质量
 */

import { hqTranslateBatch } from '@/api/translate'
import { useSettingsStore } from '@/stores/settingsStore'
import type { ImageData } from '@/types/image'

// ============================================================
// 类型定义
// ============================================================

/** AI翻译任务 */
export interface AiTranslateTask {
    imageIndex: number
    image: ImageData
    // 高质量翻译需要
    originalTexts?: string[]
    autoDirections?: string[]
}

/** AI翻译输入 */
export interface AiTranslateInput {
    mode: 'hq' | 'proofread'
    tasks: AiTranslateTask[]
}

/** AI翻译输出 */
export interface AiTranslateOutput {
    results: Array<{
        imageIndex: number
        translatedTexts: string[]
        textboxTexts: string[]
    }>
}

/** 翻译JSON数据格式 */
interface TranslationJsonData {
    imageIndex: number
    bubbles: Array<{
        bubbleIndex: number
        original: string
        translated: string
        textDirection: string
    }>
}

// ============================================================
// 主函数
// ============================================================

/**
 * 执行AI翻译步骤
 * 
 * @param input - AI翻译输入
 * @returns AI翻译输出
 */
export async function executeAiTranslate(input: AiTranslateInput): Promise<AiTranslateOutput> {
    const settingsStore = useSettingsStore()
    const settings = settingsStore.settings
    const isProofread = input.mode === 'proofread'

    // 1. 收集 JSON 数据
    const jsonData: TranslationJsonData[] = input.tasks.map(t => {
        if (isProofread) {
            // 校对模式：使用已有译文
            return {
                imageIndex: t.imageIndex,
                bubbles: (t.image.bubbleStates || []).map((state, idx) => ({
                    bubbleIndex: idx,
                    original: state.originalText || '',
                    translated: settings.useTextboxPrompt
                        ? (state.textboxText || state.translatedText || '')
                        : (state.translatedText || ''),
                    // 【简化设计】直接使用 textDirection，它已经是具体方向值
                    textDirection: (state.textDirection === 'vertical' || state.textDirection === 'horizontal')
                        ? state.textDirection
                        : (state.autoTextDirection === 'vertical' || state.autoTextDirection === 'horizontal')
                            ? state.autoTextDirection
                            : 'vertical'
                }))
            }
        } else {
            // 高质量翻译：使用 OCR 结果
            return {
                imageIndex: t.imageIndex,
                bubbles: (t.originalTexts || []).map((text, idx) => ({
                    bubbleIndex: idx,
                    original: text,
                    translated: '',
                    textDirection: (t.autoDirections?.[idx]) || 'vertical'
                }))
            }
        }
    })

    // 2. 收集图片 Base64
    const imageBase64Array = input.tasks.map(t => {
        const dataUrl = isProofread
            ? (t.image.translatedDataURL || t.image.originalDataURL)
            : t.image.originalDataURL
        return extractBase64(dataUrl)
    })

    // 3. 获取配置
    const aiConfig = isProofread ? settings.proofreading.rounds[0] : settings.hqTranslation
    const prompt = isProofread ? aiConfig?.prompt : settings.hqTranslation.prompt
    const systemPrompt = isProofread
        ? '你是一个专业的漫画翻译校对助手，能够根据漫画图像内容检查和修正翻译。'
        : '你是一个专业的漫画翻译助手，能够根据漫画图像内容和上下文提供高质量的翻译。'

    // 4. 调用 API - 第一轮
    const hqConfig = settings.hqTranslation
    const roundConfig = isProofread ? aiConfig : null
    const response = await hqTranslateBatch({
        provider: (isProofread ? roundConfig?.provider : hqConfig.provider) || 'openai',
        api_key: (isProofread ? roundConfig?.apiKey : hqConfig.apiKey) || '',
        model_name: (isProofread ? roundConfig?.modelName : hqConfig.modelName) || '',
        custom_base_url: isProofread ? roundConfig?.customBaseUrl : hqConfig.customBaseUrl,
        // 新接口：传数据，后端构建消息
        jsonData,
        imageBase64Array,
        prompt: prompt || '',
        systemPrompt,
        isProofreading: isProofread,
        enableDebugLogs: settings.enableVerboseLogs,
        // 其他参数
        low_reasoning: isProofread ? roundConfig?.lowReasoning : hqConfig.lowReasoning,
        force_json_output: isProofread ? roundConfig?.forceJsonOutput : hqConfig.forceJsonOutput,
        no_thinking_method: isProofread ? roundConfig?.noThinkingMethod : hqConfig.noThinkingMethod,
        use_stream: isProofread ? (roundConfig?.useStream ?? true) : hqConfig.useStream,
        max_retries: isProofread ? (settings.proofreading.maxRetries || 2) : (hqConfig.maxRetries || 2)
    })

    // 5. 解析结果
    const forceJson = isProofread ? (roundConfig?.forceJsonOutput || false) : hqConfig.forceJsonOutput
    const translatedData = parseHqResponse(response, forceJson)

    // 6. 校对模式可能有多轮
    let currentData = translatedData || jsonData
    if (isProofread && settings.proofreading.rounds.length > 1) {
        for (let i = 1; i < settings.proofreading.rounds.length; i++) {
            const round = settings.proofreading.rounds[i]!

            const roundResponse = await hqTranslateBatch({
                provider: round.provider,
                api_key: round.apiKey,
                model_name: round.modelName,
                custom_base_url: round.customBaseUrl,
                // 使用新接口
                jsonData: currentData as any[],
                imageBase64Array,
                prompt: round.prompt,
                systemPrompt,
                isProofreading: true,
                enableDebugLogs: settings.enableVerboseLogs,
                // 其他参数
                low_reasoning: round.lowReasoning,
                force_json_output: round.forceJsonOutput,
                no_thinking_method: round.noThinkingMethod,
                use_stream: round.useStream ?? true,
                max_retries: round.maxRetries || settings.proofreading.maxRetries || 2
            })

            const roundResult = parseHqResponse(roundResponse, round.forceJsonOutput)
            if (roundResult) {
                currentData = roundResult
            }
        }
    }

    // 7. 构建输出结果
    const results = input.tasks.map(t => {
        const taskData = (currentData as any[])?.find((d: any) => d.imageIndex === t.imageIndex)
        if (taskData) {
            return {
                imageIndex: t.imageIndex,
                translatedTexts: taskData.bubbles.map((b: any) => b.translated),
                textboxTexts: [] as string[]
            }
        } else {
            return {
                imageIndex: t.imageIndex,
                translatedTexts: [] as string[],
                textboxTexts: [] as string[]
            }
        }
    })

    return { results }
}

// ============================================================
// 辅助函数
// ============================================================

/**
 * 提取 Base64 数据
 */
function extractBase64(dataUrl: string): string {
    if (dataUrl.includes('base64,')) {
        return dataUrl.split('base64,')[1] || ''
    }
    return dataUrl
}

/**
 * 解析高质量翻译响应
 */
function parseHqResponse(
    response: { success: boolean; results?: any[]; content?: string; error?: string },
    forceJsonOutput: boolean
): any[] | null {
    if (!response.success) {
        console.error('API调用失败:', response.error)
        return null
    }

    // 优先使用后端已解析的 results
    if (response.results && response.results.length > 0) {
        const firstItem = response.results[0]
        if (firstItem && 'imageIndex' in firstItem && 'bubbles' in firstItem) {
            return response.results
        }
    }

    // 尝试从 content 解析
    const content = (response as { content?: string }).content
    if (content) {
        let parsed: any = null
        if (forceJsonOutput) {
            try {
                parsed = JSON.parse(content)
            } catch {
                return null
            }
        } else {
            const jsonMatch = content.match(/```json\s*([\s\S]*?)\s*```/)
            if (jsonMatch?.[1]) {
                try {
                    parsed = JSON.parse(jsonMatch[1])
                } catch {
                    return null
                }
            }
        }

        // 兼容单张图片格式：{imageIndex, bubbles} -> [{imageIndex, bubbles}]
        if (parsed) {
            if (Array.isArray(parsed)) {
                return parsed
            } else if (typeof parsed === 'object' && 'imageIndex' in parsed && 'bubbles' in parsed) {
                console.log('[executeAiTranslate] 检测到单张图片格式，自动包装为数组')
                return [parsed]
            }
        }
    }

    return null
}
