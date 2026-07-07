import { hqTranslateBatch } from '@/api/translate'
import type { HqTranslateJsonData } from '@/api/translate'
import type { BookTranslationConstraints } from '@/types/bookTranslationConstraints'
import type { ImageData } from '@/types/image'
import type { TranslationSettings } from '@/types/settings'
import type { HqTranslateResponse } from '@/types/api'
import type { TranslationWarning } from '@/types/translationConstraints'
import { resolveConstraintPayloadForTranslation } from '@/utils/bookTranslationConstraints'
import { extractBase64Payload } from '@/utils/dataUrl'
import { serializeOpenAICompatibleOptionsForApi } from '@/utils/openaiOptions'

export interface AiTranslateTask {
    imageIndex: number
    image: ImageData
    originalTexts?: string[]
    autoDirections?: string[]
}

export interface AiTranslateInput {
    mode: 'hq' | 'proofread'
    tasks: AiTranslateTask[]
    settingsSnapshot: TranslationSettings
    bookTranslationConstraints: BookTranslationConstraints
    isBookshelfMode: boolean
}

export interface AiTranslateOutput {
    results: Array<{
        imageIndex: number
        translatedTexts: string[]
        textboxTexts: string[]
        warnings: TranslationWarning[]
    }>
}

interface AiTranslateResultData {
    imageIndex: number
    bubbles: Array<{
        bubbleIndex?: number
        original?: string
        translated: string
        textDirection?: string
    }>
}


export async function executeAiTranslate(input: AiTranslateInput): Promise<AiTranslateOutput> {
    const settings = input.settingsSnapshot
    const isProofread = input.mode === 'proofread'
    const constraintPayload = resolveConstraintPayloadForTranslation({
        isBookshelfMode: input.isBookshelfMode,
        constraints: input.bookTranslationConstraints,
    })

    const jsonData: HqTranslateJsonData[] = input.tasks.map(t => {
        if (isProofread) {
            return {
                imageIndex: t.imageIndex,
                bubbles: (t.image.bubbleStates || []).map((state, idx) => ({
                    bubbleIndex: idx,
                    original: state.originalText || '',
                    translated: settings.useTextboxPrompt
                        ? (state.textboxText || state.translatedText || '')
                        : (state.translatedText || ''),
                    textDirection: (state.textDirection === 'vertical' || state.textDirection === 'horizontal')
                        ? state.textDirection
                        : (state.autoTextDirection === 'vertical' || state.autoTextDirection === 'horizontal')
                            ? state.autoTextDirection
                            : 'vertical'
                }))
            }
        } else {
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

    const imageBase64Array = input.tasks.map(t => {
        const dataUrl = isProofread
            ? (t.image.translatedDataURL || t.image.originalDataURL)
            : t.image.originalDataURL
        return extractBase64Payload(dataUrl)
    })

    const aiConfig = isProofread ? settings.proofreading.rounds[0] : settings.hqTranslation
    const hqConfig = settings.hqTranslation
    const roundConfig = isProofread ? aiConfig : null
    const prompt = isProofread ? aiConfig?.prompt : settings.hqTranslation.prompt
    const systemPrompt = isProofread
        ? '你是一个专业的漫画翻译校对助手，能够根据漫画图像内容检查和修正翻译。'
        : '你是一个专业的漫画翻译助手，能够根据漫画图像内容和上下文提供高质量的翻译。'
    const requestProvider = isProofread ? (roundConfig?.provider ?? '') : (hqConfig.provider ?? '')

    const response = await hqTranslateBatch({
        provider: requestProvider,
        api_key: (isProofread ? roundConfig?.apiKey : hqConfig.apiKey) || '',
        model_name: (isProofread ? roundConfig?.modelName : hqConfig.modelName) || '',
        custom_base_url: isProofread ? roundConfig?.customBaseUrl : hqConfig.customBaseUrl,
        translation_mode: isProofread ? 'proofread' : 'hq',
        translation_scope: 'batch',
        jsonData,
        imageBase64Array,
        target_language: settings.targetLanguage,
        prompt: prompt || '',
        systemPrompt,
        isProofreading: isProofread,
        enableDebugLogs: settings.enableVerboseLogs,
        ...constraintPayload,
        openai_options: serializeOpenAICompatibleOptionsForApi((isProofread ? roundConfig?.openaiOptions : hqConfig.openaiOptions)!)
    })

    const forceJsonOutput = isProofread ? (roundConfig?.openaiOptions.request.forceJsonOutput || false) : hqConfig.openaiOptions.request.forceJsonOutput
    const translatedData = parseHqResponse(response, forceJsonOutput)
    let latestWarnings = response.warnings || []

    let currentData: AiTranslateResultData[] = translatedData || jsonData
    if (isProofread && settings.proofreading.rounds.length > 1) {
        for (let i = 1; i < settings.proofreading.rounds.length; i++) {
            const round = settings.proofreading.rounds[i]!

            const roundResponse = await hqTranslateBatch({
                provider: round.provider,
                api_key: round.apiKey,
                model_name: round.modelName,
                custom_base_url: round.customBaseUrl,
                translation_mode: 'proofread',
                translation_scope: 'batch',
                jsonData: currentData,
                imageBase64Array,
                target_language: settings.targetLanguage,
                prompt: round.prompt,
                systemPrompt,
                isProofreading: true,
                enableDebugLogs: settings.enableVerboseLogs,
                ...constraintPayload,
                openai_options: serializeOpenAICompatibleOptionsForApi(round.openaiOptions)
            })

            const roundResult = parseHqResponse(roundResponse, round.openaiOptions.request.forceJsonOutput)
            latestWarnings = roundResponse.warnings || latestWarnings
            if (roundResult) {
                currentData = roundResult
            }
        }
    }

    const results = input.tasks.map(t => {
        const taskData = currentData.find((d) => d.imageIndex === t.imageIndex)
        const taskWarnings = latestWarnings.filter((warning) => warning.imageIndex === t.imageIndex)
        if (taskData) {
            return {
                imageIndex: t.imageIndex,
                translatedTexts: taskData.bubbles.map((b) => b.translated),
                textboxTexts: [] as string[],
                warnings: taskWarnings
            }
        } else {
            return {
                imageIndex: t.imageIndex,
                translatedTexts: [] as string[],
                textboxTexts: [] as string[],
                warnings: taskWarnings
            }
        }
    })

    return { results }
}

function parseHqResponse(
    response: Pick<HqTranslateResponse, 'success' | 'results' | 'content' | 'error'>,
    forceJsonOutput: boolean
): AiTranslateResultData[] | null {
    if (!response.success) {
        return null
    }

    if (response.results && response.results.length > 0) {
        return normalizeAiTranslateResultData(response.results)
    }

    const content = (response as { content?: string }).content
    if (content) {
        let parsed: unknown = null
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

        if (parsed) {
            if (Array.isArray(parsed)) {
                return normalizeAiTranslateResultData(parsed)
            } else if (typeof parsed === 'object' && 'imageIndex' in parsed && 'bubbles' in parsed) {
                return normalizeAiTranslateResultData([parsed])
            }
        }
    }

    return null
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null
}

function normalizeAiTranslateResultData(value: unknown): AiTranslateResultData[] | null {
    if (!Array.isArray(value)) {
        return null
    }

    const normalized: AiTranslateResultData[] = []
    for (const item of value) {
        if (!isRecord(item) || typeof item.imageIndex !== 'number' || !Array.isArray(item.bubbles)) {
            return null
        }

        const bubbles: AiTranslateResultData['bubbles'] = []
        for (const bubble of item.bubbles) {
            if (!isRecord(bubble) || typeof bubble.translated !== 'string') {
                return null
            }

            bubbles.push({
                ...(typeof bubble.bubbleIndex === 'number' ? { bubbleIndex: bubble.bubbleIndex } : {}),
                ...(typeof bubble.original === 'string' ? { original: bubble.original } : {}),
                translated: bubble.translated,
                ...(typeof bubble.textDirection === 'string' ? { textDirection: bubble.textDirection } : {}),
            })
        }

        normalized.push({
            imageIndex: item.imageIndex,
            bubbles,
        })
    }

    return normalized
}
