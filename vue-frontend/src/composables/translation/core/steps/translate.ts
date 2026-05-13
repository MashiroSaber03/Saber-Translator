/**
 * 翻译步骤（普通翻译）
 * 提取自 SequentialPipeline.ts Line 348-468
 * 
 * 注意：这是最复杂的步骤，包含两种翻译模式
 */
import { parallelTranslate, type ParallelTranslateResponse } from '@/api/parallelTranslate'
import { translateSingleText } from '@/api/translate'
import type { BookTranslationConstraints } from '@/types/bookTranslationConstraints'
import type { TranslationSettings } from '@/types/settings'
import type { TranslationWarning } from '@/types/translationConstraints'
import { resolveConstraintPayloadForTranslation } from '@/utils/bookTranslationConstraints'
import { serializeOpenAICompatibleOptionsForApi } from '@/utils/openaiOptions'

export interface TranslateInput {
    imageIndex: number
    translationMode?: string
    originalTexts: string[]
    settingsSnapshot: TranslationSettings
    bookTranslationConstraints: BookTranslationConstraints
    isBookshelfMode: boolean
}

export interface TranslateOutput {
    translatedTexts: string[]
    textboxTexts: string[]
    warnings: TranslationWarning[]
}

function mapWarnings(
    warnings: TranslationWarning[] | undefined,
    input: { imageIndex: number; bubbleIndex: number },
): TranslationWarning[] {
    return (warnings || []).map(warning => ({
        imageIndex: warning.imageIndex ?? input.imageIndex,
        bubbleIndex: warning.bubbleIndex ?? input.bubbleIndex,
        source: warning.source,
        expectedTarget: warning.expectedTarget,
        actualTranslation: warning.actualTranslation,
    }))
}

function resolveEffectiveTranslationWarnings(params: {
    primaryResponseWarnings: TranslationWarning[] | undefined
    textboxResponseWarnings: TranslationWarning[] | undefined
    useTextboxPrompt: boolean
    textboxText: string
}): TranslationWarning[] {
    if (!params.useTextboxPrompt) {
        return params.primaryResponseWarnings || []
    }
    if (params.textboxText) {
        return params.textboxResponseWarnings || []
    }
    return params.primaryResponseWarnings || []
}

export async function executeTranslate(input: TranslateInput): Promise<TranslateOutput> {
    const { originalTexts, translationMode: pluginMode = 'standard', settingsSnapshot } = input

    if (originalTexts.length === 0) {
        return {
            translatedTexts: [],
            textboxTexts: [],
            warnings: []
        }
    }

    const settings = settingsSnapshot
    const requestMode = settings.translation.translationMode || 'batch'
    const constraintPayload = resolveConstraintPayloadForTranslation({
        isBookshelfMode: input.isBookshelfMode,
        constraints: input.bookTranslationConstraints,
    })

    if (requestMode === 'single') {
        // ==================== 逐气泡翻译模式 ====================
        console.log(`[翻译] 使用逐气泡翻译模式，共 ${originalTexts.length} 个气泡`)

        const translatedTexts: string[] = []
        const textboxTexts: string[] = []
        const warnings: TranslationWarning[] = []

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
                // 固定使用逐气泡翻译的提示词
                const promptContent = settings.translation.openaiOptions.request.forceJsonOutput
                    ? settings.translation.singleJsonPrompt
                    : settings.translation.singleNormalPrompt

                const response = await translateSingleText({
                    original_text: originalText,
                    translation_mode: pluginMode,
                    translation_scope: 'bubble',
                    model_provider: settings.translation.provider,
                    model_name: settings.translation.modelName,
                    api_key: settings.translation.apiKey,
                    custom_base_url: settings.translation.customBaseUrl,
                    target_language: settings.targetLanguage,
                    prompt_content: promptContent,
                    ...constraintPayload,
                    openai_options: serializeOpenAICompatibleOptionsForApi(settings.translation.openaiOptions)
                })

                if (response.success && response.data) {
                    translatedTexts.push(response.data.translated_text || '')
                } else {
                    console.warn(`[翻译] 气泡 ${i + 1} 翻译失败: ${response.error}`)
                    translatedTexts.push(`【翻译失败】请检查终端中的错误日志`)
                }

                // 文本框提示词（如果启用）
                if (settings.useTextboxPrompt && settings.textboxPrompt) {
                    const textboxResponse = await translateSingleText({
                        original_text: originalText,
                        translation_mode: pluginMode,
                        translation_scope: 'bubble',
                        model_provider: settings.translation.provider,
                        model_name: settings.translation.modelName,
                        api_key: settings.translation.apiKey,
                        custom_base_url: settings.translation.customBaseUrl,
                        target_language: settings.targetLanguage,
                        prompt_content: settings.textboxPrompt,
                        ...constraintPayload,
                        openai_options: serializeOpenAICompatibleOptionsForApi({
                            ...settings.translation.openaiOptions,
                            request: {
                                ...settings.translation.openaiOptions.request,
                                forceJsonOutput: false
                            }
                        })
                    })

                    if (textboxResponse.success && textboxResponse.data) {
                        const textboxText = textboxResponse.data.translated_text || ''
                        textboxTexts.push(textboxText)
                        warnings.push(...mapWarnings(resolveEffectiveTranslationWarnings({
                            primaryResponseWarnings: response.data?.warnings,
                            textboxResponseWarnings: textboxResponse.data.warnings,
                            useTextboxPrompt: settings.useTextboxPrompt,
                            textboxText,
                        }), {
                            imageIndex: input.imageIndex,
                            bubbleIndex: i,
                        }))
                    } else {
                        textboxTexts.push('')
                        warnings.push(...mapWarnings(resolveEffectiveTranslationWarnings({
                            primaryResponseWarnings: response.data?.warnings,
                            textboxResponseWarnings: undefined,
                            useTextboxPrompt: settings.useTextboxPrompt,
                            textboxText: '',
                        }), {
                            imageIndex: input.imageIndex,
                            bubbleIndex: i,
                        }))
                    }
                } else if (response.success && response.data) {
                    warnings.push(...mapWarnings(response.data.warnings, {
                        imageIndex: input.imageIndex,
                        bubbleIndex: i,
                    }))
                }

            } catch (error) {
                console.error(`[翻译] 气泡 ${i + 1} 翻译出错:`, error)
                translatedTexts.push(`【翻译失败】请检查终端中的错误日志`)
                if (settings.useTextboxPrompt) {
                    textboxTexts.push('')
                }
            }
        }

        console.log(`[翻译] 逐气泡翻译完成，成功 ${translatedTexts.filter(t => t && !t.startsWith('[翻译')).length}/${originalTexts.length}`)
        return { translatedTexts, textboxTexts, warnings }

    } else {
        // ==================== 整页批量翻译模式 ====================
        console.log(`[翻译] 使用整页批量翻译模式，共 ${originalTexts.length} 个气泡`)

        const response: ParallelTranslateResponse = await parallelTranslate({
            original_texts: originalTexts,
            translation_mode: pluginMode,
            translation_scope: 'image',
            target_language: settings.targetLanguage,
            source_language: settings.sourceLanguage,
            model_provider: settings.translation.provider,
            model_name: settings.translation.modelName,
            api_key: settings.translation.apiKey,
            custom_base_url: settings.translation.customBaseUrl,
            prompt_content: settings.translatePrompt,
            textbox_prompt_content: settings.textboxPrompt,
            use_textbox_prompt: settings.useTextboxPrompt,
            ...constraintPayload,
            openai_options: serializeOpenAICompatibleOptionsForApi(settings.translation.openaiOptions)
        })

        if (!response.success) {
            throw new Error(response.error || '翻译失败')
        }

        return {
            translatedTexts: response.translated_texts || [],
            textboxTexts: response.textbox_texts || [],
            warnings: response.warnings || []
        }
    }
}
