/**
 * 翻译步骤（普通翻译）
 * 提取自 SequentialPipeline.ts Line 348-468
 * 
 * 注意：这是最复杂的步骤，包含两种翻译模式
 */
import { parallelTranslate, type ParallelTranslateResponse } from '@/api/parallelTranslate'
import { translateSingleText } from '@/api/translate'
import { useSettingsStore } from '@/stores/settingsStore'
import type { RateLimiter } from '@/utils/rateLimiter'

export interface TranslateInput {
    imageIndex: number
    originalTexts: string[]
    rateLimiter?: RateLimiter | null
}

export interface TranslateOutput {
    translatedTexts: string[]
    textboxTexts: string[]
}

export async function executeTranslate(input: TranslateInput): Promise<TranslateOutput> {
    const { originalTexts, rateLimiter = null } = input

    if (originalTexts.length === 0) {
        return {
            translatedTexts: [],
            textboxTexts: []
        }
    }

    const settingsStore = useSettingsStore()
    const settings = settingsStore.settings
    const translationMode = settings.translation.translationMode || 'batch'

    if (translationMode === 'single') {
        // ==================== 逐气泡翻译模式 ====================
        console.log(`[翻译] 使用逐气泡翻译模式，共 ${originalTexts.length} 个气泡`)

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
                // 固定使用逐气泡翻译的提示词
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
                    prompt_content: promptContent,
                    use_json_format: settings.translation.isJsonMode,
                    rpm_limit_translation: settings.translation.rpmLimit,
                    max_retries: settings.translation.maxRetries
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

                // RPM限制等待
                if (rateLimiter && i < originalTexts.length - 1) {
                    await rateLimiter.acquire()
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
        return { translatedTexts, textboxTexts }

    } else {
        // ==================== 整页批量翻译模式 ====================
        console.log(`[翻译] 使用整页批量翻译模式，共 ${originalTexts.length} 个气泡`)

        const response: ParallelTranslateResponse = await parallelTranslate({
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

        return {
            translatedTexts: response.translated_texts || [],
            textboxTexts: response.textbox_texts || []
        }
    }
}
