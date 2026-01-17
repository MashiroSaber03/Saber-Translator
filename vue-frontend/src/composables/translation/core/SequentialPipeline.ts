/**
 * 顺序翻译管线 - 原子步骤版本
 * 
 * 设计理念：与并行管线完全一致的原子步骤
 * 
 * 7个原子步骤：
 * 1. detection - 气泡检测
 * 2. ocr - 文字识别
 * 3. color - 颜色提取
 * 4. translate - 普通翻译
 * 5. aiTranslate - AI翻译（高质量翻译和校对共用）
 * 6. inpaint - 背景修复
 * 7. render - 渲染译文
 */

import { ref, computed } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { useValidation } from '../../useValidation'
import { useToast } from '@/utils/toast'
import { createRateLimiter, type RateLimiter } from '@/utils/rateLimiter'
import { createProgressManager } from './progressManager'
import type {
    PipelineConfig,
    PipelineResult,
    SavedTextStyles,
    TranslationMode
} from './types'
import type { ImageData as AppImageData } from '@/types/image'
import type { BubbleState, BubbleCoords } from '@/types/bubble'

// 分步 API
import {
    parallelDetect,
    parallelOcr,
    parallelColor,
    parallelTranslate,
    parallelInpaint,
    parallelRender,
    type ParallelDetectResponse,
    type ParallelOcrResponse,
    type ParallelColorResponse,
    type ParallelTranslateResponse,
    type ParallelInpaintResponse,
    type ParallelRenderResponse
} from '@/api/parallelTranslate'
import { hqTranslateBatch } from '@/api/translate'

// 自动保存模块
import {
    shouldEnableAutoSave,
    preSaveOriginalImages,
    saveTranslatedImage,
    finalizeSave,
    resetSaveState
} from './saveStep'

// ============================================================
// 原子步骤类型
// ============================================================

export type AtomicStepType =
    | 'detection'     // 气泡检测
    | 'ocr'           // 文字识别
    | 'color'         // 颜色提取
    | 'translate'     // 普通翻译
    | 'aiTranslate'   // AI翻译（高质量翻译 & 校对共用）
    | 'inpaint'       // 背景修复
    | 'render'        // 渲染
    | 'save'          // 自动保存（书架模式）

/**
 * 步骤链配置
 */
export const STEP_CHAIN_CONFIGS: Record<TranslationMode, AtomicStepType[]> = {
    standard: ['detection', 'ocr', 'color', 'translate', 'inpaint', 'render'],
    hq: ['detection', 'ocr', 'color', 'aiTranslate', 'inpaint', 'render'],
    proofread: ['aiTranslate', 'render'],
    removeText: ['detection', 'inpaint', 'render']
}

/** 步骤显示名称 */
const STEP_LABELS: Record<AtomicStepType, string> = {
    detection: '气泡检测',
    ocr: '文字识别',
    color: '颜色提取',
    translate: '翻译',
    aiTranslate: 'AI翻译',
    inpaint: '背景修复',
    render: '渲染',
    save: '保存'
}

// ============================================================
// 任务状态
// ============================================================

interface TaskState {
    imageIndex: number
    image: AppImageData

    // 检测结果
    bubbleCoords: BubbleCoords[]
    bubbleAngles: number[]
    bubblePolygons: number[][][]
    autoDirections: string[]
    rawMask?: string
    textlinesPerBubble: any[]

    // OCR结果
    originalTexts: string[]

    // 颜色结果
    colors: Array<{
        textColor: string
        bgColor: string
        autoFgColor?: [number, number, number] | null
        autoBgColor?: [number, number, number] | null
    }>

    // 翻译结果
    translatedTexts: string[]
    textboxTexts: string[]

    // 修复结果
    cleanImage?: string

    // 渲染结果
    finalImage?: string
    bubbleStates?: BubbleState[]
}

// ============================================================
// 顺序管线 Composable
// ============================================================

export function useSequentialPipeline() {
    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()
    const settingsStore = useSettingsStore()
    const validation = useValidation()
    const toast = useToast()

    const { progress, reporter } = createProgressManager()
    const isExecuting = ref(false)
    const rateLimiter = ref<RateLimiter | null>(null)
    let savedTextStyles: SavedTextStyles | null = null
    let currentMode: TranslationMode = 'standard'

    const isTranslating = computed(() => isExecuting.value || imageStore.isBatchTranslationInProgress)
    const progressPercent = computed(() => progress.value.percentage || 0)

    // ============================================================
    // 工具函数
    // ============================================================

    function initRateLimiter(): void {
        const rpm = settingsStore.settings.translation.rpmLimit
        if (!rateLimiter.value) {
            rateLimiter.value = createRateLimiter(rpm)
        } else {
            rateLimiter.value.setRpm(rpm)
        }
    }

    function validateConfig(config: PipelineConfig): boolean {
        const validationType = config.mode === 'hq' ? 'hq'
            : config.mode === 'proofread' ? 'proofread'
                : config.mode === 'removeText' ? 'ocr'
                    : 'normal'
        return validation.validateBeforeTranslation(validationType)
    }

    function saveCurrentStyles(): void {
        const { textStyle } = settingsStore.settings
        const layoutDirectionValue = textStyle.layoutDirection
        savedTextStyles = {
            fontFamily: textStyle.fontFamily,
            fontSize: textStyle.fontSize,
            autoFontSize: textStyle.autoFontSize,
            autoTextDirection: layoutDirectionValue === 'auto',
            textDirection: layoutDirectionValue === 'auto' ? 'vertical' : layoutDirectionValue,
            layoutDirection: layoutDirectionValue,  // 保存用户原始选择（包括 'auto'）
            fillColor: textStyle.fillColor,
            textColor: textStyle.textColor,
            rotationAngle: 0,
            strokeEnabled: textStyle.strokeEnabled,
            strokeColor: textStyle.strokeColor,
            strokeWidth: textStyle.strokeWidth,
            useAutoTextColor: textStyle.useAutoTextColor,
            inpaintMethod: textStyle.inpaintMethod
        }
    }

    function extractBase64(dataUrl: string): string {
        if (dataUrl.includes('base64,')) {
            return dataUrl.split('base64,')[1] || ''
        }
        return dataUrl
    }

    function getImagesToProcess(config: PipelineConfig): { image: AppImageData; index: number }[] {
        const images = imageStore.images
        if (config.scope === 'current') {
            const currentImage = imageStore.currentImage
            return currentImage ? [{ image: currentImage, index: imageStore.currentImageIndex }] : []
        }
        if (config.scope === 'failed') {
            return imageStore.getFailedImageIndices()
                .map(index => ({ image: images[index]!, index }))
                .filter(item => item.image !== undefined)
        }
        if (config.scope === 'range' && config.pageRange) {
            // 页码从1开始，转换为0索引
            const startIndex = Math.max(0, config.pageRange.startPage - 1)
            const endIndex = Math.min(images.length - 1, config.pageRange.endPage - 1)

            if (startIndex > endIndex || startIndex >= images.length) {
                return []
            }

            return images
                .slice(startIndex, endIndex + 1)
                .map((image, idx) => ({ image, index: startIndex + idx }))
        }
        return images.map((image, index) => ({ image, index }))
    }

    // ============================================================
    // 原子步骤执行器
    // ============================================================

    async function executeDetection(task: TaskState): Promise<void> {
        const settings = settingsStore.settings
        const base64 = extractBase64(task.image.originalDataURL)

        const response: ParallelDetectResponse = await parallelDetect({
            image: base64,
            detector_type: settings.textDetector,
            box_expand_ratio: settings.boxExpand.ratio,
            box_expand_top: settings.boxExpand.top,
            box_expand_bottom: settings.boxExpand.bottom,
            box_expand_left: settings.boxExpand.left,
            box_expand_right: settings.boxExpand.right
        })

        if (!response.success) {
            throw new Error(response.error || '检测失败')
        }

        task.bubbleCoords = (response.bubble_coords || []) as BubbleCoords[]
        task.bubbleAngles = response.bubble_angles || []
        task.bubblePolygons = response.bubble_polygons || []
        task.autoDirections = response.auto_directions || []
        task.rawMask = response.raw_mask
        task.textlinesPerBubble = response.textlines_per_bubble || []
    }

    async function executeOcr(task: TaskState): Promise<void> {
        if (task.bubbleCoords.length === 0) {
            task.originalTexts = []
            return
        }

        const settings = settingsStore.settings
        const base64 = extractBase64(task.image.originalDataURL)

        // PaddleOCR-VL 使用独立的源语言设置
        const ocrSourceLanguage = settings.ocrEngine === 'paddleocr_vl'
            ? settings.paddleOcrVl?.sourceLanguage || 'japanese'
            : settings.sourceLanguage

        const response: ParallelOcrResponse = await parallelOcr({
            image: base64,
            bubble_coords: task.bubbleCoords,
            source_language: ocrSourceLanguage,
            ocr_engine: settings.ocrEngine,
            baidu_api_key: settings.baiduOcr?.apiKey,
            baidu_secret_key: settings.baiduOcr?.secretKey,
            baidu_version: settings.baiduOcr?.version,
            baidu_ocr_language: settings.baiduOcr?.sourceLanguage,
            ai_vision_provider: settings.aiVisionOcr?.provider,
            ai_vision_api_key: settings.aiVisionOcr?.apiKey,
            ai_vision_model_name: settings.aiVisionOcr?.modelName,
            ai_vision_ocr_prompt: settings.aiVisionOcr?.prompt,
            custom_ai_vision_base_url: settings.aiVisionOcr?.customBaseUrl,
            textlines_per_bubble: task.textlinesPerBubble
        })

        if (!response.success) {
            throw new Error(response.error || 'OCR失败')
        }

        task.originalTexts = response.original_texts || []
    }

    async function executeColor(task: TaskState): Promise<void> {
        if (task.bubbleCoords.length === 0) {
            task.colors = []
            return
        }

        const base64 = extractBase64(task.image.originalDataURL)

        const response: ParallelColorResponse = await parallelColor({
            image: base64,
            bubble_coords: task.bubbleCoords,
            textlines_per_bubble: task.textlinesPerBubble
        })

        if (!response.success) {
            throw new Error(response.error || '颜色提取失败')
        }

        task.colors = response.colors || []
    }

    async function executeTranslate(task: TaskState): Promise<void> {
        if (task.originalTexts.length === 0) {
            task.translatedTexts = []
            task.textboxTexts = []
            return
        }

        const settings = settingsStore.settings

        const response: ParallelTranslateResponse = await parallelTranslate({
            original_texts: task.originalTexts,
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

        task.translatedTexts = response.translated_texts || []
        task.textboxTexts = response.textbox_texts || []
    }

    /**
     * AI翻译步骤（高质量翻译 & 校对共用）
     * 根据 currentMode 决定使用哪种配置
     */
    async function executeAiTranslate(tasks: TaskState[]): Promise<void> {
        const settings = settingsStore.settings
        const isProofread = currentMode === 'proofread'

        // 收集 JSON 数据
        const jsonData = tasks.map(t => {
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
                    bubbles: t.originalTexts.map((text, idx) => ({
                        bubbleIndex: idx,
                        original: text,
                        translated: '',
                        textDirection: t.autoDirections[idx] || 'vertical'
                    }))
                }
            }
        })

        // 收集图片
        const imageBase64Array = tasks.map(t => {
            const dataUrl = isProofread
                ? (t.image.translatedDataURL || t.image.originalDataURL)
                : t.image.originalDataURL
            return extractBase64(dataUrl)
        })

        // 获取配置
        const aiConfig = isProofread ? settings.proofreading.rounds[0] : settings.hqTranslation
        const prompt = isProofread ? aiConfig?.prompt : settings.hqTranslation.prompt
        const systemPrompt = isProofread
            ? '你是一个专业的漫画翻译校对助手，能够根据漫画图像内容检查和修正翻译。'
            : '你是一个专业的漫画翻译助手，能够根据漫画图像内容和上下文提供高质量的翻译。'

        // 构建消息
        const jsonString = JSON.stringify(jsonData, null, 2)
        type MessageContent = { type: 'text'; text: string } | { type: 'image_url'; image_url: { url: string } }
        const userContent: MessageContent[] = [
            {
                type: 'text',
                text: (prompt || '') + '\n\n以下是JSON数据:\n```json\n' + jsonString + '\n```'
            }
        ]
        for (const imgBase64 of imageBase64Array) {
            userContent.push({
                type: 'image_url',
                image_url: { url: `data:image/png;base64,${imgBase64}` }
            })
        }

        const messages = [
            { role: 'system' as const, content: systemPrompt },
            { role: 'user' as const, content: userContent }
        ]

        // 调用 API
        const hqConfig = settings.hqTranslation
        const roundConfig = isProofread ? aiConfig : null
        const response = await hqTranslateBatch({
            provider: (isProofread ? roundConfig?.provider : hqConfig.provider) || 'openai',
            api_key: (isProofread ? roundConfig?.apiKey : hqConfig.apiKey) || '',
            model_name: (isProofread ? roundConfig?.modelName : hqConfig.modelName) || '',
            custom_base_url: isProofread ? roundConfig?.customBaseUrl : hqConfig.customBaseUrl,
            messages,
            low_reasoning: isProofread ? roundConfig?.lowReasoning : hqConfig.lowReasoning,
            force_json_output: isProofread ? roundConfig?.forceJsonOutput : hqConfig.forceJsonOutput,
            no_thinking_method: isProofread ? roundConfig?.noThinkingMethod : hqConfig.noThinkingMethod,
            use_stream: isProofread ? false : hqConfig.useStream,
            max_retries: isProofread ? (settings.proofreading.maxRetries || 2) : (hqConfig.maxRetries || 2)
        })

        // 解析结果
        const forceJson = isProofread ? (roundConfig?.forceJsonOutput || false) : hqConfig.forceJsonOutput
        const translatedData = parseHqResponse(response, forceJson)

        // 校对模式可能有多轮
        let currentData = translatedData || jsonData
        if (isProofread && settings.proofreading.rounds.length > 1) {
            for (let i = 1; i < settings.proofreading.rounds.length; i++) {
                const round = settings.proofreading.rounds[i]!
                const roundJsonString = JSON.stringify(currentData, null, 2)
                const roundUserContent: MessageContent[] = [
                    {
                        type: 'text',
                        text: round.prompt + '\n\n以下是JSON数据:\n```json\n' + roundJsonString + '\n```'
                    }
                ]
                for (const imgBase64 of imageBase64Array) {
                    roundUserContent.push({
                        type: 'image_url',
                        image_url: { url: `data:image/png;base64,${imgBase64}` }
                    })
                }

                const roundMessages = [
                    { role: 'system' as const, content: systemPrompt },
                    { role: 'user' as const, content: roundUserContent }
                ]

                const roundResponse = await hqTranslateBatch({
                    provider: round.provider,
                    api_key: round.apiKey,
                    model_name: round.modelName,
                    custom_base_url: round.customBaseUrl,
                    messages: roundMessages,
                    low_reasoning: round.lowReasoning,
                    force_json_output: round.forceJsonOutput,
                    no_thinking_method: round.noThinkingMethod,
                    use_stream: false,
                    max_retries: round.maxRetries || settings.proofreading.maxRetries || 2
                })

                const roundResult = parseHqResponse(roundResponse, round.forceJsonOutput)
                if (roundResult) {
                    currentData = roundResult
                }
            }
        }

        // 填充结果
        for (const t of tasks) {
            const taskData = (currentData as any[])?.find((d: any) => d.imageIndex === t.imageIndex)
            if (taskData) {
                t.translatedTexts = taskData.bubbles.map((b: any) => b.translated)
            } else {
                t.translatedTexts = []
            }
            t.textboxTexts = []
        }
    }

    async function executeInpaint(task: TaskState): Promise<void> {
        if (task.bubbleCoords.length === 0) {
            task.cleanImage = extractBase64(task.image.originalDataURL)
            return
        }

        const settings = settingsStore.settings
        const { textStyle, preciseMask } = settings
        const base64 = extractBase64(task.image.originalDataURL)

        const response: ParallelInpaintResponse = await parallelInpaint({
            image: base64,
            bubble_coords: task.bubbleCoords,
            bubble_polygons: task.bubblePolygons,
            raw_mask: task.rawMask,
            method: textStyle.inpaintMethod === 'solid' ? 'solid' : 'lama',
            lama_model: textStyle.inpaintMethod === 'litelama' ? 'litelama' : 'lama_mpe',
            fill_color: textStyle.fillColor,
            mask_dilate_size: preciseMask.dilateSize,
            mask_box_expand_ratio: preciseMask.boxExpandRatio
        })

        if (!response.success) {
            throw new Error(response.error || '背景修复失败')
        }

        task.cleanImage = response.clean_image
    }

    async function executeRender(task: TaskState): Promise<void> {
        if (!task.cleanImage) {
            // 校对模式下，如果没有干净背景图，说明图片没有被翻译过
            if (currentMode === 'proofread') {
                throw new Error('此图片尚未翻译，请先翻译后再进行校对')
            }
            throw new Error('缺少干净背景图片')
        }

        const { textStyle } = settingsStore.settings

        // 【简化设计】计算 textDirection：
        // - 如果全局设置是 'auto'，使用检测结果
        // - 否则使用全局设置的值
        const globalTextDir = savedTextStyles?.autoTextDirection
            ? 'auto'  // autoTextDirection 为 true 表示用户选择了 'auto'
            : (savedTextStyles?.textDirection || textStyle.layoutDirection)

        // 构建 bubbleStates
        const bubbleStates: BubbleState[] = task.bubbleCoords.map((coords, idx) => {
            const autoDir = task.autoDirections[idx] || 'vertical'
            // 将后端返回的 'v'/'h' 格式转换为 'vertical'/'horizontal'
            const mappedAutoDir: 'vertical' | 'horizontal' = autoDir === 'v' ? 'vertical'
                : autoDir === 'h' ? 'horizontal'
                    : (autoDir === 'vertical' || autoDir === 'horizontal') ? autoDir : 'vertical'

            // 【简化设计】textDirection 直接使用具体方向值
            const textDirection =
                (globalTextDir === 'vertical' || globalTextDir === 'horizontal')
                    ? globalTextDir
                    : mappedAutoDir

            // 【修复】颜色处理：根据 useAutoTextColor 设置决定是否使用自动提取的颜色
            const useAutoColor = savedTextStyles?.useAutoTextColor ?? textStyle.useAutoTextColor
            let finalTextColor = savedTextStyles?.textColor || textStyle.textColor
            let finalFillColor = savedTextStyles?.fillColor || textStyle.fillColor
            const colorInfo = task.colors[idx]

            if (useAutoColor && colorInfo) {
                if (colorInfo.textColor) finalTextColor = colorInfo.textColor
                if (colorInfo.bgColor) finalFillColor = colorInfo.bgColor
            }

            return {
                coords,
                polygon: [] as number[][],
                position: { x: 0, y: 0 },
                rotationAngle: task.bubbleAngles[idx] || 0,
                originalText: task.originalTexts[idx] || '',
                translatedText: task.translatedTexts[idx] || '',
                textboxText: task.textboxTexts[idx] || '',
                textDirection: textDirection as 'vertical' | 'horizontal',  // 渲染用的具体方向
                autoTextDirection: mappedAutoDir as 'vertical' | 'horizontal',  // 备份检测结果
                fontSize: savedTextStyles?.fontSize || textStyle.fontSize,
                fontFamily: savedTextStyles?.fontFamily || textStyle.fontFamily,
                autoFontSize: savedTextStyles?.autoFontSize ?? textStyle.autoFontSize,
                textColor: finalTextColor,
                fillColor: finalFillColor,
                strokeEnabled: savedTextStyles?.strokeEnabled ?? textStyle.strokeEnabled,
                strokeColor: savedTextStyles?.strokeColor || textStyle.strokeColor,
                strokeWidth: savedTextStyles?.strokeWidth || textStyle.strokeWidth,
                inpaintMethod: savedTextStyles?.inpaintMethod || textStyle.inpaintMethod,
                autoFgColor: task.colors[idx]?.autoFgColor || null,
                autoBgColor: task.colors[idx]?.autoBgColor || null
            }
        })

        const response: ParallelRenderResponse = await parallelRender({
            clean_image: task.cleanImage,
            bubble_states: bubbleStates,
            fontSize: savedTextStyles?.fontSize || textStyle.fontSize,
            fontFamily: savedTextStyles?.fontFamily || textStyle.fontFamily,
            textDirection: savedTextStyles?.textDirection || textStyle.layoutDirection,
            textColor: savedTextStyles?.textColor || textStyle.textColor,
            strokeEnabled: savedTextStyles?.strokeEnabled ?? textStyle.strokeEnabled,
            strokeColor: savedTextStyles?.strokeColor || textStyle.strokeColor,
            strokeWidth: savedTextStyles?.strokeWidth || textStyle.strokeWidth,
            autoFontSize: savedTextStyles?.autoFontSize ?? textStyle.autoFontSize,
            use_individual_styles: true
        })

        if (!response.success) {
            throw new Error(response.error || '渲染失败')
        }

        task.finalImage = response.final_image
        task.bubbleStates = response.bubble_states || bubbleStates
    }

    // ============================================================
    // 辅助函数
    // ============================================================

    /**
     * 执行单个步骤（通用函数，消除重复代码）
     * 注意：aiTranslate 步骤在 executeBatchMode 中有特殊处理，不会通过此函数调用
     */
    async function executeStep(step: AtomicStepType, task: TaskState): Promise<void> {
        switch (step) {
            case 'detection':
                await executeDetection(task)
                break
            case 'ocr':
                await executeOcr(task)
                break
            case 'color':
                await executeColor(task)
                break
            case 'translate':
                await executeTranslate(task)
                break
            case 'inpaint':
                await executeInpaint(task)
                break
            case 'render':
                await executeRender(task)
                break
            case 'save':
                // 保存步骤：保存当前已渲染的图片（仅书架模式）
                await saveTranslatedImage(task.imageIndex)
                break
            case 'aiTranslate':
                // 此分支仅作为类型完整性保留，实际不会被调用
                // aiTranslate 在 executeBatchMode 中有批量处理逻辑
                throw new Error('aiTranslate 应通过批量处理逻辑调用')
        }
    }

    function parseHqResponse(
        response: { success: boolean; results?: any[]; content?: string; error?: string },
        forceJsonOutput: boolean
    ): any[] | null {
        if (!response.success) {
            console.error('API调用失败:', response.error)
            return null
        }

        if (response.results && response.results.length > 0) {
            const firstItem = response.results[0]
            if (firstItem && 'imageIndex' in firstItem && 'bubbles' in firstItem) {
                return response.results
            }
        }

        const content = (response as { content?: string }).content
        if (content) {
            if (forceJsonOutput) {
                try {
                    return JSON.parse(content)
                } catch {
                    return null
                }
            } else {
                const jsonMatch = content.match(/```json\s*([\s\S]*?)\s*```/)
                if (jsonMatch?.[1]) {
                    try {
                        return JSON.parse(jsonMatch[1])
                    } catch {
                        return null
                    }
                }
            }
        }

        return null
    }

    function updateImageStore(task: TaskState): void {
        const translatedDataURL = task.finalImage
            ? `data:image/png;base64,${task.finalImage}`
            : task.cleanImage
                ? `data:image/png;base64,${task.cleanImage}`
                : null

        const { textStyle } = settingsStore.settings

        imageStore.updateImageByIndex(task.imageIndex, {
            translatedDataURL,
            cleanImageData: task.cleanImage || null,
            bubbleStates: task.bubbleStates,
            bubbleCoords: task.bubbleCoords,
            bubbleAngles: task.bubbleAngles,
            originalTexts: task.originalTexts,
            textboxTexts: task.textboxTexts,
            bubbleTexts: task.translatedTexts,
            translationStatus: 'completed',
            translationFailed: false,
            showOriginal: false,
            hasUnsavedChanges: true,
            // 保存用户翻译时选择的设置（用于切换图片时恢复）
            // 【修复】保存完整的文字设置，避免切换图片后侧边栏显示默认值
            fontSize: savedTextStyles?.fontSize ?? textStyle.fontSize,
            autoFontSize: savedTextStyles?.autoFontSize ?? textStyle.autoFontSize,
            fontFamily: savedTextStyles?.fontFamily ?? textStyle.fontFamily,
            layoutDirection: savedTextStyles?.layoutDirection ?? textStyle.layoutDirection,
            textColor: savedTextStyles?.textColor ?? textStyle.textColor,
            fillColor: savedTextStyles?.fillColor ?? textStyle.fillColor,
            strokeEnabled: savedTextStyles?.strokeEnabled ?? textStyle.strokeEnabled,
            strokeColor: savedTextStyles?.strokeColor ?? textStyle.strokeColor,
            strokeWidth: savedTextStyles?.strokeWidth ?? textStyle.strokeWidth,
            inpaintMethod: savedTextStyles?.inpaintMethod ?? textStyle.inpaintMethod,
            useAutoTextColor: savedTextStyles?.useAutoTextColor ?? textStyle.useAutoTextColor
        })

        if (task.imageIndex === imageStore.currentImageIndex && task.bubbleStates) {
            bubbleStore.setBubbles(task.bubbleStates)
        }
    }

    // ============================================================
    // 主执行函数
    // ============================================================

    /**
     * 判断是否使用逐张处理模式
     * - standard / removeText: 逐张处理（每张图完成全部步骤后再处理下一张）
     * - hq / proofread: 按批次处理（批次内保持按步骤批量处理）
     */
    function shouldUsePerImageMode(mode: TranslationMode): boolean {
        return mode === 'standard' || mode === 'removeText'
    }

    /**
     * 获取批次大小配置
     * 仅在 executeBatchMode 中调用，用于 hq 和 proofread 模式
     */
    function getBatchSize(mode: TranslationMode): number {
        const settings = settingsStore.settings
        if (mode === 'hq') {
            return settings.hqTranslation.batchSize || 5
        }
        if (mode === 'proofread') {
            // 使用第一轮校对的批次大小，如果没有则使用默认值
            return settings.proofreading.rounds[0]?.batchSize || 5
        }
        // 防御性代码：standard 和 removeText 模式不应调用此函数
        return 1
    }

    /**
     * 逐张处理模式（标准翻译/消除文字）
     * 每张图片走完全部步骤后再处理下一张
     */
    async function executePerImageMode(
        tasks: TaskState[],
        stepChain: AtomicStepType[],
        config: PipelineConfig,
        errors: string[]
    ): Promise<{ completed: number; failed: number }> {
        let completed = 0
        let failed = 0

        for (let imageIdx = 0; imageIdx < tasks.length; imageIdx++) {
            const task = tasks[imageIdx]!

            // 检查是否取消
            if (config.scope === 'all' && !imageStore.isBatchTranslationInProgress) {
                console.log(`⏹️ 批量翻译已取消，停止处理`)
                break
            }

            const imageProgress = Math.floor((imageIdx / tasks.length) * 90)
            reporter.setPercentage(imageProgress, `处理图片 ${imageIdx + 1}/${tasks.length}`)
            toast.info(`处理图片 ${imageIdx + 1}/${tasks.length}...`)

            imageStore.setTranslationStatus(task.imageIndex, 'processing')
            let taskFailed = false

            // 对当前图片执行全部步骤
            for (let stepIdx = 0; stepIdx < stepChain.length; stepIdx++) {
                const step = stepChain[stepIdx]!

                if (taskFailed) break

                if (rateLimiter.value) {
                    await rateLimiter.value.acquire()
                }

                try {
                    const stepProgress = imageProgress + Math.floor((stepIdx / stepChain.length) * (90 / tasks.length))
                    reporter.setPercentage(stepProgress, `图片 ${imageIdx + 1}: ${STEP_LABELS[step]}`)

                    await executeStep(step, task)
                } catch (err) {
                    const msg = err instanceof Error ? err.message : '未知错误'
                    errors.push(`图片 ${task.imageIndex + 1}: ${step} - ${msg}`)
                    imageStore.setTranslationStatus(task.imageIndex, 'failed', msg)
                    taskFailed = true
                    failed++
                }
            }

            // 这张图片处理完成，立即更新 store
            if (!taskFailed) {
                updateImageStore(task)
                completed++
                console.log(`✅ 图片 ${imageIdx + 1}/${tasks.length} 处理完成`)
            }
        }

        return { completed, failed }
    }

    /**
     * 批次处理模式（高质量翻译/AI校对）
     * 
     * 处理流程：
     * 1. 对每张图片逐张执行 aiTranslate 之前的步骤
     * 2. 批量发送 aiTranslate（利用 AI 的多图上下文能力）
     * 3. 对每张图片逐张执行 aiTranslate 之后的步骤
     * 
     * 这样设计的好处：
     * - 除 aiTranslate 外，其他步骤都是逐张处理，代码简单
     * - 未来添加新步骤更容易
     * - aiTranslate 仍然保持批量发送，利用 AI 的上下文理解能力
     */
    async function executeBatchMode(
        tasks: TaskState[],
        stepChain: AtomicStepType[],
        config: PipelineConfig,
        errors: string[]
    ): Promise<{ completed: number; failed: number }> {
        let completed = 0
        let failed = 0

        const batchSize = getBatchSize(config.mode)
        const totalBatches = Math.ceil(tasks.length / batchSize)

        // 找到 aiTranslate 步骤的位置
        const aiTranslateIdx = stepChain.indexOf('aiTranslate')
        const stepsBeforeAi = aiTranslateIdx >= 0 ? stepChain.slice(0, aiTranslateIdx) : stepChain
        const stepsAfterAi = aiTranslateIdx >= 0 ? stepChain.slice(aiTranslateIdx + 1) : []

        console.log(`📦 批次处理模式：共 ${tasks.length} 张图片，每批 ${batchSize} 张，共 ${totalBatches} 批`)
        console.log(`   AI翻译前步骤: [${stepsBeforeAi.join(' → ')}]`)
        console.log(`   AI翻译后步骤: [${stepsAfterAi.join(' → ')}]`)

        for (let batchIdx = 0; batchIdx < totalBatches; batchIdx++) {
            // 检查是否取消
            if (config.scope === 'all' && !imageStore.isBatchTranslationInProgress) {
                console.log(`⏹️ 批量翻译已取消，停止处理`)
                break
            }

            const batchStart = batchIdx * batchSize
            const batchEnd = Math.min(batchStart + batchSize, tasks.length)
            const batchTasks = tasks.slice(batchStart, batchEnd)

            const batchProgress = Math.floor((batchIdx / totalBatches) * 90)
            reporter.setPercentage(batchProgress, `处理批次 ${batchIdx + 1}/${totalBatches}`)
            toast.info(`处理批次 ${batchIdx + 1}/${totalBatches}（图片 ${batchStart + 1}-${batchEnd}）...`)

            // 标记批次内图片为处理中
            for (const task of batchTasks) {
                imageStore.setTranslationStatus(task.imageIndex, 'processing')
            }

            // 跟踪批次内失败的任务索引
            const batchFailedIndices = new Set<number>()

            // ========== 阶段1：逐张执行 aiTranslate 之前的步骤 ==========
            for (let i = 0; i < batchTasks.length; i++) {
                const task = batchTasks[i]!

                for (const step of stepsBeforeAi) {
                    if (batchFailedIndices.has(task.imageIndex)) break

                    if (rateLimiter.value) {
                        await rateLimiter.value.acquire()
                    }

                    try {
                        const stepProgress = batchProgress + Math.floor((i / batchTasks.length) * 30)
                        reporter.setPercentage(stepProgress, `图片 ${batchStart + i + 1}: ${STEP_LABELS[step]}`)
                        await executeStep(step, task)
                    } catch (err) {
                        const msg = err instanceof Error ? err.message : '未知错误'
                        errors.push(`图片 ${task.imageIndex + 1}: ${step} - ${msg}`)
                        imageStore.setTranslationStatus(task.imageIndex, 'failed', msg)
                        batchFailedIndices.add(task.imageIndex)
                    }
                }
            }

            // ========== 阶段2：批量执行 aiTranslate ==========
            if (aiTranslateIdx >= 0) {
                const stepProgress = batchProgress + 40
                reporter.setPercentage(stepProgress, `批次 ${batchIdx + 1}: ${STEP_LABELS['aiTranslate']}`)

                try {
                    const validTasks = batchTasks.filter(t => !batchFailedIndices.has(t.imageIndex))
                    if (validTasks.length > 0) {
                        await executeAiTranslate(validTasks)
                    }
                } catch (err) {
                    const msg = err instanceof Error ? err.message : '未知错误'
                    errors.push(`批次 ${batchIdx + 1} AI翻译失败: ${msg}`)
                    // AI翻译失败，标记所有未失败的任务为失败
                    for (const task of batchTasks) {
                        if (!batchFailedIndices.has(task.imageIndex)) {
                            imageStore.setTranslationStatus(task.imageIndex, 'failed', msg)
                            batchFailedIndices.add(task.imageIndex)
                        }
                    }
                }
            }

            // ========== 阶段3：逐张执行 aiTranslate 之后的步骤 ==========
            for (let i = 0; i < batchTasks.length; i++) {
                const task = batchTasks[i]!

                if (batchFailedIndices.has(task.imageIndex)) continue

                for (const step of stepsAfterAi) {
                    if (batchFailedIndices.has(task.imageIndex)) break

                    if (rateLimiter.value) {
                        await rateLimiter.value.acquire()
                    }

                    try {
                        const stepProgress = batchProgress + 50 + Math.floor((i / batchTasks.length) * 40)
                        reporter.setPercentage(stepProgress, `图片 ${batchStart + i + 1}: ${STEP_LABELS[step]}`)
                        await executeStep(step, task)
                    } catch (err) {
                        const msg = err instanceof Error ? err.message : '未知错误'
                        errors.push(`图片 ${task.imageIndex + 1}: ${step} - ${msg}`)
                        imageStore.setTranslationStatus(task.imageIndex, 'failed', msg)
                        batchFailedIndices.add(task.imageIndex)
                    }
                }

                // 这张图片处理完成（aiTranslate 后的步骤都完成了），立即更新 store
                if (!batchFailedIndices.has(task.imageIndex)) {
                    updateImageStore(task)
                    completed++
                    console.log(`✅ 图片 ${batchStart + i + 1} 处理完成`)
                }
            }

            // 统计失败数量
            failed += batchFailedIndices.size

            console.log(`✅ 批次 ${batchIdx + 1}/${totalBatches} 处理完成`)
        }

        return { completed, failed }
    }

    async function execute(config: PipelineConfig): Promise<PipelineResult> {
        if (!validateConfig(config)) {
            return { success: false, completed: 0, failed: 0, errors: ['配置验证失败'] }
        }

        const images = imageStore.images
        if (images.length === 0) {
            toast.error('请先上传图片')
            return { success: false, completed: 0, failed: 0, errors: ['没有图片'] }
        }

        currentMode = config.mode
        const usePerImageMode = shouldUsePerImageMode(config.mode)

        isExecuting.value = true
        if (config.scope === 'all' || config.scope === 'failed') {
            imageStore.setBatchTranslationInProgress(true)
        }
        initRateLimiter()
        saveCurrentStyles()

        const imagesToProcess = getImagesToProcess(config)
        const errors: string[] = []

        // 判断是否启用自动保存（书架模式 + 设置开启）
        const enableAutoSave = shouldEnableAutoSave()

        // 动态生成步骤链：如果启用自动保存，追加 save 步骤
        const stepChain = [...STEP_CHAIN_CONFIGS[config.mode]]
        if (enableAutoSave) {
            stepChain.push('save')
        }

        console.log(`🚀 顺序管线启动`)
        console.log(`   模式: ${config.mode}`)
        console.log(`   处理方式: ${usePerImageMode ? '逐张处理' : '批次处理'}`)
        console.log(`   步骤链: [${stepChain.join(' → ')}]`)
        console.log(`   自动保存: ${enableAutoSave ? '启用' : '禁用'}`)

        // 创建任务状态
        const tasks: TaskState[] = imagesToProcess.map(({ image, index }) => {
            const task: TaskState = {
                imageIndex: index,
                image,
                bubbleCoords: [],
                bubbleAngles: [],
                bubblePolygons: [],
                autoDirections: [],
                textlinesPerBubble: [],
                originalTexts: [],
                colors: [],
                translatedTexts: [],
                textboxTexts: []
            }

            // 校对模式需要从已有数据初始化
            if (config.mode === 'proofread' && image.bubbleStates && image.bubbleStates.length > 0) {
                task.bubbleCoords = image.bubbleStates.map(s => s.coords)
                task.bubbleAngles = image.bubbleStates.map(s => s.rotationAngle || 0)
                task.autoDirections = image.bubbleStates.map(s => s.autoTextDirection || s.textDirection || 'vertical')
                task.originalTexts = image.bubbleStates.map(s => s.originalText || '')
                task.translatedTexts = image.bubbleStates.map(s => s.translatedText || '')
                task.textboxTexts = image.bubbleStates.map(s => s.textboxText || '')
                task.colors = image.bubbleStates.map(s => ({
                    textColor: s.textColor || '',
                    bgColor: s.fillColor || '',
                    autoFgColor: s.autoFgColor || null,
                    autoBgColor: s.autoBgColor || null
                }))
                // 使用已有的干净背景图
                if (image.cleanImageData) {
                    task.cleanImage = image.cleanImageData
                }
            }

            return task
        })

        try {
            reporter.init(imagesToProcess.length, `${config.mode} 模式启动...`)

            // 如果启用自动保存，先执行预保存（保存所有原始图片）
            if (enableAutoSave) {
                reporter.setPercentage(0, '预保存原始图片...')
                const preSaveSuccess = await preSaveOriginalImages({
                    onStart: (total) => {
                        reporter.setPercentage(0, `预保存原始图片 0/${total}...`)
                    },
                    onProgress: (current, total) => {
                        const percent = Math.round((current / total) * 10) // 预保存占 0-10%
                        reporter.setPercentage(percent, `预保存原始图片 ${current}/${total}...`)
                    },
                    onComplete: () => {
                        reporter.setPercentage(10, '预保存完成，开始翻译...')
                    },
                    onError: (error) => {
                        reporter.setPercentage(0, `预保存失败: ${error}`)
                    }
                })
                if (!preSaveSuccess) {
                    // 预保存失败，提示用户但不阻止翻译
                    toast.warning('预保存失败，翻译完成后请手动保存')
                }
            }

            let result: { completed: number; failed: number }

            if (usePerImageMode) {
                // 逐张处理模式
                result = await executePerImageMode(tasks, stepChain, config, errors)
            } else {
                // 批次处理模式
                result = await executeBatchMode(tasks, stepChain, config, errors)
            }

            reporter.setPercentage(100, '完成！')

            const modeLabels: Record<TranslationMode, string> = {
                standard: '翻译',
                hq: '高质量翻译',
                proofread: 'AI校对',
                removeText: '消除文字'
            }
            toast.success(`${modeLabels[config.mode]}完成！`)

            return {
                success: result.failed === 0,
                completed: result.completed,
                failed: result.failed,
                errors: errors.length > 0 ? errors : undefined
            }

        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : '执行失败'
            toast.error(errorMessage)
            errors.push(errorMessage)
            return {
                success: false,
                completed: 0,
                failed: imagesToProcess.length,
                errors
            }

        } finally {
            isExecuting.value = false
            imageStore.setBatchTranslationInProgress(false)

            // 如果启用了自动保存，完成保存会话
            if (enableAutoSave) {
                await finalizeSave()
            }

            const currentIndex = imageStore.currentImageIndex
            const currentImage = imageStore.images[currentIndex]
            if (currentImage?.bubbleStates && currentImage.bubbleStates.length > 0) {
                bubbleStore.setBubbles(currentImage.bubbleStates)
            }

            setTimeout(() => reporter.finish(), 1000)
        }
    }

    function cancel(): void {
        if (imageStore.isBatchTranslationInProgress) {
            imageStore.setBatchTranslationInProgress(false)
            // 重置自动保存状态
            resetSaveState()
            toast.info('操作已取消')
        }
    }

    return {
        progress,
        isExecuting,
        isTranslating,
        progressPercent,
        execute,
        cancel,
        STEP_CHAIN_CONFIGS
    }
}
