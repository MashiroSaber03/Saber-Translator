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
    render: '渲染'
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
            fillColor: textStyle.fillColor,
            textColor: textStyle.textColor,
            rotationAngle: 0,
            strokeEnabled: textStyle.strokeEnabled,
            strokeColor: textStyle.strokeColor,
            strokeWidth: textStyle.strokeWidth
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

        const response: ParallelOcrResponse = await parallelOcr({
            image: base64,
            bubble_coords: task.bubbleCoords,
            source_language: settings.sourceLanguage,
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
                textColor: task.colors[idx]?.textColor || savedTextStyles?.textColor || textStyle.textColor,
                fillColor: savedTextStyles?.fillColor || textStyle.fillColor,
                strokeEnabled: savedTextStyles?.strokeEnabled ?? textStyle.strokeEnabled,
                strokeColor: savedTextStyles?.strokeColor || textStyle.strokeColor,
                strokeWidth: savedTextStyles?.strokeWidth || textStyle.strokeWidth,
                inpaintMethod: textStyle.inpaintMethod,
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
            hasUnsavedChanges: true
        })

        if (task.imageIndex === imageStore.currentImageIndex && task.bubbleStates) {
            bubbleStore.setBubbles(task.bubbleStates)
        }
    }

    // ============================================================
    // 主执行函数
    // ============================================================

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
        const stepChain = STEP_CHAIN_CONFIGS[config.mode]
        console.log(`🚀 顺序管线启动，模式: ${config.mode}, 步骤链: [${stepChain.join(' → ')}]`)

        isExecuting.value = true
        if (config.scope === 'all' || config.scope === 'failed') {
            imageStore.setBatchTranslationInProgress(true)
        }
        initRateLimiter()
        saveCurrentStyles()

        const imagesToProcess = getImagesToProcess(config)
        const errors: string[] = []
        let completed = 0
        let failed = 0

        // 创建任务状态
        const tasks: TaskState[] = imagesToProcess.map(({ image, index }) => ({
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
        }))

        try {
            reporter.init(imagesToProcess.length, `${config.mode} 模式启动...`)

            for (let stepIdx = 0; stepIdx < stepChain.length; stepIdx++) {
                const step = stepChain[stepIdx]!
                const stepProgress = Math.floor((stepIdx / stepChain.length) * 90)
                reporter.setPercentage(stepProgress, `执行: ${STEP_LABELS[step]}`)
                toast.info(`步骤 ${stepIdx + 1}/${stepChain.length}: ${STEP_LABELS[step]}...`)

                if (step === 'aiTranslate') {
                    // 批量步骤
                    await executeAiTranslate(tasks)
                } else {
                    // 逐张执行
                    for (let i = 0; i < tasks.length; i++) {
                        const task = tasks[i]!

                        if (config.scope === 'all' && !imageStore.isBatchTranslationInProgress) {
                            break
                        }

                        if (rateLimiter.value) {
                            await rateLimiter.value.acquire()
                        }

                        try {
                            imageStore.setTranslationStatus(task.imageIndex, 'processing')

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
                                    updateImageStore(task)
                                    break
                            }
                        } catch (err) {
                            const msg = err instanceof Error ? err.message : '未知错误'
                            errors.push(`图片 ${task.imageIndex}: ${step} - ${msg}`)
                            imageStore.setTranslationStatus(task.imageIndex, 'failed', msg)
                        }

                        const taskProgress = Math.floor(((i + 1) / tasks.length) * 100)
                        const overallProgress = stepProgress + Math.floor((taskProgress / 100) * (90 / stepChain.length))
                        reporter.setPercentage(overallProgress, `${STEP_LABELS[step]}: ${i + 1}/${tasks.length}`)
                    }
                }
            }

            // 统计结果
            for (const task of tasks) {
                const status = imageStore.images[task.imageIndex]?.translationStatus
                if (status === 'completed') {
                    completed++
                } else if (status === 'failed') {
                    failed++
                } else {
                    if (!stepChain.includes('render')) {
                        updateImageStore(task)
                        completed++
                    }
                }
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
                success: failed === 0,
                completed,
                failed,
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
