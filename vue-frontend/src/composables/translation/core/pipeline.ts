/**
 * 翻译管线执行引擎
 * 
 * 提供统一的翻译流程编排和执行
 */

import { ref, computed } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { useValidation } from '../../useValidation'
import { useToast } from '@/utils/toast'
import { createRateLimiter, type RateLimiter } from '@/utils/rateLimiter'
import { createProgressManager } from './progressManager'
import { getPrepareStepExecutor } from '../steps/prepareStep'
import { executeMultimodalTranslation } from '../steps/multimodalTranslate'
import { executeProofreadingTranslation } from '../steps/proofreadTranslate'
import { importTranslationData, renderAllImages } from '../steps/renderStep'
import { generateSessionId } from '../utils'
import type {
    PipelineConfig,
    PipelineResult,
    ImageExecutionContext,
    BatchExecutionContext,
    SavedTextStyles,
    PrepareStepOptions
} from './types'

/**
 * 翻译管线 composable
 */
export function usePipeline() {
    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()
    const settingsStore = useSettingsStore()
    const validation = useValidation()
    const toast = useToast()

    // 进度管理
    const { progress, reporter } = createProgressManager()

    // 状态
    const isExecuting = ref(false)
    const rateLimiter = ref<RateLimiter | null>(null)

    // 保存的样式（用于高质量翻译和校对）
    let savedTextStyles: SavedTextStyles | null = null

    // 计算属性
    const isTranslating = computed(() => isExecuting.value || imageStore.isBatchTranslationInProgress)
    const progressPercent = computed(() => progress.value.percentage || 0)

    /**
     * 初始化或更新限速器
     */
    function initRateLimiter(): void {
        const rpm = settingsStore.settings.translation.rpmLimit
        if (!rateLimiter.value) {
            rateLimiter.value = createRateLimiter(rpm)
        } else {
            rateLimiter.value.setRpm(rpm)
        }
    }

    /**
     * 验证配置
     */
    function validateConfig(config: PipelineConfig): boolean {
        // 根据模式选择验证类型
        const validationType = config.mode === 'hq' ? 'hq'
            : config.mode === 'proofread' ? 'proofread'
                : config.mode === 'removeText' ? 'ocr'
                    : 'normal'

        return validation.validateBeforeTranslation(validationType)
    }

    /**
     * 保存当前样式设置
     */
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
        console.log('管线: 保存当前样式设置', savedTextStyles)
    }

    /**
     * 获取要处理的图片列表
     */
    function getImagesToProcess(config: PipelineConfig) {
        const images = imageStore.images

        if (config.scope === 'current') {
            const currentImage = imageStore.currentImage
            return currentImage ? [{ image: currentImage, index: imageStore.currentImageIndex }] : []
        }

        if (config.scope === 'failed') {
            const failedIndices = imageStore.getFailedImageIndices()
            return failedIndices
                .map(index => ({ image: images[index], index }))
                .filter(item => item.image !== undefined) as { image: NonNullable<typeof images[0]>; index: number }[]
        }

        // scope === 'all'
        return images.map((image, index) => ({ image, index }))
    }

    /**
     * 执行标准翻译（单张或批量）
     */
    async function executeStandardMode(config: PipelineConfig): Promise<PipelineResult> {
        const imagesToProcess = getImagesToProcess(config)
        if (imagesToProcess.length === 0) {
            toast.warning('没有可处理的图片')
            return { success: false, completed: 0, failed: 0, errors: ['没有可处理的图片'] }
        }

        const isRemoveTextMode = config.mode === 'removeText'
        const modeLabel = isRemoveTextMode ? '消除文字' : '翻译'

        reporter.init(imagesToProcess.length, `${modeLabel}: 0/${imagesToProcess.length}`)
        initRateLimiter()

        const errors: string[] = []
        let completed = 0
        let failed = 0

        const prepareStep = getPrepareStepExecutor()
        const prepareOptions: PrepareStepOptions = {
            skipTranslation: isRemoveTextMode
        }

        for (let i = 0; i < imagesToProcess.length; i++) {
            const { image, index } = imagesToProcess[i] || { image: undefined, index: -1 }
            if (!image || index < 0) continue

            // 检查是否已取消
            if (config.scope === 'all' && !imageStore.isBatchTranslationInProgress) {
                console.log('管线: 批量操作已取消')
                break
            }

            reporter.update(i + 1, `${modeLabel}: ${i + 1}/${imagesToProcess.length}`)

            // 等待限速
            if (rateLimiter.value) {
                await rateLimiter.value.acquire()
            }

            // 创建执行上下文
            const context: ImageExecutionContext = {
                imageIndex: index,
                image,
                config,
                progress: reporter
            }

            // 执行准备步骤
            const result = await prepareStep.execute(context, prepareOptions)

            if (result.success) {
                completed++
            } else {
                failed++
                if (result.error) {
                    errors.push(`图片 ${index}: ${result.error}`)
                }
            }
        }

        return {
            success: failed === 0,
            completed,
            failed,
            errors: errors.length > 0 ? errors : undefined
        }
    }

    /**
     * 执行高质量翻译模式
     */
    async function executeHqMode(config: PipelineConfig): Promise<PipelineResult> {
        const images = imageStore.images
        if (images.length === 0) {
            toast.warning('请先添加图片')
            return { success: false, completed: 0, failed: 0, errors: ['没有图片'] }
        }

        // 保存样式设置
        saveCurrentStyles()

        reporter.init(images.length, '高质量翻译: 准备中...')
        initRateLimiter()

        const errors: string[] = []

        try {
            // Step 1: 消除所有图片文字（准备步骤）
            toast.info('步骤 1/4: 消除所有图片文字...')
            reporter.setPercentage(0, '消除文字: 0/' + images.length)

            let prepareFailCount = 0
            const prepareStep = getPrepareStepExecutor()

            for (let i = 0; i < images.length; i++) {
                const image = images[i]
                if (!image) continue

                reporter.setPercentage(Math.floor((i / images.length) * 25), `消除文字: ${i + 1}/${images.length}`)

                // 等待限速
                if (rateLimiter.value) {
                    await rateLimiter.value.acquire()
                }

                const context: ImageExecutionContext = {
                    imageIndex: i,
                    image,
                    config,
                    progress: reporter
                }

                const result = await prepareStep.execute(context, { skipTranslation: true })

                if (!result.success) {
                    prepareFailCount++
                    if (result.error) {
                        errors.push(`图片 ${i} 消除文字失败: ${result.error}`)
                    }
                }
            }

            if (prepareFailCount > 0) {
                toast.warning(`消除文字完成，但有 ${prepareFailCount} 张图片失败`)
            }

            reporter.setPercentage(25, '消除文字完成')

            // Step 2: 准备翻译数据
            toast.info('步骤 2/4: 导出文本数据...')
            reporter.setPercentage(30, '导出文本数据...')

            // Step 3: 调用多模态 AI 翻译
            toast.info('步骤 3/4: 发送到 AI 进行翻译...')
            reporter.setPercentage(40, '准备批量翻译...')

            const batchContext: BatchExecutionContext = {
                images,
                config,
                progress: reporter,
                sessionId: generateSessionId()
            }

            const translateResult = await executeMultimodalTranslation(batchContext)

            if (!translateResult.success || !translateResult.translationData) {
                throw new Error(translateResult.error || 'AI 翻译失败')
            }

            // Step 4: 导入翻译结果并渲染
            toast.info('步骤 4/4: 导入翻译结果并渲染...')
            reporter.setPercentage(90, '导入翻译结果...')

            await importTranslationData(translateResult.translationData, savedTextStyles || undefined)

            reporter.setPercentage(95, '渲染图片...')

            const renderResult = await renderAllImages(
                translateResult.translationData,
                savedTextStyles || undefined,
                (current, total) => {
                    const percentage = 95 + Math.floor((current / total) * 5)
                    reporter.setPercentage(percentage, `渲染图片 ${current}/${total}`)
                }
            )

            reporter.setPercentage(100, '翻译完成！')
            toast.success('高质量翻译完成！')

            // 刷新当前图片的气泡
            const currentIdx = imageStore.currentImageIndex
            const currentImage = imageStore.images[currentIdx]
            if (currentImage?.bubbleStates) {
                bubbleStore.setBubbles(currentImage.bubbleStates)
            }

            return {
                success: true,
                completed: renderResult.success,
                failed: renderResult.failed + prepareFailCount,
                errors: errors.length > 0 ? errors : undefined
            }
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : '高质量翻译失败'
            toast.error(errorMessage)
            errors.push(errorMessage)
            return {
                success: false,
                completed: 0,
                failed: images.length,
                errors
            }
        }
    }

    /**
     * 执行 AI 校对模式
     */
    async function executeProofreadMode(config: PipelineConfig): Promise<PipelineResult> {
        const images = imageStore.images
        if (images.length === 0) {
            toast.warning('请先添加图片')
            return { success: false, completed: 0, failed: 0, errors: ['没有图片'] }
        }

        // 检查是否有翻译结果
        const hasTranslation = images.some(img => img.bubbleStates && img.bubbleStates.length > 0)
        if (!hasTranslation) {
            toast.warning('请先翻译图片')
            return { success: false, completed: 0, failed: 0, errors: ['没有翻译结果'] }
        }

        // 保存样式设置
        saveCurrentStyles()

        reporter.init(images.length, 'AI 校对: 准备中...')
        initRateLimiter()

        const errors: string[] = []

        try {
            // Step 1: 调用 AI 校对
            toast.info('步骤 1/2: 发送到 AI 进行校对...')
            reporter.setPercentage(10, '准备校对数据...')

            const batchContext: BatchExecutionContext = {
                images,
                config,
                progress: reporter,
                sessionId: generateSessionId()
            }

            const proofreadResult = await executeProofreadingTranslation(batchContext)

            if (!proofreadResult.success || !proofreadResult.translationData) {
                throw new Error(proofreadResult.error || 'AI 校对失败')
            }

            // Step 2: 导入校对结果并渲染
            toast.info('步骤 2/2: 导入校对结果并渲染...')
            reporter.setPercentage(90, '导入校对结果...')

            await importTranslationData(proofreadResult.translationData, savedTextStyles || undefined)

            reporter.setPercentage(95, '渲染图片...')

            const renderResult = await renderAllImages(
                proofreadResult.translationData,
                savedTextStyles || undefined,
                (current, total) => {
                    const percentage = 95 + Math.floor((current / total) * 5)
                    reporter.setPercentage(percentage, `渲染图片 ${current}/${total}`)
                }
            )

            reporter.setPercentage(100, '校对完成！')
            toast.success('AI 校对完成！')

            // 刷新当前图片的气泡
            const currentIdx = imageStore.currentImageIndex
            const currentImage = imageStore.images[currentIdx]
            if (currentImage?.bubbleStates) {
                bubbleStore.setBubbles(currentImage.bubbleStates)
            }

            return {
                success: true,
                completed: renderResult.success,
                failed: renderResult.failed,
                errors: errors.length > 0 ? errors : undefined
            }
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : 'AI 校对失败'
            toast.error(errorMessage)
            errors.push(errorMessage)
            return {
                success: false,
                completed: 0,
                failed: images.length,
                errors
            }
        }
    }

    /**
     * 执行翻译管线
     */
    async function execute(config: PipelineConfig): Promise<PipelineResult> {
        // 验证配置
        if (!validateConfig(config)) {
            return { success: false, completed: 0, failed: 0, errors: ['配置验证失败'] }
        }

        // 检查图片
        if (imageStore.images.length === 0) {
            toast.error('请先上传图片')
            return { success: false, completed: 0, failed: 0, errors: ['没有图片'] }
        }

        // 设置状态
        isExecuting.value = true
        if (config.scope === 'all' || config.scope === 'failed') {
            imageStore.setBatchTranslationInProgress(true)
        }

        try {
            let result: PipelineResult

            switch (config.mode) {
                case 'hq':
                    result = await executeHqMode(config)
                    break
                case 'proofread':
                    result = await executeProofreadMode(config)
                    break
                case 'standard':
                case 'removeText':
                default:
                    result = await executeStandardMode(config)
                    break
            }

            // 显示结果
            if (result.success) {
                if (config.mode === 'removeText') {
                    toast.success(config.scope === 'all' ? '所有图片文字消除完成' : '文字消除完成')
                } else if (config.mode !== 'hq' && config.mode !== 'proofread') {
                    toast.success(config.scope === 'all' ? '所有图片翻译完成' : '翻译成功！')
                }
            } else if (result.failed > 0 && result.completed > 0) {
                toast.warning(`完成 ${result.completed} 张，失败 ${result.failed} 张`)
            }

            return result
        } finally {
            isExecuting.value = false
            imageStore.setBatchTranslationInProgress(false)

            // 刷新当前显示的图片
            const currentIndex = imageStore.currentImageIndex
            if (currentIndex >= 0 && currentIndex < imageStore.images.length) {
                const currentImage = imageStore.images[currentIndex]
                if (currentImage?.bubbleStates && currentImage.bubbleStates.length > 0) {
                    bubbleStore.setBubbles(currentImage.bubbleStates)
                }
                imageStore.setCurrentImageIndex(currentIndex)
            }

            // 延迟隐藏进度条
            setTimeout(() => {
                reporter.finish()
            }, 1000)
        }
    }

    /**
     * 取消当前操作
     */
    function cancel(): void {
        if (imageStore.isBatchTranslationInProgress) {
            imageStore.setBatchTranslationInProgress(false)
            toast.info('操作已取消')
        }
    }

    return {
        // 状态
        progress,
        isExecuting,
        isTranslating,
        progressPercent,

        // 方法
        execute,
        cancel
    }
}
