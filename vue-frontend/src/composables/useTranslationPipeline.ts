/**
 * 翻译功能组合式函数 - 重构版
 * 
 * 使用统一的翻译入口，支持：
 * - 单页翻译
 * - 指定页翻译
 * - 全部翻译
 * - 重试失败页面
 * 
 * 核心设计：统一使用 translatePages(pageIndexes, mode) 入口
 */

import { computed } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useToast } from '@/utils/toast'

// 导入管线和模式配置
import {
    usePipeline,
    getStandardModeConfig,
    getHqModeConfig,
    getProofreadModeConfig,
    getRemoveTextModeConfig
} from './translation'
import type { PageSelection } from './translation/core/types'
import { pageIndexesToSelection, pageSelectionToPageIndexes } from '@/utils/pageSelection'

// 重新导出类型供外部使用
export type { TranslationProgress, PageSelection } from './translation/core/types'

/** 翻译模式 */
export type TranslationMode = 'standard' | 'hq' | 'proofread' | 'removeText'

/** 翻译结果 */
export interface TranslateResult {
    success: boolean
    completed: number
    failed: number
    errors: string[]
}

// ============================================================
// 辅助函数
// ============================================================

/**
 * 生成连续的页面索引数组
 */
export function range(start: number, end: number): number[] {
    const result: number[] = []
    for (let i = start; i < end; i++) {
        result.push(i)
    }
    return result
}

// ============================================================
// 组合式函数
// ============================================================

/**
 * 翻译功能组合式函数
 */
export function useTranslation() {
    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()
    const toast = useToast()

    // 使用管线
    const pipeline = usePipeline()

    // ============================================================
    // 状态（代理到管线）
    // ============================================================

    const progress = pipeline.progress
    const isTranslatingSingle = pipeline.isExecuting
    const isTranslating = pipeline.isTranslating
    const progressPercent = pipeline.progressPercent

    // 高质量翻译和校对状态（向后兼容）
    const isHqTranslating = computed(() => pipeline.isExecuting.value)
    const isProofreading = computed(() => pipeline.isExecuting.value)

    // ============================================================
    // 统一翻译入口（核心）
    // ============================================================

    /**
     * 统一翻译入口
     * 
     * @param pageIndexes 要翻译的页面索引数组（0-based）
     * @param mode 翻译模式
     * @returns 翻译结果
     * 
     * @example
     * // 翻译当前页
     * translatePages([currentIndex], 'standard')
     * 
     * // 翻译全部
     * translatePages(range(0, totalImages), 'standard')
     * 
     * // 翻译指定页
     * translatePages([0, 2, 7, 9], 'standard')
     * 
     * // 翻译单页
     * translatePages([7], 'standard')
     * 
     * // 重试失败
     * translatePages(failedIndexes, 'standard')
     */
    async function translatePages(
        pageIndexes: number[],
        mode: TranslationMode
    ): Promise<TranslateResult> {
        // 验证
        if (pageIndexes.length === 0) {
            toast.error('没有指定要翻译的页面')
            return { success: false, completed: 0, failed: 0, errors: ['没有指定要翻译的页面'] }
        }

        const totalImages = imageStore.images.length
        if (totalImages === 0) {
            toast.error('请先上传图片')
            return { success: false, completed: 0, failed: 0, errors: ['请先上传图片'] }
        }

        // 验证索引有效性
        for (const idx of pageIndexes) {
            if (idx < 0 || idx >= totalImages) {
                toast.error(`无效的页面索引: ${idx}`)
                return { success: false, completed: 0, failed: 0, errors: [`无效的页面索引: ${idx}`] }
            }
        }

        // 确定执行范围
        const isSinglePage = pageIndexes.length === 1
        const isAllPages = pageIndexes.length === totalImages

        // 获取模式配置
        let config
        if (isSinglePage) {
            // 单页翻译：设置当前索引后执行 'current' 模式
            const originalIndex = imageStore.currentImageIndex
            const targetIndex = pageIndexes[0]
            if (targetIndex !== undefined) {
                imageStore.setCurrentImageIndex(targetIndex)
            }

            config = getModeConfig(mode, 'current')
            const pipelineResult = await pipeline.execute(config)

            imageStore.setCurrentImageIndex(originalIndex)
            return {
                success: pipelineResult.success,
                completed: pipelineResult.completed,
                failed: pipelineResult.failed,
                errors: pipelineResult.errors ?? []
            }
        } else if (isAllPages) {
            // 全部翻译
            config = getModeConfig(mode, 'all')
        } else {
            // 指定页翻译
            const pageSelection = { pages: pageIndexesToSelection(pageIndexes) }
            config = getModeConfig(mode, 'selection', pageSelection)
        }

        const pipelineResult = await pipeline.execute(config)
        return {
            success: pipelineResult.success,
            completed: pipelineResult.completed,
            failed: pipelineResult.failed,
            errors: pipelineResult.errors ?? []
        }
    }

    /**
     * 根据模式获取配置
     */
    function getModeConfig(
        mode: TranslationMode,
        scope: 'current' | 'all' | 'selection' | 'failed',
        pageSelection?: PageSelection
    ) {
        switch (mode) {
            case 'standard':
                return getStandardModeConfig(scope, pageSelection ? { pageSelection } : undefined)
            case 'hq':
                return getHqModeConfig(scope, pageSelection ? { pageSelection } : undefined)
            case 'proofread':
                return getProofreadModeConfig(scope, pageSelection ? { pageSelection } : undefined)
            case 'removeText':
                return getRemoveTextModeConfig(scope, pageSelection ? { pageSelection } : undefined)
            default:
                return getStandardModeConfig(scope, pageSelection ? { pageSelection } : undefined)
        }
    }

    // ============================================================
    // 便捷方法（基于 translatePages 实现）
    // ============================================================

    /**
     * 翻译当前图片
     */
    async function translateCurrentImage(): Promise<boolean> {
        const result = await translatePages([imageStore.currentImageIndex], 'standard')
        return result.success
    }

    /**
     * 翻译指定索引的图片
     */
    async function translateImageByIndex(index: number): Promise<boolean> {
        const result = await translatePages([index], 'standard')
        return result.success
    }

    /**
     * 翻译所有图片
     */
    async function translateAllImages(): Promise<boolean> {
        const result = await translatePages(range(0, imageStore.images.length), 'standard')
        return result.success
    }

    /**
     * 翻译指定页码的图片
     * @param pageSelection 页面选择（页码从1开始）
     */
    async function translateSelectedImages(pageSelection: PageSelection): Promise<boolean> {
        const pageIndexes = pageSelectionToPageIndexes(pageSelection.pages)
        const result = await translatePages(pageIndexes, 'standard')
        return result.success
    }

    /**
     * 取消翻译
     */
    function cancelBatchTranslation(): void {
        pipeline.cancel()
    }

    // ============================================================
    // 仅消除文字
    // ============================================================

    /**
     * 仅消除当前图片文字
     */
    async function removeTextOnly(): Promise<boolean> {
        const result = await translatePages([imageStore.currentImageIndex], 'removeText')
        return result.success
    }

    /**
     * 消除所有图片文字
     */
    async function removeAllTexts(): Promise<boolean> {
        const result = await translatePages(range(0, imageStore.images.length), 'removeText')
        return result.success
    }

    /**
     * 消除指定页码图片的文字
     */
    async function removeTextSelection(pageSelection: PageSelection): Promise<boolean> {
        const pageIndexes = pageSelectionToPageIndexes(pageSelection.pages)
        const result = await translatePages(pageIndexes, 'removeText')
        return result.success
    }

    // ============================================================
    // 重新翻译失败图片
    // ============================================================

    /**
     * 重新翻译所有失败的图片
     */
    async function retryFailedImages(): Promise<boolean> {
        const failedIndices = imageStore.getFailedImageIndices()
        if (failedIndices.length === 0) {
            toast.info('没有失败的图片需要重新翻译')
            return true
        }
        const result = await translatePages(failedIndices, 'standard')
        return result.success
    }

    // ============================================================
    // 高质量翻译
    // ============================================================

    /**
     * 执行高质量翻译
     * @param pageSelection 可选的页面选择
     */
    async function executeHqTranslation(pageSelection?: PageSelection): Promise<boolean> {
        const pageIndexes = pageSelection
            ? pageSelectionToPageIndexes(pageSelection.pages)
            : range(0, imageStore.images.length)
        const result = await translatePages(pageIndexes, 'hq')
        return result.success
    }

    // ============================================================
    // AI 校对
    // ============================================================

    /**
     * 执行 AI 校对
     * @param pageSelection 可选的页面选择
     */
    async function executeProofreading(pageSelection?: PageSelection): Promise<boolean> {
        const pageIndexes = pageSelection
            ? pageSelectionToPageIndexes(pageSelection.pages)
            : range(0, imageStore.images.length)
        const result = await translatePages(pageIndexes, 'proofread')
        return result.success
    }

    // ============================================================
    // 使用已有气泡框翻译
    // ============================================================

    /**
     * 使用当前手动标注的气泡框进行翻译
     */
    async function translateWithCurrentBubbles(): Promise<boolean> {
        const currentImage = imageStore.currentImage
        if (!currentImage) {
            toast.error('请先上传图片')
            return false
        }

        const bubbles = bubbleStore.bubbles
        if (!bubbles || bubbles.length === 0) {
            toast.error('当前图片没有气泡框，请先检测或手动添加')
            return false
        }

        // 标记为手动标注，然后执行翻译
        imageStore.setManuallyAnnotated(true)
        return translateCurrentImage()
    }

    // ============================================================
    // 返回
    // ============================================================

    return {
        // 状态
        progress,
        isTranslatingSingle,
        isHqTranslating,
        isProofreading,

        // 计算属性
        isTranslating,
        progressPercent,

        // 统一入口（推荐使用）
        translatePages,
        range,

        // 单张翻译
        translateCurrentImage,
        translateImageByIndex,

        // 批量翻译
        translateAllImages,
        translateSelectedImages,
        cancelBatchTranslation,

        // 仅消除文字
        removeTextOnly,
        removeAllTexts,
        removeTextSelection,

        // 重新翻译失败图片
        retryFailedImages,

        // 高质量翻译
        executeHqTranslation,

        // AI 校对
        executeProofreading,

        // 使用已有气泡框翻译
        translateWithCurrentBubbles
    }
}
