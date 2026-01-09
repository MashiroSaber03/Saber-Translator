/**
 * 翻译功能组合式函数
 * 
 * 使用统一的管线架构，提供所有翻译模式的统一入口
 * 
 * 架构：
 * - usePipeline: 核心管线执行引擎（自动选择顺序/并行模式）
 * - 模式配置: 通过配置区分不同翻译模式
 * - 原子步骤: detection, ocr, color, translate, aiTranslate, inpaint, render
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

// 重新导出类型供外部使用
export type { TranslationProgress } from './translation/core/types'

// ============================================================
// 组合式函数
// ============================================================

/**
 * 翻译功能组合式函数
 * 
 * 提供统一的翻译 API，内部使用管线架构执行
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
    // 单张翻译
    // ============================================================

    /**
     * 翻译当前图片
     */
    async function translateCurrentImage(): Promise<boolean> {
        const result = await pipeline.execute(getStandardModeConfig('current'))
        return result.success
    }

    /**
     * 翻译指定索引的图片（内部使用，用于批量翻译）
     */
    async function translateImageByIndex(index: number): Promise<boolean> {
        // 临时设置当前图片索引
        const originalIndex = imageStore.currentImageIndex
        imageStore.setCurrentImageIndex(index)

        const result = await pipeline.execute(getStandardModeConfig('current'))

        // 恢复原索引
        imageStore.setCurrentImageIndex(originalIndex)

        return result.success
    }

    // ============================================================
    // 批量翻译
    // ============================================================

    /**
     * 翻译所有图片
     */
    async function translateAllImages(): Promise<boolean> {
        const result = await pipeline.execute(getStandardModeConfig('all'))
        return result.success
    }

    /**
     * 取消批量翻译
     */
    function cancelBatchTranslation(): void {
        pipeline.cancel()
    }

    // ============================================================
    // 仅消除文字
    // ============================================================

    /**
     * 仅消除当前图片文字（不翻译）
     */
    async function removeTextOnly(): Promise<boolean> {
        const result = await pipeline.execute(getRemoveTextModeConfig('current'))
        return result.success
    }

    /**
     * 消除所有图片文字（不翻译）
     */
    async function removeAllTexts(): Promise<boolean> {
        const result = await pipeline.execute(getRemoveTextModeConfig('all'))
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

        const result = await pipeline.execute(getStandardModeConfig('failed'))
        return result.success
    }

    // ============================================================
    // 高质量翻译
    // ============================================================

    /**
     * 执行高质量翻译
     */
    async function executeHqTranslation(): Promise<boolean> {
        const result = await pipeline.execute(getHqModeConfig())
        return result.success
    }

    // ============================================================
    // AI 校对
    // ============================================================

    /**
     * 执行 AI 校对
     */
    async function executeProofreading(): Promise<boolean> {
        const result = await pipeline.execute(getProofreadModeConfig())
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

        // 单张翻译
        translateCurrentImage,
        translateImageByIndex,

        // 批量翻译
        translateAllImages,
        cancelBatchTranslation,

        // 仅消除文字
        removeTextOnly,
        removeAllTexts,

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
