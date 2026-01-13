/**
 * 翻译管线执行引擎 - 统一入口
 * 
 * 重构后的设计：
 * - 此文件作为统一入口，根据配置委托给具体的管线实现
 * - SequentialPipeline: 顺序执行（适用于单张或需要严格顺序的场景）
 * - ParallelPipeline: 并行执行（适用于批量处理，提高效率）
 * 
 * 核心设计理念：
 * - 所有模式统一使用步骤链配置
 * - 消除 executeStandardMode, executeHqMode 等重复代码
 * - 简化选项传递（skipTranslation, skipOcr 等）
 */

import { computed } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { useToast } from '@/utils/toast'
import { useSequentialPipeline } from './SequentialPipeline'
import { useParallelTranslation } from '../parallel'
import {
    shouldEnableAutoSave,
    preSaveOriginalImages,
    finalizeSave,
    resetSaveState
} from './saveStep'
import type { PipelineConfig, PipelineResult } from './types'
import type { ParallelTranslationMode } from '../parallel/types'

/**
 * 翻译管线 composable - 统一入口
 * 
 * 使用示例：
 * ```typescript
 * const pipeline = usePipeline()
 * 
 * // 标准翻译（单张）
 * await pipeline.execute({ mode: 'standard', scope: 'current' })
 * 
 * // 高质量翻译（批量）
 * await pipeline.execute({ mode: 'hq', scope: 'all' })
 * 
 * // 消除文字
 * await pipeline.execute({ mode: 'removeText', scope: 'current' })
 * ```
 */
export function usePipeline() {
    const imageStore = useImageStore()
    const settingsStore = useSettingsStore()
    const toast = useToast()

    // 获取两种管线实现
    const sequentialPipeline = useSequentialPipeline()
    const parallelTranslation = useParallelTranslation()

    // 统一状态
    const isTranslating = computed(() =>
        sequentialPipeline.isTranslating.value || imageStore.isBatchTranslationInProgress
    )
    const progressPercent = computed(() => sequentialPipeline.progressPercent.value)

    /**
     * 执行翻译管线
     * 
     * 自动选择执行引擎：
     * - 并行模式开启 + 批量操作 → 使用 ParallelPipeline
     * - 其他情况 → 使用 SequentialPipeline
     */
    async function execute(config: PipelineConfig): Promise<PipelineResult> {
        // 检查图片
        if (imageStore.images.length === 0) {
            toast.error('请先上传图片')
            return { success: false, completed: 0, failed: 0, errors: ['没有图片'] }
        }

        // 检查是否使用并行模式
        const parallelConfig = settingsStore.settings.parallel
        const shouldUseParallel = parallelConfig?.enabled && config.scope === 'all'

        if (shouldUseParallel) {
            console.log(`🚀 使用并行管线，模式: ${config.mode}`)
            return executeParallelMode(config)
        }

        // 使用顺序管线
        console.log(`🚀 使用顺序管线，模式: ${config.mode}`)
        return sequentialPipeline.execute(config)
    }

    /**
     * 执行并行模式
     */
    async function executeParallelMode(config: PipelineConfig): Promise<PipelineResult> {
        const images = imageStore.images

        // 判断是否启用自动保存（书架模式 + 设置开启）
        const enableAutoSave = shouldEnableAutoSave()

        try {
            // 如果启用自动保存，先执行预保存（保存所有原始图片）
            if (enableAutoSave) {
                console.log('[ParallelPipeline] 执行预保存...')
                toast.info('开始预保存原始图片...')

                // 通过进度回调更新预保存进度
                const preSaveSuccess = await preSaveOriginalImages({
                    onStart: (total) => {
                        // 更新全局进度的预保存状态
                        const progress = parallelTranslation.progress.value
                        progress.preSave = {
                            isRunning: true,
                            current: 0,
                            total
                        }
                    },
                    onProgress: (current, total) => {
                        const progress = parallelTranslation.progress.value
                        if (progress.preSave) {
                            progress.preSave.current = current
                            progress.preSave.total = total
                        }
                    },
                    onComplete: () => {
                        const progress = parallelTranslation.progress.value
                        if (progress.preSave) {
                            progress.preSave.isRunning = false
                        }
                        toast.success('预保存完成，开始翻译...')
                    },
                    onError: (error) => {
                        const progress = parallelTranslation.progress.value
                        progress.preSave = undefined
                        toast.warning(`预保存失败：${error}，翻译完成后请手动保存`)
                    }
                })

                if (!preSaveSuccess) {
                    // 预保存失败，清除预保存进度状态
                    const progress = parallelTranslation.progress.value
                    progress.preSave = undefined
                }
            }

            // 映射模式
            const parallelMode: ParallelTranslationMode = config.mode as ParallelTranslationMode

            console.log(`🚀 启动并行翻译模式: ${parallelMode}`)
            console.log(`   自动保存: ${enableAutoSave ? '启用' : '禁用'}`)

            // 初始化保存进度
            if (enableAutoSave) {
                const progress = parallelTranslation.progress.value
                progress.save = {
                    completed: 0,
                    total: images.length
                }
            }

            const result = await parallelTranslation.executeParallel(parallelMode)

            // 显示结果
            if (result.success > 0 && result.failed === 0) {
                toast.success(`并行翻译完成，成功处理 ${result.success} 张图片`)
            } else if (result.success > 0 && result.failed > 0) {
                toast.warning(`并行翻译完成，成功 ${result.success} 张，失败 ${result.failed} 张`)
            } else {
                toast.error('并行翻译失败')
            }

            return {
                success: result.failed === 0,
                completed: result.success,
                failed: result.failed,
                errors: result.errors
            }
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : '并行翻译出错'
            toast.error(errorMessage)
            return {
                success: false,
                completed: 0,
                failed: images.length,
                errors: [errorMessage]
            }
        } finally {
            // 清除预保存和保存进度状态
            const progress = parallelTranslation.progress.value
            progress.preSave = undefined
            progress.save = undefined

            // 如果启用了自动保存，完成保存会话
            if (enableAutoSave) {
                console.log('[ParallelPipeline] 完成保存...')
                await finalizeSave()
            }
        }
    }

    /**
     * 取消当前操作
     */
    function cancel(): void {
        sequentialPipeline.cancel()
        parallelTranslation.cancel()
        // 重置自动保存状态
        resetSaveState()
    }

    return {
        // 状态
        progress: sequentialPipeline.progress,
        isExecuting: sequentialPipeline.isExecuting,
        isTranslating,
        progressPercent,

        // 方法
        execute,
        cancel,

        // 导出步骤链配置（便于调试）
        STEP_CHAIN_CONFIGS: sequentialPipeline.STEP_CHAIN_CONFIGS
    }
}

// 导出类型
export type { PipelineConfig, PipelineResult }
