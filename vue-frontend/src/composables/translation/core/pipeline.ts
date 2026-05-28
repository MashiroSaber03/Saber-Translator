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
import { useValidation } from '@/composables/useValidation'
import { useSequentialPipeline } from './SequentialPipeline'
import { useParallelTranslation } from '../parallel'
import {
    shouldEnableAutoSave,
    preSaveOriginalImages,
    finalizeSave,
    resetSaveState
} from './saveStep'
import type { PipelineConfig, PipelineResult, TranslationMode } from './types'
import type { ParallelTranslationMode } from '../parallel/types'
import {
    notifyPipelineAfter,
    notifyPipelineBefore,
    PipelineCancelledError,
    type PipelineMode,
    type PipelineScope,
} from '@/api/pipeline'

/** 把前端 TranslationMode 映射为后端 PLUGIN_MODES（仅 'removeText' → 'remove_text'）。 */
function toBackendMode(mode: TranslationMode): PipelineMode {
    return mode === 'removeText' ? 'remove_text' : (mode as PipelineMode)
}

/**
 * 计算本次任务的 0-based 页面索引数组。
 * 与 SequentialPipeline.getImagesToProcess 语义保持一致。
 */
function resolvePageIndexes(
    config: PipelineConfig,
    totalImages: number,
    currentIndex: number,
    failedIndices: number[]
): number[] {
    if (totalImages === 0) {
        return []
    }
    if (config.scope === 'current') {
        return currentIndex >= 0 && currentIndex < totalImages ? [currentIndex] : []
    }
    if (config.scope === 'failed') {
        return failedIndices.filter(idx => idx >= 0 && idx < totalImages)
    }
    if (config.scope === 'selection' && config.pageSelection) {
        return config.pageSelection.pages
            .map(page => page - 1)
            .filter(idx => idx >= 0 && idx < totalImages)
    }
    // 'all'
    return Array.from({ length: totalImages }, (_, i) => i)
}

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
    const validation = useValidation()

    // 获取两种管线实现
    const sequentialPipeline = useSequentialPipeline()
    const parallelTranslation = useParallelTranslation()

    // 统一状态
    const isTranslating = computed(() =>
        sequentialPipeline.isTranslating.value || imageStore.isBatchTranslationInProgress
    )
    const progressPercent = computed(() => sequentialPipeline.progressPercent.value)

    function validatePipelineConfig(config: PipelineConfig): boolean {
        if (config.mode === 'removeText') {
            if (!settingsStore.settings.removeTextWithOcr) {
                return true
            }
            return validation.validateBeforeTranslation('ocr')
        }

        const validationType = config.mode === 'hq'
            ? 'hq'
            : config.mode === 'proofread'
                ? 'proofread'
                : 'normal'
        return validation.validateBeforeTranslation(validationType)
    }

    /**
     * 执行翻译管线
     *
     * 自动选择执行引擎：
     * - 并行模式开启 + 批量操作 → 使用 ParallelPipeline
     * - 其他情况 → 使用 SequentialPipeline
     *
     * 调用前后统一触发后端的 before_pipeline / after_pipeline 钩子。
     */
    async function execute(config: PipelineConfig): Promise<PipelineResult> {
        // 检查图片
        if (imageStore.images.length === 0) {
            toast.error('请先上传图片')
            return { success: false, completed: 0, failed: 0, errors: ['没有图片'] }
        }

        if (!validatePipelineConfig(config)) {
            return { success: false, completed: 0, failed: 0, errors: ['配置验证失败'] }
        }

        // 1. 生成 pipeline_id 并通知 before_pipeline
        const pipelineId =
            typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
                ? crypto.randomUUID()
                : `pipeline-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`

        const failedIndices = imageStore.getFailedImageIndices()
        const pageIndexes = resolvePageIndexes(
            config,
            imageStore.images.length,
            imageStore.currentImageIndex,
            failedIndices
        )
        const backendMode = toBackendMode(config.mode)
        const backendScope = config.scope as PipelineScope

        try {
            await notifyPipelineBefore({
                pipeline_id: pipelineId,
                mode: backendMode,
                scope: backendScope,
                page_indexes: pageIndexes,
                total_images: pageIndexes.length,
            })
        } catch (err) {
            if (err instanceof PipelineCancelledError) {
                toast.error(`翻译被插件取消：${err.message}`)
                console.warn('[pipeline.before] 插件取消任务', err.details)
                return {
                    success: false,
                    completed: 0,
                    failed: 0,
                    errors: [`插件取消任务: ${err.message}`],
                }
            }
            console.warn('[pipeline.before] 通知失败（继续执行翻译）:', err)
        }

        // 2. 执行真实翻译；3. 无论成败都通知 after_pipeline
        const startedAt = Date.now()
        const sumWarnings = () => imageStore.images.reduce(
            (total, image) => total + (image.translationWarnings?.length || 0),
            0
        )
        const sendAfter = (r: PipelineResult) => notifyPipelineAfter({
            pipeline_id: pipelineId,
            mode: backendMode,
            scope: backendScope,
            completed: r.completed,
            failed: r.failed,
            errors: r.errors,
            warnings_count: sumWarnings(),
            duration_ms: Date.now() - startedAt,
        })

        try {
            const parallelConfig = settingsStore.settings.parallel
            const isBatchScope = config.scope === 'all' || config.scope === 'selection'
            const shouldUseParallel = parallelConfig?.enabled && isBatchScope

            const result = shouldUseParallel
                ? await executeParallelMode(config)
                : await sequentialPipeline.execute(config)

            console.log(`🚀 ${shouldUseParallel ? '并行' : '顺序'}管线完成，模式: ${config.mode}, pipeline_id=${pipelineId}`)
            void sendAfter(result)
            return result
        } catch (err) {
            const message = err instanceof Error ? err.message : '翻译执行出错'
            void sendAfter({
                success: false,
                completed: 0,
                failed: pageIndexes.length,
                errors: [message],
            })
            throw err
        }
    }

    /**
     * 执行并行模式
     */
    async function executeParallelMode(config: PipelineConfig): Promise<PipelineResult> {
        const pageIndexes = resolvePageIndexes(
            config,
            imageStore.images.length,
            imageStore.currentImageIndex,
            imageStore.getFailedImageIndices()
        )
        const imagesToProcess = pageIndexes.map((index) => imageStore.images[index]).filter(Boolean)

        if (imagesToProcess.length === 0) {
            toast.error('没有可处理的页码')
            return { success: false, completed: 0, failed: 0, errors: ['没有可处理的页码'] }
        }

        // 【修复】批量翻译开始时，将当前文字设置预先写入到所有待翻译的图片
        // 这样用户在翻译过程中切换图片时，侧边栏不会显示默认值，翻译也不会受影响
        if (imagesToProcess.length > 1) {
            const { textStyle } = settingsStore.settings
            console.log(`📝 [并行模式] 预分发文字设置到 ${imagesToProcess.length} 张待翻译图片...`)
            for (const imageIndex of pageIndexes) {
                imageStore.updateImageByIndex(imageIndex, {
                    fontSize: textStyle.fontSize,
                    autoFontSize: textStyle.autoFontSize,
                    fontFamily: textStyle.fontFamily,
                    layoutDirection: textStyle.layoutDirection,
                    textColor: textStyle.textColor,
                    fillColor: textStyle.fillColor,
                    strokeEnabled: textStyle.strokeEnabled,
                    strokeColor: textStyle.strokeColor,
                    strokeWidth: textStyle.strokeWidth,
                    lineSpacing: textStyle.lineSpacing,
                    textAlign: textStyle.textAlign,
                    inpaintMethod: textStyle.inpaintMethod,
                    useAutoTextColor: textStyle.useAutoTextColor
                })
            }
        }

        // 判断是否启用自动保存（书架模式 + 设置开启）
        const enableAutoSave = shouldEnableAutoSave()

        try {
            // 初始化进度状态（用于显示预保存进度条）
            // 注意：不设置 isRunning，避免与 executeParallel 冲突
            parallelTranslation.progress.value.totalPages = imagesToProcess.length
            parallelTranslation.progress.value.totalCompleted = 0
            parallelTranslation.progress.value.totalFailed = 0

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
            console.log(`   图片数量: ${imagesToProcess.length}`)
            console.log(`   页码索引: [${pageIndexes.join(', ')}]`)
            console.log(`   自动保存: ${enableAutoSave ? '启用' : '禁用'}`)

            // 初始化保存进度
            if (enableAutoSave) {
                const progress = parallelTranslation.progress.value
                progress.save = {
                    completed: 0,
                    total: imagesToProcess.length
                }
            }

            // 传入过滤后的图片数组和原始索引数组
            const result = await parallelTranslation.executeParallel(parallelMode, imagesToProcess, pageIndexes)

            // 显示结果
            if (result.success > 0 && result.failed === 0) {
                toast.success(`并行翻译完成，成功处理 ${result.success} 张图片`)
            } else if (result.success > 0 && result.failed > 0) {
                toast.warning(`并行翻译完成，成功 ${result.success} 张，失败 ${result.failed} 张`)
            } else {
                toast.error('并行翻译失败')
            }

            const warningCount = imagesToProcess.reduce(
                (total, image) => total + (image.translationWarnings?.length || 0),
                0
            )
            if (warningCount > 0) {
                toast.warning(`有 ${warningCount} 处术语未遵守`)
                console.warn('[TranslationWarnings]', imagesToProcess.flatMap(image => image.translationWarnings || []))
            }

            const autoGlossaryStats = result.autoGlossaryStats || {
                added: 0,
                duplicates: 0,
                failedPages: 0,
            }
            if (autoGlossaryStats.added > 0 || autoGlossaryStats.duplicates > 0 || autoGlossaryStats.failedPages > 0) {
                toast.info(`自动添加术语：新增 ${autoGlossaryStats.added} 条，跳过重复 ${autoGlossaryStats.duplicates} 条，失败 ${autoGlossaryStats.failedPages} 页`)
            }

            return {
                success: result.failed === 0,
                completed: result.success,
                failed: result.failed,
                errors: result.errors,
                autoGlossaryStats,
            }
        } catch (error) {
            const errorMessage = error instanceof Error ? error.message : '并行翻译出错'
            toast.error(errorMessage)
            return {
                success: false,
                completed: 0,
                failed: imagesToProcess.length,
                errors: [errorMessage],
                autoGlossaryStats: {
                    added: 0,
                    duplicates: 0,
                    failedPages: 0,
                },
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
