import { computed } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import { useToast } from '@/utils/toast'
import { useValidation } from '@/composables/useValidation'
import { useSequentialPipeline } from './SequentialPipeline'
import { useParallelTranslation } from '../parallel'
import {
  shouldEnableAutoSave,
  preSaveOriginalImages,
  finalizeSave,
  resetSaveState,
} from './saveStep'
import {
  resolvePipelineImageSelection,
  resolvePipelinePageIndexes,
} from './pageScope'
import type { PipelineConfig, PipelineResult, TranslationMode } from './types'
import type { ParallelTranslationMode } from '../parallel/types'
import {
  notifyPipelineAfter,
  notifyPipelineBefore,
  PipelineCancelledError,
  type PipelineMode,
} from '@/api/pipeline'

function toBackendMode(mode: TranslationMode): PipelineMode {
  return mode === 'removeText' ? 'remove_text' : (mode as PipelineMode)
}

export function usePipeline() {
    const imageStore = useImageStore()
    const settingsStore = useSettingsStore()
    const toast = useToast()
    const validation = useValidation()

    const sequentialPipeline = useSequentialPipeline()
    const parallelTranslation = useParallelTranslation()

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

    async function execute(config: PipelineConfig): Promise<PipelineResult> {
        if (imageStore.images.length === 0) {
            toast.error('请先上传图片')
            return { success: false, completed: 0, failed: 0, errors: ['没有图片'] }
        }

        if (!validatePipelineConfig(config)) {
            return { success: false, completed: 0, failed: 0, errors: ['配置验证失败'] }
        }

        const pipelineId =
            typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
                ? crypto.randomUUID()
                : `pipeline-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`

        const failedIndices = imageStore.getFailedImageIndices()
        const pageIndexes = resolvePipelinePageIndexes(
            config,
            imageStore.images.length,
            imageStore.currentImageIndex,
            failedIndices
        )
        const backendMode = toBackendMode(config.mode)
        const backendScope = config.scope

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
                return {
                    success: false,
                    completed: 0,
                    failed: 0,
                    errors: [`插件取消任务: ${err.message}`],
                }
            }
        }

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

    async function executeParallelMode(config: PipelineConfig): Promise<PipelineResult> {
        const pageSelection = resolvePipelineImageSelection(
            config,
            imageStore.images,
            imageStore.currentImageIndex,
            imageStore.getFailedImageIndices(),
        )
        const pageIndexes = pageSelection.map(({ index }) => index)
        const imagesToProcess = pageSelection.map(({ image }) => image)

        if (imagesToProcess.length === 0) {
            toast.error('没有可处理的页码')
            return { success: false, completed: 0, failed: 0, errors: ['没有可处理的页码'] }
        }

        if (imagesToProcess.length > 1) {
            const { textStyle } = settingsStore.settings
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

        const enableAutoSave = shouldEnableAutoSave()

        try {
            parallelTranslation.progress.value.totalPages = imagesToProcess.length
            parallelTranslation.progress.value.totalCompleted = 0
            parallelTranslation.progress.value.totalFailed = 0

            if (enableAutoSave) {
                toast.info('开始预保存原始图片...')

                const preSaveSuccess = await preSaveOriginalImages({
                    onStart: (total) => {
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
                    const progress = parallelTranslation.progress.value
                    progress.preSave = undefined
                }
            }

            const parallelMode: ParallelTranslationMode = config.mode as ParallelTranslationMode

            if (enableAutoSave) {
                const progress = parallelTranslation.progress.value
                progress.save = {
                    completed: 0,
                    total: imagesToProcess.length
                }
            }

            const result = await parallelTranslation.executeParallel(parallelMode, imagesToProcess, pageIndexes)

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
            const progress = parallelTranslation.progress.value
            progress.preSave = undefined
            progress.save = undefined

            if (enableAutoSave) {
                await finalizeSave()
            }
        }
    }

    function cancel(): void {
        sequentialPipeline.cancel()
        parallelTranslation.cancel()
        resetSaveState()
    }

    return {
        progress: sequentialPipeline.progress,
        isExecuting: sequentialPipeline.isExecuting,
        isTranslating,
        progressPercent,

        execute,
        cancel,

        STEP_CHAIN_CONFIGS: sequentialPipeline.STEP_CHAIN_CONFIGS
    }
}

export type { PipelineConfig, PipelineResult }
