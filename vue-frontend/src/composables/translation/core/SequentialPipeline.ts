import { computed, getCurrentInstance, onUnmounted, ref } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useSettingsStore } from '@/stores/settings'
import { useSessionStore } from '@/stores/sessionStore'
import { useValidation } from '../../useValidation'
import { useToast } from '@/utils/toast'
import { createProgressManager } from './progressManager'
import type {
  PipelineConfig,
  PipelineResult,
  SavedTextStyles,
  TranslationMode,
} from './types'
import type { TaskContext, PipelineRuntime } from './runtime'
import {
  resolvePipelineImageSelection,
  type PipelineImageSelection,
} from './pageScope'
import {
  buildSavedTextStylesFromSettings,
  createPipelineRuntime,
  hydrateTaskContextFromImage,
} from './runtime'
import { executeAtomicStep, executeBatchAtomicStep, type AtomicStepName } from './atomicSteps'
import { projectTaskContext } from './taskProjector'
import {
  STEP_CHAIN_CONFIGS as BASE_STEP_CHAIN_CONFIGS,
  getStepLabel,
  resolveSequentialStepChain,
  type AtomicStepType,
} from './pipelineRegistry'
import {
  shouldEnableAutoSave,
  preSaveOriginalImages,
  finalizeSave,
  resetSaveState,
} from './saveStep'

export type { AtomicStepType } from './pipelineRegistry'
export const STEP_CHAIN_CONFIGS = BASE_STEP_CHAIN_CONFIGS

export function useSequentialPipeline() {
  const imageStore = useImageStore()
  const bubbleStore = useBubbleStore()
  const settingsStore = useSettingsStore()
  const validation = useValidation()
  const toast = useToast()

  const { progress, reporter } = createProgressManager()
  const isExecuting = ref(false)
  let savedTextStyles: SavedTextStyles | null = null
  let finishTimer: ReturnType<typeof setTimeout> | null = null

  const isTranslating = computed(() => isExecuting.value || imageStore.isBatchTranslationInProgress)
  const progressPercent = computed(() => progress.value.percentage || 0)

  function clearFinishTimer(): void {
    if (finishTimer !== null) {
      clearTimeout(finishTimer)
      finishTimer = null
    }
  }

  function scheduleFinish(): void {
    clearFinishTimer()
    finishTimer = setTimeout(() => {
      finishTimer = null
      reporter.finish()
    }, 1000)
  }

  function validateConfig(config: PipelineConfig): boolean {
    if (config.mode === 'removeText' && !settingsStore.settings.removeTextWithOcr) {
      return true
    }

    const validationType = config.mode === 'hq' ? 'hq'
      : config.mode === 'proofread' ? 'proofread'
        : config.mode === 'removeText' ? 'ocr'
          : 'normal'
    return validation.validateBeforeTranslation(validationType)
  }

  function saveCurrentStyles(): void {
    savedTextStyles = buildSavedTextStylesFromSettings(settingsStore.settings)
  }

  function getImagesToProcess(config: PipelineConfig): PipelineImageSelection[] {
    return resolvePipelineImageSelection(
      config,
      imageStore.images,
      imageStore.currentImageIndex,
      imageStore.getFailedImageIndices(),
    )
  }

  function shouldUsePerImageMode(mode: TranslationMode): boolean {
    return mode === 'standard' || mode === 'removeText'
  }

  function getBatchSize(mode: TranslationMode, runtime: PipelineRuntime): number {
    if (mode === 'hq') {
      return runtime.settingsSnapshot.hqTranslation.batchSize || 3
    }
    if (mode === 'proofread') {
      return runtime.settingsSnapshot.proofreading.rounds[0]?.batchSize || 3
    }
    return 1
  }

  async function executeSingleStep(
    step: AtomicStepType,
    task: TaskContext,
    runtime: PipelineRuntime
  ): Promise<TaskContext> {
    if (step === 'aiTranslate') {
      throw new Error('aiTranslate 应通过批量处理逻辑调用')
    }

    return await executeAtomicStep(step as AtomicStepName, task, runtime)
  }

  function projectAfterStep(step: AtomicStepType, task: TaskContext, runtime: PipelineRuntime): void {
    if (step === 'render' || step === 'save') {
      projectTaskContext(task, runtime)
    }
  }

  async function executePerImageMode(
    tasks: TaskContext[],
    stepChain: AtomicStepType[],
    config: PipelineConfig,
    errors: string[],
    runtime: PipelineRuntime
  ): Promise<{ completed: number; failed: number }> {
    let completed = 0
    let failed = 0

    for (let imageIdx = 0; imageIdx < tasks.length; imageIdx++) {
      let task = tasks[imageIdx]!

      if ((config.scope === 'all' || config.scope === 'selection') && !imageStore.isBatchTranslationInProgress) {
        break
      }

      const imageProgress = Math.floor((imageIdx / tasks.length) * 90)
      reporter.setPercentage(imageProgress, `处理图片 ${imageIdx + 1}/${tasks.length}`)
      toast.info(`处理图片 ${imageIdx + 1}/${tasks.length}...`)
      imageStore.setTranslationStatus(task.imageIndex, 'processing')

      let taskFailed = false
      for (let stepIdx = 0; stepIdx < stepChain.length; stepIdx++) {
        const step = stepChain[stepIdx]!
        if (taskFailed) break

        try {
          const stepProgress = imageProgress + Math.floor((stepIdx / stepChain.length) * (90 / tasks.length))
          reporter.setPercentage(stepProgress, `图片 ${imageIdx + 1}: ${getStepLabel(step)}`)
          task = await executeSingleStep(step, task, runtime)
          tasks[imageIdx] = task
          projectAfterStep(step, task, runtime)
        } catch (err) {
          const msg = err instanceof Error ? err.message : '未知错误'
          errors.push(`图片 ${task.imageIndex + 1}: ${step} - ${msg}`)
          imageStore.setTranslationStatus(task.imageIndex, 'failed', msg)
          taskFailed = true
          failed++
        }
      }

      if (!taskFailed) {
        const completedTask: TaskContext = { ...task, status: 'completed' }
        tasks[imageIdx] = completedTask
        projectTaskContext(completedTask, runtime)
        completed++
      }
    }

    return { completed, failed }
  }

  async function executeBatchMode(
    tasks: TaskContext[],
    stepChain: AtomicStepType[],
    config: PipelineConfig,
    errors: string[],
    runtime: PipelineRuntime
  ): Promise<{ completed: number; failed: number }> {
    let completed = 0
    let failed = 0

    const batchSize = getBatchSize(config.mode, runtime)
    const totalBatches = Math.ceil(tasks.length / batchSize)
    const aiTranslateIdx = stepChain.indexOf('aiTranslate')
    const stepsBeforeAi = aiTranslateIdx >= 0 ? stepChain.slice(0, aiTranslateIdx) : stepChain
    const stepsAfterAi = aiTranslateIdx >= 0 ? stepChain.slice(aiTranslateIdx + 1) : []

    for (let batchIdx = 0; batchIdx < totalBatches; batchIdx++) {
      if ((config.scope === 'all' || config.scope === 'selection') && !imageStore.isBatchTranslationInProgress) {
        break
      }

      const batchStart = batchIdx * batchSize
      const batchEnd = Math.min(batchStart + batchSize, tasks.length)
      const batchTasks = tasks.slice(batchStart, batchEnd)

      const batchProgress = Math.floor((batchIdx / totalBatches) * 90)
      reporter.setPercentage(batchProgress, `处理批次 ${batchIdx + 1}/${totalBatches}`)
      toast.info(`处理批次 ${batchIdx + 1}/${totalBatches}（图片 ${batchStart + 1}-${batchEnd}）...`)

      const failedIndices = new Set<number>()

      for (let index = 0; index < batchTasks.length; index++) {
        let task = batchTasks[index]!
        imageStore.setTranslationStatus(task.imageIndex, 'processing')

        for (const step of stepsBeforeAi) {
          if (failedIndices.has(task.imageIndex)) break
          try {
            const stepProgress = batchProgress + Math.floor((index / batchTasks.length) * 30)
            reporter.setPercentage(stepProgress, `图片 ${batchStart + index + 1}: ${getStepLabel(step)}`)
            task = await executeSingleStep(step, task, runtime)
            tasks[batchStart + index] = task
            projectAfterStep(step, task, runtime)
            batchTasks[index] = task
          } catch (err) {
            const msg = err instanceof Error ? err.message : '未知错误'
            errors.push(`图片 ${task.imageIndex + 1}: ${step} - ${msg}`)
            imageStore.setTranslationStatus(task.imageIndex, 'failed', msg)
            failedIndices.add(task.imageIndex)
          }
        }
      }

      if (aiTranslateIdx >= 0) {
        try {
          const validTasks = batchTasks.filter((task) => !failedIndices.has(task.imageIndex))
          if (validTasks.length > 0) {
            const translatedTasks = await executeBatchAtomicStep('aiTranslate', validTasks, runtime)
            for (const translatedTask of translatedTasks) {
              const localIndex = batchTasks.findIndex((task) => task.imageIndex === translatedTask.imageIndex)
              if (localIndex >= 0) {
                batchTasks[localIndex] = translatedTask
                tasks[batchStart + localIndex] = translatedTask
              }
            }
          }
        } catch (err) {
          const msg = err instanceof Error ? err.message : '未知错误'
          errors.push(`批次 ${batchIdx + 1} AI翻译失败: ${msg}`)
          for (const task of batchTasks) {
            if (!failedIndices.has(task.imageIndex)) {
              imageStore.setTranslationStatus(task.imageIndex, 'failed', msg)
              failedIndices.add(task.imageIndex)
            }
          }
        }
      }

      for (let index = 0; index < batchTasks.length; index++) {
        let task = batchTasks[index]!
        if (failedIndices.has(task.imageIndex)) continue

        for (const step of stepsAfterAi) {
          if (failedIndices.has(task.imageIndex)) break
          try {
            const stepProgress = batchProgress + 50 + Math.floor((index / batchTasks.length) * 40)
            reporter.setPercentage(stepProgress, `图片 ${batchStart + index + 1}: ${getStepLabel(step)}`)
            task = await executeSingleStep(step, task, runtime)
            tasks[batchStart + index] = task
            projectAfterStep(step, task, runtime)
            batchTasks[index] = task
          } catch (err) {
            const msg = err instanceof Error ? err.message : '未知错误'
            errors.push(`图片 ${task.imageIndex + 1}: ${step} - ${msg}`)
            imageStore.setTranslationStatus(task.imageIndex, 'failed', msg)
            failedIndices.add(task.imageIndex)
          }
        }

        if (!failedIndices.has(task.imageIndex)) {
          const completedTask: TaskContext = { ...task, status: 'completed' }
          tasks[batchStart + index] = completedTask
          projectTaskContext(completedTask, runtime)
          completed++
        }
      }

      failed += failedIndices.size
    }

    return { completed, failed }
  }

  async function execute(config: PipelineConfig): Promise<PipelineResult> {
    if (!validateConfig(config)) {
      return { success: false, completed: 0, failed: 0, errors: ['配置验证失败'] }
    }

    if (imageStore.images.length === 0) {
      toast.error('请先上传图片')
      return { success: false, completed: 0, failed: 0, errors: ['没有图片'] }
    }

    const usePerImageMode = shouldUsePerImageMode(config.mode)
    clearFinishTimer()
    isExecuting.value = true
    if (config.scope === 'all' || config.scope === 'failed' || config.scope === 'selection') {
      imageStore.setBatchTranslationInProgress(true)
    }
    saveCurrentStyles()

    const imagesToProcess = getImagesToProcess(config)
    const errors: string[] = []

    if (savedTextStyles && imagesToProcess.length > 1) {
      for (const { index } of imagesToProcess) {
        imageStore.updateImageByIndex(index, {
          fontSize: savedTextStyles.fontSize,
          autoFontSize: savedTextStyles.autoFontSize,
          fontFamily: savedTextStyles.fontFamily,
          layoutDirection: savedTextStyles.layoutDirection,
          textColor: savedTextStyles.textColor,
          fillColor: savedTextStyles.fillColor,
          strokeEnabled: savedTextStyles.strokeEnabled,
          strokeColor: savedTextStyles.strokeColor,
          strokeWidth: savedTextStyles.strokeWidth,
          lineSpacing: savedTextStyles.lineSpacing,
          textAlign: savedTextStyles.textAlign,
          inpaintMethod: savedTextStyles.inpaintMethod,
          useAutoTextColor: savedTextStyles.useAutoTextColor,
        })
      }
    }

    const enableAutoSave = shouldEnableAutoSave()
    const sessionStore = useSessionStore()
    const runtime = createPipelineRuntime(config.mode, {
      savedTextStyles,
      autoSaveEnabled: enableAutoSave,
      bookId: sessionStore.currentBookId,
      chapterId: sessionStore.currentChapterId,
    })

    const stepChain = resolveSequentialStepChain(config.mode, {
      removeTextWithOcr: runtime.settingsSnapshot.removeTextWithOcr,
      autoSaveEnabled: enableAutoSave,
    })

    const tasks: TaskContext[] = imagesToProcess.map(({ image, index }) =>
      hydrateTaskContextFromImage(index, image, config.mode, runtime),
    )

    try {
      reporter.init(imagesToProcess.length, `${config.mode} 模式启动...`)

      if (enableAutoSave) {
        reporter.setPercentage(0, '预保存原始图片...')
        const preSaveSuccess = await preSaveOriginalImages({
          onStart: (total) => reporter.setPercentage(0, `预保存原始图片 0/${total}...`),
          onProgress: (current, total) => {
            const percent = Math.round((current / total) * 10)
            reporter.setPercentage(percent, `预保存原始图片 ${current}/${total}...`)
          },
          onComplete: () => reporter.setPercentage(10, '预保存完成，开始翻译...'),
          onError: (error) => reporter.setPercentage(0, `预保存失败: ${error}`),
        })
        if (!preSaveSuccess) {
          toast.warning('预保存失败，翻译完成后请手动保存')
        }
      }

      const result = usePerImageMode
        ? await executePerImageMode(tasks, stepChain, config, errors, runtime)
        : await executeBatchMode(tasks, stepChain, config, errors, runtime)

      reporter.setPercentage(100, '完成！')

      const modeLabels: Record<TranslationMode, string> = {
        standard: '翻译',
        hq: '高质量翻译',
        proofread: 'AI校对',
        removeText: '消除文字',
      }
      toast.success(`${modeLabels[config.mode]}完成！`)

      const warningCount = tasks.reduce((total, task) => total + task.warnings.length, 0)
      if (warningCount > 0) {
        toast.warning(`有 ${warningCount} 处术语未遵守`)
      }

      const autoGlossaryStats = tasks.reduce((total, task) => ({
        added: total.added + task.autoGlossaryStats.added,
        duplicates: total.duplicates + task.autoGlossaryStats.duplicates,
        failedPages: total.failedPages + task.autoGlossaryStats.failedPages,
      }), {
        added: 0,
        duplicates: 0,
        failedPages: 0,
      })
      if (autoGlossaryStats.added > 0 || autoGlossaryStats.duplicates > 0 || autoGlossaryStats.failedPages > 0) {
        toast.info(
          `自动添加术语：新增 ${autoGlossaryStats.added} 条，跳过重复 ${autoGlossaryStats.duplicates} 条，失败 ${autoGlossaryStats.failedPages} 页`
        )
      }

      return {
        success: result.failed === 0,
        completed: result.completed,
        failed: result.failed,
        errors: errors.length > 0 ? errors : undefined,
        autoGlossaryStats,
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : '执行失败'
      toast.error(errorMessage)
      errors.push(errorMessage)
      return {
        success: false,
        completed: 0,
        failed: imagesToProcess.length,
        errors,
        autoGlossaryStats: {
          added: 0,
          duplicates: 0,
          failedPages: 0,
        },
      }
    } finally {
      isExecuting.value = false
      imageStore.setBatchTranslationInProgress(false)

      if (enableAutoSave) {
        await finalizeSave()
      }

      const currentIndex = imageStore.currentImageIndex
      const currentImage = imageStore.images[currentIndex]
      if (currentImage?.bubbleStates && currentImage.bubbleStates.length > 0) {
        bubbleStore.setBubbles(currentImage.bubbleStates, true)
      }

      scheduleFinish()
    }
  }

  function cancel(): void {
    clearFinishTimer()
    if (imageStore.isBatchTranslationInProgress) {
      imageStore.setBatchTranslationInProgress(false)
      resetSaveState()
      toast.info('操作已取消')
    }
  }

  if (getCurrentInstance()) {
    onUnmounted(() => {
      clearFinishTimer()
    })
  }

  return {
    progress,
    isExecuting,
    isTranslating,
    progressPercent,
    execute,
    cancel,
    STEP_CHAIN_CONFIGS,
  }
}
