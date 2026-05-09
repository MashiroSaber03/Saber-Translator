/**
 * 顺序翻译管线
 *
 * 串行/批次只负责调度，具体业务步骤统一交给 atomic steps。
 */

import { ref, computed } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { useValidation } from '../../useValidation'
import { useToast } from '@/utils/toast'
import { createProgressManager } from './progressManager'
import type {
  PipelineConfig,
  PipelineResult,
  SavedTextStyles,
  TranslationMode,
} from './types'
import type { ImageData as AppImageData } from '@/types/image'
import type { TaskContext, PipelineRuntime } from './runtime'
import {
  buildSavedTextStylesFromSettings,
  createPipelineRuntime,
  hydrateTaskContextFromImage,
} from './runtime'
import { executeAtomicStep, executeBatchAtomicStep, type AtomicStepName } from './atomicSteps'
import { projectTaskContext } from './taskProjector'
import {
  shouldEnableAutoSave,
  preSaveOriginalImages,
  finalizeSave,
  resetSaveState,
} from './saveStep'

export type AtomicStepType =
  | 'detection'
  | 'ocr'
  | 'color'
  | 'translate'
  | 'aiTranslate'
  | 'inpaint'
  | 'render'
  | 'save'

export const STEP_CHAIN_CONFIGS: Record<TranslationMode, AtomicStepType[]> = {
  standard: ['detection', 'ocr', 'color', 'translate', 'inpaint', 'render'],
  hq: ['detection', 'ocr', 'color', 'aiTranslate', 'inpaint', 'render'],
  proofread: ['aiTranslate', 'render'],
  removeText: ['detection', 'inpaint', 'render'],
}

const STEP_LABELS: Record<AtomicStepType, string> = {
  detection: '气泡检测',
  ocr: '文字识别',
  color: '颜色提取',
  translate: '翻译',
  aiTranslate: 'AI翻译',
  inpaint: '背景修复',
  render: '渲染',
  save: '保存',
}

export function useSequentialPipeline() {
  const imageStore = useImageStore()
  const bubbleStore = useBubbleStore()
  const settingsStore = useSettingsStore()
  const validation = useValidation()
  const toast = useToast()

  const { progress, reporter } = createProgressManager()
  const isExecuting = ref(false)
  let savedTextStyles: SavedTextStyles | null = null

  const isTranslating = computed(() => isExecuting.value || imageStore.isBatchTranslationInProgress)
  const progressPercent = computed(() => progress.value.percentage || 0)

  function validateConfig(config: PipelineConfig): boolean {
    const validationType = config.mode === 'hq' ? 'hq'
      : config.mode === 'proofread' ? 'proofread'
        : config.mode === 'removeText' ? 'ocr'
          : 'normal'
    return validation.validateBeforeTranslation(validationType)
  }

  function saveCurrentStyles(): void {
    savedTextStyles = buildSavedTextStylesFromSettings(settingsStore.settings)
  }

  function getImagesToProcess(config: PipelineConfig): { image: AppImageData; index: number }[] {
    const images = imageStore.images
    if (config.scope === 'current') {
      const currentImage = imageStore.currentImage
      return currentImage ? [{ image: currentImage, index: imageStore.currentImageIndex }] : []
    }
    if (config.scope === 'failed') {
      return imageStore.getFailedImageIndices()
        .map((index) => ({ image: images[index]!, index }))
        .filter((item) => item.image !== undefined)
    }
    if (config.scope === 'range' && config.pageRange) {
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

      if (config.scope === 'all' && !imageStore.isBatchTranslationInProgress) {
        console.log('⏹️ 批量翻译已取消，停止处理')
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
          reporter.setPercentage(stepProgress, `图片 ${imageIdx + 1}: ${STEP_LABELS[step]}`)
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
        console.log(`✅ 图片 ${imageIdx + 1}/${tasks.length} 处理完成`)
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
      if (config.scope === 'all' && !imageStore.isBatchTranslationInProgress) {
        console.log('⏹️ 批量翻译已取消，停止处理')
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
            reporter.setPercentage(stepProgress, `图片 ${batchStart + index + 1}: ${STEP_LABELS[step]}`)
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
            reporter.setPercentage(stepProgress, `图片 ${batchStart + index + 1}: ${STEP_LABELS[step]}`)
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
          console.log(`✅ 图片 ${batchStart + index + 1} 处理完成`)
        }
      }

      failed += failedIndices.size
      console.log(`✅ 批次 ${batchIdx + 1}/${totalBatches} 处理完成`)
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
    isExecuting.value = true
    if (config.scope === 'all' || config.scope === 'failed') {
      imageStore.setBatchTranslationInProgress(true)
    }
    saveCurrentStyles()

    const imagesToProcess = getImagesToProcess(config)
    const errors: string[] = []

    if (savedTextStyles && imagesToProcess.length > 1) {
      console.log(`📝 预分发文字设置到 ${imagesToProcess.length} 张待翻译图片...`)
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
    const runtime = createPipelineRuntime(config.mode, {
      savedTextStyles,
      autoSaveEnabled: enableAutoSave,
    })

    const stepChain = [...STEP_CHAIN_CONFIGS[config.mode]]
    if (config.mode === 'removeText' && runtime.settingsSnapshot.removeTextWithOcr) {
      const detectionIdx = stepChain.indexOf('detection')
      if (detectionIdx !== -1) {
        stepChain.splice(detectionIdx + 1, 0, 'ocr')
      }
    }
    if (enableAutoSave) {
      stepChain.push('save')
    }

    console.log('🚀 顺序管线启动')
    console.log(`   模式: ${config.mode}`)
    console.log(`   处理方式: ${usePerImageMode ? '逐张处理' : '批次处理'}`)
    console.log(`   步骤链: [${stepChain.join(' → ')}]`)
    console.log(`   自动保存: ${enableAutoSave ? '启用' : '禁用'}`)

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
        console.warn('[TranslationWarnings]', tasks.flatMap((task) => task.warnings))
      }

      return {
        success: result.failed === 0,
        completed: result.completed,
        failed: result.failed,
        errors: errors.length > 0 ? errors : undefined,
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

      setTimeout(() => reporter.finish(), 1000)
    }
  }

  function cancel(): void {
    if (imageStore.isBatchTranslationInProgress) {
      imageStore.setBatchTranslationInProgress(false)
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
    STEP_CHAIN_CONFIGS,
  }
}
