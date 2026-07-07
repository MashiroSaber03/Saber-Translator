import { ref, computed, shallowRef, reactive } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import {
  providerRequiresApiKey,
  providerRequiresBaseUrl,
  providerSupportsCapability,
} from '@/config/aiProviders'
import { createInitialParallelProgress } from './progressDefaults'
import { ParallelPipeline, createParallelPipeline } from './ParallelPipeline'
import type { ParallelTranslationMode, ParallelExecutionResult, ParallelProgress } from './types'

const globalProgress = reactive<ParallelProgress>(createInitialParallelProgress())

const globalIsRunning = ref(false)

export function useParallelTranslation() {
  const imageStore = useImageStore()
  const settingsStore = useSettingsStore()

  const pipeline = shallowRef<ParallelPipeline | null>(null)

  const config = computed(() => settingsStore.settings.parallel)

  const isEnabled = computed(() => config.value?.enabled ?? false)

  const isRunning = globalIsRunning

  const progress = computed<ParallelProgress>(() => globalProgress)

  function determineMode(): ParallelTranslationMode {
    const settings = settingsStore.settings

    if (settings.proofreading?.enabled && settings.proofreading.rounds.length > 0) {
      return 'proofread'
    }

    const hqProvider = settings.hqTranslation?.provider || ''
    if (providerSupportsCapability(hqProvider, 'hqTranslation')) {
      const hasApiKey = !providerRequiresApiKey(hqProvider) || Boolean(settings.hqTranslation?.apiKey?.trim())
      const hasModelName = Boolean(settings.hqTranslation?.modelName?.trim())
      const hasBaseUrl = !providerRequiresBaseUrl(hqProvider) || Boolean(settings.hqTranslation?.customBaseUrl?.trim())

      if (hasApiKey && hasModelName && hasBaseUrl) {
        return 'hq'
      }
    }

    return 'standard'
  }

  function syncProgress(): void {
    if (!pipeline.value) return
    const pipelineProgress = pipeline.value.progress
    if (!pipelineProgress) return

    globalProgress.pools = pipelineProgress.pools.map(p => ({ ...p }))
    globalProgress.totalCompleted = pipelineProgress.totalCompleted
    globalProgress.totalFailed = pipelineProgress.totalFailed
    globalProgress.totalPages = pipelineProgress.totalPages
    globalProgress.estimatedTimeRemaining = pipelineProgress.estimatedTimeRemaining
    globalProgress.save = pipelineProgress.save ? { ...pipelineProgress.save } : undefined
  }

  async function executeParallel(
    mode?: ParallelTranslationMode,
    imagesToProcess?: typeof imageStore.images,
    imageIndexes: number[] = [],
  ): Promise<ParallelExecutionResult> {
    if (isRunning.value) {
      return { success: 0, failed: 0, errors: ['翻译正在进行中'] }
    }

    const images = imagesToProcess ?? imageStore.images
    if (images.length === 0) {
      return { success: 0, failed: 0, errors: ['没有图片'] }
    }

    isRunning.value = true

    globalProgress.totalPages = images.length
    globalProgress.totalCompleted = 0
    globalProgress.totalFailed = 0

    const syncInterval = setInterval(syncProgress, 200)

    try {
      pipeline.value = createParallelPipeline({
        enabled: true,
        deepLearningLockSize: config.value?.deepLearningLockSize ?? 1,
      })

      const translationMode = mode ?? determineMode()

      const result = await pipeline.value.execute(images, translationMode, imageIndexes)

      syncProgress()

      return result

    } catch (error) {
      return {
        success: 0,
        failed: images.length,
        errors: [(error as Error).message],
      }
    } finally {
      clearInterval(syncInterval)
      isRunning.value = false
    }
  }

  function cancel(): void {
    if (pipeline.value) {
      pipeline.value.cancel()
    }
    isRunning.value = false
  }

  function reset(): void {
    pipeline.value = null
    isRunning.value = false
  }

  return {
    isEnabled,
    isRunning,
    progress,
    executeParallel,
    cancel,
    reset,
    determineMode,
  }
}
