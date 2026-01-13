/**
 * 并行翻译 Composable
 * 
 * 提供并行翻译的入口函数和状态管理
 */

import { ref, computed, shallowRef, reactive } from 'vue'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { ParallelPipeline, createParallelPipeline } from './ParallelPipeline'
import type { ParallelTranslationMode, ParallelExecutionResult, ParallelProgress } from './types'

// 全局响应式进度状态
const globalProgress = reactive<ParallelProgress>({
  pools: [
    { name: '检测', icon: '📍', waiting: 0, processing: false, completed: 0, isWaitingLock: false },
    { name: 'OCR', icon: '📖', waiting: 0, processing: false, completed: 0, isWaitingLock: false },
    { name: '颜色', icon: '🎨', waiting: 0, processing: false, completed: 0, isWaitingLock: false },
    { name: '翻译', icon: '🌐', waiting: 0, processing: false, completed: 0, isWaitingLock: false },
    { name: '修复', icon: '🖌️', waiting: 0, processing: false, completed: 0, isWaitingLock: false },
    { name: '渲染', icon: '✨', waiting: 0, processing: false, completed: 0, isWaitingLock: false }
  ],
  totalCompleted: 0,
  totalFailed: 0,
  totalPages: 0,
  estimatedTimeRemaining: 0,
  // 预保存进度
  preSave: undefined,
  // 保存进度
  save: undefined
})

const globalIsRunning = ref(false)

export function useParallelTranslation() {
  const imageStore = useImageStore()
  const settingsStore = useSettingsStore()

  const pipeline = shallowRef<ParallelPipeline | null>(null)

  // 获取当前配置
  const config = computed(() => settingsStore.settings.parallel)

  // 是否启用并行模式
  const isEnabled = computed(() => config.value?.enabled ?? false)

  // 是否正在运行（使用全局状态）
  const isRunning = globalIsRunning

  // 进度（使用全局响应式状态）
  const progress = computed<ParallelProgress>(() => globalProgress)

  /**
   * 确定翻译模式
   */
  function determineMode(): ParallelTranslationMode {
    const settings = settingsStore.settings

    // 检查是否启用AI校对
    if (settings.proofreading?.enabled && settings.proofreading.rounds.length > 0) {
      return 'proofread'
    }

    // 检查是否使用高质量翻译（根据provider判断）
    const hqProviders = ['gemini', 'openai', 'claude', 'deepseek']
    if (hqProviders.includes(settings.hqTranslation?.provider || '')) {
      // 检查是否配置了高质量翻译API
      if (settings.hqTranslation?.apiKey) {
        return 'hq'
      }
    }

    return 'standard'
  }

  /**
   * 同步管线进度到全局状态
   */
  function syncProgress(): void {
    if (!pipeline.value) return
    const pipelineProgress = pipeline.value.progress
    if (!pipelineProgress) return

    // 同步池子状态
    globalProgress.pools = pipelineProgress.pools.map(p => ({ ...p }))
    globalProgress.totalCompleted = pipelineProgress.totalCompleted
    globalProgress.totalFailed = pipelineProgress.totalFailed
    globalProgress.totalPages = pipelineProgress.totalPages
    globalProgress.estimatedTimeRemaining = pipelineProgress.estimatedTimeRemaining
    // 注意：preSave 和 save 字段是直接在 globalProgress 上操作的，不需要从 pipelineProgress 同步
  }

  /**
   * 执行并行翻译
   */
  async function executeParallel(
    mode?: ParallelTranslationMode
  ): Promise<ParallelExecutionResult> {
    if (isRunning.value) {
      return { success: 0, failed: 0, errors: ['翻译正在进行中'] }
    }

    const images = imageStore.images
    if (images.length === 0) {
      return { success: 0, failed: 0, errors: ['没有图片'] }
    }

    isRunning.value = true

    // 初始化全局进度
    globalProgress.totalPages = images.length
    globalProgress.totalCompleted = 0
    globalProgress.totalFailed = 0

    // 启动进度同步定时器
    const syncInterval = setInterval(syncProgress, 200)

    try {
      // 创建管线
      pipeline.value = createParallelPipeline({
        enabled: true,
        deepLearningLockSize: config.value?.deepLearningLockSize ?? 1
      })

      // 确定模式
      const translationMode = mode ?? determineMode()

      console.log(`🚀 开始并行翻译，模式: ${translationMode}，图片数: ${images.length}`)

      // 执行
      const result = await pipeline.value.execute(images, translationMode)

      // 最后同步一次
      syncProgress()

      console.log(`✅ 并行翻译完成，成功: ${result.success}，失败: ${result.failed}`)

      return result

    } catch (error) {
      console.error('并行翻译出错:', error)
      return {
        success: 0,
        failed: images.length,
        errors: [(error as Error).message]
      }
    } finally {
      clearInterval(syncInterval)
      isRunning.value = false
    }
  }

  /**
   * 取消翻译
   */
  function cancel(): void {
    if (pipeline.value) {
      pipeline.value.cancel()
    }
    isRunning.value = false
  }

  /**
   * 重置
   */
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
    determineMode
  }
}
