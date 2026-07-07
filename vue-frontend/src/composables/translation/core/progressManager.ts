import { ref, type Ref } from 'vue'
import type { ProgressReporter, TranslationProgress } from './types'

function clampPercentage(value: number): number {
  return Math.min(100, Math.max(0, value))
}

export function createProgressManager(): {
  progress: Ref<TranslationProgress>
  reporter: ProgressReporter
} {
  const progress = ref<TranslationProgress>({
    current: 0,
    total: 0,
    completed: 0,
    failed: 0,
    isInProgress: false,
    label: '',
    percentage: 0,
  })

  const reporter: ProgressReporter = {
    init(total: number, label?: string) {
      progress.value = {
        current: 0,
        total,
        completed: 0,
        failed: 0,
        isInProgress: true,
        label: label || '准备中...',
        percentage: 0,
      }
    },

    update(current: number, label?: string) {
      progress.value.current = current
      if (label !== undefined) {
        progress.value.label = label
      }
      if (progress.value.total > 0) {
        progress.value.percentage = clampPercentage(Math.round((current / progress.value.total) * 100))
      }
    },

    setPercentage(percentage: number, label?: string) {
      progress.value.percentage = clampPercentage(percentage)
      if (label !== undefined) {
        progress.value.label = label
      }
    },

    incrementCompleted() {
      progress.value.completed++
    },

    incrementFailed() {
      progress.value.failed++
    },

    finish() {
      progress.value.isInProgress = false
      progress.value.percentage = 100
    },

    getProgress() {
      return { ...progress.value }
    },
  }

  return { progress, reporter }
}
