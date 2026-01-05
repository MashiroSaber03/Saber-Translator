/**
 * 进度管理器
 * 
 * 提供统一的进度报告和管理功能
 */

import { ref, type Ref } from 'vue'
import type { TranslationProgress, ProgressReporter } from './types'

/**
 * 创建进度管理器
 */
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
        percentage: 0
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
                percentage: 0
            }
        },

        update(current: number, label?: string) {
            progress.value.current = current
            if (label !== undefined) {
                progress.value.label = label
            }
            // 自动计算百分比
            if (progress.value.total > 0) {
                progress.value.percentage = Math.round((current / progress.value.total) * 100)
            }
        },

        setPercentage(percentage: number, label?: string) {
            progress.value.percentage = Math.min(100, Math.max(0, percentage))
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
        }
    }

    return { progress, reporter }
}
