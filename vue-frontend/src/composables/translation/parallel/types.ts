/**
 * 并行翻译模块类型定义
 */

import type { TaskContext } from '@/composables/translation/core/runtime'

/**
 * 翻译模式
 */
export type ParallelTranslationMode = 'standard' | 'hq' | 'proofread' | 'removeText'

export type PipelineTask = TaskContext

/**
 * 池子状态
 */
export interface PoolStatus {
  name: string
  icon: string
  waiting: number
  processing: boolean
  currentPage?: number
  completed: number
  isWaitingLock: boolean
}

/**
 * 并行进度
 */
export interface ParallelProgress {
  pools: PoolStatus[]
  totalCompleted: number
  totalFailed: number
  totalPages: number
  estimatedTimeRemaining: number
  // 预保存进度
  preSave?: {
    isRunning: boolean
    current: number
    total: number
  }
  // 保存进度（翻译过程中的保存）
  save?: {
    completed: number
    total: number
  }
}

/**
 * 并行配置
 */
export interface ParallelConfig {
  enabled: boolean
  deepLearningLockSize: number  // 深度学习锁大小（并发数）
}

/**
 * 池子进度更新
 */
export interface PoolProgressUpdate {
  waiting?: number
  isProcessing?: boolean
  currentPage?: number
  completed?: number
  isWaitingLock?: boolean
}

/**
 * 并行执行结果
 */
export interface ParallelExecutionResult {
  success: number
  failed: number
  errors?: string[]
  autoGlossaryStats?: {
    added: number
    duplicates: number
    failedPages: number
  }
}
