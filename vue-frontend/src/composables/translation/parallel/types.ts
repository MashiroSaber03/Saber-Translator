import type { TaskContext } from '@/composables/translation/core/runtime'
import type { UiIconName } from '@/components/ui/iconRegistry'

export type ParallelTranslationMode = 'standard' | 'hq' | 'proofread' | 'removeText'

export type PipelineTask = TaskContext

export interface PoolStatus {
  name: string
  icon: UiIconName
  waiting: number
  processing: boolean
  currentPage?: number
  completed: number
  isWaitingLock: boolean
}

export interface ParallelProgress {
  pools: PoolStatus[]
  totalCompleted: number
  totalFailed: number
  totalPages: number
  estimatedTimeRemaining: number
  preSave?: {
    isRunning: boolean
    current: number
    total: number
  }
  save?: {
    completed: number
    total: number
  }
}

export interface ParallelConfig {
  enabled: boolean
  deepLearningLockSize: number
}

export interface PoolProgressUpdate {
  waiting?: number
  isProcessing?: boolean
  currentPage?: number
  completed?: number
  isWaitingLock?: boolean
}

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
