/**
 * 并行翻译模块类型定义
 */

import type { ImageData } from '@/types/image'
import type { BubbleState } from '@/types/bubble'

/**
 * 翻译模式
 */
export type ParallelTranslationMode = 'standard' | 'hq' | 'proofread' | 'removeText'

/**
 * 任务状态
 */
export type TaskStatus = 'pending' | 'processing' | 'completed' | 'failed' | 'buffered'

/**
 * 流水线任务
 */
export interface PipelineTask {
  id: string
  imageIndex: number
  imageData: ImageData
  status: TaskStatus
  error?: string

  // 检测结果
  detectionResult?: {
    bubbleCoords: number[][]  // [[x1, y1, x2, y2], ...]
    bubbleAngles: number[]
    bubblePolygons: number[][][]
    autoDirections: string[]
    textMask?: string
    textlinesPerBubble?: any[]
  }

  // OCR结果
  ocrResult?: {
    originalTexts: string[]
    textlinesPerBubble?: any[]
  }

  // 颜色提取结果
  colorResult?: {
    colors: Array<{
      textColor: string
      bgColor: string
      autoFgColor?: [number, number, number] | null
      autoBgColor?: [number, number, number] | null
    }>
  }

  // 翻译结果
  translateResult?: {
    translatedTexts: string[]
    textboxTexts: string[]
  }

  // 修复结果
  inpaintResult?: {
    cleanImage: string
  }

  // 渲染结果
  renderResult?: {
    finalImage: string
    bubbleStates: BubbleState[]
  }
}

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
 * 池子链配置
 */
export interface PoolChainConfig {
  pools: string[]
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
 * 翻译JSON数据（用于高质量翻译和AI校对）
 */
export interface TranslationJsonData {
  imageIndex: number
  bubbles: Array<{
    bubbleIndex: number
    original: string
    translated: string
    textDirection: string
  }>
}

/**
 * 并行执行结果
 */
export interface ParallelExecutionResult {
  success: number
  failed: number
  errors?: string[]
}
