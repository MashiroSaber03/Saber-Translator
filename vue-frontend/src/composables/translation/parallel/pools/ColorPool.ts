/**
 * 颜色提取池
 * 
 * 负责调用后端颜色提取API，识别文字和背景颜色
 */

import { TaskPool } from '../TaskPool'
import type { PipelineTask } from '../types'
import type { DeepLearningLock } from '../DeepLearningLock'
import type { ParallelProgressTracker } from '../ParallelProgressTracker'
import { parallelColor } from '@/api/parallelTranslate'

export class ColorPool extends TaskPool {
  constructor(
    nextPool: TaskPool | null,
    lock: DeepLearningLock,
    progressTracker: ParallelProgressTracker,
    onTaskComplete?: (task: PipelineTask) => void
  ) {
    super('颜色', '🎨', nextPool, lock, progressTracker, onTaskComplete)
  }

  protected async process(task: PipelineTask): Promise<PipelineTask> {
    const { imageData, detectionResult, ocrResult } = task

    if (!detectionResult || detectionResult.bubbleCoords.length === 0) {
      task.colorResult = { colors: [] }
      task.status = 'processing'
      return task
    }

    const base64 = this.extractBase64(imageData.originalDataURL)

    const response = await parallelColor({
      image: base64,
      bubble_coords: detectionResult.bubbleCoords,
      textlines_per_bubble: ocrResult?.textlinesPerBubble || detectionResult.textlinesPerBubble
    })

    if (!response.success) {
      throw new Error(response.error || '颜色提取失败')
    }

    task.colorResult = {
      colors: response.colors || []
    }

    task.status = 'processing'
    return task
  }

  private extractBase64(dataUrl: string): string {
    if (dataUrl.includes('base64,')) {
      return dataUrl.split('base64,')[1] || ''
    }
    return dataUrl
  }
}
