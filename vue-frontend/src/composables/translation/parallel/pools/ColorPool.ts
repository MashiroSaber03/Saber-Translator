/**
 * 颜色提取池
 * 
 * 负责调用后端颜色提取API，识别文字和背景颜色
 */

import { TaskPool } from '../TaskPool'
import type { PipelineTask } from '../types'
import type { DeepLearningLock } from '../DeepLearningLock'
import type { ParallelProgressTracker } from '../ParallelProgressTracker'
import { executeColor } from '@/composables/translation/core/steps'

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

    // 调用独立的颜色提取步骤模块
    const result = await executeColor({
      imageIndex: task.imageIndex,
      image: imageData,
      bubbleCoords: detectionResult.bubbleCoords as any,
      textlinesPerBubble: ocrResult?.textlinesPerBubble || detectionResult.textlinesPerBubble || []
    })

    task.colorResult = {
      colors: result.colors
    }

    task.status = 'processing'
    return task
  }
}
