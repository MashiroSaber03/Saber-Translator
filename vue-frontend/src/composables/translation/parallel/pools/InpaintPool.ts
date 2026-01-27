/**
 * 修复池
 * 
 * 负责调用后端修复API，生成干净的背景图
 */

import { TaskPool } from '../TaskPool'
import type { PipelineTask } from '../types'
import type { DeepLearningLock } from '../DeepLearningLock'
import type { ParallelProgressTracker } from '../ParallelProgressTracker'
import { executeInpaint } from '@/composables/translation/core/steps'

export class InpaintPool extends TaskPool {
  constructor(
    nextPool: TaskPool | null,
    lock: DeepLearningLock,
    progressTracker: ParallelProgressTracker,
    onTaskComplete?: (task: PipelineTask) => void
  ) {
    super('修复', '🖌️', nextPool, lock, progressTracker, onTaskComplete)
  }

  protected async process(task: PipelineTask): Promise<PipelineTask> {
    const { imageData, detectionResult } = task

    if (!detectionResult || detectionResult.bubbleCoords.length === 0) {
      // 没有气泡，使用原图作为干净图
      const extractBase64 = (dataUrl: string): string => {
        if (dataUrl.includes('base64,')) {
          return dataUrl.split('base64,')[1] || ''
        }
        return dataUrl
      }

      task.inpaintResult = {
        cleanImage: extractBase64(imageData.originalDataURL)
      }
      task.status = 'processing'
      return task
    }

    // 调用独立的修复步骤模块
    const result = await executeInpaint({
      imageIndex: task.imageIndex,
      image: imageData,
      bubbleCoords: detectionResult.bubbleCoords as any,
      bubblePolygons: detectionResult.bubblePolygons,
      textMask: detectionResult.textMask,
      userMask: imageData.userMask || undefined  // ✅ 传递用户掩膜
    })

    task.inpaintResult = {
      cleanImage: result.cleanImage
    }

    task.status = 'processing'
    return task
  }
}
