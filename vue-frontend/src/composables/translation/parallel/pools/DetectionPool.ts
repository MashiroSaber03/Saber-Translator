/**
 * 检测池
 * 
 * 负责调用后端检测API，获取气泡坐标、角度、多边形等信息
 */

import { TaskPool } from '../TaskPool'
import type { PipelineTask } from '../types'
import type { DeepLearningLock } from '../DeepLearningLock'
import type { ParallelProgressTracker } from '../ParallelProgressTracker'
import { executeDetection } from '@/composables/translation/core/steps'

export class DetectionPool extends TaskPool {
  constructor(
    nextPool: TaskPool | null,
    lock: DeepLearningLock,
    progressTracker: ParallelProgressTracker,
    onTaskComplete?: (task: PipelineTask) => void
  ) {
    super('检测', '📍', nextPool, lock, progressTracker, onTaskComplete)
  }

  protected async process(task: PipelineTask): Promise<PipelineTask> {
    const { imageData } = task

    // 调用独立的检测步骤模块
    const result = await executeDetection({
      imageIndex: task.imageIndex,
      image: imageData,
      forceDetect: false
    })

    // 保存检测结果
    task.detectionResult = {
      bubbleCoords: result.bubbleCoords,
      bubbleAngles: result.bubbleAngles,
      bubblePolygons: result.bubblePolygons,
      autoDirections: result.autoDirections,
      textMask: result.textMask,
      textlinesPerBubble: result.textlinesPerBubble,
      bubbleStates: result.bubbleStates
    }
    task.imageData.bubbleStates = result.bubbleStates

    task.status = 'processing'
    return task
  }
}
