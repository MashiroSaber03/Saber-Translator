/**
 * OCR池
 * 
 * 负责调用后端OCR API，识别气泡中的文字
 */

import { TaskPool } from '../TaskPool'
import type { PipelineTask } from '../types'
import type { DeepLearningLock } from '../DeepLearningLock'
import type { ParallelProgressTracker } from '../ParallelProgressTracker'
import { executeOcr } from '@/composables/translation/core/steps'

export class OcrPool extends TaskPool {
  constructor(
    nextPool: TaskPool | null,
    lock: DeepLearningLock,
    progressTracker: ParallelProgressTracker,
    onTaskComplete?: (task: PipelineTask) => void
  ) {
    super('OCR', '📖', nextPool, lock, progressTracker, onTaskComplete)
  }

  protected async process(task: PipelineTask): Promise<PipelineTask> {
    const { imageData, detectionResult } = task

    if (!detectionResult || detectionResult.bubbleCoords.length === 0) {
      // 没有检测到气泡，直接跳过
      task.ocrResult = {
        originalTexts: [],
        ocrResults: [],
        textlinesPerBubble: []
      }
      task.status = 'processing'
      return task
    }

    // 调用独立的OCR步骤模块
    const result = await executeOcr({
      imageIndex: task.imageIndex,
      image: imageData,
      bubbleCoords: detectionResult.bubbleCoords as any,
      bubbleStates: imageData.bubbleStates,
      textlinesPerBubble: detectionResult.textlinesPerBubble || []
    })

    if (imageData.bubbleStates) {
      imageData.bubbleStates = imageData.bubbleStates.map((bubble, index) => ({
        ...bubble,
        originalText: result.originalTexts[index] || '',
        ocrResult: result.ocrResults[index] || null
      }))
    }

    task.ocrResult = {
      originalTexts: result.originalTexts,
      ocrResults: result.ocrResults,
      textlinesPerBubble: imageData.bubbleStates?.map((bubble) => bubble.textlines || []) || detectionResult.textlinesPerBubble || []
    }

    task.status = 'processing'
    return task
  }
}
