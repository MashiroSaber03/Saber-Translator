/**
 * 检测池
 * 
 * 负责调用后端检测API，获取气泡坐标、角度、多边形等信息
 */

import { TaskPool } from '../TaskPool'
import type { PipelineTask } from '../types'
import type { DeepLearningLock } from '../DeepLearningLock'
import type { ParallelProgressTracker } from '../ParallelProgressTracker'
import { parallelDetect } from '@/api/parallelTranslate'
import { useSettingsStore } from '@/stores/settingsStore'

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
    const settingsStore = useSettingsStore()
    const settings = settingsStore.settings

    // 提取Base64
    const base64 = this.extractBase64(imageData.originalDataURL)

    // 调用后端检测API
    // 注意：精准掩膜参数(usePreciseMask等)只在修复阶段使用，检测阶段不需要
    const response = await parallelDetect({
      image: base64,
      detector_type: settings.textDetector,
      box_expand_ratio: settings.boxExpand.ratio,
      box_expand_top: settings.boxExpand.top,
      box_expand_bottom: settings.boxExpand.bottom,
      box_expand_left: settings.boxExpand.left,
      box_expand_right: settings.boxExpand.right
    })

    if (!response.success) {
      throw new Error(response.error || '检测失败')
    }

    // 保存检测结果
    task.detectionResult = {
      bubbleCoords: response.bubble_coords || [],
      bubbleAngles: response.bubble_angles || [],
      bubblePolygons: response.bubble_polygons || [],
      autoDirections: response.auto_directions || [],
      rawMask: response.raw_mask,
      textlinesPerBubble: response.textlines_per_bubble || []
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
