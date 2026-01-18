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

    // 如果图片已有 bubbleStates 数据（包括空数组），跳过检测
    // - bubbleStates === null/undefined: 从未处理过，需要自动检测
    // - bubbleStates === []: 用户主动清空，跳过检测（避免"框复活"）
    // - bubbleStates.length > 0: 有气泡数据，复用已有数据
    const existingBubbles = imageData.bubbleStates
    if (existingBubbles !== null && existingBubbles !== undefined) {
      if (existingBubbles.length > 0) {
        console.log(`[检测池] 图片 ${task.imageIndex + 1} 已有 ${existingBubbles.length} 个气泡，跳过检测`)
        task.detectionResult = {
          // 坐标需要转换为整数，后端 numpy 切片需要整数索引
          bubbleCoords: existingBubbles.map(s => s.coords.map(c => Math.round(c))),
          bubbleAngles: existingBubbles.map(s => s.rotationAngle || 0),
          bubblePolygons: existingBubbles.map(s => s.polygon || []),
          autoDirections: existingBubbles.map(s => s.autoTextDirection || s.textDirection || 'vertical'),
          rawMask: undefined,
          textlinesPerBubble: []
        }
      } else {
        console.log(`[检测池] 图片 ${task.imageIndex + 1} 气泡已被清空，跳过检测`)
        task.detectionResult = {
          bubbleCoords: [],
          bubbleAngles: [],
          bubblePolygons: [],
          autoDirections: [],
          rawMask: undefined,
          textlinesPerBubble: []
        }
      }
      task.status = 'processing'
      return task
    }

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
