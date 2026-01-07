/**
 * 修复池
 * 
 * 负责调用后端修复API，生成干净的背景图
 */

import { TaskPool } from '../TaskPool'
import type { PipelineTask } from '../types'
import type { DeepLearningLock } from '../DeepLearningLock'
import type { ParallelProgressTracker } from '../ParallelProgressTracker'
import { parallelInpaint } from '@/api/parallelTranslate'
import { useSettingsStore } from '@/stores/settingsStore'

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
    const settingsStore = useSettingsStore()
    const settings = settingsStore.settings

    if (!detectionResult || detectionResult.bubbleCoords.length === 0) {
      // 没有气泡，使用原图作为干净图
      task.inpaintResult = {
        cleanImage: this.extractBase64(imageData.originalDataURL)
      }
      task.status = 'processing'
      return task
    }

    const base64 = this.extractBase64(imageData.originalDataURL)

    // 确定修复方法和模型
    const inpaintMethod = settings.textStyle.inpaintMethod
    const useLama = inpaintMethod === 'lama_mpe' || inpaintMethod === 'litelama'

    const response = await parallelInpaint({
      image: base64,
      bubble_coords: detectionResult.bubbleCoords,
      bubble_polygons: detectionResult.bubblePolygons,
      raw_mask: detectionResult.rawMask,
      method: useLama ? 'lama' : 'solid',
      lama_model: useLama ? inpaintMethod : undefined,
      fill_color: settings.textStyle.fillColor,
      mask_dilate_size: settings.preciseMask.dilateSize,
      mask_box_expand_ratio: settings.preciseMask.boxExpandRatio
    })

    if (!response.success) {
      throw new Error(response.error || '修复失败')
    }

    task.inpaintResult = {
      cleanImage: response.clean_image || ''
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
