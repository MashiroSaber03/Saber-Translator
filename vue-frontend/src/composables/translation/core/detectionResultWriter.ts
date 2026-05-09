import { useImageStore } from '@/stores/imageStore'
import type { DetectionOutput } from './steps/detection'

/**
 * 统一保存检测结果到 ImageData
 * 确保所有检测结果字段都被正确保存，避免遗漏
 */
export function saveDetectionResultToImage(
  imageIndex: number,
  result: DetectionOutput,
  options?: {
    updateBubbleStates?: boolean
    bubbleStates?: any[]
  }
): void {
  const imageStore = useImageStore()

  const updateData: Record<string, any> = {
    bubbleCoords: result.bubbleCoords,
    bubbleAngles: result.bubbleAngles,
    textMask: result.textMask || null,
    textlinesPerBubble: result.textlinesPerBubble || [],
  }

  if (options?.updateBubbleStates && options.bubbleStates) {
    updateData.bubbleStates = options.bubbleStates.map((state, index) => ({
      ...state,
      textlines: state.textlines && state.textlines.length > 0
        ? state.textlines
        : (result.textlinesPerBubble[index] || [])
    }))
  } else if (result.bubbleStates.length > 0) {
    updateData.bubbleStates = result.bubbleStates
  }

  imageStore.updateImageByIndex(imageIndex, updateData)
}
