import { useImageStore } from '@/stores/imageStore'
import type { BubbleState } from '@/types/bubble'
import type { ImageDataUpdates } from '@/types/image'
import type { DetectionOutput } from './steps/detection'

interface DetectionResultWriterOptions {
  imageStore?: ReturnType<typeof useImageStore>
  updateBubbleStates?: boolean
  bubbleStates?: BubbleState[]
}

export function saveDetectionResultToImage(
  imageIndex: number,
  result: DetectionOutput,
  options: DetectionResultWriterOptions = {},
): void {
  const imageStore = options.imageStore ?? useImageStore()

  const updateData: ImageDataUpdates = {
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
        : (result.textlinesPerBubble[index] || []),
    }))
  } else if (result.bubbleStates.length > 0) {
    updateData.bubbleStates = result.bubbleStates
  }

  imageStore.updateImageByIndex(imageIndex, updateData)
}
