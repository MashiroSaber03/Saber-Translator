import type { BubbleState } from '@/types/bubble'
import type { ImageData } from '@/types/image'
import { getTextlinesPerBubbleFromStates } from '@/utils/bubbleFactory'

export function applyImageBubbleMirrors(target: ImageData, bubbleStates: BubbleState[] | null): void {
  target.bubbleStates = bubbleStates
  if (!bubbleStates) {
    target.bubbleCoords = undefined
    target.bubbleAngles = undefined
    target.originalTexts = undefined
    target.bubbleTexts = undefined
    target.textboxTexts = undefined
    target.textlinesPerBubble = undefined
    target.ocrResults = undefined
    return
  }

  target.bubbleCoords = bubbleStates.map(bubble => bubble.coords)
  target.bubbleAngles = bubbleStates.map(bubble => bubble.rotationAngle || 0)
  target.originalTexts = bubbleStates.map(bubble => bubble.originalText || '')
  target.bubbleTexts = bubbleStates.map(bubble => bubble.translatedText || '')
  target.textboxTexts = bubbleStates.map(bubble => bubble.textboxText || '')
  target.textlinesPerBubble = getTextlinesPerBubbleFromStates(bubbleStates)
  target.ocrResults = bubbleStates.map(
    bubble =>
      bubble.ocrResult || {
        text: bubble.originalText || '',
        confidence: null,
        confidenceSupported: false,
        engine: '',
        primaryEngine: '',
        fallbackUsed: false,
      },
  )
}
