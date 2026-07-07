import type { BubbleCoords, BubbleState, BubbleTextline } from '@/types/bubble'
import type { ImageData } from '@/types/image'
import type { OcrResult } from '@/types/ocr'
import { normalizeImageTextStyleFields } from '@/defaults/textStyleDefaults'
import { getTextlinesPerBubbleFromStates } from '@/utils/bubbleFactory'

export interface SessionImagePayload {
  originalDataURL: string
  translatedDataURL?: string
  cleanImageData?: string
  bubbleStates?: unknown[]
  fileName: string
  [key: string]: unknown
}

function isBubbleTextline(value: unknown): value is BubbleTextline {
  if (!value || typeof value !== 'object') return false

  const textline = value as Partial<BubbleTextline>
  return (
    Array.isArray(textline.polygon) &&
    (textline.direction === 'h' || textline.direction === 'v') &&
    typeof textline.confidence === 'number'
  )
}

function readSessionTextlinesPerBubble(value: unknown): BubbleTextline[][] | undefined {
  if (!Array.isArray(value)) return undefined

  return value.map((group) => (
    Array.isArray(group) ? group.filter(isBubbleTextline) : []
  ))
}

export function hydrateSessionImages(sessionImages: SessionImagePayload[]): ImageData[] {
  return sessionImages.map((img, index) => {
    const restoredTextlines = readSessionTextlinesPerBubble(img.textlinesPerBubble)
    const bubbleStates = (img.bubbleStates !== undefined && img.bubbleStates !== null)
      ? (img.bubbleStates as BubbleState[]).map((state, bubbleIndex) => ({
          ...state,
          textlines: state.textlines && state.textlines.length > 0
            ? state.textlines
            : restoredTextlines?.[bubbleIndex] ?? []
        }))
      : null

    return {
      id: `session-${index}-${Date.now()}`,
      originalDataURL: img.originalDataURL,
      translatedDataURL: img.translatedDataURL || null,
      cleanImageData: img.cleanImageData || null,
      width: (img.width as number) || undefined,
      height: (img.height as number) || undefined,
      bubbleStates,
      bubbleCoords: bubbleStates
        ? bubbleStates.map((state) => state.coords)
        : (img.bubbleCoords !== undefined ? (img.bubbleCoords as BubbleCoords[]) : undefined),
      bubbleAngles: bubbleStates
        ? bubbleStates.map((state) => state.rotationAngle || 0)
        : (img.bubbleAngles !== undefined ? (img.bubbleAngles as number[]) : undefined),
      originalTexts: bubbleStates
        ? bubbleStates.map((state) => state.originalText || '')
        : (img.originalTexts !== undefined ? (img.originalTexts as string[]) : undefined),
      bubbleTexts: bubbleStates
        ? bubbleStates.map((state) => state.translatedText || '')
        : (img.bubbleTexts !== undefined ? (img.bubbleTexts as string[]) : undefined),
      textboxTexts: bubbleStates
        ? bubbleStates.map((state) => state.textboxText || '')
        : (img.textboxTexts !== undefined ? (img.textboxTexts as string[]) : undefined),
      textlinesPerBubble: bubbleStates
        ? getTextlinesPerBubbleFromStates(bubbleStates)
        : restoredTextlines,
      isManuallyAnnotated: Boolean(img.isManuallyAnnotated),
      relativePath: (img.relativePath as string) || undefined,
      folderPath: (img.folderPath as string) || undefined,
      ocrResults: bubbleStates
        ? bubbleStates.map((state, bubbleIndex) => state.ocrResult || ((img.ocrResults as OcrResult[] | undefined)?.[bubbleIndex] ?? {
            text: state.originalText || '',
            confidence: null,
            confidenceSupported: false,
            engine: '',
            primaryEngine: '',
            fallbackUsed: false
          }))
        : (img.ocrResults !== undefined ? (img.ocrResults as OcrResult[]) : undefined),
      fileName: img.fileName || `image-${index + 1}.png`,
      translationStatus: (img.translationStatus as 'pending' | 'processing' | 'completed' | 'failed') || 'pending',
      translationFailed: Boolean(img.translationFailed),
      hasUnsavedChanges: false,
      ...normalizeImageTextStyleFields(img),
      textMask: (img.textMask as string) || null,
      userMask: (img.userMask as string) || null,
    }
  })
}
