import type { BubbleState } from './bubble'

export interface SessionData {
  name: string
  version: string
  savedAt: string
  imageCount: number
  ui_settings: Record<string, unknown>
  images: Array<{
    originalDataURL: string
    translatedDataURL?: string
    cleanImageData?: string
    bubbleStates?: BubbleState[]
    fileName: string
    [key: string]: unknown
  }>
  currentImageIndex: number
}

export interface SessionListItem {
  name: string
  savedAt: string
  imageCount: number
  version: string
}
