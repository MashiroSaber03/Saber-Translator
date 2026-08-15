import type { OcrResult } from './ocr'

export type BubbleCoords = [number, number, number, number]

export type PolygonCoords = number[][]

export interface BubbleTextline {
  polygon: PolygonCoords
  direction: 'h' | 'v'
  confidence: number
}

export type TextDirection = 'vertical' | 'horizontal' | 'auto'

export type ResolvedTextDirection = Exclude<TextDirection, 'auto'>

export type TextAlign = 'start' | 'center' | 'end'

export type InpaintMethod = 'solid' | 'lama_mpe' | 'litelama'

export interface BubblePosition {
  x: number
  y: number
}

export interface BubbleState {
  /** Stable backend identity. UI-only bubbles do not have one until persisted. */
  backendBubbleId?: string
  /** Correlates an unsaved UI bubble with the backend-created identity. */
  clientMutationId?: string
  originalText: string
  translatedText: string
  textboxText: string

  coords: BubbleCoords
  polygon: PolygonCoords

  fontSize: number
  fontFamily: string
  textDirection: ResolvedTextDirection
  autoTextDirection: ResolvedTextDirection
  textColor: string
  fillColor: string
  rotationAngle: number
  position: BubblePosition

  strokeEnabled: boolean
  strokeColor: string
  strokeWidth: number

  lineSpacing: number
  textAlign: TextAlign

  inpaintMethod: InpaintMethod

  autoFgColor: [number, number, number] | null
  autoBgColor: [number, number, number] | null
  colorConfidence: number
  textlines: BubbleTextline[]
  ocrResult: OcrResult | null
}

export type BubbleStateOverrides = Partial<BubbleState>

export type BubbleStateUpdates = Partial<BubbleState>
