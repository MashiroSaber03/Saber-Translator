import type { OcrResult } from './ocr'

export type BubbleCoords = [number, number, number, number]

export type PolygonCoords = number[][]

export interface BubbleTextline {
  polygon: PolygonCoords
  direction: 'h' | 'v'
  confidence: number
}

export type TextDirection = 'vertical' | 'horizontal' | 'auto'

export type TextAlign = 'start' | 'center' | 'end'

export type InpaintMethod = 'solid' | 'lama_mpe' | 'litelama'

export interface BubblePosition {
  x: number
  y: number
}

export interface BubbleState {
  originalText: string
  translatedText: string
  textboxText: string

  coords: BubbleCoords
  polygon: PolygonCoords

  fontSize: number
  fontFamily: string
  textDirection: TextDirection
  autoTextDirection: TextDirection
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

  autoFgColor?: [number, number, number] | null
  autoBgColor?: [number, number, number] | null
  colorConfidence?: number
  textlines: BubbleTextline[]
  ocrResult?: OcrResult | null
}

export type BubbleStateOverrides = Partial<BubbleState>

export type BubbleStateUpdates = Partial<BubbleState>

export interface BubbleApiResponse {
  bubble_coords?: BubbleCoords[]
  bubble_states?: BubbleState[]
  original_texts?: string[]
  ocr_results?: OcrResult[]
  textlines_per_bubble?: BubbleTextline[][]
  bubble_texts?: string[]
  textbox_texts?: string[]
  bubble_angles?: number[]
  auto_directions?: ('v' | 'h')[]
}

export interface BubbleGlobalDefaults {
  fontSize?: number
  fontFamily?: string
  textDirection?: TextDirection
  textColor?: string
  fillColor?: string
  inpaintMethod?: InpaintMethod
  strokeEnabled?: boolean
  strokeColor?: string
  strokeWidth?: number
  lineSpacing?: number
  textAlign?: TextAlign
}

export function getEffectiveDirection(
  bubble: Pick<BubbleState, 'textDirection' | 'autoTextDirection' | 'coords'>
): 'vertical' | 'horizontal' {
  if (bubble.textDirection === 'vertical' || bubble.textDirection === 'horizontal') {
    return bubble.textDirection
  }

  // 异常输入会按检测方向和气泡宽高比回退。
  if (bubble.autoTextDirection === 'vertical' || bubble.autoTextDirection === 'horizontal') {
    return bubble.autoTextDirection
  }

  if (bubble.coords) {
    const [x1, y1, x2, y2] = bubble.coords
    return (y2 - y1) > (x2 - x1) ? 'vertical' : 'horizontal'
  }
  return 'vertical'
}
