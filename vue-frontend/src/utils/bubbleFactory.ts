import type {
  BubbleState,
  BubbleCoords,
  BubbleTextline,
  BubbleStateOverrides,
  ResolvedTextDirection,
} from '@/types/bubble'
import { TEXT_STYLE_DEFAULTS } from '@/defaults/textStyleDefaults'

export const DEFAULT_BUBBLE_STATE: BubbleState = {
  originalText: '',
  translatedText: '',
  textboxText: '',

  coords: [0, 0, 100, 100],
  polygon: [],

  fontSize: TEXT_STYLE_DEFAULTS.fontSize,
  fontFamily: TEXT_STYLE_DEFAULTS.fontFamily,
  textDirection: 'vertical',
  autoTextDirection: 'vertical',
  textColor: TEXT_STYLE_DEFAULTS.textColor,
  fillColor: TEXT_STYLE_DEFAULTS.fillColor,
  rotationAngle: 0,
  position: { x: 0, y: 0 },

  strokeEnabled: TEXT_STYLE_DEFAULTS.strokeEnabled,
  strokeColor: TEXT_STYLE_DEFAULTS.strokeColor,
  strokeWidth: TEXT_STYLE_DEFAULTS.strokeWidth,

  lineSpacing: TEXT_STYLE_DEFAULTS.lineSpacing,
  inlineAlign: TEXT_STYLE_DEFAULTS.inlineAlign,
  blockAlign: TEXT_STYLE_DEFAULTS.blockAlign,

  inpaintMethod: TEXT_STYLE_DEFAULTS.inpaintMethod,

  autoFgColor: null,
  autoBgColor: null,
  colorConfidence: 0,
  textlines: [],
  ocrResult: null,
}

type BubbleColorTuple = [number, number, number]

function clonePolygon(polygon: number[][]): number[][] {
  return polygon.map(point => [...point])
}

function cloneColorTuple(color: BubbleColorTuple | null): BubbleColorTuple | null {
  return color ? ([...color] as BubbleColorTuple) : null
}

export function cloneBubbleTextlines(textlines: BubbleTextline[]): BubbleTextline[] {
  return textlines.map(line => ({
    polygon: clonePolygon(line.polygon),
    direction: line.direction,
    confidence: line.confidence,
  }))
}

export function createBubbleState(overrides?: BubbleStateOverrides): BubbleState {
  const state: BubbleState = {
    ...DEFAULT_BUBBLE_STATE,
    ...overrides,
  }

  return {
    ...state,
    coords: [...state.coords] as BubbleCoords,
    polygon: clonePolygon(state.polygon),
    textlines: cloneBubbleTextlines(state.textlines),
    position: { ...state.position },
    autoFgColor: cloneColorTuple(state.autoFgColor),
    autoBgColor: cloneColorTuple(state.autoBgColor),
    ocrResult: state.ocrResult ? { ...state.ocrResult } : null,
  }
}

export function detectTextDirection(coords: BubbleCoords): ResolvedTextDirection {
  const [x1, y1, x2, y2] = coords
  const width = Math.abs(x2 - x1)
  const height = Math.abs(y2 - y1)
  return height > width ? 'vertical' : 'horizontal'
}

function cloneBubbleStateFields(state: BubbleState): BubbleState {
  return {
    ...state,
    coords: [...state.coords] as BubbleCoords,
    polygon: clonePolygon(state.polygon),
    position: { ...state.position },
    textlines: cloneBubbleTextlines(state.textlines),
    autoFgColor: cloneColorTuple(state.autoFgColor),
    autoBgColor: cloneColorTuple(state.autoBgColor),
    ocrResult: state.ocrResult ? { ...state.ocrResult } : null,
  }
}

export function cloneBubbleStates(states: BubbleState[]): BubbleState[] {
  return states.map(state => cloneBubbleStateFields(state))
}
