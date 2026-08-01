import type {
  BubbleState,
  BubbleCoords,
  BubbleTextline,
  BubbleStateOverrides,
  TextDirection,
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
  textAlign: TEXT_STYLE_DEFAULTS.textAlign,

  inpaintMethod: TEXT_STYLE_DEFAULTS.inpaintMethod,

  autoFgColor: null,
  autoBgColor: null,
  colorConfidence: 0,
  textlines: [],
  ocrResult: null,
}

type BubbleColorTuple = [number, number, number]

function clonePolygon(polygon?: number[][] | null): number[][] {
  return Array.isArray(polygon) ? polygon.map(point => [...point]) : []
}

function cloneColorTuple(color?: BubbleColorTuple | null): BubbleColorTuple | null {
  return color ? ([...color] as BubbleColorTuple) : null
}

export function cloneBubbleTextlines(textlines?: BubbleTextline[] | null): BubbleTextline[] {
  if (!textlines || !Array.isArray(textlines)) {
    return []
  }
  return textlines.map(line => ({
    polygon: clonePolygon(line.polygon),
    direction: line.direction === 'v' ? 'v' : 'h',
    confidence: Number(line.confidence) || 0,
  }))
}

export function createBubbleState(overrides?: BubbleStateOverrides): BubbleState {
  const base = {
    ...DEFAULT_BUBBLE_STATE,
    ...overrides,
  }

  return {
    ...base,
    coords: overrides?.coords
      ? ([...overrides.coords] as BubbleCoords)
      : ([...DEFAULT_BUBBLE_STATE.coords] as BubbleCoords),
    polygon: clonePolygon(overrides?.polygon),
    textlines: cloneBubbleTextlines(overrides?.textlines),
    position: overrides?.position
      ? { ...DEFAULT_BUBBLE_STATE.position, ...overrides.position }
      : { ...DEFAULT_BUBBLE_STATE.position },
  }
}

export function detectTextDirection(coords: BubbleCoords): TextDirection {
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
