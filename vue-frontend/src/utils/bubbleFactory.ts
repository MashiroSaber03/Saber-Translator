import type {
  BubbleState,
  BubbleCoords,
  BubbleTextline,
  BubbleStateOverrides,
  BubbleStateUpdates,
  BubbleApiResponse,
  BubbleGlobalDefaults,
  TextDirection,
  InpaintMethod,
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
  return Array.isArray(polygon) ? polygon.map((point) => [...point]) : []
}

function cloneColorTuple(color?: BubbleColorTuple | null): BubbleColorTuple | null {
  return color ? ([...color] as BubbleColorTuple) : null
}

export function cloneBubbleTextlines(textlines?: BubbleTextline[] | null): BubbleTextline[] {
  if (!textlines || !Array.isArray(textlines)) {
    return []
  }
  return textlines.map((line) => ({
    polygon: clonePolygon(line.polygon),
    direction: line.direction === 'v' ? 'v' : 'h',
    confidence: Number(line.confidence) || 0,
  }))
}

export function getTextlinesPerBubbleFromStates(states?: BubbleState[] | null): BubbleTextline[][] {
  if (!states || !Array.isArray(states)) {
    return []
  }
  return states.map((state) => cloneBubbleTextlines(state.textlines))
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

export function createBubbleStatesFromResponse(
  response: BubbleApiResponse,
  globalDefaults?: BubbleGlobalDefaults,
): BubbleState[] {
  const {
    bubble_coords = [],
    bubble_states = [],
    original_texts = [],
    ocr_results = [],
    textlines_per_bubble = [],
    bubble_texts = [],
    textbox_texts = [],
    bubble_angles = [],
    auto_directions = [],
  } = response

  if (bubble_states.length > 0) {
    return bubble_states.map((state, index) => ({
      ...createBubbleState(globalDefaults),
      ...state,
      coords: state.coords || bubble_coords[index] || [0, 0, 100, 100],
      textlines: cloneBubbleTextlines(
        state.textlines && state.textlines.length > 0
          ? state.textlines
          : textlines_per_bubble[index],
      ),
      ocrResult: state.ocrResult || ocr_results[index] || null,
    }))
  }

  return bubble_coords.map((coords, index) => {
    let autoDirection: TextDirection
    if (auto_directions[index]) {
      autoDirection = auto_directions[index] === 'v' ? 'vertical' : 'horizontal'
    } else {
      autoDirection = detectTextDirection(coords)
    }

    const globalTextDir = globalDefaults?.textDirection
    const textDirection: TextDirection =
      (globalTextDir === 'vertical' || globalTextDir === 'horizontal')
        ? globalTextDir
        : autoDirection

    return createBubbleState({
      coords,
      originalText: original_texts[index] || ocr_results[index]?.text || '',
      textlines: cloneBubbleTextlines(textlines_per_bubble[index]),
      ocrResult: ocr_results[index] || null,
      translatedText: bubble_texts[index] || '',
      textboxText: textbox_texts[index] || '',
      rotationAngle: bubble_angles[index] || 0,
      ...globalDefaults,
      autoTextDirection: autoDirection,
      textDirection,
    })
  })
}

export function bubbleStatesToApiRequest(states: BubbleState[]): {
  bubble_coords: BubbleCoords[]
  bubble_states: BubbleState[]
  original_texts: string[]
  translated_texts: string[]
  textbox_texts: string[]
} {
  return {
    bubble_coords: states.map((s) => s.coords),
    bubble_states: states,
    original_texts: states.map((s) => s.originalText),
    translated_texts: states.map((s) => s.translatedText),
    textbox_texts: states.map((s) => s.textboxText),
  }
}

export function updateBubbleState(
  state: BubbleState,
  updates: BubbleStateUpdates,
): BubbleState {
  return {
    ...state,
    ...updates,
    position: updates.position
      ? { ...state.position, ...updates.position }
      : state.position,
  }
}

export function updateAllBubbleStates(
  states: BubbleState[],
  updates: BubbleStateUpdates,
): BubbleState[] {
  return states.map((state) => updateBubbleState(state, updates))
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
  return states.map((state) => cloneBubbleStateFields(state))
}

export function cloneBubbleState(state: BubbleState): BubbleState {
  return cloneBubbleStateFields(state)
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

function isString(value: unknown): value is string {
  return typeof value === 'string'
}

function isValidPolygon(value: unknown): value is number[][] {
  return (
    Array.isArray(value) &&
    value.every((point) =>
      Array.isArray(point) &&
      point.length >= 2 &&
      point.every((coordinate) => isFiniteNumber(coordinate))
    )
  )
}

function isValidPosition(value: unknown): value is { x: number; y: number } {
  if (!value || typeof value !== 'object') {
    return false
  }
  const position = value as Record<string, unknown>
  return isFiniteNumber(position.x) && isFiniteNumber(position.y)
}

function isValidBubbleTextline(value: unknown): value is BubbleTextline {
  if (!value || typeof value !== 'object') {
    return false
  }
  const line = value as Record<string, unknown>
  return (
    isValidPolygon(line.polygon) &&
    line.polygon.length === 4 &&
    (line.direction === 'h' || line.direction === 'v') &&
    isFiniteNumber(line.confidence)
  )
}

export function isValidBubbleState(state: unknown): state is BubbleState {
  if (!state || typeof state !== 'object') {
    return false
  }

  const s = state as Record<string, unknown>
  const requiredFields = [
    'originalText',
    'translatedText',
    'textboxText',
    'coords',
    'polygon',
    'fontSize',
    'fontFamily',
    'textDirection',
    'autoTextDirection',
    'textColor',
    'fillColor',
    'rotationAngle',
    'position',
    'strokeEnabled',
    'strokeColor',
    'strokeWidth',
    'lineSpacing',
    'textAlign',
    'inpaintMethod',
    'textlines',
  ]

  if (!requiredFields.every((field) => Object.prototype.hasOwnProperty.call(s, field))) {
    return false
  }

  if (!Array.isArray(s.coords) || s.coords.length !== 4) {
    return false
  }

  if (!s.coords.every((v) => isFiniteNumber(v))) {
    return false
  }

  if (
    !isString(s.originalText) ||
    !isString(s.translatedText) ||
    !isString(s.textboxText)
  ) {
    return false
  }

  if (!isValidPolygon(s.polygon)) {
    return false
  }

  if (!isString(s.fontFamily)) {
    return false
  }

  if (!isString(s.textColor) || !isString(s.fillColor) || !isString(s.strokeColor)) {
    return false
  }

  if (
    !isFiniteNumber(s.fontSize) ||
    s.fontSize <= 0 ||
    !isFiniteNumber(s.rotationAngle) ||
    !isFiniteNumber(s.strokeWidth) ||
    s.strokeWidth < 0 ||
    !isFiniteNumber(s.lineSpacing) ||
    s.lineSpacing <= 0
  ) {
    return false
  }

  if (typeof s.strokeEnabled !== 'boolean') {
    return false
  }

  if (!isValidPosition(s.position)) {
    return false
  }

  const validDirections: TextDirection[] = ['vertical', 'horizontal', 'auto']
  if (
    typeof s.textDirection !== 'string' ||
    !validDirections.includes(s.textDirection as TextDirection)
  ) {
    return false
  }

  if (
    typeof s.autoTextDirection !== 'string' ||
    !validDirections.includes(s.autoTextDirection as TextDirection)
  ) {
    return false
  }

  const validTextAlignments = ['start', 'center', 'end']
  if (typeof s.textAlign !== 'string' || !validTextAlignments.includes(s.textAlign)) {
    return false
  }

  const validInpaintMethods: InpaintMethod[] = ['solid', 'lama_mpe', 'litelama']
  if (
    typeof s.inpaintMethod !== 'string' ||
    !validInpaintMethods.includes(s.inpaintMethod as InpaintMethod)
  ) {
    return false
  }

  if (s.ocrResult !== undefined && s.ocrResult !== null) {
    const ocrResult = s.ocrResult as Record<string, unknown>
    if (typeof ocrResult.text !== 'string') {
      return false
    }
  }

  if (!Array.isArray(s.textlines) || !s.textlines.every((line) => isValidBubbleTextline(line))) {
    return false
  }

  return true
}

export function getBubbleCenter(state: BubbleState): { x: number; y: number } {
  const [x1, y1, x2, y2] = state.coords
  return {
    x: (x1 + x2) / 2,
    y: (y1 + y2) / 2,
  }
}

export function getBubbleSize(state: BubbleState): { width: number; height: number } {
  const [x1, y1, x2, y2] = state.coords
  return {
    width: Math.abs(x2 - x1),
    height: Math.abs(y2 - y1),
  }
}

export function isPointInBubble(state: BubbleState, x: number, y: number): boolean {
  const [x1, y1, x2, y2] = state.coords
  const minX = Math.min(x1, x2)
  const maxX = Math.max(x1, x2)
  const minY = Math.min(y1, y2)
  const maxY = Math.max(y1, y2)
  return x >= minX && x <= maxX && y >= minY && y <= maxY
}

export function isPointInPolygon(polygon: number[][], x: number, y: number): boolean {
  if (polygon.length < 3) {
    return false
  }

  let inside = false
  const n = polygon.length

  for (let i = 0, j = n - 1; i < n; j = i++) {
    const pointI = polygon[i]
    const pointJ = polygon[j]
    if (!pointI || !pointJ || pointI.length < 2 || pointJ.length < 2) {
      continue
    }
    const xi = pointI[0] as number
    const yi = pointI[1] as number
    const xj = pointJ[0] as number
    const yj = pointJ[1] as number

    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) {
      inside = !inside
    }
  }

  return inside
}

export function isPointInBubbleArea(state: BubbleState, x: number, y: number): boolean {
  if (state.polygon && state.polygon.length >= 3) {
    return isPointInPolygon(state.polygon, x, y)
  }
  return isPointInBubble(state, x, y)
}

export function getDefaultBubbleSettings(globalSettings?: {
  fontSize?: number
  fontFamily?: string
  layoutDirection?: 'auto' | 'vertical' | 'horizontal'
  textColor?: string
  fillColor?: string
  strokeEnabled?: boolean
  strokeColor?: string
  strokeWidth?: number
  inpaintMethod?: InpaintMethod
  lineSpacing?: number
  textAlign?: 'start' | 'center' | 'end'
}): BubbleStateOverrides {
  if (!globalSettings) {
    return {
      fontSize: DEFAULT_BUBBLE_STATE.fontSize,
      fontFamily: DEFAULT_BUBBLE_STATE.fontFamily,
      textDirection: DEFAULT_BUBBLE_STATE.textDirection,
      textColor: DEFAULT_BUBBLE_STATE.textColor,
      fillColor: DEFAULT_BUBBLE_STATE.fillColor,
      strokeEnabled: DEFAULT_BUBBLE_STATE.strokeEnabled,
      strokeColor: DEFAULT_BUBBLE_STATE.strokeColor,
      strokeWidth: DEFAULT_BUBBLE_STATE.strokeWidth,
      inpaintMethod: DEFAULT_BUBBLE_STATE.inpaintMethod,
      lineSpacing: DEFAULT_BUBBLE_STATE.lineSpacing,
      textAlign: DEFAULT_BUBBLE_STATE.textAlign,
    }
  }

  return {
    fontSize: globalSettings.fontSize ?? DEFAULT_BUBBLE_STATE.fontSize,
    fontFamily: globalSettings.fontFamily ?? DEFAULT_BUBBLE_STATE.fontFamily,
    textDirection: globalSettings.layoutDirection ?? DEFAULT_BUBBLE_STATE.textDirection,
    textColor: globalSettings.textColor ?? DEFAULT_BUBBLE_STATE.textColor,
    fillColor: globalSettings.fillColor ?? DEFAULT_BUBBLE_STATE.fillColor,
    strokeEnabled: globalSettings.strokeEnabled ?? DEFAULT_BUBBLE_STATE.strokeEnabled,
    strokeColor: globalSettings.strokeColor ?? DEFAULT_BUBBLE_STATE.strokeColor,
    strokeWidth: globalSettings.strokeWidth ?? DEFAULT_BUBBLE_STATE.strokeWidth,
    inpaintMethod: globalSettings.inpaintMethod ?? DEFAULT_BUBBLE_STATE.inpaintMethod,
    lineSpacing: globalSettings.lineSpacing ?? DEFAULT_BUBBLE_STATE.lineSpacing,
    textAlign: globalSettings.textAlign ?? DEFAULT_BUBBLE_STATE.textAlign,
  }
}

export function initBubbleStates(
  savedStates: BubbleState[] | undefined,
  coords: BubbleCoords[] | undefined,
  globalDefaults?: BubbleGlobalDefaults,
): BubbleState[] {
  if (savedStates && savedStates.length > 0) {
    if (!coords || savedStates.length === coords.length) {
      return cloneBubbleStates(savedStates)
    }
  }

  if (!coords || coords.length === 0) {
    return []
  }

  return coords.map((coord) => {
    const autoDirection = detectTextDirection(coord)

    const globalTextDir = globalDefaults?.textDirection
    const textDirection: TextDirection =
      (globalTextDir === 'vertical' || globalTextDir === 'horizontal')
        ? globalTextDir
        : autoDirection

    return createBubbleState({
      coords: coord,
      ...globalDefaults,
      autoTextDirection: autoDirection,
      textDirection,
    })
  })
}
