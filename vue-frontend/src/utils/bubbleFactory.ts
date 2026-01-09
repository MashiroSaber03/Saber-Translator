/**
 * 气泡状态工厂函数
 * 提供创建、更新、验证气泡状态的工具函数
 */

import type {
  BubbleState,
  BubbleCoords,
  BubbleStateOverrides,
  BubbleStateUpdates,
  BubbleApiResponse,
  BubbleGlobalDefaults,
  TextDirection,
  InpaintMethod
} from '@/types/bubble'
import {
  DEFAULT_FILL_COLOR,
  DEFAULT_STROKE_ENABLED,
  DEFAULT_STROKE_COLOR,
  DEFAULT_STROKE_WIDTH
} from '@/constants'

/**
 * 默认气泡状态值
 * 与后端 BubbleState 默认值保持一致
 */
export const DEFAULT_BUBBLE_STATE: BubbleState = {
  // 文本内容
  originalText: '',
  translatedText: '',
  textboxText: '',

  // 坐标信息
  coords: [0, 0, 100, 100],
  polygon: [],

  // 渲染参数
  fontSize: 25,
  fontFamily: 'fonts/STSONG.TTF',
  textDirection: 'vertical',  // 简化设计：不再使用 'auto'，始终是具体方向
  autoTextDirection: 'vertical',
  textColor: '#000000',
  fillColor: DEFAULT_FILL_COLOR,
  rotationAngle: 0,
  position: { x: 0, y: 0 },

  // 描边参数
  strokeEnabled: DEFAULT_STROKE_ENABLED,
  strokeColor: DEFAULT_STROKE_COLOR,
  strokeWidth: DEFAULT_STROKE_WIDTH,

  // 修复参数
  inpaintMethod: 'solid',

  // 自动颜色提取（可选字段，翻译时由后端填充）
  autoFgColor: null,
  autoBgColor: null,
  colorConfidence: 0
}

/**
 * 创建气泡状态
 * @param overrides - 覆盖默认值的参数
 * @returns 完整的气泡状态对象
 */
export function createBubbleState(overrides?: BubbleStateOverrides): BubbleState {
  // 先合并基础属性
  const base = {
    ...DEFAULT_BUBBLE_STATE,
    ...overrides
  }

  // 确保数组和对象是独立的副本，覆盖可能被共享的引用
  return {
    ...base,
    coords: overrides?.coords
      ? ([...overrides.coords] as BubbleCoords)
      : ([...DEFAULT_BUBBLE_STATE.coords] as BubbleCoords),
    polygon: overrides?.polygon ? overrides.polygon.map((point) => [...point]) : [],
    position: overrides?.position
      ? { ...DEFAULT_BUBBLE_STATE.position, ...overrides.position }
      : { ...DEFAULT_BUBBLE_STATE.position }
  }
}

/**
 * 根据气泡宽高比自动检测排版方向
 * @param coords - 气泡坐标 [x1, y1, x2, y2]
 * @returns 自动检测的排版方向
 */
export function detectTextDirection(coords: BubbleCoords): TextDirection {
  const [x1, y1, x2, y2] = coords
  const width = Math.abs(x2 - x1)
  const height = Math.abs(y2 - y1)
  // 高度大于宽度时使用垂直排版
  return height > width ? 'vertical' : 'horizontal'
}

/**
 * 从后端响应创建气泡状态数组
 * @param response - 后端 API 响应
 * @param globalDefaults - 全局默认设置
 * @returns 气泡状态数组
 */
export function createBubbleStatesFromResponse(
  response: BubbleApiResponse,
  globalDefaults?: BubbleGlobalDefaults
): BubbleState[] {
  const {
    bubble_coords = [],
    bubble_states = [],
    original_texts = [],
    bubble_texts = [],
    textbox_texts = [],
    bubble_angles = [],
    auto_directions = []  // 后端基于文本行分析的排版方向
  } = response

  // 如果后端返回了完整的 bubble_states，直接使用
  if (bubble_states.length > 0) {
    return bubble_states.map((state, index) => ({
      ...createBubbleState(globalDefaults),
      ...state,
      // 确保坐标存在
      coords: state.coords || bubble_coords[index] || [0, 0, 100, 100]
    }))
  }

  // 否则根据坐标创建新的状态
  return bubble_coords.map((coords, index) => {
    // 【简化设计】获取后端检测的方向（备份）
    let autoDirection: TextDirection
    if (auto_directions[index]) {
      autoDirection = auto_directions[index] === 'v' ? 'vertical' : 'horizontal'
    } else {
      // 降级方案：根据合并后大框的宽高比判断
      autoDirection = detectTextDirection(coords)
    }

    // 【简化设计】textDirection 直接使用具体方向值：
    // - 如果全局设置是 'auto'，使用检测结果
    // - 否则使用全局设置的值
    const globalTextDir = globalDefaults?.textDirection
    const textDirection: TextDirection =
      (globalTextDir === 'vertical' || globalTextDir === 'horizontal')
        ? globalTextDir
        : autoDirection

    return createBubbleState({
      coords,
      originalText: original_texts[index] || '',
      translatedText: bubble_texts[index] || '',
      textboxText: textbox_texts[index] || '',
      rotationAngle: bubble_angles[index] || 0,
      ...globalDefaults,
      // 这两个必须在 globalDefaults 之后，确保不被覆盖
      autoTextDirection: autoDirection,  // 备份检测结果
      textDirection: textDirection,       // 渲染用的方向
    })
  })
}

/**
 * 将气泡状态数组转换为 API 请求格式
 * @param states - 气泡状态数组
 * @returns API 请求格式的数据
 */
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
    textbox_texts: states.map((s) => s.textboxText)
  }
}

/**
 * 更新单个气泡状态
 * @param state - 原始气泡状态
 * @param updates - 更新的字段
 * @returns 更新后的气泡状态
 */
export function updateBubbleState(
  state: BubbleState,
  updates: BubbleStateUpdates
): BubbleState {
  return {
    ...state,
    ...updates,
    // 如果更新了 position，需要合并而不是替换
    position: updates.position
      ? { ...state.position, ...updates.position }
      : state.position
  }
}

/**
 * 批量更新所有气泡状态
 * @param states - 气泡状态数组
 * @param updates - 要应用到所有气泡的更新
 * @returns 更新后的气泡状态数组
 */
export function updateAllBubbleStates(
  states: BubbleState[],
  updates: BubbleStateUpdates
): BubbleState[] {
  return states.map((state) => updateBubbleState(state, updates))
}

/**
 * 深拷贝气泡状态数组
 * @param states - 气泡状态数组
 * @returns 深拷贝后的数组
 */
export function cloneBubbleStates(states: BubbleState[]): BubbleState[] {
  return states.map((state) => ({
    ...state,
    coords: [...state.coords] as BubbleCoords,
    polygon: state.polygon ? state.polygon.map((point) => [...point]) : [],
    position: { ...state.position },
    // 深拷贝颜色数组（如果存在）
    autoFgColor: state.autoFgColor ? [...state.autoFgColor] as [number, number, number] : null,
    autoBgColor: state.autoBgColor ? [...state.autoBgColor] as [number, number, number] : null
  }))
}

/**
 * 深拷贝单个气泡状态
 * @param state - 气泡状态
 * @returns 深拷贝后的状态
 */
export function cloneBubbleState(state: BubbleState): BubbleState {
  return {
    ...state,
    coords: [...state.coords] as BubbleCoords,
    polygon: state.polygon ? state.polygon.map((point) => [...point]) : [],
    position: { ...state.position },
    // 深拷贝颜色数组（如果存在）
    autoFgColor: state.autoFgColor ? [...state.autoFgColor] as [number, number, number] : null,
    autoBgColor: state.autoBgColor ? [...state.autoBgColor] as [number, number, number] : null
  }
}

/**
 * 验证气泡状态是否有效
 * @param state - 要验证的状态
 * @returns 是否有效
 */
export function isValidBubbleState(state: unknown): state is BubbleState {
  if (!state || typeof state !== 'object') {
    return false
  }

  const s = state as Record<string, unknown>

  // 检查必需的坐标字段
  if (!Array.isArray(s.coords) || s.coords.length !== 4) {
    return false
  }

  // 检查坐标值是否为数字
  if (!s.coords.every((v) => typeof v === 'number' && !isNaN(v))) {
    return false
  }

  // 检查字号是否为正数
  if (typeof s.fontSize !== 'number' || s.fontSize <= 0) {
    return false
  }

  // 检查文本方向是否有效
  const validDirections: TextDirection[] = ['vertical', 'horizontal', 'auto']
  if (
    typeof s.textDirection !== 'string' ||
    !validDirections.includes(s.textDirection as TextDirection)
  ) {
    return false
  }

  // 检查修复方式是否有效
  const validInpaintMethods: InpaintMethod[] = ['solid', 'lama_mpe', 'litelama']
  if (
    typeof s.inpaintMethod !== 'string' ||
    !validInpaintMethods.includes(s.inpaintMethod as InpaintMethod)
  ) {
    return false
  }

  return true
}

/**
 * 获取气泡的中心点坐标
 * @param state - 气泡状态
 * @returns 中心点坐标
 */
export function getBubbleCenter(state: BubbleState): { x: number; y: number } {
  const [x1, y1, x2, y2] = state.coords
  return {
    x: (x1 + x2) / 2,
    y: (y1 + y2) / 2
  }
}

/**
 * 获取气泡的宽高
 * @param state - 气泡状态
 * @returns 宽高对象
 */
export function getBubbleSize(state: BubbleState): { width: number; height: number } {
  const [x1, y1, x2, y2] = state.coords
  return {
    width: Math.abs(x2 - x1),
    height: Math.abs(y2 - y1)
  }
}

/**
 * 检查点是否在气泡矩形内
 * @param state - 气泡状态
 * @param x - 点的 x 坐标
 * @param y - 点的 y 坐标
 * @returns 是否在矩形内
 */
export function isPointInBubble(state: BubbleState, x: number, y: number): boolean {
  const [x1, y1, x2, y2] = state.coords
  const minX = Math.min(x1, x2)
  const maxX = Math.max(x1, x2)
  const minY = Math.min(y1, y2)
  const maxY = Math.max(y1, y2)
  return x >= minX && x <= maxX && y >= minY && y <= maxY
}

/**
 * 检查点是否在多边形内（射线法）
 * @param polygon - 多边形坐标数组
 * @param x - 点的 x 坐标
 * @param y - 点的 y 坐标
 * @returns 是否在多边形内
 */
export function isPointInPolygon(polygon: number[][], x: number, y: number): boolean {
  if (polygon.length < 3) {
    return false
  }

  let inside = false
  const n = polygon.length

  for (let i = 0, j = n - 1; i < n; j = i++) {
    const pointI = polygon[i]
    const pointJ = polygon[j]
    // 确保点坐标存在
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

/**
 * 检查点是否在气泡内（优先使用多边形，否则使用矩形）
 * @param state - 气泡状态
 * @param x - 点的 x 坐标
 * @param y - 点的 y 坐标
 * @returns 是否在气泡内
 */
export function isPointInBubbleArea(state: BubbleState, x: number, y: number): boolean {
  // 如果有多边形坐标，优先使用多边形检测
  if (state.polygon && state.polygon.length >= 3) {
    return isPointInPolygon(state.polygon, x, y)
  }
  // 否则使用矩形检测
  return isPointInBubble(state, x, y)
}

/**
 * 获取默认气泡设置（从全局 UI 设置读取）
 * 用于创建新气泡时应用当前的默认样式设置
 * @param globalSettings - 全局设置对象（可选，如果不传则使用默认值）
 * @returns 气泡默认设置
 */
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
}): BubbleStateOverrides {
  if (!globalSettings) {
    // 返回默认值
    return {
      fontSize: DEFAULT_BUBBLE_STATE.fontSize,
      fontFamily: DEFAULT_BUBBLE_STATE.fontFamily,
      textDirection: DEFAULT_BUBBLE_STATE.textDirection,
      textColor: DEFAULT_BUBBLE_STATE.textColor,
      fillColor: DEFAULT_BUBBLE_STATE.fillColor,
      strokeEnabled: DEFAULT_BUBBLE_STATE.strokeEnabled,
      strokeColor: DEFAULT_BUBBLE_STATE.strokeColor,
      strokeWidth: DEFAULT_BUBBLE_STATE.strokeWidth,
      inpaintMethod: DEFAULT_BUBBLE_STATE.inpaintMethod
    }
  }

  // 从全局设置构建默认值
  return {
    fontSize: globalSettings.fontSize ?? DEFAULT_BUBBLE_STATE.fontSize,
    fontFamily: globalSettings.fontFamily ?? DEFAULT_BUBBLE_STATE.fontFamily,
    textDirection: globalSettings.layoutDirection ?? DEFAULT_BUBBLE_STATE.textDirection,
    textColor: globalSettings.textColor ?? DEFAULT_BUBBLE_STATE.textColor,
    fillColor: globalSettings.fillColor ?? DEFAULT_BUBBLE_STATE.fillColor,
    strokeEnabled: globalSettings.strokeEnabled ?? DEFAULT_BUBBLE_STATE.strokeEnabled,
    strokeColor: globalSettings.strokeColor ?? DEFAULT_BUBBLE_STATE.strokeColor,
    strokeWidth: globalSettings.strokeWidth ?? DEFAULT_BUBBLE_STATE.strokeWidth,
    inpaintMethod: globalSettings.inpaintMethod ?? DEFAULT_BUBBLE_STATE.inpaintMethod
  }
}

/**
 * 初始化气泡状态数组
 * 如果已有保存的状态且数量匹配，直接使用；否则创建默认状态
 * @param savedStates - 已保存的气泡状态
 * @param coords - 气泡坐标数组
 * @param globalDefaults - 全局默认设置
 * @returns 初始化后的气泡状态数组
 */
export function initBubbleStates(
  savedStates: BubbleState[] | undefined,
  coords: BubbleCoords[] | undefined,
  globalDefaults?: BubbleGlobalDefaults
): BubbleState[] {
  // 如果有保存的状态且数量匹配坐标数量，直接使用
  if (savedStates && savedStates.length > 0) {
    if (!coords || savedStates.length === coords.length) {
      return cloneBubbleStates(savedStates)
    }
  }

  // 如果没有坐标，返回空数组（允许无气泡进入编辑模式）
  if (!coords || coords.length === 0) {
    return []
  }

  // 根据坐标创建新的状态
  return coords.map((coord) => {
    const autoDirection = detectTextDirection(coord)

    // 【简化设计】textDirection 直接使用具体方向值
    const globalTextDir = globalDefaults?.textDirection
    const textDirection: TextDirection =
      (globalTextDir === 'vertical' || globalTextDir === 'horizontal')
        ? globalTextDir
        : autoDirection

    return createBubbleState({
      coords: coord,
      ...globalDefaults,
      autoTextDirection: autoDirection,
      textDirection: textDirection,
    })
  })
}
