/**
 * 气泡状态类型定义
 * 与后端 BubbleState 数据类对应，用于前后端统一管理气泡渲染参数
 */

/**
 * 气泡坐标类型 [x1, y1, x2, y2]
 */
export type BubbleCoords = [number, number, number, number]

/**
 * 多边形坐标类型 [[x1, y1], [x2, y2], ...]
 */
export type PolygonCoords = number[][]

/**
 * 文本排版方向
 */
export type TextDirection = 'vertical' | 'horizontal' | 'auto'

/**
 * 修复方式
 */
export type InpaintMethod = 'solid' | 'lama_mpe' | 'litelama'

/**
 * 气泡位置
 */
export interface BubblePosition {
  x: number
  y: number
}

/**
 * 气泡状态接口
 * 包含气泡的所有渲染参数
 */
export interface BubbleState {
  // 文本内容
  /** 原文文本 */
  originalText: string
  /** 翻译后文本 */
  translatedText: string
  /** 文本框提示词处理后的文本 */
  textboxText: string

  // 坐标信息
  /** 矩形坐标 [x1, y1, x2, y2] */
  coords: BubbleCoords
  /** 多边形坐标（可选，用于非矩形气泡） */
  polygon: PolygonCoords

  // 渲染参数
  /** 字号 */
  fontSize: number
  /** 字体路径 */
  fontFamily: string
  /** 用户设置的排版方向 */
  textDirection: TextDirection
  /** 自动检测的排版方向（根据宽高比） */
  autoTextDirection: TextDirection
  /** 文字颜色 */
  textColor: string
  /** 填充颜色 */
  fillColor: string
  /** 旋转角度（度） */
  rotationAngle: number
  /** 位置偏移 */
  position: BubblePosition

  // 描边参数
  /** 是否启用描边 */
  strokeEnabled: boolean
  /** 描边颜色 */
  strokeColor: string
  /** 描边宽度 */
  strokeWidth: number

  // 修复参数
  /** 修复方式 */
  inpaintMethod: InpaintMethod
}

/**
 * 创建气泡状态时的可选参数
 */
export type BubbleStateOverrides = Partial<BubbleState>

/**
 * 气泡状态更新参数
 */
export type BubbleStateUpdates = Partial<BubbleState>

/**
 * 后端 API 响应中的气泡数据
 */
export interface BubbleApiResponse {
  bubble_coords?: BubbleCoords[]
  bubble_states?: BubbleState[]
  original_texts?: string[]
  bubble_texts?: string[]
  textbox_texts?: string[]
  bubble_angles?: number[]
  /** 自动检测的排版方向数组（'v' 表示垂直，'h' 表示水平）- 后端基于文本行分析 */
  auto_directions?: ('v' | 'h')[]
}

/**
 * 全局默认设置
 */
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
}

// ============================================================
// 工具函数
// ============================================================

/**
 * 获取气泡的有效渲染方向
 * 【原版设计】渲染时直接使用气泡的 textDirection 字段
 * - 翻译时：后端会根据自动检测结果同时设置 textDirection 和 autoTextDirection
 * - 用户修改时：只修改 textDirection
 * - 渲染时：直接用 textDirection，不需要复杂判断
 * 
 * @param bubble - 气泡状态（只需包含 textDirection, autoTextDirection, coords）
 * @returns 有效的渲染方向 'vertical' | 'horizontal'
 */
export function getEffectiveDirection(
  bubble: Pick<BubbleState, 'textDirection' | 'autoTextDirection' | 'coords'>
): 'vertical' | 'horizontal' {
  // 直接使用气泡自己的 textDirection
  if (bubble.textDirection && bubble.textDirection !== 'auto') {
    return bubble.textDirection as 'vertical' | 'horizontal'
  }
  // 回退：使用 autoTextDirection（后端检测结果）
  if (bubble.autoTextDirection && bubble.autoTextDirection !== 'auto') {
    return bubble.autoTextDirection as 'vertical' | 'horizontal'
  }
  // 最后回退：根据宽高比判断
  if (bubble.coords) {
    const [x1, y1, x2, y2] = bubble.coords
    return (y2 - y1) > (x2 - x1) ? 'vertical' : 'horizontal'
  }
  return 'vertical'
}
