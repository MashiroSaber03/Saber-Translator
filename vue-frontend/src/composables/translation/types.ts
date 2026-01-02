/**
 * 翻译功能共享类型定义
 */

import type { BubbleState, BubbleCoords } from '@/types/bubble'

// ============================================================
// 翻译进度类型
// ============================================================

/** 翻译进度信息 */
export interface TranslationProgress {
  /** 当前处理的图片索引 */
  current: number
  /** 总图片数 */
  total: number
  /** 已完成数量 */
  completed: number
  /** 失败数量 */
  failed: number
  /** 是否正在进行 */
  isInProgress: boolean
  /** 是否暂停 */
  isPaused: boolean
  /** 自定义进度标签（复刻原版） */
  label?: string
  /** 进度百分比（0-100，用于精确控制进度条） */
  percentage?: number
}

/** 翻译选项 */
export interface TranslationOptions {
  /** 仅消除文字，不翻译 */
  removeTextOnly?: boolean
  /** 使用已有气泡框 */
  useExistingBubbles?: boolean
  /**
   * 已有气泡状态数组（推荐使用）
   * 从中自动提取坐标和角度，避免遗漏参数
   */
  existingBubbleStates?: BubbleState[]
  /** 已有气泡坐标（兼容旧接口，优先使用 existingBubbleStates） */
  existingBubbleCoords?: BubbleCoords[]
  /** 已有气泡角度（兼容旧接口，优先使用 existingBubbleStates） */
  existingBubbleAngles?: number[]
}

// ============================================================
// 高质量翻译类型
// ============================================================

/** 保存翻译前的文本样式设置（复刻原版） */
export interface SavedTextStyles {
  fontFamily: string
  fontSize: number
  autoFontSize: boolean
  textDirection: string
  autoTextDirection: boolean
  fillColor: string
  textColor: string
  rotationAngle: number
  strokeEnabled: boolean
  strokeColor: string
  strokeWidth: number
}

/** 翻译/校对JSON数据格式（高质量翻译和AI校对共用） */
export interface TranslationJsonData {
  imageIndex: number
  bubbles: Array<{
    bubbleIndex: number
    original: string
    translated: string
    textDirection: string
  }>
}

/** 高质量翻译JSON数据（别名，保持兼容） */
export type HqJsonData = TranslationJsonData

/** 校对JSON数据（别名，保持兼容） */
export type ProofreadingJsonData = TranslationJsonData

// ============================================================
// 已有气泡数据提取结果
// ============================================================

/** 已有气泡数据提取结果 */
export interface ExistingBubbleData {
  coords: BubbleCoords[]
  angles: number[] | undefined
  isEmpty: boolean
  isManual: boolean
}
