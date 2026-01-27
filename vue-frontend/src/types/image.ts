/**
 * 图片数据类型定义
 * 定义翻译页面中图片的数据结构
 */

import type { BubbleState, BubbleCoords, TextDirection, InpaintMethod } from './bubble'

/**
 * 翻译状态
 */
export type TranslationStatus = 'pending' | 'processing' | 'completed' | 'failed'

/**
 * 图片数据接口
 * 包含图片的所有状态信息
 */
export interface ImageData {
  /** 唯一标识符 */
  id: string
  /** 文件名 */
  fileName: string

  // 图片尺寸
  /** 图片宽度 */
  width?: number
  /** 图片高度 */
  height?: number

  // 图片数据（Base64）
  /** 原始图片数据 */
  originalDataURL: string
  /** 翻译后图片数据 */
  translatedDataURL: string | null
  /** 干净背景图片数据（用于笔刷修复） */
  cleanImageData: string | null

  // 气泡状态
  /** 气泡状态数组 */
  bubbleStates: BubbleState[] | null
  /** 气泡坐标（兼容旧数据） */
  bubbleCoords?: BubbleCoords[]
  /** 气泡角度（兼容旧数据） */
  bubbleAngles?: number[]
  /** 原文文本数组（兼容旧数据） */
  originalTexts?: string[]
  /** 译文文本数组（兼容旧数据） */
  bubbleTexts?: string[]
  /** 文本框文本数组（兼容旧数据） */
  textboxTexts?: string[]

  // 双掩膜系统
  /** 文字检测掩膜 - Default检测器生成的精确文字区域（Base64格式） */
  textMask?: string | null
  /** 用户笔刷掩膜 - 记录用户手动修改意图（Base64格式）
   * 像素值：白色(255)=消除区域，黑色(0)=还原区域，灰色(127)=未修改 */
  userMask?: string | null

  // 手动标注标记
  /** 是否经过手动标注（用户在编辑模式中操作过） */
  isManuallyAnnotated?: boolean

  // 翻译状态
  /** 翻译状态 */
  translationStatus: TranslationStatus
  /** 翻译是否失败 */
  translationFailed: boolean
  /** 错误信息 */
  errorMessage?: string

  // 图片级别设置
  /** 字号 */
  fontSize: number
  /** 是否自动字号 */
  autoFontSize: boolean
  /** 字体 */
  fontFamily: string
  /** 排版方向（用户选择，包括 'auto'） */
  layoutDirection: TextDirection
  /** 文字颜色 */
  textColor: string
  /** 填充颜色 */
  fillColor: string
  /** 修复方式 */
  inpaintMethod: InpaintMethod
  /** 是否启用描边 */
  strokeEnabled: boolean
  /** 描边颜色 */
  strokeColor: string
  /** 描边宽度 */
  strokeWidth: number
  /** 是否使用自动检测的文字颜色 */
  useAutoTextColor?: boolean

  // 元数据
  /** 是否有未保存的更改 */
  hasUnsavedChanges: boolean
  /** 是否为手动标注模式 */
  isManualAnnotation?: boolean
  /** 【修复6】是否显示原图（按图片持久化，切换图片时保留状态） */
  showOriginal?: boolean

  // 文件夹导入信息
  /** 文件的原始路径（用于文件夹分组） */
  relativePath?: string
  /** 所属文件夹路径 */
  folderPath?: string
}


/**
 * 创建图片数据时的可选参数
 */
export type ImageDataOverrides = Partial<ImageData>

/**
 * 图片数据更新参数
 */
export type ImageDataUpdates = Partial<ImageData>

/**
 * 图片上传结果
 */
export interface ImageUploadResult {
  success: boolean
  images: ImageData[]
  errors?: string[]
}

/**
 * PDF 解析会话
 */
export interface PdfParseSession {
  sessionId: string
  totalPages: number
  currentPage: number
}

/**
 * MOBI 解析会话
 */
export interface MobiParseSession {
  sessionId: string
  totalPages: number
  currentPage: number
}
