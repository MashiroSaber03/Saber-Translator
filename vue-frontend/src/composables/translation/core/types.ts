/**
 * 翻译管线核心类型定义
 * 
 * 统一所有翻译模式的类型系统
 */

import type { BubbleState, BubbleCoords, TextDirection } from '@/types/bubble'
import type { ImageData as AppImageData } from '@/types/image'

// ============================================================
// 翻译模式与范围
// ============================================================

/** 翻译模式 */
export type TranslationMode = 'standard' | 'hq' | 'proofread' | 'removeText'

/** 执行范围 */
export type ExecutionScope = 'current' | 'all' | 'failed'

/** 步骤类型 */
export type StepType = 'prepare' | 'translate' | 'render'

/** 翻译方法 */
export type TranslateMethod = 'backend' | 'multimodal' | 'proofread'

// ============================================================
// 进度管理
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
    /** 自定义进度标签 */
    label?: string
    /** 进度百分比（0-100） */
    percentage?: number
}

/** 进度报告器接口 */
export interface ProgressReporter {
    /** 初始化进度 */
    init(total: number, label?: string): void
    /** 更新进度 */
    update(current: number, label?: string): void
    /** 设置百分比 */
    setPercentage(percentage: number, label?: string): void
    /** 增加完成计数 */
    incrementCompleted(): void
    /** 增加失败计数 */
    incrementFailed(): void
    /** 完成进度 */
    finish(): void
    /** 获取当前进度 */
    getProgress(): TranslationProgress
}

// ============================================================
// 步骤配置
// ============================================================

/** 准备步骤选项 */
export interface PrepareStepOptions {
    /** 跳过翻译（仅检测/OCR/修复） */
    skipTranslation?: boolean
    /** 跳过渲染（仅获取干净背景） */
    skipRender?: boolean
}

/** 翻译步骤选项 */
export interface TranslateStepOptions {
    /** 翻译方法 */
    method: TranslateMethod
}

/** 渲染步骤选项 */
export interface RenderStepOptions {
    /** 使用个别样式 */
    useIndividualStyles?: boolean
}

/** 步骤配置 */
export interface StepConfig {
    type: StepType
    enabled: boolean
    options?: PrepareStepOptions | TranslateStepOptions | RenderStepOptions
}

// ============================================================
// 批量处理选项
// ============================================================

/** 批量处理选项 */
export interface BatchOptions {
    /** 批次大小（用于多模态翻译） */
    batchSize?: number
    /** 最大重试次数 */
    maxRetries?: number
    /** RPM 限制 */
    rpmLimit?: number
    /** 会话重置频率 */
    sessionResetFrequency?: number
}

// ============================================================
// 管线配置
// ============================================================

/** 管线配置 */
export interface PipelineConfig {
    /** 翻译模式 */
    mode: TranslationMode
    /** 执行范围 */
    scope: ExecutionScope
    /** 步骤序列 */
    steps: StepConfig[]
    /** 批量处理选项 */
    batchOptions?: BatchOptions
}

// ============================================================
// 执行上下文
// ============================================================

/** 单张图片执行上下文 */
export interface ImageExecutionContext {
    /** 图片索引 */
    imageIndex: number
    /** 图片数据 */
    image: AppImageData
    /** 管线配置 */
    config: PipelineConfig
    /** 进度报告器 */
    progress: ProgressReporter
}

/** 批量执行上下文 */
export interface BatchExecutionContext {
    /** 所有图片数据 */
    images: AppImageData[]
    /** 管线配置 */
    config: PipelineConfig
    /** 进度报告器 */
    progress: ProgressReporter
    /** 会话 ID */
    sessionId: string
}

// ============================================================
// 步骤结果
// ============================================================

/** 准备步骤结果 */
export interface PrepareStepResult {
    success: boolean
    /** 翻译后的图片（Base64） */
    translatedImage?: string
    /** 干净背景图片（Base64） */
    cleanImage?: string
    /** 气泡坐标 */
    bubbleCoords?: BubbleCoords[]
    /** 气泡角度 */
    bubbleAngles?: number[]
    /** 原文文本 */
    originalTexts?: string[]
    /** 翻译文本 */
    bubbleTexts?: string[]
    /** 文本框文本 */
    textboxTexts?: string[]
    /** 气泡状态 */
    bubbleStates?: BubbleState[]
    /** 自动排版方向 */
    autoDirections?: TextDirection[]
    /** 错误信息 */
    error?: string
}

/** 翻译数据（JSON格式） */
export interface TranslationJsonData {
    imageIndex: number
    bubbles: Array<{
        bubbleIndex: number
        original: string
        translated: string
        textDirection: string
    }>
}

/** 翻译步骤结果 */
export interface TranslateStepResult {
    success: boolean
    /** 翻译结果数据 */
    translationData?: TranslationJsonData[]
    /** 错误信息 */
    error?: string
}

/** 渲染步骤结果 */
export interface RenderStepResult {
    success: boolean
    /** 渲染后的图片（Base64） */
    renderedImage?: string
    /** 更新后的气泡状态 */
    bubbleStates?: BubbleState[]
    /** 错误信息 */
    error?: string
}

/** 步骤结果联合类型 */
export type StepResult = PrepareStepResult | TranslateStepResult | RenderStepResult

// ============================================================
// 管线执行结果
// ============================================================

/** 管线执行结果 */
export interface PipelineResult {
    success: boolean
    /** 完成的图片数量 */
    completed: number
    /** 失败的图片数量 */
    failed: number
    /** 错误信息 */
    errors?: string[]
}

// ============================================================
// 步骤执行器接口
// ============================================================

/** 步骤选项联合类型 */
export type StepOptions = PrepareStepOptions | TranslateStepOptions | RenderStepOptions

/** 步骤执行器接口 */
export interface StepExecutor<T extends StepResult = StepResult> {
    /** 步骤名称 */
    name: string
    /** 步骤类型 */
    type: StepType
    /** 执行步骤 */
    execute(context: ImageExecutionContext, options?: StepOptions): Promise<T>
}

// ============================================================
// 保存的样式设置（高质量翻译使用）
// ============================================================

/** 保存的文本样式设置 */
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

// ============================================================
// 已有气泡数据
// ============================================================

/** 已有气泡数据提取结果 */
export interface ExistingBubbleData {
    coords: BubbleCoords[]
    angles: number[] | undefined
    isEmpty: boolean
    isManual: boolean
}

// ============================================================
// 翻译选项（用于 buildTranslateParams）
// ============================================================

/** 翻译选项 */
export interface TranslationOptions {
    /** 仅消除文字（不翻译） */
    removeTextOnly?: boolean
    /** 已有气泡坐标 */
    existingBubbleCoords?: BubbleCoords[]
    /** 已有气泡角度 */
    existingBubbleAngles?: number[]
    /** 已有气泡状态 */
    existingBubbleStates?: BubbleState[]
    /** 使用已有气泡 */
    useExistingBubbles?: boolean
}
