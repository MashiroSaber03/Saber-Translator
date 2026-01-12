/**
 * 翻译管线核心类型定义
 */

// ============================================================
// 翻译模式与范围
// ============================================================

/** 翻译模式 */
export type TranslationMode = 'standard' | 'hq' | 'proofread' | 'removeText'

/** 执行范围 */
export type ExecutionScope = 'current' | 'all' | 'failed'

// ============================================================
// 进度管理
// ============================================================

/** 翻译进度信息 */
export interface TranslationProgress {
    current: number
    total: number
    completed: number
    failed: number
    isInProgress: boolean
    label?: string
    percentage?: number
}

/** 进度报告器接口 */
export interface ProgressReporter {
    init(total: number, label?: string): void
    update(current: number, label?: string): void
    setPercentage(percentage: number, label?: string): void
    incrementCompleted(): void
    incrementFailed(): void
    finish(): void
    getProgress(): TranslationProgress
}

// ============================================================
// 批量处理选项
// ============================================================

/** 批量处理选项 */
export interface BatchOptions {
    batchSize?: number
    maxRetries?: number
    rpmLimit?: number
    sessionResetFrequency?: number
}

// ============================================================
// 管线配置
// ============================================================

/** 管线配置 */
export interface PipelineConfig {
    mode: TranslationMode
    scope: ExecutionScope
    batchOptions?: BatchOptions
}

// ============================================================
// 管线执行结果
// ============================================================

/** 管线执行结果 */
export interface PipelineResult {
    success: boolean
    completed: number
    failed: number
    errors?: string[]
}

// ============================================================
// 保存的样式设置
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
    useAutoTextColor: boolean
}
