/**
 * 高质量翻译模式配置
 * 
 * 步骤链由 SequentialPipeline.ts 中的 STEP_CHAIN_CONFIGS 定义
 */

import type { PipelineConfig, ExecutionScope, PageRange } from '../core/types'

/** 高质量模式配置选项 */
export interface HqModeOptions {
    batchSize?: number
    maxRetries?: number
    rpmLimit?: number
    sessionResetFrequency?: number
    /** 页面范围（仅当 scope = 'range' 时使用） */
    pageRange?: PageRange
}

/**
 * 获取高质量翻译模式配置
 * @param scope 执行范围：'all' | 'range'（高质量翻译默认处理多张图片）
 * @param options 可选的批量处理选项覆盖
 */
export function getHqModeConfig(
    scope: ExecutionScope = 'all',
    options?: HqModeOptions
): PipelineConfig {
    return {
        mode: 'hq',
        scope,
        pageRange: options?.pageRange,
        batchOptions: {
            batchSize: options?.batchSize ?? 3,
            maxRetries: options?.maxRetries ?? 2,
            rpmLimit: options?.rpmLimit ?? 10,
            sessionResetFrequency: options?.sessionResetFrequency ?? 5
        }
    }
}
