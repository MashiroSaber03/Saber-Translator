/**
 * 高质量翻译模式配置
 * 
 * 步骤链由 core/pipelineRegistry.ts 中的统一注册表定义
 */

import type { PipelineConfig, ExecutionScope, PageSelection } from '../core/types'

/** 高质量模式配置选项 */
export interface HqModeOptions {
    batchSize?: number
    maxRetries?: number
    rpmLimit?: number
    /** 页面选择（仅当 scope = 'selection' 时使用） */
    pageSelection?: PageSelection
}

/**
 * 获取高质量翻译模式配置
 * @param scope 执行范围：'all' | 'selection'（高质量翻译默认处理多张图片）
 * @param options 可选的批量处理选项覆盖
 */
export function getHqModeConfig(
    scope: ExecutionScope = 'all',
    options?: HqModeOptions
): PipelineConfig {
    return {
        mode: 'hq',
        scope,
        pageSelection: options?.pageSelection,
        batchOptions: {
            batchSize: options?.batchSize ?? 3,
            maxRetries: options?.maxRetries ?? 2,
            rpmLimit: options?.rpmLimit ?? 10
        }
    }
}
