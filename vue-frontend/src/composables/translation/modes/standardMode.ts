/**
 * 标准翻译模式配置
 * 
 * 步骤链由 SequentialPipeline.ts 中的 STEP_CHAIN_CONFIGS 定义
 */

import type { PipelineConfig, ExecutionScope, PageRange } from '../core/types'

/** 标准模式配置选项 */
export interface StandardModeOptions {
    /** 页面范围（仅当 scope = 'range' 时使用） */
    pageRange?: PageRange
}

/**
 * 获取标准翻译模式配置
 * @param scope 执行范围：'current' | 'all' | 'failed' | 'range'
 * @param options 可选配置
 */
export function getStandardModeConfig(
    scope: ExecutionScope = 'current',
    options?: StandardModeOptions
): PipelineConfig {
    return {
        mode: 'standard',
        scope,
        pageRange: options?.pageRange
    }
}
