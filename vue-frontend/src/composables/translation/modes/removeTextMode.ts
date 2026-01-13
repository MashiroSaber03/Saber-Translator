/**
 * 仅消除文字模式配置
 * 
 * 步骤链由 SequentialPipeline.ts 中的 STEP_CHAIN_CONFIGS 定义
 */

import type { PipelineConfig, ExecutionScope, PageRange } from '../core/types'

/** 消除文字模式配置选项 */
export interface RemoveTextModeOptions {
    /** 页面范围（仅当 scope = 'range' 时使用） */
    pageRange?: PageRange
}

/**
 * 获取仅消除文字模式配置
 * @param scope 执行范围：'current' | 'all' | 'range'
 * @param options 可选配置
 */
export function getRemoveTextModeConfig(
    scope: ExecutionScope = 'current',
    options?: RemoveTextModeOptions
): PipelineConfig {
    return {
        mode: 'removeText',
        scope,
        pageRange: options?.pageRange
    }
}
