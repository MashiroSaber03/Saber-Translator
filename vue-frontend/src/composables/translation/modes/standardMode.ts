/**
 * 标准翻译模式配置
 * 
 * 步骤链由 core/pipelineRegistry.ts 中的统一注册表定义
 */

import type { PipelineConfig, ExecutionScope, PageSelection } from '../core/types'

/** 标准模式配置选项 */
export interface StandardModeOptions {
    /** 页面选择（仅当 scope = 'selection' 时使用） */
    pageSelection?: PageSelection
}

/**
 * 获取标准翻译模式配置
 * @param scope 执行范围：'current' | 'all' | 'failed' | 'selection'
 * @param options 可选配置
 */
export function getStandardModeConfig(
    scope: ExecutionScope = 'current',
    options?: StandardModeOptions
): PipelineConfig {
    return {
        mode: 'standard',
        scope,
        pageSelection: options?.pageSelection
    }
}
