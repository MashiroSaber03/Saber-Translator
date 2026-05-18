/**
 * 仅消除文字模式配置
 * 
 * 步骤链由 core/pipelineRegistry.ts 中的统一注册表定义
 */

import type { PipelineConfig, ExecutionScope, PageSelection } from '../core/types'

/** 消除文字模式配置选项 */
export interface RemoveTextModeOptions {
    /** 页面选择（仅当 scope = 'selection' 时使用） */
    pageSelection?: PageSelection
}

/**
 * 获取仅消除文字模式配置
 * @param scope 执行范围：'current' | 'all' | 'selection'
 * @param options 可选配置
 */
export function getRemoveTextModeConfig(
    scope: ExecutionScope = 'current',
    options?: RemoveTextModeOptions
): PipelineConfig {
    return {
        mode: 'removeText',
        scope,
        pageSelection: options?.pageSelection
    }
}
