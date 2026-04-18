/**
 * 全部重渲染模式配置
 */

import type { PipelineConfig, ExecutionScope, PageRange } from '../core/types'

export interface RerenderModeOptions {
    pageRange?: PageRange
}

export function getRerenderModeConfig(
    scope: ExecutionScope = 'current',
    options?: RerenderModeOptions
): PipelineConfig {
    return {
        mode: 'rerender',
        scope,
        pageRange: options?.pageRange
    }
}
