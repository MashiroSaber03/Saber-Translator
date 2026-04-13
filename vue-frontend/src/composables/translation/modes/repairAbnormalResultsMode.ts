/**
 * 复查异常结果并重翻模式配置
 */

import type { PipelineConfig, ExecutionScope, PageRange } from '../core/types'

export interface RepairAbnormalResultsModeOptions {
    pageRange?: PageRange
}

export function getRepairAbnormalResultsModeConfig(
    scope: ExecutionScope = 'current',
    options?: RepairAbnormalResultsModeOptions
): PipelineConfig {
    return {
        mode: 'repairAbnormalResults',
        scope,
        pageRange: options?.pageRange
    }
}
