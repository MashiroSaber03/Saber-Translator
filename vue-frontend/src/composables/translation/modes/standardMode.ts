/**
 * 标准翻译模式配置
 * 
 * 使用后端 API 一体化完成全部流程（检测/OCR/翻译/修复/渲染）
 */

import type { PipelineConfig } from '../core/types'

/**
 * 标准翻译模式配置（单张）
 */
export const standardModeConfig: PipelineConfig = {
    mode: 'standard',
    scope: 'current',
    steps: [
        {
            type: 'prepare',
            enabled: true,
            options: {
                skipTranslation: false  // 后端 API 一次性完成全部流程
            }
        }
        // 无需额外的 translate 和 render 步骤
    ]
}

/**
 * 标准翻译模式配置（批量）
 */
export const standardBatchModeConfig: PipelineConfig = {
    ...standardModeConfig,
    scope: 'all'
}

/**
 * 获取标准翻译模式配置
 * @param scope 执行范围
 */
export function getStandardModeConfig(scope: 'current' | 'all' | 'failed' = 'current'): PipelineConfig {
    return {
        ...standardModeConfig,
        scope
    }
}
