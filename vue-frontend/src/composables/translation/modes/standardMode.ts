/**
 * 标准翻译模式配置
 * 
 * 步骤链由 SequentialPipeline.ts 中的 STEP_CHAIN_CONFIGS 定义
 */

import type { PipelineConfig } from '../core/types'

/**
 * 获取标准翻译模式配置
 * @param scope 执行范围：'current' | 'all' | 'failed'
 */
export function getStandardModeConfig(scope: 'current' | 'all' | 'failed' = 'current'): PipelineConfig {
    return {
        mode: 'standard',
        scope
    }
}
