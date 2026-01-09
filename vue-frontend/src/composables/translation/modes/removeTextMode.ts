/**
 * 仅消除文字模式配置
 * 
 * 步骤链由 SequentialPipeline.ts 中的 STEP_CHAIN_CONFIGS 定义
 */

import type { PipelineConfig } from '../core/types'

/**
 * 获取仅消除文字模式配置
 * @param scope 执行范围：'current' | 'all'
 */
export function getRemoveTextModeConfig(scope: 'current' | 'all' = 'current'): PipelineConfig {
    return {
        mode: 'removeText',
        scope
    }
}
