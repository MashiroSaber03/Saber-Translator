/**
 * AI 校对模式配置
 * 
 * 步骤链由 SequentialPipeline.ts 中的 STEP_CHAIN_CONFIGS 定义
 */

import type { PipelineConfig } from '../core/types'

/**
 * 获取 AI 校对模式配置
 * @param options 可选的批量处理选项覆盖
 */
export function getProofreadModeConfig(options?: {
    batchSize?: number
    maxRetries?: number
    rpmLimit?: number
    sessionResetFrequency?: number
}): PipelineConfig {
    return {
        mode: 'proofread',
        scope: 'all',
        batchOptions: {
            batchSize: options?.batchSize ?? 3,
            maxRetries: options?.maxRetries ?? 2,
            rpmLimit: options?.rpmLimit ?? 10,
            sessionResetFrequency: options?.sessionResetFrequency ?? 5
        }
    }
}
