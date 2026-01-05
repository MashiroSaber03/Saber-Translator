/**
 * AI 校对模式配置
 * 
 * 使用多模态 AI 对已翻译的文本进行校对
 * 
 * 流程：
 * 1. 导出已翻译文本
 * 2. 发送给 AI 进行校对（支持双轮校对）
 * 3. 渲染校对后的结果
 */

import type { PipelineConfig } from '../core/types'

/**
 * AI 校对模式配置
 */
export const proofreadModeConfig: PipelineConfig = {
    mode: 'proofread',
    scope: 'all',  // 校对必须处理所有图片
    steps: [
        // 无需 prepare 步骤，因为图片已经翻译过了
        {
            type: 'translate',
            enabled: true,
            options: {
                method: 'proofread'  // 使用校对模式
            }
        },
        {
            type: 'render',
            enabled: true,
            options: {
                useIndividualStyles: true
            }
        }
    ],
    batchOptions: {
        batchSize: 3,
        maxRetries: 2,
        rpmLimit: 10,
        sessionResetFrequency: 5
    }
}

/**
 * 获取 AI 校对模式配置
 * @param options 可选配置覆盖
 */
export function getProofreadModeConfig(options?: {
    batchSize?: number
    maxRetries?: number
    rpmLimit?: number
    sessionResetFrequency?: number
}): PipelineConfig {
    return {
        ...proofreadModeConfig,
        batchOptions: {
            ...proofreadModeConfig.batchOptions,
            ...options
        }
    }
}
