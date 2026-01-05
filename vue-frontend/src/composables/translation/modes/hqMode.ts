/**
 * 高质量翻译模式配置
 * 
 * 使用多模态 AI（如 GPT-4V、Gemini）进行批量上下文翻译
 * 
 * 流程：
 * 1. 消除所有图片文字（获取干净背景和原文）
 * 2. 批量发送给多模态 AI 进行翻译
 * 3. 渲染翻译结果
 */

import type { PipelineConfig } from '../core/types'

/**
 * 高质量翻译模式配置
 */
export const hqModeConfig: PipelineConfig = {
    mode: 'hq',
    scope: 'all',  // 高质量翻译必须处理所有图片
    steps: [
        {
            type: 'prepare',
            enabled: true,
            options: {
                skipTranslation: true  // 只检测/OCR/修复，不翻译
            }
        },
        {
            type: 'translate',
            enabled: true,
            options: {
                method: 'multimodal'  // 使用多模态 AI 翻译
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
 * 获取高质量翻译模式配置
 * @param options 可选配置覆盖
 */
export function getHqModeConfig(options?: {
    batchSize?: number
    maxRetries?: number
    rpmLimit?: number
    sessionResetFrequency?: number
}): PipelineConfig {
    return {
        ...hqModeConfig,
        batchOptions: {
            ...hqModeConfig.batchOptions,
            ...options
        }
    }
}
