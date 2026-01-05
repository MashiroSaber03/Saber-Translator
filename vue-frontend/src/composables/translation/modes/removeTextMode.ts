/**
 * 仅消除文字模式配置
 * 
 * 只进行检测、OCR 和背景修复，不翻译也不渲染
 */

import type { PipelineConfig } from '../core/types'

/**
 * 仅消除文字模式配置（单张）
 */
export const removeTextModeConfig: PipelineConfig = {
    mode: 'removeText',
    scope: 'current',
    steps: [
        {
            type: 'prepare',
            enabled: true,
            options: {
                skipTranslation: true,  // 跳过翻译
                skipRender: true        // 跳过渲染
            }
        }
        // 无需 translate 和 render 步骤
    ]
}

/**
 * 仅消除文字模式配置（批量）
 */
export const removeTextBatchModeConfig: PipelineConfig = {
    ...removeTextModeConfig,
    scope: 'all'
}

/**
 * 获取仅消除文字模式配置
 * @param scope 执行范围
 */
export function getRemoveTextModeConfig(scope: 'current' | 'all' = 'current'): PipelineConfig {
    return {
        ...removeTextModeConfig,
        scope
    }
}
