/**
 * 仅嵌字模式配置
 *
 * 使用已有文本与气泡状态进行修复+渲染，不调用翻译 API。
 */

import type { PipelineConfig, ExecutionScope, PageRange } from '../core/types'

/** 仅嵌字模式配置选项 */
export interface EmbedModeOptions {
    /** 页面范围（仅当 scope = 'range' 时使用） */
    pageRange?: PageRange
}

/**
 * 获取仅嵌字模式配置
 * @param scope 执行范围：'current' | 'all' | 'range'
 * @param options 可选配置
 */
export function getEmbedModeConfig(
    scope: ExecutionScope = 'current',
    options?: EmbedModeOptions
): PipelineConfig {
    return {
        mode: 'embed',
        scope,
        pageRange: options?.pageRange
    }
}

