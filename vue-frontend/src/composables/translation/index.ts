/**
 * 翻译功能模块索引
 * 
 * 模块结构：
 * - core/: 核心类型定义、进度管理和管线执行引擎
 * - steps/: 独立步骤执行器（准备、翻译、渲染）
 * - modes/: 模式配置（标准、高质量、校对、消除文字）
 * - utils.ts: 共享工具函数
 * 
 * 使用方式：
 * ```typescript
 * import { usePipeline, getStandardModeConfig, getHqModeConfig } from '@/composables/translation'
 * 
 * const { execute, progress } = usePipeline()
 * 
 * // 翻译当前图片
 * await execute(getStandardModeConfig('current'))
 * 
 * // 翻译所有图片
 * await execute(getStandardModeConfig('all'))
 * 
 * // 高质量翻译
 * await execute(getHqModeConfig())
 * ```
 */

// ============================================================
// 核心模块导出
// ============================================================
export * from './core'

// ============================================================
// 步骤模块导出
// ============================================================
export * from './steps'

// ============================================================
// 模式配置导出
// ============================================================
export * from './modes'

// ============================================================
// 工具函数导出
// ============================================================
export {
  extractBase64FromDataUrl,
  extractExistingBubbleData,
  buildTranslateParams,
  createBubbleStatesFromApiResponse,
  generateSessionId,
  filterJsonForBatch,
  mergeJsonResults
} from './utils'
