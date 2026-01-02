/**
 * 翻译功能模块索引
 * 
 * 拆分后的模块结构：
 * - types.ts: 共享类型定义
 * - utils.ts: 共享工具函数
 * - useHqTranslation.ts: 高质量翻译
 * - useProofreading.ts: AI校对
 * 
 * 主入口 useTranslation 保留核心翻译功能（单张、批量、消除文字等）
 */

// 类型导出
export * from './types'

// 工具函数导出
export {
  extractBase64FromDataUrl,
  extractExistingBubbleData,
  buildTranslateParams,
  createBubbleStatesFromApiResponse,
  generateSessionId,
  filterJsonForBatch,
  mergeJsonResults
} from './utils'

// 组合式函数导出
export { useHqTranslation } from './useHqTranslation'
export { useProofreading } from './useProofreading'
