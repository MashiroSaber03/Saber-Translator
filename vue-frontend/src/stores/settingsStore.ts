/**
 * 设置状态管理 Store（兼容层）
 * 
 * 此文件现在作为兼容层，从模块化的 settings 目录重新导出
 * 实际实现已迁移到 ./settings/ 目录
 * 
 * 模块结构：
 * - settings/types.ts: 类型定义
 * - settings/defaults.ts: 默认值
 * - settings/modules/ocr.ts: OCR识别设置
 * - settings/modules/translation.ts: 翻译服务设置
 * - settings/modules/detection.ts: 检测设置
 * - settings/modules/hqTranslation.ts: 高质量翻译设置
 * - settings/modules/proofreading.ts: AI校对设置
 * - settings/modules/prompts.ts: 提示词管理
 * - settings/modules/misc.ts: 更多设置
 * - settings/index.ts: 主入口
 */

// 从模块化目录重新导出
export { useSettingsStore } from './settings'
export type { ProviderConfigsCache } from './settings'
