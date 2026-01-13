/**
 * API 服务层索引文件
 * 统一导出所有 API 服务
 */

// HTTP 客户端
export { apiClient, type ApiError, type ApiResponse } from './client'

// 系统级 API
export * from './system'

// 书架 API
export * from './bookshelf'

// 翻译 API
export * from './translate'

// 会话 API
export * from './session'

// 单页存储 API（新版）- 使用重命名导出避免冲突
export {
    saveSessionMeta,
    loadSessionMeta,
    savePageImage,
    getPageImageUrl,
    savePageMeta,
    loadPageMeta,
    presaveAllPages,
    saveTranslatedPage,
    loadSession as loadSessionV2,  // 重命名避免与 session.ts 冲突
    extractBase64,
    saveAllPagesSequentially,
    type PageData,
    type PageInfo,
    type SessionData as SessionDataV2,
    type ImageDataForSave,
    type SaveProgressCallback
} from './pageStorage'

// 漫画分析 API
export * from './insight'

// 插件 API
export * from './plugin'

// 配置 API
export * from './config'
