/**
 * 网页导入相关类型定义
 */

import type { TranslationProvider } from './settings'

/** 图片预处理设置 */
export interface ImagePreprocessSettings {
    enabled: boolean
    autoRotate: boolean
    compression: {
        enabled: boolean
        quality: number
        maxWidth: number
        maxHeight: number
    }
    formatConvert: {
        enabled: boolean
        targetFormat: 'jpeg' | 'png' | 'webp' | 'original'
    }
}

/** 网页导入完整设置 */
export interface WebImportSettings {
    /** Firecrawl 配置 */
    firecrawl: {
        apiKey: string
    }

    /** AI Agent 配置 */
    agent: {
        provider: TranslationProvider
        apiKey: string
        customBaseUrl: string
        modelName: string
        useStream: boolean
        forceJsonOutput: boolean
        maxRetries: number
        timeout: number
    }

    /** 提取设置 */
    extraction: {
        prompt: string
        maxIterations: number
    }

    /** 下载设置 */
    download: {
        concurrency: number
        timeout: number
        retries: number
        delay: number
        useReferer: boolean
    }

    /** 图片预处理 */
    imagePreprocess: ImagePreprocessSettings

    /** 高级设置 */
    advanced: {
        customCookie: string
        customHeaders: string
        bypassProxy: boolean
    }

    /** 界面设置 */
    ui: {
        showAgentLogs: boolean
        autoImport: boolean
    }
}

/** 漫画页面 */
export interface ComicPage {
    pageNumber: number
    imageUrl: string
}

/** 提取结果 */
export interface ExtractResult {
    success: boolean
    comicTitle: string
    chapterTitle: string
    pages: ComicPage[]
    totalPages: number
    sourceUrl: string
    error?: string
}

/** 下载的图片 */
export interface DownloadedImage {
    index: number
    filename: string
    dataUrl: string
    size: number
}

/** 下载结果 */
export interface DownloadResult {
    success: boolean
    images: DownloadedImage[]
    failedCount: number
}

/** Agent 日志 */
export interface AgentLog {
    timestamp: string
    type: 'info' | 'tool_call' | 'tool_result' | 'thinking' | 'error'
    message: string
}

/** 网页导入运行时状态 */
export interface WebImportState {
    status: 'idle' | 'extracting' | 'extracted' | 'downloading' | 'completed' | 'error'
    url: string
    logs: AgentLog[]
    extractResult: ExtractResult | null
    selectedPages: Set<number>
    downloadProgress: {
        current: number
        total: number
    }
    downloadedImages: DownloadedImage[]
    error: string | null
}
