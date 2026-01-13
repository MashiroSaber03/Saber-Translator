/**
 * 单页存储 API
 * 
 * 提供单页独立保存/加载的前端 API 调用
 */

import { apiClient } from './client'

// ============================================================
// 响应类型定义
// ============================================================

interface BaseResponse {
    success: boolean
    error?: string
}

interface DataResponse<T> extends BaseResponse {
    data?: T
}

// ============================================================
// 会话元数据 API
// ============================================================

/**
 * 保存会话元数据
 */
export async function saveSessionMeta(
    sessionPath: string,
    metadata: {
        ui_settings?: Record<string, unknown>
        total_pages?: number
        currentImageIndex?: number
    }
): Promise<{ success: boolean; error?: string }> {
    return apiClient.post<BaseResponse>(`/api/sessions/meta/${sessionPath}`, metadata)
}

/**
 * 加载会话元数据
 */
export async function loadSessionMeta(
    sessionPath: string
): Promise<{ success: boolean; data?: Record<string, unknown>; error?: string }> {
    return apiClient.get<DataResponse<Record<string, unknown>>>(`/api/sessions/meta/${sessionPath}`)
}


// ============================================================
// 单页图片 API
// ============================================================

/**
 * 保存单页图片
 */
export async function savePageImage(
    sessionPath: string,
    pageIndex: number,
    imageType: 'original' | 'clean' | 'translated',
    base64Data: string
): Promise<{ success: boolean; error?: string }> {
    return apiClient.post<BaseResponse>(
        `/api/sessions/page/${sessionPath}/${pageIndex}/${imageType}`,
        { data: base64Data }
    )
}

/**
 * 获取单页图片 URL（用于 img 标签）
 */
export function getPageImageUrl(
    sessionPath: string,
    pageIndex: number,
    imageType: 'original' | 'clean' | 'translated'
): string {
    return `/api/sessions/page/${sessionPath}/${pageIndex}/${imageType}`
}


// ============================================================
// 单页元数据 API
// ============================================================

/**
 * 保存单页元数据
 */
export async function savePageMeta(
    sessionPath: string,
    pageIndex: number,
    meta: Record<string, unknown>
): Promise<{ success: boolean; error?: string }> {
    return apiClient.post<BaseResponse>(
        `/api/sessions/page/${sessionPath}/${pageIndex}/meta`,
        meta
    )
}

/**
 * 加载单页元数据
 */
export async function loadPageMeta(
    sessionPath: string,
    pageIndex: number
): Promise<{ success: boolean; data?: Record<string, unknown>; error?: string }> {
    return apiClient.get<DataResponse<Record<string, unknown>>>(
        `/api/sessions/page/${sessionPath}/${pageIndex}/meta`
    )
}


// ============================================================
// 批量操作 API
// ============================================================

/** 页面数据（用于预保存） */
export interface PageData {
    originalDataURL?: string
    translatedDataURL?: string
    cleanImageData?: string
    [key: string]: unknown
}

/** 预保存响应 */
interface PresaveResponse extends BaseResponse {
    saved?: number
}

/**
 * 预保存所有页面
 */
export async function presaveAllPages(
    sessionPath: string,
    images: PageData[],
    uiSettings: Record<string, unknown>
): Promise<{ success: boolean; saved?: number; error?: string }> {
    return apiClient.post<PresaveResponse>(
        `/api/sessions/presave/${sessionPath}`,
        { images, ui_settings: uiSettings }
    )
}

/**
 * 保存翻译完成的页面
 */
export async function saveTranslatedPage(
    sessionPath: string,
    pageIndex: number,
    data: {
        translated?: string
        clean?: string
        meta?: Record<string, unknown>
    }
): Promise<{ success: boolean; error?: string }> {
    return apiClient.post<BaseResponse>(
        `/api/sessions/save_translated/${sessionPath}/${pageIndex}`,
        data
    )
}

/** 页面信息（加载时返回） */
export interface PageInfo {
    index: number
    hasOriginal: boolean
    hasTranslated: boolean
    hasClean: boolean
    originalUrl?: string
    translatedUrl?: string
    cleanUrl?: string
    [key: string]: unknown
}

/** 完整会话数据 */
export interface SessionData {
    ui_settings: Record<string, unknown>
    total_pages: number
    currentImageIndex: number
    pages: PageInfo[]
}

/**
 * 加载完整会话
 */
export async function loadSession(
    sessionPath: string
): Promise<{ success: boolean; data?: SessionData; error?: string }> {
    return apiClient.get<DataResponse<SessionData>>(`/api/sessions/load/${sessionPath}`)
}


// ============================================================
// 公共工具函数
// ============================================================

/**
 * 提取纯 Base64 数据（去掉 data:image/...;base64, 前缀）
 */
export function extractBase64(dataUrl: string | null | undefined): string | null {
    if (!dataUrl || typeof dataUrl !== 'string') return null
    if (dataUrl.startsWith('/api/')) return null // URL 格式不保存
    if (dataUrl.startsWith('data:')) {
        const parts = dataUrl.split(',')
        return parts.length > 1 ? (parts[1] ?? null) : null
    }
    return dataUrl // 可能已经是纯 base64
}

/** 图片数据接口 */
export interface ImageDataForSave {
    originalDataURL?: string | null
    translatedDataURL?: string | null
    cleanImageData?: string | null
    fileName?: string
    translationStatus?: string
    translationFailed?: boolean
    bubbleStates?: unknown[]
    isManuallyAnnotated?: boolean
    relativePath?: string
    folderPath?: string
    fontSize?: number
    autoFontSize?: boolean
    fontFamily?: string
    layoutDirection?: string
    useAutoTextColor?: boolean
    textColor?: string
    fillColor?: string
    inpaintMethod?: string
    strokeEnabled?: boolean
    strokeColor?: string
    strokeWidth?: number
}

/** 保存进度回调 */
export interface SaveProgressCallback {
    onProgress?: (current: number, total: number) => void
}

/**
 * 逐页保存所有图片（公共函数）
 * 
 * @param sessionPath 会话路径
 * @param images 图片数组
 * @param callback 进度回调
 * @returns 成功保存的数量
 */
export async function saveAllPagesSequentially(
    sessionPath: string,
    images: ImageDataForSave[],
    callback?: SaveProgressCallback
): Promise<number> {
    const totalImages = images.length
    let savedCount = 0

    for (let idx = 0; idx < totalImages; idx++) {
        const img = images[idx]
        if (!img) continue

        // 报告进度
        callback?.onProgress?.(idx + 1, totalImages)

        try {
            // 保存原图
            const originalBase64 = extractBase64(img.originalDataURL)
            if (originalBase64) {
                await savePageImage(sessionPath, idx, 'original', originalBase64)
            }

            // 保存译图
            const translatedBase64 = extractBase64(img.translatedDataURL)
            if (translatedBase64) {
                await savePageImage(sessionPath, idx, 'translated', translatedBase64)
            }

            // 保存干净背景
            const cleanBase64 = extractBase64(img.cleanImageData)
            if (cleanBase64) {
                await savePageImage(sessionPath, idx, 'clean', cleanBase64)
            }

            // 保存页面元数据
            const pageMeta: Record<string, unknown> = {
                fileName: img.fileName,
                translationStatus: img.translationStatus,
                translationFailed: img.translationFailed,
                bubbleStates: img.bubbleStates,
                isManuallyAnnotated: img.isManuallyAnnotated,
                relativePath: img.relativePath,
                folderPath: img.folderPath,
                fontSize: img.fontSize,
                autoFontSize: img.autoFontSize,
                fontFamily: img.fontFamily,
                layoutDirection: img.layoutDirection,
                useAutoTextColor: img.useAutoTextColor,
                textColor: img.textColor,
                fillColor: img.fillColor,
                inpaintMethod: img.inpaintMethod,
                strokeEnabled: img.strokeEnabled,
                strokeColor: img.strokeColor,
                strokeWidth: img.strokeWidth,
            }
            await savePageMeta(sessionPath, idx, pageMeta)

            savedCount++
        } catch (pageError) {
            console.error(`保存第 ${idx + 1} 页失败:`, pageError)
            // 继续保存其他页
        }
    }

    return savedCount
}
