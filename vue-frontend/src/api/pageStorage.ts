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

