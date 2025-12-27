/**
 * 系统级 API
 * 包含 PDF/MOBI 解析、批量下载、调试文件清理等系统功能
 */

import { apiClient } from './client'
import type {
  ApiResponse,
  PdfParseStartResponse,
  PdfParseBatchResponse,
  DownloadSessionResponse,
  DownloadFinalizeResponse,
  ServerInfoResponse,
} from '@/types'

// ==================== PDF 解析 API ====================

/**
 * 开始 PDF 分批解析
 * @param file PDF 文件
 * @param batchSize 每批页数
 */
export async function parsePdfStart(
  file: File,
  batchSize: number = 5
): Promise<PdfParseStartResponse> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('batch_size', batchSize.toString())
  return apiClient.upload<PdfParseStartResponse>('/api/parse_pdf_start', formData)
}

/**
 * 获取 PDF 解析批次
 * @param sessionId 解析会话 ID
 * @param startIndex 起始索引（复刻原版）
 * @param count 批次数量（复刻原版）
 */
export async function parsePdfBatch(
  sessionId: string,
  startIndex: number,
  count: number
): Promise<PdfParseBatchResponse> {
  return apiClient.post<PdfParseBatchResponse>('/api/parse_pdf_batch', {
    session_id: sessionId,
    start_index: startIndex,
    count: count,
  })
}

/**
 * 清理 PDF 解析会话
 * @param sessionId 解析会话 ID
 */
export async function parsePdfCleanup(sessionId: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(`/api/parse_pdf_cleanup/${sessionId}`)
}

// ==================== MOBI/AZW 解析 API ====================

/**
 * MOBI 解析开始响应
 */
export interface MobiParseStartResponse {
  success: boolean
  session_id?: string
  total_images?: number
  error?: string
}

/**
 * MOBI 解析批次响应
 */
export interface MobiParseBatchResponse {
  success: boolean
  images?: string[]
  has_more?: boolean
  error?: string
}

/**
 * 开始 MOBI/AZW 分批解析
 * @param file MOBI/AZW 文件
 * @param batchSize 每批图片数
 */
export async function parseMobiStart(
  file: File,
  batchSize: number = 5
): Promise<MobiParseStartResponse> {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('batch_size', batchSize.toString())
  return apiClient.upload<MobiParseStartResponse>('/api/parse_mobi_start', formData)
}

/**
 * 获取 MOBI/AZW 解析批次
 * @param sessionId 解析会话 ID
 */
export async function parseMobiBatch(sessionId: string): Promise<MobiParseBatchResponse> {
  return apiClient.post<MobiParseBatchResponse>('/api/parse_mobi_batch', {
    session_id: sessionId,
  })
}

/**
 * 清理 MOBI/AZW 解析会话
 * @param sessionId 解析会话 ID
 */
export async function parseMobiCleanup(sessionId: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(`/api/parse_mobi_cleanup/${sessionId}`)
}

// ==================== 批量下载 API ====================

/**
 * 开始下载会话
 * @param totalImages 总图片数量
 */
export async function downloadStartSession(
  totalImages: number
): Promise<DownloadSessionResponse> {
  return apiClient.post<DownloadSessionResponse>('/api/download_start_session', { total_images: totalImages })
}

/**
 * 上传图片到下载会话
 * @param sessionId 下载会话 ID
 * @param imageData Base64 图片数据
 * @param imageIndex 图片索引
 */
export async function downloadUploadImage(
  sessionId: string,
  imageData: string,
  imageIndex: number
): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>('/api/download_upload_image', {
    session_id: sessionId,
    image_data: imageData,
    image_index: imageIndex,
  })
}

/**
 * 完成下载打包
 * @param sessionId 下载会话 ID
 * @param format 下载格式
 */
export async function downloadFinalize(
  sessionId: string,
  format: 'zip' | 'pdf' | 'cbz'
): Promise<DownloadFinalizeResponse> {
  return apiClient.post<DownloadFinalizeResponse>('/api/download_finalize', {
    session_id: sessionId,
    format,
  })
}

/**
 * 获取下载文件 URL
 * @param fileId 文件 ID
 * @param format 下载格式
 */
export function getDownloadFileUrl(fileId: string, format: 'zip' | 'pdf' | 'cbz'): string {
  return `/api/download_file/${fileId}?format=${format}`
}

// ==================== 系统维护 API ====================

/**
 * 清理调试文件
 */
export async function cleanDebugFiles(): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>('/api/clean_debug_files')
}

/**
 * 清理临时文件
 */
export async function cleanTempFiles(): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>('/api/clean_temp_files')
}

/**
 * 获取服务器信息（局域网地址等）
 */
export async function getServerInfo(): Promise<ServerInfoResponse> {
  return apiClient.get<ServerInfoResponse>('/api/server-info')
}
