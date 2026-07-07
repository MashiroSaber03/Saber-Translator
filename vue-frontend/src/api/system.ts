import { apiClient } from './client'
import type {
  ApiResponse,
  PdfParseStartResponse,
  PdfParseBatchResponse,
  DownloadSessionResponse,
  DownloadFinalizeResponse,
  ServerInfoResponse,
} from '@/types'

export interface MobiParseStartResponse {
  success: boolean
  session_id?: string
  total_pages?: number
  total_images?: number
  error?: string
}

export interface MobiBatchImage {
  success: boolean
  data_url?: string
  width?: number
  height?: number
  filename?: string
  page_index?: number
  error?: string
}

export interface MobiParseBatchResponse {
  success: boolean
  images?: MobiBatchImage[]
  start_index?: number
  end_index?: number
  total_pages?: number
  has_more?: boolean
  error?: string
}

export interface GpuCleanupResponse {
  success: boolean
  message?: string
  unloaded_models?: string[]
  cuda_available?: boolean
  memory_allocated_mb?: number
  memory_reserved_mb?: number
  error?: string
}

export interface GpuStatusResponse {
  success: boolean
  cuda_available?: boolean
  device_name?: string
  memory_allocated_mb?: number
  memory_reserved_mb?: number
  memory_total_mb?: number
  models_loaded?: string[]
  error?: string
}

function createBatchParseFormData(file: File, batchSize: number): FormData {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('batch_size', batchSize.toString())
  return formData
}

export async function parsePdfStart(
  file: File,
  batchSize: number = 5
): Promise<PdfParseStartResponse> {
  return apiClient.upload<PdfParseStartResponse>(
    '/api/parse_pdf_start',
    createBatchParseFormData(file, batchSize)
  )
}

export async function parsePdfBatch(
  sessionId: string,
  startIndex: number,
  count: number
): Promise<PdfParseBatchResponse> {
  return apiClient.post<PdfParseBatchResponse>('/api/parse_pdf_batch', {
    session_id: sessionId,
    start_index: startIndex,
    count,
  })
}

export async function parsePdfCleanup(sessionId: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(`/api/parse_pdf_cleanup/${sessionId}`)
}

export async function parseMobiStart(
  file: File,
  batchSize: number = 5
): Promise<MobiParseStartResponse> {
  return apiClient.upload<MobiParseStartResponse>(
    '/api/parse_mobi_start',
    createBatchParseFormData(file, batchSize)
  )
}

export async function parseMobiBatch(
  sessionId: string,
  startIndex: number = 0,
  count: number = 5
): Promise<MobiParseBatchResponse> {
  return apiClient.post<MobiParseBatchResponse>('/api/parse_mobi_batch', {
    session_id: sessionId,
    start_index: startIndex,
    count,
  })
}

export async function parseMobiCleanup(sessionId: string): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>(`/api/parse_mobi_cleanup/${sessionId}`)
}

export async function downloadStartSession(totalImages: number): Promise<DownloadSessionResponse> {
  return apiClient.post<DownloadSessionResponse>('/api/download_start_session', {
    total_images: totalImages,
  })
}

export async function downloadUploadImage(
  sessionId: string,
  imageData: string,
  imageIndex: number,
  filePath?: string
): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>('/api/download_upload_image', {
    session_id: sessionId,
    image_data: imageData,
    image_index: imageIndex,
    file_path: filePath,
  })
}

export async function downloadFinalize(
  sessionId: string,
  format: 'zip' | 'pdf' | 'cbz'
): Promise<DownloadFinalizeResponse> {
  return apiClient.post<DownloadFinalizeResponse>('/api/download_finalize', {
    session_id: sessionId,
    format,
  })
}

export function getDownloadFileUrl(fileId: string, format: 'zip' | 'pdf' | 'cbz'): string {
  return `/api/download_file/${fileId}?format=${format}`
}

export async function cleanDebugFiles(): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>('/api/clean_debug_files')
}

export async function cleanTempFiles(): Promise<ApiResponse> {
  return apiClient.post<ApiResponse>('/api/clean_temp_files')
}

export async function getServerInfo(): Promise<ServerInfoResponse> {
  return apiClient.get<ServerInfoResponse>('/api/server-info')
}

export async function cleanupGpu(): Promise<GpuCleanupResponse> {
  return apiClient.post<GpuCleanupResponse>('/api/cleanup-gpu')
}

export async function getGpuStatus(): Promise<GpuStatusResponse> {
  return apiClient.get<GpuStatusResponse>('/api/gpu-status')
}
