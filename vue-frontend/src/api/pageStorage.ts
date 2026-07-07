import { apiClient } from './client'

interface BaseResponse {
  success: boolean
  error?: string
}

interface DataResponse<T> extends BaseResponse {
  data?: T
}

export async function saveSessionMeta(
  sessionPath: string,
  metadata: {
    ui_settings?: Record<string, unknown>
    total_pages?: number
    currentImageIndex?: number
  }
): Promise<BaseResponse> {
  return apiClient.post<BaseResponse>(`/api/sessions/meta/${sessionPath}`, metadata)
}

export async function loadSessionMeta(
  sessionPath: string
): Promise<DataResponse<Record<string, unknown>>> {
  return apiClient.get<DataResponse<Record<string, unknown>>>(`/api/sessions/meta/${sessionPath}`)
}

export async function savePageImage(
  sessionPath: string,
  pageIndex: number,
  imageType: 'original' | 'clean' | 'translated',
  base64Data: string
): Promise<BaseResponse> {
  return apiClient.post<BaseResponse>(
    `/api/sessions/page/${sessionPath}/${pageIndex}/${imageType}`,
    {
      data: base64Data,
    }
  )
}

export async function savePageMeta(
  sessionPath: string,
  pageIndex: number,
  meta: Record<string, unknown>
): Promise<BaseResponse> {
  return apiClient.post<BaseResponse>(`/api/sessions/page/${sessionPath}/${pageIndex}/meta`, meta)
}
