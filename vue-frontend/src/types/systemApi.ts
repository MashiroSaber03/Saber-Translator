export interface ServerInfoResponse {
  success: boolean
  local_url?: string
  lan_url?: string
  lan_ip?: string
  port?: number
  error?: string
}

export interface DownloadSessionResponse {
  success: boolean
  session_id?: string
  error?: string
}

export interface DownloadFinalizeResponse {
  success: boolean
  file_id?: string
  error?: string
}

export interface PdfParseStartResponse {
  success: boolean
  session_id?: string
  total_pages?: number
  error?: string
}

export interface PdfParseBatchResponse {
  success: boolean
  images?: Array<{
    page_index: number
    data_url: string
  }>
  has_more?: boolean
  error?: string
}
