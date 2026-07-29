export interface ModelInfoItem {
  id: string
  name: string
}

export interface FetchModelsResponse {
  success: boolean
  models?: ModelInfoItem[]
  message?: string
  error?: string
}

export interface DiagnosticConnectionTestResponse {
  success: boolean
  message?: string
  models?: string[]
  error?: string
}
