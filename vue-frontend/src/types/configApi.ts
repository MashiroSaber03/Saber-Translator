export interface FontInfo {
  id?: string
  kind?: 'builtin' | 'uploaded'
  file_name: string
  display_name: string
  path: string
  is_default: boolean
}

export interface FontListResponse {
  success?: boolean
  fonts?: FontInfo[]
  default_fonts?: Record<string, string>
  error?: string
}

export interface PromptListResponse {
  success?: boolean
  prompt_names?: string[]
  default_prompt_content?: string
  error?: string
}

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

export interface ConnectionTestResponse {
  success: boolean
  message?: string
  models?: string[]
  error?: string
}
