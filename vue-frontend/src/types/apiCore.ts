export interface ApiResponse<T = unknown> {
  success: boolean
  data?: T
  error?: string
  message?: string
}

export interface ApiError extends Error {
  code: string
  message: string
  status: number
  details?: Record<string, unknown>
}
