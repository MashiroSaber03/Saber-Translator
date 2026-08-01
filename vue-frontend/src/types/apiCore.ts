export interface ApiError extends Error {
  code: string
  message: string
  status: number
  details?: Record<string, unknown>
}
