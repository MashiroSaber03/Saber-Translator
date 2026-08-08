import axios, { type AxiosInstance, type AxiosRequestConfig, type AxiosError } from 'axios'
import type { ApiError } from '@/types'

interface ApiClientErrorInit {
  code: string
  message: string
  status: number
  details?: Record<string, unknown>
}

export class ApiClientError extends Error implements ApiError {
  readonly code: string
  readonly status: number
  readonly details?: Record<string, unknown>

  constructor({ code, message, status, details }: ApiClientErrorInit) {
    super(message)
    this.name = 'ApiClientError'
    Object.setPrototypeOf(this, new.target.prototype)
    this.code = code
    this.status = status
    this.details = details
  }
}

function createApiError(error: AxiosError): ApiError {
  const response = error.response
  const data = response?.data
  const dataObject = data && typeof data === 'object' && !Array.isArray(data)
    ? data as Record<string, unknown>
    : undefined
  const backendError = dataObject?.error && typeof dataObject.error === 'object'
    ? dataObject.error as Record<string, unknown>
    : undefined
  const backendCode = typeof backendError?.code === 'string'
    ? backendError.code
    : undefined
  const backendMessage = typeof backendError?.message === 'string'
    ? backendError.message
    : undefined
  const backendDetails = backendError?.details
  const contentType = String(response?.headers?.['content-type'] || '').toLowerCase()
  const isNetworkFailure = response === undefined
  const isDevelopmentProxyFailure = response?.status === 500
    && backendError === undefined
    && (data === '' || data === undefined || contentType.startsWith('text/plain'))

  return new ApiClientError({
    code: backendCode
      || (isNetworkFailure ? 'network_error' : undefined)
      || (isDevelopmentProxyFailure ? 'proxy_connection_error' : undefined)
      || error.code
      || 'UNKNOWN_ERROR',
    message: backendMessage
      || (isNetworkFailure ? '无法连接后端服务，请稍后重试' : undefined)
      || (isDevelopmentProxyFailure ? '开发代理与后端连接中断，请稍后重试' : undefined)
      || error.message,
    status: response?.status ?? 0,
    details: backendDetails && typeof backendDetails === 'object' && !Array.isArray(backendDetails)
      ? backendDetails as Record<string, unknown>
      : undefined,
  })
}

class ApiClient {
  private instance: AxiosInstance

  constructor() {
    this.instance = axios.create({
      baseURL: '',
      timeout: 300000,
    })

    this.instance.interceptors.response.use(
      response => response,
      (error: AxiosError) => {
        const apiError = createApiError(error)
        return Promise.reject(apiError)
      }
    )
  }

  async get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.instance.get<T>(url, config)
    return response.data
  }

  async post<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.instance.post<T>(url, data, config)
    return response.data
  }

  async put<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.instance.put<T>(url, data, config)
    return response.data
  }

  async patch<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.instance.patch<T>(url, data, config)
    return response.data
  }

  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.instance.delete<T>(url, config)
    return response.data
  }

  async upload<T>(
    url: string,
    formData: FormData,
    config?: AxiosRequestConfig,
    method: 'post' | 'put' = 'post',
  ): Promise<T> {
    const response = method === 'put'
      ? await this.instance.put<T>(url, formData, config)
      : await this.instance.post<T>(url, formData, config)
    return response.data
  }

}

export const apiClient = new ApiClient()

export type { ApiError }
