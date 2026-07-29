import axios, { type AxiosInstance, type AxiosRequestConfig, type AxiosError } from 'axios'
import type { ApiError, ApiResponse } from '@/types'
import { assertBackendActionAllowed } from '@/services/backendAccessGate'

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
  const data = response?.data as Record<string, unknown> | undefined
  const structuredError = data?.error && typeof data.error === 'object'
    ? data.error as Record<string, unknown>
    : undefined

  return new ApiClientError({
    code: (structuredError?.code as string) || (data?.code as string) || (data?.error_code as string) || error.code || 'UNKNOWN_ERROR',
    message: (structuredError?.message as string) || (data?.error as string) || (data?.message as string) || error.message,
    status: response?.status || 500,
    details: (
      structuredError?.details ?? data?.details
    ) as Record<string, unknown> | undefined,
  })
}

class ApiClient {
  private instance: AxiosInstance

  constructor() {
    this.instance = axios.create({
      baseURL: '',
      timeout: 300000,
      headers: {
        'Content-Type': 'application/json',
      },
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
    assertBackendActionAllowed()
    const response = await this.instance.post<T>(url, data, config)
    return response.data
  }

  async put<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    assertBackendActionAllowed()
    const response = await this.instance.put<T>(url, data, config)
    return response.data
  }

  async patch<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    assertBackendActionAllowed()
    const response = await this.instance.patch<T>(url, data, config)
    return response.data
  }

  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    assertBackendActionAllowed()
    const response = await this.instance.delete<T>(url, config)
    return response.data
  }

  async upload<T>(url: string, formData: FormData, config?: AxiosRequestConfig): Promise<T> {
    assertBackendActionAllowed()
    const response = await this.instance.post<T>(url, formData, {
      ...config,
      headers: {
        ...config?.headers,
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  }

  getAxiosInstance(): AxiosInstance {
    return this.instance
  }
}

export const apiClient = new ApiClient()

export type { ApiError, ApiResponse }

export default apiClient
