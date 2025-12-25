/**
 * HTTP 客户端
 * 封装 Axios，提供统一的错误处理和类型安全
 */

import axios, { type AxiosInstance, type AxiosRequestConfig, type AxiosError } from 'axios'
import type { ApiError, ApiResponse } from '@/types'

/**
 * 创建 API 错误对象
 */
function createApiError(error: AxiosError): ApiError {
  const response = error.response
  const data = response?.data as Record<string, unknown> | undefined

  return {
    code: (data?.code as string) || error.code || 'UNKNOWN_ERROR',
    message: (data?.error as string) || (data?.message as string) || error.message,
    status: response?.status || 500,
    details: data?.details as Record<string, unknown> | undefined,
  }
}

/**
 * API 客户端类
 * 提供类型安全的 HTTP 请求方法
 */
class ApiClient {
  private instance: AxiosInstance

  constructor() {
    this.instance = axios.create({
      // 基础 URL（开发时通过 Vite 代理，生产时直接访问）
      baseURL: '',
      timeout: 300000, // 5分钟超时（翻译可能需要较长时间）
      headers: {
        'Content-Type': 'application/json',
      },
    })

    // 请求拦截器
    this.instance.interceptors.request.use(
      config => {
        // 可以在这里添加认证 token 等
        return config
      },
      error => {
        return Promise.reject(error)
      }
    )

    // 响应拦截器
    this.instance.interceptors.response.use(
      response => response,
      (error: AxiosError) => {
        const apiError = createApiError(error)
        console.error('API 错误:', apiError)
        return Promise.reject(apiError)
      }
    )
  }

  /**
   * GET 请求
   */
  async get<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.instance.get<T>(url, config)
    return response.data
  }

  /**
   * POST 请求
   */
  async post<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.instance.post<T>(url, data, config)
    return response.data
  }

  /**
   * PUT 请求
   */
  async put<T>(url: string, data?: unknown, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.instance.put<T>(url, data, config)
    return response.data
  }

  /**
   * DELETE 请求
   */
  async delete<T>(url: string, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.instance.delete<T>(url, config)
    return response.data
  }

  /**
   * 上传文件
   */
  async upload<T>(url: string, formData: FormData, config?: AxiosRequestConfig): Promise<T> {
    const response = await this.instance.post<T>(url, formData, {
      ...config,
      headers: {
        ...config?.headers,
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  }

  /**
   * 获取原始 Axios 实例（用于特殊场景）
   */
  getAxiosInstance(): AxiosInstance {
    return this.instance
  }
}

// 导出单例实例
export const apiClient = new ApiClient()

// 导出类型
export type { ApiError, ApiResponse }

// 默认导出
export default apiClient
