import { afterEach, describe, expect, it, vi } from 'vitest'
import type { AxiosError } from 'axios'

const {
  requestUseMock,
  responseUseMock,
  createMock,
  deleteRequestMock,
  getRequestMock,
  patchRequestMock,
  postRequestMock,
  putRequestMock,
} = vi.hoisted(() => ({
  requestUseMock: vi.fn(),
  responseUseMock: vi.fn(),
  createMock: vi.fn(),
  deleteRequestMock: vi.fn(),
  getRequestMock: vi.fn(),
  patchRequestMock: vi.fn(),
  postRequestMock: vi.fn(),
  putRequestMock: vi.fn(),
}))

vi.mock('axios', () => {
  const instance = {
    interceptors: {
      request: { use: requestUseMock },
      response: { use: responseUseMock },
    },
    delete: deleteRequestMock,
    get: getRequestMock,
    patch: patchRequestMock,
    post: postRequestMock,
    put: putRequestMock,
  }

  createMock.mockReturnValue(instance)

  return {
    default: {
      create: createMock,
    },
    create: createMock,
  }
})

type ResponseErrorHandler = (error: AxiosError) => Promise<never>

describe('apiClient error normalization', () => {
  afterEach(() => {
    deleteRequestMock.mockReset()
    getRequestMock.mockReset()
    patchRequestMock.mockReset()
    postRequestMock.mockReset()
    putRequestMock.mockReset()
    vi.restoreAllMocks()
  })

  it('rejects API failures as real Error instances with backend metadata', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const { ApiClientError } = await import('@/api/client')
    await import('@/api/client')

    const responseErrorHandler = responseUseMock.mock.calls[0]?.[1] as ResponseErrorHandler | undefined
    if (!responseErrorHandler) {
      throw new Error('response error interceptor was not registered')
    }

    const backendError = Object.assign(new Error('Request failed with status code 400'), {
      code: 'ERR_BAD_REQUEST',
      message: 'Request failed with status code 400',
      response: {
        status: 400,
        data: {
          error: {
            code: 'validation_error',
            message: 'AI 生成结果缺少 identity。',
            details: {
              section: 'full',
            },
          },
        },
      },
    }) as AxiosError
    const error = await responseErrorHandler(backendError).catch((value: unknown) => value)

    expect(error).toBeInstanceOf(Error)
    expect(error).toBeInstanceOf(ApiClientError)
    expect(error).toMatchObject({
      message: 'AI 生成结果缺少 identity。',
      code: 'validation_error',
      status: 400,
      details: {
        section: 'full',
      },
    })
  })

  it('distinguishes a missing backend response from an application 500', async () => {
    const { ApiClientError } = await import('@/api/client')
    const responseErrorHandler = responseUseMock.mock.calls[0]?.[1] as ResponseErrorHandler | undefined
    if (!responseErrorHandler) throw new Error('response error interceptor was not registered')

    const error = await responseErrorHandler(Object.assign(new Error('Network Error'), {
      code: 'ERR_NETWORK',
      message: 'Network Error',
    }) as AxiosError).catch((value: unknown) => value)

    expect(error).toBeInstanceOf(ApiClientError)
    expect(error).toMatchObject({
      code: 'network_error',
      message: '无法连接后端服务，请稍后重试',
      status: 0,
    })
  })

  it('does not report an expected unauthenticated session probe as an expired session', async () => {
    await import('@/api/client')
    const responseErrorHandler = responseUseMock.mock.calls[0]?.[1] as ResponseErrorHandler | undefined
    if (!responseErrorHandler) throw new Error('response error interceptor was not registered')
    const authenticationRequired = vi.fn()
    window.addEventListener('saber:authentication-required', authenticationRequired)

    const unauthorized = (url: string) => Object.assign(new Error('Unauthorized'), {
      config: { url },
      response: {
        status: 401,
        data: { error: { code: 'authentication_required', message: '请先登录' } },
      },
    }) as AxiosError

    try {
      await responseErrorHandler(unauthorized('/api/v2/auth/me')).catch(() => undefined)
      expect(authenticationRequired).not.toHaveBeenCalled()

      await responseErrorHandler(unauthorized('/api/v2/books')).catch(() => undefined)
      expect(authenticationRequired).toHaveBeenCalledTimes(1)
    } finally {
      window.removeEventListener('saber:authentication-required', authenticationRequired)
    }
  })

  it('preserves request cancellation instead of reporting a network outage', async () => {
    const { ApiClientError, isRequestCanceled } = await import('@/api/client')
    const responseErrorHandler = responseUseMock.mock.calls[0]?.[1] as ResponseErrorHandler | undefined
    if (!responseErrorHandler) throw new Error('response error interceptor was not registered')

    const error = await responseErrorHandler(Object.assign(new Error('canceled'), {
      code: 'ERR_CANCELED',
      name: 'CanceledError',
    }) as AxiosError).catch((value: unknown) => value)

    expect(error).toBeInstanceOf(ApiClientError)
    expect(error).toMatchObject({
      code: 'request_canceled',
      message: '请求已取消',
      status: 0,
    })
    expect(isRequestCanceled(error)).toBe(true)
    expect(isRequestCanceled(new DOMException('aborted', 'AbortError'))).toBe(true)
    expect(isRequestCanceled(new Error('普通错误'))).toBe(false)
  })

  it('labels the Vite empty text response as a proxy connection failure', async () => {
    const responseErrorHandler = responseUseMock.mock.calls[0]?.[1] as ResponseErrorHandler | undefined
    if (!responseErrorHandler) throw new Error('response error interceptor was not registered')

    const error = await responseErrorHandler(Object.assign(new Error('status 500'), {
      code: 'ERR_BAD_RESPONSE',
      response: {
        status: 500,
        data: '',
        headers: { 'content-type': 'text/plain' },
      },
    }) as AxiosError).catch((value: unknown) => value)

    expect(error).toMatchObject({
      code: 'proxy_connection_error',
      message: '开发代理与后端连接中断，请稍后重试',
      status: 500,
    })
  })

})
