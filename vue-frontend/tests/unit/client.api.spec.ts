import { describe, expect, it, vi } from 'vitest'

const {
  requestUseMock,
  responseUseMock,
  createMock,
} = vi.hoisted(() => ({
  requestUseMock: vi.fn(),
  responseUseMock: vi.fn(),
  createMock: vi.fn(),
}))

vi.mock('axios', () => {
  const instance = {
    interceptors: {
      request: { use: requestUseMock },
      response: { use: responseUseMock },
    },
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
    delete: vi.fn(),
  }

  createMock.mockReturnValue(instance)

  return {
    default: {
      create: createMock,
    },
    create: createMock,
  }
})

describe('apiClient error normalization', () => {
  it('rejects API failures as real Error instances with backend metadata', async () => {
    const { ApiClientError } = await import('@/api/client')
    await import('@/api/client')

    const responseErrorHandler = responseUseMock.mock.calls[0]?.[1]
    if (!responseErrorHandler) {
      throw new Error('response error interceptor was not registered')
    }

    const error = await responseErrorHandler({
      code: 'ERR_BAD_REQUEST',
      message: 'Request failed with status code 400',
      response: {
        status: 400,
        data: {
          error_code: 'VALIDATION_FAILED',
          error: 'AI 生成结果缺少 identity。',
          details: {
            section: 'full',
          },
        },
      },
    } as any).catch((value: unknown) => value)

    expect(error).toBeInstanceOf(Error)
    expect(error).toBeInstanceOf(ApiClientError)
    expect(error).toMatchObject({
      message: 'AI 生成结果缺少 identity。',
      code: 'VALIDATION_FAILED',
      status: 400,
      details: {
        section: 'full',
      },
    })
  })
})
