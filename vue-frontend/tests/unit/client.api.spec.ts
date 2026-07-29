import { afterEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
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

  it('keeps API client error fixtures typed to Axios error contracts', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/unit/client.api.spec.ts'), 'utf8')

    expect(source).not.toMatch(/\bas any\b|:\s*any\b|any\[\]/)
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
          error_code: 'VALIDATION_FAILED',
          error: 'AI 生成结果缺少 identity。',
          details: {
            section: 'full',
          },
        },
      },
    }) as AxiosError
    const error = await responseErrorHandler(backendError).catch((value: unknown) => value)

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

  it('allows reads but blocks every mutation while settings are restricted', async () => {
    const { apiClient } = await import('@/api/client')
    const {
      BackendAccessRestrictedError,
      setBackendAccessRestricted,
    } = await import('@/services/backendAccessGate')
    getRequestMock.mockResolvedValue({ data: { ok: true } })
    setBackendAccessRestricted(true, '设置加载失败')

    try {
      await expect(apiClient.get('/api/v2/books')).resolves.toEqual({ ok: true })
      await expect(apiClient.post('/api/v2/jobs', {})).rejects.toBeInstanceOf(
        BackendAccessRestrictedError,
      )
      await expect(apiClient.put('/api/v2/settings', {})).rejects.toBeInstanceOf(
        BackendAccessRestrictedError,
      )
      await expect(apiClient.patch('/api/v2/pages/one', {})).rejects.toBeInstanceOf(
        BackendAccessRestrictedError,
      )
      await expect(apiClient.delete('/api/v2/books/one')).rejects.toBeInstanceOf(
        BackendAccessRestrictedError,
      )
      await expect(
        apiClient.upload('/api/v2/fonts', new FormData()),
      ).rejects.toBeInstanceOf(BackendAccessRestrictedError)

      expect(getRequestMock).toHaveBeenCalledTimes(1)
      expect(postRequestMock).not.toHaveBeenCalled()
      expect(putRequestMock).not.toHaveBeenCalled()
      expect(patchRequestMock).not.toHaveBeenCalled()
      expect(deleteRequestMock).not.toHaveBeenCalled()
    } finally {
      setBackendAccessRestricted(false)
    }
  })
})
