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

  it('keeps unrelated mutations open while task APIs enforce the settings gate', async () => {
    const { apiClient } = await import('@/api/client')
    const { createChapterTranslationJob } = await import('@/api/v2/translation')
    const {
      BackendAccessRestrictedError,
      setBackendAccessRestricted,
    } = await import('@/services/backendAccessGate')
    getRequestMock.mockResolvedValue({ data: { ok: true } })
    postRequestMock.mockResolvedValue({ data: { ok: true } })
    putRequestMock.mockResolvedValue({ data: { ok: true } })
    patchRequestMock.mockResolvedValue({ data: { ok: true } })
    deleteRequestMock.mockResolvedValue({ data: { ok: true } })
    setBackendAccessRestricted(true, '设置加载失败')

    try {
      await expect(apiClient.get('/api/v2/books')).resolves.toEqual({ ok: true })
      await expect(apiClient.post('/api/v2/books', {})).resolves.toEqual({ ok: true })
      await expect(apiClient.put('/api/v2/books/one', {})).resolves.toEqual({ ok: true })
      await expect(apiClient.patch('/api/v2/pages/one/document', {})).resolves.toEqual({ ok: true })
      await expect(apiClient.delete('/api/v2/books/one')).resolves.toEqual({ ok: true })
      await expect(apiClient.upload('/api/v2/pages', new FormData())).resolves.toEqual({ ok: true })
      await expect(
        createChapterTranslationJob('chapter', [], { mode: 'standard' }),
      ).rejects.toBeInstanceOf(BackendAccessRestrictedError)

      expect(getRequestMock).toHaveBeenCalledTimes(1)
      expect(postRequestMock).toHaveBeenCalledTimes(2)
      expect(putRequestMock).toHaveBeenCalledTimes(1)
      expect(patchRequestMock).toHaveBeenCalledTimes(1)
      expect(deleteRequestMock).toHaveBeenCalledTimes(1)
    } finally {
      setBackendAccessRestricted(false)
    }
  })
})
