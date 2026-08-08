import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { V2PageImportResult } from '@/api/v2/content'

const mocks = vi.hoisted(() => ({
  upload: vi.fn(),
}))

vi.mock('@/api/client', () => {
  class ApiClientError extends Error {
    readonly code: string
    readonly status: number

    constructor(options: { code: string; message: string; status: number }) {
      super(options.message)
      this.name = 'ApiClientError'
      Object.setPrototypeOf(this, new.target.prototype)
      this.code = options.code
      this.status = options.status
    }
  }

  return {
    ApiClientError,
    apiClient: { upload: mocks.upload },
  }
})

function pageImportResult(id: string): V2PageImportResult {
  return {
    page: {
      chapterId: '00000000-0000-0000-0000-000000000001',
      cleanUrl: null,
      detectionState: 'unprocessed',
      documentRevision: 1,
      height: 20,
      id,
      logicalSourcePath: `${id}.png`,
      ordinal: 1,
      renderedRevision: null,
      renderStatus: 'not_rendered',
      sourceRevision: 1,
      sourceUrl: `/source/${id}`,
      thumbnailSourceUrl: `/thumbnail/${id}`,
      translatedUrl: null,
      width: 20,
    },
    pageOrderRevision: 1,
  }
}

describe('ordinary image upload API', () => {
  beforeEach(() => {
    mocks.upload.mockReset()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('retries a transport failure with the same per-image idempotency key', async () => {
    vi.useFakeTimers()
    const { ApiClientError } = await import('@/api/client')
    const { importImagesSequentially } = await import('@/api/v2/content')
    mocks.upload
      .mockRejectedValueOnce(new ApiClientError({
        code: 'network_error',
        message: 'offline',
        status: 0,
      }))
      .mockResolvedValueOnce(pageImportResult('page-1'))

    const promise = importImagesSequentially(
      '00000000-0000-0000-0000-000000000002',
      [new File(['image'], '001.png', { type: 'image/png' })],
    )
    await vi.advanceTimersByTimeAsync(250)
    const summary = await promise

    expect(summary).toMatchObject({ failures: [], results: [pageImportResult('page-1')] })
    expect(mocks.upload).toHaveBeenCalledTimes(2)
    const firstConfig = mocks.upload.mock.calls[0]?.[2]
    const secondConfig = mocks.upload.mock.calls[1]?.[2]
    expect(firstConfig.headers['Idempotency-Key']).toBe(secondConfig.headers['Idempotency-Key'])
  })

  it('keeps importing after one image has a non-retryable validation failure', async () => {
    const { ApiClientError } = await import('@/api/client')
    const { importImagesSequentially } = await import('@/api/v2/content')
    mocks.upload
      .mockRejectedValueOnce(new ApiClientError({
        code: 'unsupported_image',
        message: 'not an image',
        status: 422,
      }))
      .mockResolvedValueOnce(pageImportResult('page-2'))

    const summary = await importImagesSequentially(
      '00000000-0000-0000-0000-000000000002',
      [
        new File(['bad'], '001.png', { type: 'image/png' }),
        new File(['good'], '002.png', { type: 'image/png' }),
      ],
    )

    expect(mocks.upload).toHaveBeenCalledTimes(2)
    expect(summary.results).toEqual([pageImportResult('page-2')])
    expect(summary.failures).toHaveLength(1)
    expect(summary.failures[0]?.entry.logicalPath).toBe('001.png')
  })

  it('reuses the original key when the user retries only failed images', async () => {
    const { ApiClientError } = await import('@/api/client')
    const {
      importImagesSequentially,
      retryFailedImageImports,
    } = await import('@/api/v2/content')
    mocks.upload.mockRejectedValueOnce(new ApiClientError({
      code: 'unsupported_image',
      message: 'temporary test failure',
      status: 422,
    }))
    const first = await importImagesSequentially(
      '00000000-0000-0000-0000-000000000002',
      [new File(['image'], '001.png', { type: 'image/png' })],
    )
    const originalKey = mocks.upload.mock.calls[0]?.[2].headers['Idempotency-Key']
    mocks.upload.mockResolvedValueOnce(pageImportResult('page-3'))

    const retried = await retryFailedImageImports(
      '00000000-0000-0000-0000-000000000002',
      first.failures,
    )

    expect(retried.failures).toEqual([])
    expect(mocks.upload.mock.calls[1]?.[2].headers['Idempotency-Key']).toBe(originalKey)
  })
})
