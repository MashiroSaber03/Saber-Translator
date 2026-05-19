import { beforeEach, describe, expect, it, vi } from 'vitest'

const {
  getMock,
  postMock,
  deleteMock,
  uploadMock,
} = vi.hoisted(() => ({
  getMock: vi.fn(),
  postMock: vi.fn(),
  deleteMock: vi.fn(),
  uploadMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  apiClient: {
    get: getMock,
    post: postMock,
    delete: deleteMock,
    upload: uploadMock,
  },
}))

describe('plugin api import/export helpers', () => {
  beforeEach(() => {
    getMock.mockReset()
    postMock.mockReset()
    deleteMock.mockReset()
    uploadMock.mockReset()
    vi.restoreAllMocks()
  })

  it('downloads plugin export and parses content-disposition filename', async () => {
    const blob = new Blob(['zip-bytes'], { type: 'application/zip' })
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      headers: {
        get: vi.fn().mockImplementation((name: string) => (
          name.toLowerCase() === 'content-disposition'
            ? 'attachment; filename="sample_plugin.zip"'
            : null
        )),
      },
      blob: vi.fn().mockResolvedValue(blob),
    })
    vi.stubGlobal('fetch', fetchMock)

    const { exportPlugin } = await import('@/api/plugin')
    const result = await exportPlugin('sample_plugin')

    expect(fetchMock).toHaveBeenCalledWith('/api/plugins/sample_plugin/export')
    expect(result.filename).toBe('sample_plugin.zip')
    expect(result.blob).toBe(blob)
  })

  it('uploads plugin package with replace flag', async () => {
    uploadMock.mockResolvedValue({
      success: true,
      plugin: {
        id: 'sample_plugin',
      },
    })

    const { importPlugin } = await import('@/api/plugin')
    const file = new File(['zip-bytes'], 'sample_plugin.zip', { type: 'application/zip' })
    const result = await importPlugin(file, true)

    expect(uploadMock).toHaveBeenCalledTimes(1)
    const [url, formData] = uploadMock.mock.calls[0] || []
    expect(url).toBe('/api/plugins/import')
    expect(formData).toBeInstanceOf(FormData)
    expect((formData as FormData).get('file')).toBe(file)
    expect((formData as FormData).get('replace')).toBe('true')
    expect(result.success).toBe(true)
  })
})
