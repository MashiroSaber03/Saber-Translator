import { describe, expect, it, vi } from 'vitest'

describe('api download helpers', () => {
  it('parses content-disposition filenames and falls back when absent', async () => {
    const { parseContentDispositionFilename } = await import('@/api/download')

    expect(
      parseContentDispositionFilename('attachment; filename="studio.export.json"', 'fallback.json')
    ).toBe('studio.export.json')
    expect(
      parseContentDispositionFilename('attachment; filename=plugin.zip', 'fallback.zip')
    ).toBe('plugin.zip')
    expect(parseContentDispositionFilename('', 'fallback.zip')).toBe('fallback.zip')
  })

  it('downloads a blob with shared filename parsing', async () => {
    const blob = new Blob(['zip-bytes'], { type: 'application/zip' })
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      headers: {
        get: vi.fn().mockImplementation((name: string) => (
          name.toLowerCase() === 'content-disposition'
            ? 'attachment; filename="plugin.zip"'
            : null
        )),
      },
      blob: vi.fn().mockResolvedValue(blob),
    })
    vi.stubGlobal('fetch', fetchMock)

    const { downloadBlob } = await import('@/api/download')
    const result = await downloadBlob({
      url: '/api/plugins/demo/export',
      fallbackFilename: 'demo.zip',
      fallbackErrorMessage: '导出失败',
    })

    expect(fetchMock).toHaveBeenCalledWith('/api/plugins/demo/export')
    expect(result).toEqual({ blob, filename: 'plugin.zip' })
  })

  it('uses the v2 error envelope or plain proxy text before fallback messages', async () => {
    const { readApiErrorMessage } = await import('@/api/download')

    const nestedJsonResponse = new Response(
      JSON.stringify({ error: { code: 'validation_error', message: 'nested json failed' } }),
      { status: 422 },
    )
    await expect(readApiErrorMessage(nestedJsonResponse, 'fallback')).resolves.toBe(
      'nested json failed',
    )

    const textResponse = new Response('plain failed', { status: 500 })
    await expect(readApiErrorMessage(textResponse, 'fallback')).resolves.toBe('plain failed')

    const emptyResponse = new Response('', { status: 500 })
    await expect(readApiErrorMessage(emptyResponse, 'fallback')).resolves.toBe('fallback')
  })
})
