import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const {
  getMock,
  postMock,
  putMock,
  deleteMock,
  uploadMock,
} = vi.hoisted(() => ({
  getMock: vi.fn(),
  postMock: vi.fn(),
  putMock: vi.fn(),
  deleteMock: vi.fn(),
  uploadMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  apiClient: {
    get: getMock,
    post: postMock,
    put: putMock,
    delete: deleteMock,
    upload: uploadMock,
  },
}))

describe('continuation api exports', () => {
  beforeEach(() => {
    getMock.mockReset()
    postMock.mockReset()
    putMock.mockReset()
    deleteMock.mockReset()
    uploadMock.mockReset()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('exports generated images with the shared blob downloader', async () => {
    const blob = new Blob(['zip'], { type: 'application/zip' })
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      headers: { get: vi.fn().mockReturnValue(null) },
      blob: vi.fn().mockResolvedValue(blob),
    })
    vi.stubGlobal('fetch', fetchMock)

    const { exportAsImages } = await import('@/api/continuation')
    await expect(exportAsImages('book-1')).resolves.toBe(blob)

    expect(fetchMock).toHaveBeenCalledWith('/api/manga-insight/book-1/continuation/export/images', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({}),
    })
  })

  it('surfaces json export errors from the shared downloader', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ error: 'pdf failed' }), { status: 500 })
    )
    vi.stubGlobal('fetch', fetchMock)

    const { exportAsPdf } = await import('@/api/continuation')

    await expect(exportAsPdf('book-1')).rejects.toThrow('pdf failed')
  })

  it('encodes character and form endpoints through shared path helpers', async () => {
    const formData = new FormData()
    const file = new File(['image'], 'ref.png', { type: 'image/png' })
    const {
      deleteCharacter,
      generateFormOrtho,
      getAvailableImages,
      toggleFormEnabled,
      updateCharacterForm,
      uploadFormImage,
    } = await import('@/api/continuation')

    await deleteCharacter('book-1', 'Saber/Alter')
    await updateCharacterForm('book-1', 'Saber/Alter', 'battle form', { form_name: 'Battle' })
    await toggleFormEnabled('book-1', 'Saber/Alter', 'battle form', true)
    await uploadFormImage('book-1', 'Saber/Alter', 'battle form', formData)
    await generateFormOrtho('book-1', 'Saber/Alter', 'battle form', [file])
    await getAvailableImages('book-1', 'image')

    const formPath = '/api/manga-insight/book-1/continuation/characters/Saber%2FAlter/forms/battle%20form'
    expect(deleteMock).toHaveBeenCalledWith('/api/manga-insight/book-1/continuation/characters/Saber%2FAlter')
    expect(putMock).toHaveBeenCalledWith(`${formPath}`, { form_name: 'Battle' })
    expect(postMock).toHaveBeenCalledWith(`${formPath}/toggle`, { enabled: true })
    expect(uploadMock).toHaveBeenCalledWith(`${formPath}/image`, formData)
    expect(postMock).toHaveBeenCalledWith(
      `${formPath}/orthographic`,
      expect.any(FormData),
      {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 0,
      }
    )
    expect(getMock).toHaveBeenCalledWith('/api/manga-insight/book-1/continuation/available-images?mode=image')
  })

  it('routes continuation book endpoints through encoded book paths', async () => {
    const {
      clearContinuationData,
      getAvailableImages,
      prepareContinuation,
      saveConfig,
    } = await import('@/api/continuation')

    const bookId = 'book/id one'
    const base = '/api/manga-insight/book%2Fid%20one/continuation'

    await prepareContinuation(bookId)
    await saveConfig(bookId, {
      page_count: 3,
      style_reference_pages: 2,
      continuation_direction: 'forward',
    })
    await clearContinuationData(bookId)
    await getAvailableImages(bookId, 'image')

    expect(getMock).toHaveBeenNthCalledWith(1, `${base}/prepare`)
    expect(postMock).toHaveBeenNthCalledWith(1, `${base}/save-config`, {
      page_count: 3,
      style_reference_pages: 2,
      continuation_direction: 'forward',
    })
    expect(deleteMock).toHaveBeenNthCalledWith(1, `${base}/clear`)
    expect(getMock).toHaveBeenNthCalledWith(2, `${base}/available-images?mode=image`)
  })

  it('keeps generation payloads and long-running timeout contracts', async () => {
    const { generatePageImage, generateScriptWithRefs } = await import('@/api/continuation')
    const page = {
      page_number: 1,
      continuity_text: 'continuity',
      story_text: 'story',
      dialogue_text: 'dialogue',
      characters: ['Saber'],
      final_prompt: 'prompt',
      image_url: '',
      previous_url: '',
      status: 'pending' as const,
    }

    await generatePageImage('book-1', 2, page, ['style-a'], 'session-1', 4)
    await generateScriptWithRefs('book-1', 'go north', 3, ['ref-a'], 6)

    expect(postMock).toHaveBeenCalledWith(
      '/api/manga-insight/book-1/continuation/generate/2',
      {
        page,
        style_reference_tokens: ['style-a'],
        session_id: 'session-1',
        style_ref_count: 4,
      },
      { timeout: 0 }
    )
    expect(postMock).toHaveBeenCalledWith(
      '/api/manga-insight/book-1/continuation/script',
      {
        direction: 'go north',
        page_count: 3,
        reference_tokens: ['ref-a'],
        reference_image_count: 6,
      },
      { timeout: 0 }
    )
  })
})
