import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const { deleteMock, getMock, patchMock, postMock, putMock, uploadMock } = vi.hoisted(() => ({
  deleteMock: vi.fn(),
  getMock: vi.fn(),
  patchMock: vi.fn(),
  postMock: vi.fn(),
  putMock: vi.fn(),
  uploadMock: vi.fn(),
}))

vi.mock('@/api/client', () => {
  const apiClient = {
    delete: deleteMock,
    get: getMock,
    patch: patchMock,
    post: postMock,
    put: putMock,
    upload: uploadMock,
  }
  return { apiClient, default: apiClient }
})

const form = {
  adoptedAssetId: null,
  characterId: 'character-1',
  formId: 'form-1',
  imageVersions: [],
  name: '战斗形态',
  payload: { description: 'armor', enabled: true },
  referenceAssetId: 'reference-1',
  referenceAssetUrl: '/api/v2/assets/reference-1',
  referenceThumbnailUrl: '/api/v2/assets/reference-thumb-1',
  revision: 2,
}

const project = {
  bookId: 'book/id one',
  characters: [
    {
      aliases: ['Saber'],
      characterId: 'character-1',
      enabled: true,
      name: '阿尔托莉雅',
      payload: { description: '骑士王' },
      projectId: 'project-1',
      revision: 3,
    },
  ],
  config: {
    direction: 'forward',
    pageCount: 3,
    styleReferencePages: 2,
  },
  pages: [
    {
      continuationPageId: 'continuation-page-1',
      imageVersions: [
        {
          active: true,
          assetId: 'generated-1',
          assetUrl: '/api/v2/assets/generated-1',
          thumbnailUrl: '/api/v2/assets/generated-thumb-1',
          version: 1,
        },
      ],
      ordinal: 1,
      payload: {
        continuityText: 'continuity',
        storyText: 'story',
        dialogueText: 'dialogue',
        characters: ['阿尔托莉雅'],
        finalPrompt: 'prompt',
        status: 'ready',
      },
      revision: 4,
    },
  ],
  projectId: 'project-1',
  referenceAssets: [
    {
      assetId: 'reference-1',
      assetUrl: '/api/v2/assets/reference-1',
      thumbnailUrl: '/api/v2/assets/reference-thumb-1',
    },
  ],
  revision: 5,
  script: {
    content: 'script body',
    revision: 2,
    scriptId: 'script-1',
  },
  sourceRunId: 'run-1',
}

const state = {
  activeRunId: 'run-1',
  bookId: 'book/id one',
  missing: [],
  project,
  ready: true,
}

function installGetResponses() {
  getMock.mockImplementation((url: string, config?: { params?: { cursor?: number } }) => {
    if (url.endsWith('/continuation')) return Promise.resolve(state)
    if (url.includes('/continuation/projects/project-1/forms')) {
      return Promise.resolve({ items: [form], nextCursor: null })
    }
    if (url.endsWith('/chapters')) {
      return Promise.resolve({
        items: [{
          chapterId: 'chapter-1',
          ordinal: 1,
          pageCount: 7,
          title: '第1章',
        }],
        nextCursor: null,
      })
    }
    if (url.endsWith('/pages') && config?.params?.cursor === 0) {
      return Promise.resolve({
        items: [
          {
            activeAnalysisId: null,
            analysisState: 'ready',
            chapterId: 'chapter-1',
            displayPageNumber: 7,
            pageId: 'page-7',
            sourceAssetId: 'source-7',
            thumbnailUrl: '/api/v2/assets/source-thumb-7',
          },
        ],
        nextCursor: null,
      })
    }
    throw new Error(`Unexpected GET ${url}`)
  })
}

describe('continuation v2 api facade', () => {
  beforeEach(() => {
    vi.resetModules()
    deleteMock.mockReset()
    getMock.mockReset()
    patchMock.mockReset()
    postMock.mockReset()
    putMock.mockReset()
    uploadMock.mockReset()
    installGetResponses()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('hydrates the existing workflow shape from backend-owned v2 state', async () => {
    const { getCharacters, prepareContinuation } = await import('@/api/continuation')

    const prepared = await prepareContinuation('book/id one')
    const characters = await getCharacters('book/id one')

    expect(getMock).toHaveBeenCalledWith('/api/v2/insight/books/book%2Fid%20one/continuation')
    expect(prepared.saved_data).toMatchObject({
      script: { script_text: 'script body', page_count: 3 },
      pages: [
        {
          page_number: 1,
          image_url: '/api/v2/assets/generated-1',
          status: 'generated',
        },
      ],
      config: {
        page_count: 3,
        style_reference_pages: 2,
        continuation_direction: 'forward',
      },
    })
    expect(characters.items[0]).toMatchObject({
      name: '阿尔托莉雅',
      description: '骑士王',
      forms: [
        {
          form_id: 'form-1',
          reference_image: '/api/v2/assets/reference-1',
        },
      ],
    })
  })

  it('preserves an existing image while exposing script-invalidated pages as stale', async () => {
    const staleState = {
      ...state,
      project: {
        ...project,
        pages: project.pages.map(page => ({
          ...page,
          payload: { ...page.payload, staleReason: 'script_changed' },
          revision: page.revision + 1,
        })),
      },
    }
    getMock
      .mockResolvedValueOnce(staleState)
      .mockResolvedValueOnce({ items: [form], nextCursor: null })
    const { prepareContinuation } = await import('@/api/continuation')

    const prepared = await prepareContinuation('book/id one')

    expect(prepared.saved_data.pages).toEqual([
      expect.objectContaining({
        image_url: '/api/v2/assets/generated-1',
        page_number: 1,
        status: 'stale',
        story_text: 'story',
      }),
    ])
  })

  it('keeps ready page stories generated when no image exists yet', async () => {
    getMock.mockResolvedValueOnce({
      ...state,
      project: {
        ...project,
        pages: project.pages.map(page => ({
          ...page,
          imageVersions: [],
        })),
      },
    })
    const { prepareContinuation } = await import('@/api/continuation')

    const prepared = await prepareContinuation('book/id one')

    expect(prepared.saved_data.pages).toEqual([
      expect.objectContaining({
        image_url: '',
        page_number: 1,
        status: 'generated',
      }),
    ])
  })

  it('loads character forms one cursor page at a time', async () => {
    const secondForm = {
      ...form,
      formId: 'form-2',
      name: '礼服形态',
      referenceAssetId: null,
      referenceAssetUrl: null,
      referenceThumbnailUrl: null,
    }
    getMock.mockImplementation((url: string, config?: { params?: { cursor?: number } }) => {
      if (url.endsWith('/continuation')) return Promise.resolve(state)
      if (url.includes('/continuation/projects/project-1/forms')) {
        return config?.params?.cursor === 100
          ? Promise.resolve({ items: [secondForm], nextCursor: null })
          : Promise.resolve({ items: [form], nextCursor: 100 })
      }
      throw new Error(`Unexpected GET ${url}`)
    })
    const { getCharacters } = await import('@/api/continuation')

    const firstPage = await getCharacters('book/id one')

    expect(firstPage.items[0]?.forms.map(item => item.form_id)).toEqual(['form-1'])
    expect(firstPage.nextCursor).toBe(100)
    expect(getMock).toHaveBeenCalledWith(
      '/api/v2/insight/continuation/projects/project-1/forms',
      { params: { cursor: 0, limit: 100 } },
    )
    expect(getMock).not.toHaveBeenCalledWith(
      '/api/v2/insight/continuation/projects/project-1/forms',
      { params: { cursor: 100, limit: 100 } },
    )

    const secondPage = await getCharacters('book/id one', firstPage.nextCursor!)

    expect(secondPage.items[0]?.forms.map(item => item.form_id)).toEqual(['form-2'])
    expect(secondPage.nextCursor).toBeNull()
    expect(getMock).toHaveBeenCalledWith(
      '/api/v2/insight/continuation/projects/project-1/forms',
      { params: { cursor: 100, limit: 100 } },
    )
  })

  it('reads the current project revision before every mutation', async () => {
    getMock
      .mockResolvedValueOnce(state)
      .mockResolvedValueOnce({
        ...state,
        project: { ...project, revision: 6 },
      })
    patchMock
      .mockResolvedValueOnce({ ...project, revision: 6 })
      .mockResolvedValueOnce({ ...project, revision: 7 })
    const { saveConfig } = await import('@/api/continuation')

    const config = {
      page_count: 3,
      style_reference_pages: 2,
      continuation_direction: 'forward',
    }
    await saveConfig('book/id one', config)
    await saveConfig('book/id one', config)

    expect(getMock).toHaveBeenCalledTimes(2)
    expect(patchMock.mock.calls.map(call => call[1].baseRevision)).toEqual([5, 6])
  })

  it('loads a bounded original-page summary without touching continuation data', async () => {
    const { getOriginalReferenceImages } = await import('@/api/continuation')

    await expect(getOriginalReferenceImages('book/id one')).resolves.toMatchObject({
      original_images: [
        {
          token: 'source-7',
          page_number: 7,
          path: '/api/v2/assets/source-thumb-7',
          has_image: true,
        },
      ],
      original_cursor: 0,
    })
    expect(getMock).toHaveBeenCalledWith(
      '/api/v2/insight/books/book%2Fid%20one/chapters',
    )
    expect(getMock).toHaveBeenCalledWith('/api/v2/insight/books/book%2Fid%20one/pages', {
      params: { cursor: 0, limit: 100 },
    })
    expect(getMock).not.toHaveBeenCalledWith(
      '/api/v2/insight/books/book%2Fid%20one/continuation',
    )
    expect(getMock).not.toHaveBeenCalledWith(
      '/api/v2/insight/continuation/projects/project-1/forms',
      expect.anything(),
    )
    expect(getMock).not.toHaveBeenCalledWith('/api/v2/assets/source-7')
  })

  it('loads continuation images and the first form cursor page for image generation', async () => {
    const { getAvailableImages } = await import('@/api/continuation')

    await expect(getAvailableImages('book/id one')).resolves.toMatchObject({
      continuation_images: [{ token: 'generated-1', page_number: 1 }],
      character_forms: [{ token: 'reference-1', form_id: 'form-1' }],
      character_forms_cursor: null,
    })
  })

  it('uses stable IDs and CAS revisions for character and form mutations', async () => {
    deleteMock.mockResolvedValue({ deleted: true })
    patchMock.mockResolvedValue({
      ...form,
      name: '更新形态',
      revision: 3,
    })
    const { deleteCharacter, updateCharacterForm } = await import('@/api/continuation')

    await updateCharacterForm('book/id one', '阿尔托莉雅', 'form-1', {
      form_name: '更新形态',
      description: 'updated',
    })
    await deleteCharacter('book/id one', '阿尔托莉雅')

    expect(patchMock).toHaveBeenCalledWith(
      '/api/v2/insight/continuation/forms/form-1',
      {
        baseRevision: 2,
        name: '更新形态',
        payload: {
          description: 'updated',
          enabled: true,
        },
      },
      { headers: { 'Idempotency-Key': expect.any(String) } }
    )
    expect(deleteMock).toHaveBeenCalledWith(
      '/api/v2/insight/continuation/characters/character-1?baseRevision=3',
      { headers: { 'Idempotency-Key': expect.any(String) } }
    )
  })

  it('uploads exactly the selected reference image before creating a character-sheet job', async () => {
    const sourceImage = new File(['reference'], 'reference.png', { type: 'image/png' })
    uploadMock.mockImplementation((_url: string, body: FormData) => {
      expect(body.get('file')).toBe(sourceImage)
      expect(body.get('baseRevision')).toBe('2')
      return Promise.resolve({ ...form, revision: 3 })
    })
    postMock.mockResolvedValue({
      batchId: 'ortho-batch',
      jobIds: ['ortho-job'],
      status: 'queued',
    })
    const { generateFormOrtho } = await import('@/api/continuation')

    await expect(
      generateFormOrtho('book/id one', '阿尔托莉雅', 'form-1', sourceImage),
    ).resolves.toBe('ortho-job')

    expect(uploadMock).toHaveBeenCalledWith(
      '/api/v2/insight/continuation/forms/form-1/reference',
      expect.any(FormData),
      { headers: { 'Idempotency-Key': expect.any(String) } },
    )
    expect(postMock).toHaveBeenCalledWith(
      '/api/v2/insight/books/book%2Fid%20one/continuation/jobs',
      { kind: 'character_sheet', formId: 'form-1' },
      { headers: { 'Idempotency-Key': expect.any(String) } },
    )
  })

  it('submits script and export work as durable jobs and downloads v2 artifacts', async () => {
    const updatedProject = {
      ...project,
      config: { ...project.config, direction: 'north' },
      revision: 6,
    }
    patchMock.mockResolvedValueOnce(updatedProject)
    putMock.mockResolvedValueOnce({ ...updatedProject, revision: 7 })
    postMock
      .mockResolvedValueOnce({ batchId: 'batch-1', jobIds: ['script-job'], status: 'queued' })
      .mockResolvedValueOnce({ batchId: 'batch-2', jobIds: ['export-job'], status: 'queued' })
    const blob = new Blob(['zip'], { type: 'application/zip' })
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      headers: { get: vi.fn().mockReturnValue(null) },
      blob: vi.fn().mockResolvedValue(blob),
    })
    vi.stubGlobal('fetch', fetchMock)
    const { createContinuationExportJob, downloadContinuationExport, generateScriptWithRefs } =
      await import('@/api/continuation')

    const script = await generateScriptWithRefs('book/id one', 'north', 3, ['reference-1'], 5)
    const exportJobId = await createContinuationExportJob('book/id one', 'zip')
    await expect(downloadContinuationExport('export-asset', 'book/id one', 'zip')).resolves.toBe(
      blob
    )

    expect(script).toBe('script-job')
    expect(exportJobId).toBe('export-job')
    expect(patchMock).toHaveBeenCalledWith(
      '/api/v2/insight/continuation/projects/project-1',
      {
        baseRevision: 5,
        config: {
          direction: 'north',
          pageCount: 3,
          styleReferencePages: 5,
        },
      },
      { headers: { 'Idempotency-Key': expect.any(String) } }
    )
    expect(putMock).toHaveBeenCalledWith(
      '/api/v2/insight/continuation/projects/project-1/references',
      { baseRevision: 6, assetIds: ['reference-1'] },
      { headers: { 'Idempotency-Key': expect.any(String) } }
    )
    expect(postMock).toHaveBeenNthCalledWith(
      1,
      '/api/v2/insight/books/book%2Fid%20one/continuation/jobs',
      { kind: 'script' },
      { headers: { 'Idempotency-Key': expect.any(String) } }
    )
    expect(postMock).toHaveBeenNthCalledWith(
      2,
      '/api/v2/insight/books/book%2Fid%20one/continuation/jobs',
      { kind: 'export', format: 'zip' },
      { headers: { 'Idempotency-Key': expect.any(String) } }
    )
    expect(fetchMock).toHaveBeenCalledWith('/api/v2/assets/export-asset')
  })
})
