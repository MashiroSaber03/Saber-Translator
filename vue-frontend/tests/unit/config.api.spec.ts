import { beforeEach, describe, expect, it, vi } from 'vitest'

const {
  deleteMock,
  getMock,
  patchMock,
  postMock,
  putMock,
  uploadMock,
} = vi.hoisted(() => ({
  deleteMock: vi.fn(),
  getMock: vi.fn(),
  patchMock: vi.fn(),
  postMock: vi.fn(),
  putMock: vi.fn(),
  uploadMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  apiClient: {
    delete: deleteMock,
    get: getMock,
    patch: patchMock,
    post: postMock,
    put: putMock,
    upload: uploadMock,
  },
}))

const idempotencyHeaders = {
  headers: { 'Idempotency-Key': expect.any(String) },
}

describe('config api v2 facade', () => {
  beforeEach(() => {
    vi.resetModules()
    deleteMock.mockReset()
    getMock.mockReset()
    patchMock.mockReset()
    postMock.mockReset()
    putMock.mockReset()
    uploadMock.mockReset()
  })

  it('adapts name-based prompt UI calls to UUID/revision v2 CRUD', async () => {
    const defaultPrompt = {
      id: 'prompt-default',
      type: 'translate',
      name: 'default',
      content: 'factory',
      revision: 4,
      isFactoryDefault: true,
    }
    getMock.mockResolvedValue({ items: [defaultPrompt] })
    putMock.mockResolvedValue({ ...defaultPrompt, content: 'updated', revision: 5 })
    postMock.mockResolvedValue({ ...defaultPrompt, content: 'factory', revision: 6 })
    deleteMock.mockResolvedValue({ deleted: true })

    const {
      getPromptContent,
      getPrompts,
      resetPromptToDefault,
      savePrompt,
    } = await import('@/api/config')

    await expect(getPrompts('translate')).resolves.toEqual({
      success: true,
      prompt_names: ['default'],
      default_prompt_content: 'factory',
    })
    await expect(getPromptContent('translate', 'default')).resolves.toEqual({
      success: true,
      prompt_content: 'factory',
    })
    await savePrompt('translate', 'default', 'updated')
    await resetPromptToDefault('default')

    expect(getMock).toHaveBeenCalledWith('/api/v2/prompts', {
      params: { type: 'translate' },
    })
    expect(putMock).toHaveBeenCalledWith(
      '/api/v2/prompts/prompt-default',
      { name: 'default', content: 'updated', baseRevision: 4 },
      idempotencyHeaders,
    )
    expect(postMock).toHaveBeenCalledWith(
      '/api/v2/prompts/prompt-default/reset',
      { baseRevision: 4 },
      idempotencyHeaders,
    )
  })

  it('uses unified v2 model and connection diagnostics without old endpoints', async () => {
    postMock.mockResolvedValue({ success: true, models: [] })
    const {
      fetchModels,
      testAiTranslateConnection,
      testAiVisionOcrConnection,
      testBaiduOcrConnection,
    } = await import('@/api/config')

    await fetchModels(' OpenAI ', 'model-key', 'https://api.example.test')
    await testAiVisionOcrConnection({
      provider: ' Gemini ',
      apiKey: 'vision-key',
      modelName: 'vision-model',
      customBaseUrl: 'https://vision.example.test',
      prompt: 'read text',
    })
    await testAiTranslateConnection({
      provider: ' DeepSeek ',
      apiKey: '',
      modelName: 'chat-model',
      domain: 'hq',
    })
    await testBaiduOcrConnection('baidu-key', 'baidu-secret')

    expect(postMock).toHaveBeenNthCalledWith(1, '/api/v2/model-catalog', {
      provider: 'openai',
      baseUrl: 'https://api.example.test',
      secret: { apiKey: 'model-key' },
    })
    expect(postMock).toHaveBeenNthCalledWith(
      2,
      '/api/v2/connection-tests/ai_vision_ocr',
      {
        provider: 'gemini',
        model: 'vision-model',
        baseUrl: 'https://vision.example.test',
        prompt: 'read text',
        secret: { apiKey: 'vision-key' },
      },
    )
    expect(postMock).toHaveBeenNthCalledWith(
      3,
      '/api/v2/connection-tests/ai_translate',
      {
        provider: 'deepseek',
        model: 'chat-model',
        baseUrl: undefined,
        domain: 'hq',
      },
    )
    expect(postMock).toHaveBeenNthCalledWith(
      4,
      '/api/v2/connection-tests/baidu_ocr',
      { secret: { apiKey: 'baidu-key', secretKey: 'baidu-secret' } },
    )
  })

  it('adapts fonts and settings to backend-owned v2 resources', async () => {
    getMock.mockImplementation((url: string) => {
      if (url === '/api/v2/fonts') {
        return Promise.resolve({
          items: [{
            id: 'font-id',
            kind: 'uploaded',
            displayName: 'Comic Font',
            builtinKey: null,
            assetUrl: '/api/v2/assets/font-asset',
          }],
        })
      }
      return Promise.resolve({
        settings: [{
          domain: 'translation',
          payload: { mode: 'standard' },
          revision: 3,
          schemaVersion: 3,
        }],
        bookSettings: [],
        providerSettings: [],
        credentials: [],
      })
    })
    uploadMock.mockResolvedValue({ id: 'uploaded-font-id', assetUrl: '/api/v2/assets/a' })
    putMock.mockResolvedValue({
      settings: [{ domain: 'translation', revision: 4 }],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
    })

    const {
      getFontList,
      getUserSettings,
      saveUserSettings,
      uploadFont,
    } = await import('@/api/config')
    const file = new File(['font'], 'comic.otf', { type: 'font/otf' })

    await expect(getFontList()).resolves.toMatchObject({
      success: true,
      fonts: [{ id: 'font-id', path: 'font-id', kind: 'uploaded' }],
    })
    await expect(uploadFont(file)).resolves.toEqual({
      success: true,
      fontPath: 'uploaded-font-id',
    })
    await expect(getUserSettings()).resolves.toEqual({
      success: true,
      settings: { mode: 'standard' },
    })
    await saveUserSettings({ mode: 'hq' })

    const [, form, uploadOptions] = uploadMock.mock.calls[0]!
    expect(uploadMock.mock.calls[0]![0]).toBe('/api/v2/fonts')
    expect((form as FormData).get('file')).toBe(file)
    expect(uploadOptions).toEqual(idempotencyHeaders)
    expect(putMock).toHaveBeenCalledWith(
      '/api/v2/settings/transactions',
      {
        settings: [{
          domain: 'translation',
          payload: { mode: 'hq' },
          baseRevision: 3,
          schemaVersion: 3,
        }],
      },
      idempotencyHeaders,
    )
  })

  it('keeps configApi as the aggregate settings facade', async () => {
    const {
      configApi,
      fetchModels,
      saveUserSettings,
      testAiTranslateConnection,
    } = await import('@/api/config')

    expect(configApi.fetchModels).toBe(fetchModels)
    expect(configApi.testAiTranslateConnection).toBe(testAiTranslateConnection)
    expect(configApi.saveUserSettings).toBe(saveUserSettings)
  })
})
