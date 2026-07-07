import { beforeEach, describe, expect, it, vi } from 'vitest'

const {
  getMock,
  postMock,
  uploadMock,
} = vi.hoisted(() => ({
  getMock: vi.fn(),
  postMock: vi.fn(),
  uploadMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  apiClient: {
    get: getMock,
    post: postMock,
    upload: uploadMock,
  },
}))

describe('config api', () => {
  beforeEach(() => {
    getMock.mockReset()
    postMock.mockReset()
    uploadMock.mockReset()
  })

  it('routes prompt endpoints through current prompt wire fields', async () => {
    const {
      deletePrompt,
      getPromptContent,
      getPrompts,
      resetPromptToDefault,
      savePrompt,
      saveTextboxPrompt,
    } = await import('@/api/config')

    await getPrompts('translate')
    await getPrompts()
    await getPromptContent('translate', 'default')
    await savePrompt('translate', 'default', 'prompt body')
    await deletePrompt('translate', 'default')
    await resetPromptToDefault('default')
    await saveTextboxPrompt('textbox', 'textbox body')

    expect(getMock).toHaveBeenNthCalledWith(1, '/api/get_prompts', {
      params: { type: 'translate' },
    })
    expect(getMock).toHaveBeenNthCalledWith(2, '/api/get_prompts', {
      params: {},
    })
    expect(getMock).toHaveBeenNthCalledWith(3, '/api/get_prompt_content', {
      params: {
        type: 'translate',
        prompt_name: 'default',
      },
    })
    expect(postMock).toHaveBeenNthCalledWith(1, '/api/save_prompt', {
      type: 'translate',
      prompt_name: 'default',
      prompt_content: 'prompt body',
    })
    expect(postMock).toHaveBeenNthCalledWith(2, '/api/delete_prompt', {
      type: 'translate',
      prompt_name: 'default',
    })
    expect(postMock).toHaveBeenNthCalledWith(3, '/api/reset_prompt_to_default', {
      prompt_name: 'default',
    })
    expect(postMock).toHaveBeenNthCalledWith(4, '/api/save_textbox_prompt', {
      prompt_name: 'textbox',
      prompt_content: 'textbox body',
    })
  })

  it('normalizes model provider ids and maps connection test payloads', async () => {
    const {
      fetchModels,
      testAiTranslateConnection,
      testAiVisionOcrConnection,
      testBaiduOcrConnection,
      testBaiduTranslateConnection,
      testYoudaoTranslateConnection,
    } = await import('@/api/config')

    await fetchModels(' OpenAI-Compatible ', 'model-key', 'https://api.example.test')
    await testAiVisionOcrConnection({
      provider: ' Gemini ',
      apiKey: 'vision-key',
      modelName: 'vision-model',
      customBaseUrl: 'https://vision.example.test',
      prompt: 'read text',
    })
    await testAiTranslateConnection({
      provider: ' DeepSeek ',
      apiKey: 'translate-key',
      modelName: 'chat-model',
      baseUrl: 'https://chat.example.test',
    })
    await testBaiduOcrConnection('baidu-key', 'baidu-secret')
    await testBaiduTranslateConnection('baidu-app', 'baidu-app-key')
    await testYoudaoTranslateConnection('youdao-key', 'youdao-secret')

    expect(postMock).toHaveBeenNthCalledWith(1, '/api/fetch_models', {
      provider: 'openai-compatible',
      api_key: 'model-key',
      base_url: 'https://api.example.test',
    })
    expect(postMock).toHaveBeenNthCalledWith(2, '/api/test_ai_vision_ocr', {
      provider: 'gemini',
      api_key: 'vision-key',
      model_name: 'vision-model',
      custom_ai_vision_base_url: 'https://vision.example.test',
      prompt: 'read text',
    })
    expect(postMock).toHaveBeenNthCalledWith(3, '/api/test_ai_translate_connection', {
      provider: 'deepseek',
      api_key: 'translate-key',
      model_name: 'chat-model',
      base_url: 'https://chat.example.test',
    })
    expect(postMock).toHaveBeenNthCalledWith(4, '/api/test_baidu_ocr_connection', {
      api_key: 'baidu-key',
      secret_key: 'baidu-secret',
    })
    expect(postMock).toHaveBeenNthCalledWith(5, '/api/test_baidu_translate_connection', {
      app_id: 'baidu-app',
      app_key: 'baidu-app-key',
    })
    expect(postMock).toHaveBeenNthCalledWith(6, '/api/test_youdao_translate', {
      appKey: 'youdao-key',
      appSecret: 'youdao-secret',
    })
  })

  it('routes font upload and user settings endpoints without leaking UI schema details', async () => {
    const {
      getFontList,
      getTextStyleDefaults,
      getUserSettings,
      saveTextStyleDefaults,
      saveTranslateWorkflowPreferences,
      saveUserSettings,
      uploadFont,
    } = await import('@/api/config')

    const file = new File(['font'], 'font.otf', { type: 'font/otf' })
    const defaults = {
      fontSize: 22,
    } as never

    await getFontList()
    await uploadFont(file)
    await getUserSettings()
    await getTextStyleDefaults()
    await saveTextStyleDefaults(defaults)
    await saveTranslateWorkflowPreferences({
      rememberWorkflowModeEnabled: true,
      lastWorkflowMode: 'parallel',
    })
    await saveUserSettings({ theme: 'dark' })

    expect(getMock).toHaveBeenNthCalledWith(1, '/api/get_font_list')
    expect(uploadMock).toHaveBeenCalledTimes(1)
    const [uploadUrl, formData] = uploadMock.mock.calls[0] || []
    expect(uploadUrl).toBe('/api/upload_font')
    expect(formData).toBeInstanceOf(FormData)
    expect((formData as FormData).get('font')).toBe(file)
    expect(getMock).toHaveBeenNthCalledWith(2, '/api/get_settings')
    expect(getMock).toHaveBeenNthCalledWith(3, '/api/config/text-style-defaults')
    expect(postMock).toHaveBeenNthCalledWith(1, '/api/config/text-style-defaults', {
      defaults,
    })
    expect(postMock).toHaveBeenNthCalledWith(2, '/api/config/translate-workflow-preferences', {
      rememberWorkflowModeEnabled: true,
      lastWorkflowMode: 'parallel',
    })
    expect(postMock).toHaveBeenNthCalledWith(3, '/api/save_settings', {
      settings: { theme: 'dark' },
    })
  })

  it('keeps configApi as the aggregate settings facade', async () => {
    const { configApi, fetchModels, saveUserSettings, testAiTranslateConnection } = await import('@/api/config')

    expect(configApi.fetchModels).toBe(fetchModels)
    expect(configApi.testAiTranslateConnection).toBe(testAiTranslateConnection)
    expect(configApi.saveUserSettings).toBe(saveUserSettings)
  })
})
