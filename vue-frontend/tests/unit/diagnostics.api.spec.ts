import { beforeEach, describe, expect, it, vi } from 'vitest'

const { postMock } = vi.hoisted(() => ({
  postMock: vi.fn(),
}))

vi.mock('@/api/client', () => ({
  apiClient: {
    post: postMock,
  },
}))

import {
  fetchModels,
  testAiTranslateConnection,
  testAiVisionOcrConnection,
  testBaiduOcrConnection,
} from '@/api/v2/diagnostics'

describe('v2 diagnostics api', () => {
  beforeEach(() => {
    postMock.mockReset()
    postMock.mockResolvedValue({ success: true, models: [] })
  })

  it('uses the unified model and connection diagnostic endpoints', async () => {
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
      secret: { api_key: 'model-key' },
    })
    expect(postMock).toHaveBeenNthCalledWith(
      2,
      '/api/v2/connection-tests/ai_vision_ocr',
      {
        provider: 'gemini',
        model: 'vision-model',
        baseUrl: 'https://vision.example.test',
        prompt: 'read text',
        secret: { ai_vision_api_key: 'vision-key' },
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
      {
        secret: {
          baidu_api_key: 'baidu-key',
          baidu_secret_key: 'baidu-secret',
        },
      },
    )
  })
})
