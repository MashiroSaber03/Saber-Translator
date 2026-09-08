// @vitest-environment jsdom
import { createApp, nextTick, reactive, type App } from 'vue'
import { afterEach, expect, it, vi } from 'vitest'
import ServiceFields from './ServiceFields.vue'
let app: App
const config = {
  provider: 'siliconflow',
  modelName: 'test-model',
  customBaseUrl: 'https://example.invalid/v1',
  openaiOptions: {
    request: { forceJsonOutput: false },
    execution: { useStream: false, rpmLimit: 0, businessRetries: 1, transportRetries: 1 },
  },
}
async function mount(overrides: Record<string, unknown> = {}) {
  document.body.innerHTML = '<div id="app"></div>'
  const api = vi.fn().mockResolvedValue({ models: [], success: true, message: '测试成功' })
  app = createApp(ServiceFields, {
    config,
    domain: 'translation',
    capability: 'translation',
    api,
    secret: '',
    configured: true,
    ...overrides,
  })
  app.mount('#app')
  await nextTick()
  return api
}
const click = (text: string) =>
  [...document.querySelectorAll('button')].find(b => b.textContent?.trim() === text)!.click()
afterEach(() => app.unmount())
it('sends only model-catalog fields, letting the backend use the saved key', async () => {
  const api = await mount()
  click('获取模型列表')
  await nextTick()
  expect(api).toHaveBeenCalledWith('/model-catalog', 'POST', {
    provider: 'siliconflow',
    domain: 'translation',
    baseUrl: 'https://example.invalid/v1',
  })
})
it('uses the OCR diagnostic and the OCR credential field', async () => {
  const api = await mount({
    domain: 'ai_vision_ocr',
    capability: 'visionOcr',
    secret: 'vision-test',
    config: { ...config, prompt: 'Read text' },
  })
  click('测试连接')
  await nextTick()
  expect(api).toHaveBeenCalledWith(
    '/connection-tests/ai_vision_ocr',
    'POST',
    expect.objectContaining({
      domain: 'ai_vision_ocr',
      model: 'test-model',
      prompt: 'Read text',
      secret: { ai_vision_api_key: 'vision-test' },
    })
  )
})
it('maps saved traditional translation credentials to the corresponding diagnostic', async () => {
  const api = await mount({
    config: { ...config, provider: 'baidu_translate', modelName: 'app-secret' },
    savedSecret: 'app-id',
  })
  click('测试连接')
  await nextTick()
  expect(api).toHaveBeenCalledWith('/connection-tests/baidu_translate', 'POST', {
    domain: 'translation',
    secret: { app_id: 'app-id', app_key: 'app-secret' },
  })
})

it('ignores a failed diagnostic after switching away from its provider', async () => {
  const draft = reactive({ ...config })
  const api = await mount({ config: draft })
  let reject!: (error: Error) => void
  api.mockImplementation(() => new Promise((_resolve, fail) => { reject = fail }))
  click('测试连接')
  draft.provider = 'ollama'
  await nextTick()
  reject(new Error('上一个服务商无法连接'))
  await new Promise(resolve => setTimeout(resolve, 0))
  expect(document.body.textContent).not.toContain('上一个服务商无法连接')
})
