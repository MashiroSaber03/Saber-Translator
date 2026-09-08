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
    ...overrides,
  })
  app.mount('#app')
  await nextTick()
  return api
}
const click = (text: string) =>
  [...document.querySelectorAll('button')].find(b => b.textContent?.trim() === text)!.click()
afterEach(() => app.unmount())
it('hides unsupported controls for traditional translators while retaining retry settings', async () => {
  const draft = reactive({ ...config, provider: 'caiyun' })
  await mount({ config: draft })
  expect(document.body.textContent).not.toContain('模型名称')
  expect(document.body.textContent).not.toContain('源语言 (可选)')
  expect(document.body.textContent).not.toContain('RPM 限制')
  expect(document.body.textContent).not.toContain('流式调用')
  expect(document.body.textContent).not.toContain('附加请求参数')
  expect(document.body.textContent).toContain('业务重试次数')
  draft.provider = 'baidu_translate'
  await nextTick()
  expect(document.body.textContent).toContain('App Key')
  draft.provider = 'siliconflow'
  await nextTick()
  expect(document.body.textContent).toContain('模型名称')
  expect(document.body.textContent).toContain('流式调用')
  expect(document.body.textContent).toContain('附加请求参数')
})
it('shows the current key and only exposes the URL for a custom provider', async () => {
  const draft = reactive({ ...config, provider: 'deepseek' })
  await mount({ config: draft, secret: 'test-current-key' })
  const input = document.querySelector<HTMLInputElement>('input[aria-label="API Key"]')!
  expect(input.value).toBe('test-current-key')
  expect(input.type).toBe('password')
  click('显示')
  await nextTick()
  expect(input.type).toBe('text')
  expect(document.body.textContent).not.toContain('留空')
  expect(document.body.textContent).not.toContain('API 地址')
  draft.provider = 'custom'
  await nextTick()
  expect(document.body.textContent).toContain('API 地址')
  draft.provider = 'gemini'
  await nextTick()
  expect(document.body.textContent).not.toContain('API 地址')
})
it('sends only model-catalog fields with the current key', async () => {
  const api = await mount()
  click('获取模型列表')
  await nextTick()
  expect(api).toHaveBeenCalledWith('/model-catalog', 'POST', {
    provider: 'siliconflow',
    domain: 'translation',
    baseUrl: 'https://example.invalid/v1',
    secret: { api_key: '' },
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
    secret: 'app-id',
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
