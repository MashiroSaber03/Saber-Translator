// @vitest-environment jsdom
import { afterEach, expect, it, vi } from 'vitest'

afterEach(() => vi.unstubAllGlobals())

it('preserves an unsaved draft across task navigation and saves provider credentials atomically', async () => {
  document.body.innerHTML =
    '<button id="tasks-tab"></button><button id="settings-tab"></button><p id="notice"></p><main id="content"></main>'
  location.hash = 'settings'
  const provider = () => ({
    provider: 'ollama',
    modelName: 'existing-model',
    customBaseUrl: '',
    openaiOptions: {
      execution: {
        useStream: false,
        rpmLimit: 0,
        businessRetries: 1,
        transportRetries: 1,
      },
      request: { forceJsonOutput: false },
    },
  })
  const settings = {
    settings: [
      {
        domain: 'translation',
        revision: 3,
        schemaVersion: 9,
        payload: {
          targetLanguage: 'zh',
          ocrEngine: 'manga_ocr',
          textDetector: 'default',
          translation: { ...provider(), translationMode: 'batch' },
          hqTranslation: { ...provider(), prompt: 'hq', batchSize: 3 },
          aiVisionOcr: {
            ...provider(),
            prompt: 'ocr',
            promptMode: 'normal',
            minImageSize: 32,
          },
          browserDomAgent: provider(),
          baiduOcr: { version: 'standard', sourceLanguage: 'JAP' },
          hybridOcr: { enabled: false },
          parallel: { enabled: false, deepLearningLockSize: 1 },
        },
      },
      {
        domain: 'text_style_defaults',
        revision: 1,
        schemaVersion: 2,
        payload: { fontFamily: 'font', strokeWidth: 1.2 },
      },
    ],
    credentials: [
      {
        domain: 'ocr',
        provider: 'baidu',
        credentialId: 'baidu-key',
        credentialVersionId: 'baidu-version',
        revision: 2,
        secret: {
          baidu_api_key: 'old-key',
          baidu_secret_key: 'keep-this-secret',
        },
      },
    ],
    providerSettings: [],
    bookSettings: [],
  }
  vi.stubGlobal('chrome', {
    storage: {
      local: {
        get: async () => ({
          'saber-extension-settings-v1': {
            token: 'test-pairing-token',
            serverPort: 5000,
            domains: {},
          },
        }),
      },
    },
  })
  const transactions: Array<Record<string, unknown>> = []
  vi.stubGlobal(
    'fetch',
    vi.fn(async (url: string, init: RequestInit) => {
      if (url.endsWith('/settings/transactions')) {
        transactions.push(JSON.parse(String(init.body)))
        return Response.json({ saved: true })
      }
      if (url.includes('/settings?')) {
        const domains = new URL(url).searchParams.get('domains')!.split(',')
        return Response.json({ ...settings, credentials: settings.credentials.filter(row => domains.includes(row.domain)) })
      }
      if (url.endsWith('/fonts'))
        return Response.json({
          items: [{ id: 'font', displayName: '测试字体' }],
        })
      if (url.includes('/jobs?'))
        return Response.json({
          items: [],
          workerOnline: true,
          queuePaused: false,
        })
      throw new Error(`Unexpected endpoint ${url}`)
    }),
  )
  await import('./panel')
  const input = (label: string) =>
    document.querySelector<HTMLInputElement | HTMLSelectElement>(
      `[aria-label="${label}"]`,
    )!
  const edit = (node: HTMLInputElement | HTMLSelectElement, value: string) => {
    node.value = value
    node.dispatchEvent(new Event('input'))
  }
  await vi.waitFor(() => expect(input('目标语言')).not.toBeNull())
  edit(input('目标语言'), 'en')
  location.hash = 'tasks'
  await vi.waitFor(() =>
    expect(document.body.textContent).toContain('暂无任务'),
  )
  location.hash = 'settings'
  await vi.waitFor(() => expect(input('目标语言')?.value).toBe('en'))
  edit(input('服务商'), 'siliconflow')
  edit(input('模型名称'), 'new-model')
  edit(input('API Key'), 'test-only-secret')
  edit(input('API Key（已配置时留空保持）'), 'new-baidu-key')
  const save = [...document.querySelectorAll('button')].find(
    (node) => node.textContent === '保存插件配置',
  )!
  save.click()
  await vi.waitFor(() => expect(transactions).toHaveLength(1))
  expect(transactions[0]).toMatchObject({
    settings: [
      {
        domain: 'translation',
        baseRevision: 3,
        payload: {
          targetLanguage: 'en',
          translation: { provider: 'siliconflow' },
        },
      },
    ],
    providerSettings: [
      {
        domain: 'translation',
        provider: 'siliconflow',
        payload: { modelName: 'new-model' },
        credentialEditRef: 'translation:siliconflow',
      },
      { domain: 'ocr', provider: 'baidu', credentialEditRef: 'ocr:baidu' },
    ],
    credentialEdits: [
      {
        domain: 'translation',
        provider: 'siliconflow',
        secret: { api_key: 'test-only-secret' },
      },
      {
        domain: 'ocr',
        provider: 'baidu',
        baseRevision: 2,
        secret: {
          baidu_api_key: 'new-baidu-key',
          baidu_secret_key: 'keep-this-secret',
        },
      },
    ],
  })
  await vi.waitFor(() =>
    expect(document.getElementById('notice')?.textContent).toContain(
      '插件配置已保存',
    ),
  )
})
