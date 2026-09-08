// @vitest-environment jsdom
import { nextTick } from 'vue'
import { afterEach, expect, it, vi } from 'vitest'
import factoryStyle from '../../src/shared/text_style_defaults_factory.json'



afterEach(() => vi.unstubAllGlobals())

it('edits only plugin text style and the optional DOM assistant, preserving drafts across tabs', async () => {
  document.body.innerHTML =
    '<div id="app"></div>'
  Element.prototype.scrollTo = vi.fn()
  location.hash = 'settings'
  const settings = {
    settings: [
      {
        domain: 'text_style_defaults',
        revision: 1,
        schemaVersion: 2,
        payload: {
          ...factoryStyle,
          fontFamily: 'font',
          autoFontSize: true,
          useAutoTextColor: false,
          inpaintMethod: 'solid',
          strokeEnabled: true,
        },
      },
      {
        domain: 'browser_dom_agent',
        revision: 0,
        schemaVersion: 1,
        payload: {
          provider: 'ollama',
          modelName: 'agent-model',
          customBaseUrl: '',
          openaiOptions: {
            request: { forceJsonOutput: false },
            execution: {
              useStream: false,
              rpmLimit: 0,
              businessRetries: 1,
              transportRetries: 1,
            },
          },
        },
      },
    ],
    credentials: [],
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
        const transaction = JSON.parse(String(init.body))
        transactions.push(transaction)
        return Response.json({
          settings: transaction.settings.map((row: { domain: string; baseRevision: number }) => ({ domain: row.domain, revision: row.baseRevision + 1 })),
          providerSettings: transaction.providerSettings.map((row: { domain: string; provider: string; baseRevision: number }) => ({ domain: row.domain, provider: row.provider, revision: row.baseRevision + 1 })),
          credentials: transaction.credentialEdits.map((row: { domain: string; provider: string; secret: unknown }) => ({ ...row, credentialId: 'key', credentialVersionId: 'version', revision: 1, currentVersion: 1, hasKey: true })),
          bookSettings: [], prompts: [],
        })
      }
      if (url.includes('/settings?')) {
        expect(new URL(url).searchParams.get('domains')).toBe(
          'text_style_defaults,browser_dom_agent',
        )
        return Response.json(settings)
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
  vi.stubGlobal('matchMedia', () => ({ matches: false, addEventListener: vi.fn() }))
  vi.stubGlobal('IntersectionObserver', class { observe() {} disconnect() {} })
  await import('./panel')
  const input = (label: string) => {
    const node = document.querySelector<HTMLInputElement | HTMLSelectElement>(`[aria-label="${label}"]`)
    const fieldLabel = [...document.querySelectorAll('label')].find(node => node.textContent?.trim() === label)
    return (node ?? fieldLabel?.querySelector<HTMLInputElement>('input') ?? null)!
  }
  const edit = async (label: string, value: string) => {
    const node = input(label)
    if (node.tagName === 'BUTTON') {
      node.click()
      await nextTick()
      document.querySelector<HTMLElement>(`[data-value="${value}"]`)!.click()
      await nextTick()
      return
    }
    node.value = value
    node.dispatchEvent(new Event('input'))
    node.dispatchEvent(new Event('change'))
    await nextTick()
  }
  const toggle = async (label: string, checked: boolean) => {
    const node = input(label) as HTMLInputElement
    node.checked = checked
    node.dispatchEvent(new Event('change'))
    await nextTick()
  }
  await vi.waitFor(() => expect(input('文本字体')).not.toBeNull())
  expect(transactions).toHaveLength(0)
  expect(document.body.textContent).not.toContain('保存插件配置')
  expect(document.body.textContent).not.toContain('重新读取')
  for (const label of [
    '目标语言',
    'OCR 引擎',
    '文本检测',
    '并行翻译',
    '普通翻译提示词',
  ])
    expect(input(label)).toBeNull()
  expect(input('字号').disabled).toBe(true)
  await toggle('自动计算初始字号', false)
  expect(input('字号').disabled).toBe(false)
  await toggle('自动识别文字颜色', true)
  expect(input('文字颜色').disabled).toBe(true)
  await edit('气泡填充方式', 'lama_mpe')
  expect(input('填充颜色')).toBeNull()
  await toggle('启用描边', false)
  expect(input('描边颜色')).toBeNull()
  await toggle('启用描边', true)
  await edit('描边宽度 (px)', '1.2')
  await edit('行内对齐', 'center')
  await edit('文本块对齐', 'end')
  location.hash = 'tasks'
  await vi.waitFor(() =>
    expect(document.body.textContent).toContain('暂无任务'),
  )
  location.hash = 'settings'
  await vi.waitFor(() => expect(input('描边宽度 (px)')?.value).toBe('1.2'))
  await edit('识别助手服务商', 'siliconflow')
  await edit('模型名称', 'new-agent-model')
  await edit('API Key', 'test-agent-only-secret')
  await vi.waitFor(() => expect(transactions).toHaveLength(1))
  expect(transactions[0]).toMatchObject({
    settings: [
      {
        domain: 'text_style_defaults',
        baseRevision: 1,
        payload: { strokeWidth: 1.2, inlineAlign: 'center', blockAlign: 'end' },
      },
      {
        domain: 'browser_dom_agent',
        baseRevision: 0,
        payload: { provider: 'siliconflow' },
      },
    ],
    providerSettings: [
      {
        domain: 'browser_dom_agent',
        provider: 'siliconflow',
        payload: { modelName: 'new-agent-model' },
        credentialEditRef: 'siliconflow',
      },
    ],
    credentialEdits: [
      {
        domain: 'browser_dom_agent',
        provider: 'siliconflow',
        secret: { api_key: 'test-agent-only-secret' },
      },
    ],
  })
  await vi.waitFor(() => expect(input('API Key')?.value).toBe('test-agent-only-secret'))
  await edit('模型名称', 'updated-model')
  await vi.waitFor(() => expect(transactions).toHaveLength(2))
  expect(transactions[1]).toMatchObject({ settings: [], credentialEdits: [], providerSettings: [{ baseRevision: 1, credentialVersionId: 'version', payload: { modelName: 'updated-model' } }] })
})
