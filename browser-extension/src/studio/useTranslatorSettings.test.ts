// @vitest-environment jsdom
import { createApp, h, ref, type App } from 'vue'
import { afterEach, expect, it, vi } from 'vitest'
import { useTranslatorSettings } from './useTranslatorSettings'
import type { PluginSettingsApi } from '../../../vue-frontend/src/types/browserExtensionSettings'
let app: App
let state: ReturnType<typeof useTranslatorSettings>
const active = ref(true)
const options = {
  request: { forceJsonOutput: false },
  execution: { useStream: false, rpmLimit: 0, businessRetries: 1, transportRetries: 1 },
}
function document() {
  return {
    settings: [
      {
        domain: 'translation',
        schemaVersion: 9,
        revision: 3,
        payload: {
          textDetector: 'default',
          ocrEngine: 'manga_ocr',
          proofreading: { enabled: true, rounds: [{ id: 'keep' }] },
          translation: {
            provider: 'ollama',
            modelName: 'old',
            customBaseUrl: '',
            openaiOptions: options,
            translationMode: 'batch',
          },
          hqTranslation: {
            provider: 'siliconflow',
            modelName: '',
            customBaseUrl: '',
            openaiOptions: options,
            batchSize: 3,
            prompt: 'hq',
          },
          aiVisionOcr: {
            provider: 'gemini',
            modelName: '',
            customBaseUrl: '',
            openaiOptions: options,
            prompt: 'ocr',
            promptMode: 'normal',
            minImageSize: 32,
          },
          baiduOcr: { version: 'standard', sourceLanguage: 'JAP' },
        },
      },
    ],
    credentials: [],
    providerSettings: [
      {
        domain: 'translation',
        provider: 'ollama',
        revision: 1,
        schemaVersion: 1,
        credentialVersionId: null,
        payload: {
          modelName: 'cached-model',
          customBaseUrl: '',
          openaiOptions: options,
          translationMode: 'batch',
        },
      },
    ],
    bookSettings: [],
  }
}
let revision = 3
const result = (body: any) => ({
  settings: body.settings.map((s: any) => ({ domain: s.domain, revision: ++revision })),
  providerSettings: body.providerSettings.map((s: any) => ({
    domain: s.domain,
    provider: s.provider,
    revision: s.baseRevision + 1,
  })),
  credentials: [],
  prompts: [],
})
const flush = async () => {
  await new Promise(resolve => setTimeout(resolve, 0))
}
async function mount(api: ReturnType<typeof vi.fn>) {
  active.value = true
  revision = 3
  globalThis.document.body.innerHTML = '<div id="app"></div>'
  app = createApp({
    setup() {
      state = useTranslatorSettings(api as PluginSettingsApi, active)
      return () => h('div')
    },
  })
  app.mount('#app')
  await flush()
}
afterEach(() => {
  app.unmount()
  vi.restoreAllMocks()
})
it('reads global provider memories and writes the same translation document without touching unrelated domains', async () => {
  const api = vi.fn(async (_path, method, body) => (method === 'PUT' ? result(body) : document()))
  await mount(api)
  expect(state.settings.value!.translation.modelName).toBe('cached-model')
  state.settings.value!.textDetector = 'ctd'
  state.change()
  expect(await state.save()).toBe(true)
  const [path, , body] = api.mock.calls.find(call => call[1] === 'PUT')!
  expect(path).toBe('/settings/transactions?scope=global')
  expect(body.settings).toHaveLength(1)
  expect(body.settings[0]).toMatchObject({
    domain: 'translation',
    baseRevision: 3,
    payload: { textDetector: 'ctd', proofreading: { enabled: true, rounds: [{ id: 'keep' }] } },
  })
  expect(body.credentialEdits).toEqual([])
})
it('queues edits made while a request is in flight without overwriting newer input', async () => {
  let finish!: (value: any) => void
  const api = vi.fn(async (_path, method, _body) =>
    method === 'PUT'
      ? new Promise(resolve => {
          finish = resolve
        })
      : document()
  )
  await mount(api)
  state.settings.value!.textDetector = 'ctd'
  state.change()
  const saved = state.save()
  const first = api.mock.calls.at(-1)![2]
  state.settings.value!.textDetector = 'yolo'
  state.change()
  finish(result(first))
  await flush()
  const second = api.mock.calls.at(-1)![2]
  expect(second.settings[0].payload.textDetector).toBe('yolo')
  expect(second.settings[0].baseRevision).toBe(first.settings[0].baseRevision + 1)
  finish(result(second))
  expect(await saved).toBe(true)
  expect(state.settings.value!.textDetector).toBe('yolo')
})
it('keeps failed drafts when hidden and resumes only on explicit retry', async () => {
  const api = vi.fn(async (_path, method, body) => (method === 'PUT' ? result(body) : document()))
  await mount(api)
  api.mockRejectedValueOnce(new Error('版本冲突'))
  state.settings.value!.textDetector = 'ctd'
  state.change()
  expect(await state.save()).toBe(false)
  active.value = false
  await flush()
  active.value = true
  await flush()
  expect(state.settings.value!.textDetector).toBe('ctd')
  expect(state.error.value).toContain('版本冲突')
  expect(api).toHaveBeenCalledTimes(2)
  expect(await state.save()).toBe(true)
})
it('keeps credentials separate from the settings payload and retains provider-specific models', async () => {
  const api = vi.fn(async (_path, method, body) => (method === 'PUT' ? result(body) : document()))
  await mount(api)
  state.selectProvider('translation', 'siliconflow')
  state.settings.value!.translation.modelName = 'cloud-model'
  state.updateSecret('translation', 'api_key', 'test-only-key')
  state.change('translation')
  state.selectProvider('translation', 'ollama')
  expect(state.settings.value!.translation.modelName).toBe('cached-model')
  state.selectProvider('translation', 'siliconflow')
  expect(state.settings.value!.translation.modelName).toBe('cloud-model')
  expect(await state.save()).toBe(true)
  const body = api.mock.calls.find(call => call[1] === 'PUT')![2]
  expect(JSON.stringify(body.settings)).not.toContain('test-only-key')
  expect(body.credentialEdits).toEqual([
    expect.objectContaining({
      domain: 'translation',
      provider: 'siliconflow',
      secret: { api_key: 'test-only-key' },
    }),
  ])
})

it('does not create provider drafts or save when selecting the current provider', async () => {
  const api = vi.fn(async (_path, method, body) => method === 'PUT' ? result(body) : document())
  await mount(api)
  state.selectProvider('translation', 'ollama')
  expect(state.dirty.value).toBe(false)
  expect(await state.save()).toBe(true)
  expect(api).toHaveBeenCalledTimes(1)
})

it('reads existing keys and changes one field without losing the other', async () => {
  const doc = document() as any
  doc.credentials = [{ domain: 'ocr', provider: 'baidu', revision: 1,
    credentialId: 'ocr-key', credentialVersionId: 'ocr-v1',
    secret: { baidu_api_key: 'existing-id', baidu_secret_key: 'existing-secret' } }]
  const api = vi.fn(async (_path, method, body) => method === 'PUT' ? {
    ...result(body), credentials: body.credentialEdits.map((row: any) => ({
      ...row, credentialId: 'ocr-key', credentialVersionId: 'ocr-v2', revision: 2,
    })),
  } : doc)
  await mount(api)
  expect(state.secretValue('ocr', 'baidu', 'baidu_api_key')).toBe('existing-id')
  state.updateSecret('baiduOcr', 'baidu_api_key', 'new-id')
  expect(await state.save()).toBe(true)
  expect(api.mock.calls.at(-1)![2].credentialEdits[0].secret).toEqual({
    baidu_api_key: 'new-id', baidu_secret_key: 'existing-secret',
  })
  expect(state.secretValue('ocr', 'baidu', 'baidu_api_key')).toBe('new-id')
  expect(state.secretValue('ocr', 'baidu', 'baidu_secret_key')).toBe('existing-secret')
  state.change('baiduOcr')
  const count = api.mock.calls.length
  expect(await state.save()).toBe(true)
  expect(api.mock.calls.length).toBe(count)
})
