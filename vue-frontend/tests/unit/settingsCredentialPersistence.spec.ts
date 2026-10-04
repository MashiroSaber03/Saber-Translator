import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent, h } from 'vue'
import { flushPromises, mount } from '@vue/test-utils'
import { useSettingsStore } from '@/stores/settings'
import { createDefaultSettings } from '@/stores/settings/defaults'
import { useBrowserExtensionSettings } from '@/composables/useBrowserExtensionSettings'

const mocks = vi.hoisted(() => ({ get: vi.fn(), save: vi.fn(), post: vi.fn(), prompts: vi.fn() }))
vi.mock('@/api/v2/settings', async (importOriginal) => ({
  ...await importOriginal<typeof import('@/api/v2/settings')>(),
  getV2Settings: mocks.get, saveV2SettingsTransaction: mocks.save, listV2Prompts: mocks.prompts,
}))
vi.mock('@/api/client', () => ({ apiClient: { post: mocks.post } }))

const roundId = '11111111-1111-4111-8111-111111111111'
const domains = ['translation', 'hq', 'ai_vision_ocr', 'plugin_agent', `proofreading_${roundId}`, 'ocr']
function authority() {
  const settings = createDefaultSettings()
  settings.aiVisionOcr.provider = 'siliconflow'
  settings.proofreading.rounds = [{ ...structuredClone(settings.hqTranslation), id: roundId, name: 'audit' }]
  const credentials = domains.map(domain => ({
    domain, provider: domain === 'ocr' ? 'baidu' : 'siliconflow', credentialId: `id-${domain}`,
    credentialVersionId: `version-${domain}`, revision: 1, currentVersion: 1, hasKey: true,
    secret: domain === 'ocr' ? { baidu_api_key: 'dummy-baidu-key', baidu_secret_key: 'dummy-baidu-secret' }
      : { [domain === 'ai_vision_ocr' ? 'ai_vision_api_key' : 'api_key']: `dummy-${domain}` },
  }))
  return {
    settings: [
      { domain: 'translation', revision: 1, payload: settings },
      { domain: 'text_style_defaults', revision: 1, payload: settings.textStyle },
      { domain: 'workflow_preferences', revision: 1, payload: { rememberWorkflowModeEnabled: false, lastWorkflowMode: 'translate-current' } },
      { domain: 'export_preferences', revision: 1, payload: { preserveOriginalFilenames: false } },
    ], providerSettings: credentials.map(credential => {
      const domain = credential.domain
      const source = domain === 'translation' ? settings.translation : domain === 'hq' ? settings.hqTranslation
        : domain === 'ai_vision_ocr' ? settings.aiVisionOcr : domain === 'plugin_agent' ? settings.pluginAgent
          : domain === 'ocr' ? settings.baiduOcr : settings.proofreading.rounds[0]!
      const fields = domain === 'translation' ? ['modelName', 'customBaseUrl', 'openaiOptions', 'translationMode']
        : domain === 'ai_vision_ocr' ? ['modelName', 'customBaseUrl', 'openaiOptions', 'prompt', 'promptMode', 'minImageSize']
          : domain === 'ocr' ? ['version', 'sourceLanguage'] : domain === 'plugin_agent' ? ['modelName', 'customBaseUrl', 'openaiOptions']
            : ['modelName', 'customBaseUrl', 'openaiOptions', 'prompt', 'batchSize']
      return { domain, provider: credential.provider, revision: 1, credentialVersionId: credential.credentialVersionId,
        payload: Object.fromEntries(fields.map(field => [field, (source as any)[field]])) }
    }), credentials, bookSettings: [],
  }
}
function section(store: ReturnType<typeof useSettingsStore>, domain: string) {
  if (domain === 'translation') return store.settings.translation
  if (domain === 'hq') return store.settings.hqTranslation
  if (domain === 'ai_vision_ocr') return store.settings.aiVisionOcr
  if (domain === 'plugin_agent') return store.settings.pluginAgent
  if (domain === 'ocr') return store.settings.baiduOcr
  return store.settings.proofreading.rounds[0]!
}

describe('shared settings credential persistence', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    mocks.get.mockReset(); mocks.save.mockReset(); mocks.post.mockReset()
    mocks.get.mockImplementation(async () => structuredClone(authority()))
    mocks.save.mockImplementation(async (tx) => ({
      settings: tx.settings.map((r: any) => ({ domain: r.domain, revision: r.baseRevision + 1 })),
      providerSettings: tx.providerSettings.map((r: any) => ({ domain: r.domain, provider: r.provider, revision: r.baseRevision + 1 })),
      credentials: [], prompts: [], bookSettings: [],
    }))
    mocks.post.mockResolvedValue({ models: [] })
  })
  afterEach(() => vi.useRealTimers())

  it.each(domains)('clearing %s writes empty values without restoring the historical key', async domain => {
    const store = useSettingsStore()
    expect(await store.loadFromBackend()).toBe(true)
    const target = section(store, domain)
    const old = target.apiKey
    expect(old).not.toBe('')
    target.apiKey = ''
    if (domain === 'ocr') store.settings.baiduOcr.secretKey = ''
    expect(await store.saveToBackend()).toBe(true)
    expect(target.apiKey).toBe('')
    const tx = mocks.save.mock.calls[0]![0]
    const secret = domain === 'ocr' ? { baidu_api_key: '', baidu_secret_key: '' }
      : { [domain === 'ai_vision_ocr' ? 'ai_vision_api_key' : 'api_key']: '' }
    expect(tx.credentialEdits).toContainEqual(expect.objectContaining({ domain, secret }))
  })

  it('loading an unbound provider ignores historical credentials and unrelated saves keep it unbound', async () => {
    const doc = authority() as any
    doc.providerSettings = [{ domain: 'translation', provider: 'siliconflow', revision: 1, credentialVersionId: null,
      payload: { modelName: 'audit', customBaseUrl: '', openaiOptions: createDefaultSettings().translation.openaiOptions, translationMode: 'batch' } }]
    mocks.get.mockResolvedValueOnce(doc)
    const store = useSettingsStore()
    expect(await store.loadFromBackend()).toBe(true)
    expect(store.settings.translation.apiKey).toBe('')
    expect(await store.saveToBackend()).toBe(true)
    expect(mocks.save).not.toHaveBeenCalled()
    store.exportPreferences.preserveOriginalFilenames = true
    expect(await store.saveToBackend()).toBe(true)
    expect(mocks.save.mock.calls[0]![0].providerSettings).toEqual([])
  })

  it('saves each Baidu field including empty values together with export preferences', async () => {
    const store = useSettingsStore()
    expect(await store.loadFromBackend()).toBe(true)
    store.settings.baiduOcr.secretKey = ''
    store.exportPreferences.preserveOriginalFilenames = true
    expect(store.settings.ocrEngine).toBe('manga_ocr')
    expect(await store.saveToBackend()).toBe(true)
    expect(mocks.save.mock.calls[0]![0].credentialEdits).toContainEqual(expect.objectContaining({
      domain: 'ocr', secret: { baidu_api_key: 'dummy-baidu-key', baidu_secret_key: '' },
    }))
    expect(mocks.save.mock.calls[0]![0].settings).toEqual([expect.objectContaining({ domain: 'export_preferences' })])
  })

  it('loads the bound key even when a newer historical credential is available', async () => {
    const doc = authority()
    doc.credentials.push({ ...doc.credentials[0]!, credentialVersionId: 'newer-version', currentVersion: 2,
      revision: 2, secret: { api_key: 'dummy-newer-key' } })
    mocks.get.mockResolvedValueOnce(doc)
    const store = useSettingsStore()
    expect(await store.loadFromBackend()).toBe(true)
    expect(store.settings.translation.apiKey).toBe('dummy-translation')
    expect(await store.saveToBackend()).toBe(true)
    expect(mocks.save).not.toHaveBeenCalled()
  })

  it('saving the plugin assistant preserves pending translation key edits', async () => {
    const store = useSettingsStore()
    expect(await store.loadFromBackend()).toBe(true)
    store.settings.translation.apiKey = 'dummy-pending-edit'
    store.settings.pluginAgent.modelName = 'new-agent-model'
    expect(await store.savePluginAgentSettings()).toBe(true)
    expect(store.settings.translation.apiKey).toBe('dummy-pending-edit')
  })

  it.each(['translation', 'hq', 'ai_vision_ocr', 'plugin_agent', `proofreading_${roundId}`])('keyed %s diagnostics retain their domain', async domain => {
    const { fetchModels } = await import('@/api/v2/diagnostics')
    await fetchModels('custom', 'dummy-diagnostic-key', 'http://127.0.0.1:65530/v1', domain)
    const body = mocks.post.mock.calls[0]![1]
    expect(body.domain).toBe(domain)
    expect(body.secret).toHaveProperty(domain === 'ai_vision_ocr' ? 'ai_vision_api_key' : 'api_key')
  })

  it('extension settings persist a cleared key and keep it empty after reload', async () => {
    vi.useFakeTimers()
    const defaults = createDefaultSettings()
    const doc = {
      settings: [
        { domain: 'text_style_defaults', revision: 1, payload: defaults.textStyle },
        { domain: 'browser_dom_agent', revision: 1, payload: { provider: 'siliconflow', modelName: 'audit', customBaseUrl: '', openaiOptions: defaults.pluginAgent.openaiOptions } },
      ], bookSettings: [],
      providerSettings: [{ domain: 'browser_dom_agent', provider: 'siliconflow', revision: 1, credentialVersionId: 'extension-version',
        payload: { modelName: 'audit', customBaseUrl: '', openaiOptions: defaults.pluginAgent.openaiOptions } }],
      credentials: [{ domain: 'browser_dom_agent', provider: 'siliconflow', credentialId: 'extension-id', credentialVersionId: 'extension-version',
        revision: 1, currentVersion: 1, hasKey: true, secret: { api_key: 'dummy-extension-key' } }],
    }
    const api = vi.fn(async (path: string, method?: string, tx?: any) => {
      if (method === 'PUT') {
        const changed = tx.providerSettings[0]
        const key = { ...doc.credentials[0]!, credentialVersionId: 'empty-extension-version',
          secret: tx.credentialEdits[0].secret, revision: 2, currentVersion: 2, hasKey: false }
        doc.credentials.push(key)
        doc.providerSettings[0]!.credentialVersionId = key.credentialVersionId
        return { settings: [], providerSettings: [{ domain: changed.domain, provider: changed.provider, revision: 2 }], credentials: [key], prompts: [], bookSettings: [] }
      }
      return path === '/fonts' ? { items: [] } : structuredClone(doc)
    })
    let hook!: ReturnType<typeof useBrowserExtensionSettings>
    const wrapper = mount(defineComponent({ setup() { hook = useBrowserExtensionSettings(api as any); return () => h('div') } }))
    await flushPromises()
    hook.updateKey('')
    expect(await hook.save()).toBe(true)
    expect(api.mock.calls.filter(([, method]: any) => method === 'PUT')).toHaveLength(1)
    expect(hook.credential.value?.secret.api_key).toBe('')
    await hook.load()
    expect(hook.credential.value?.secret.api_key).toBe('')
    wrapper.unmount()
    await flushPromises()
  })

  it.each(['vlm', 'llm', 'embedding', 'reranker', 'imageGen'] as const)('Insight %s saves an empty value when its key is cleared', async kind => {
    const doc: any = {
      settings: [{ domain: 'insight', revision: 1, payload: {
        analysis: { batch: { pagesPerBatch: 5, contextBatchCount: 3, architecturePreset: 'standard', customLayers: [] } },
        vlm: { provider: 'gemini' }, chat: { provider: 'gemini', useSameAsVlm: false },
        embedding: { provider: 'openai' }, reranker: { provider: 'jina' }, imageGen: { provider: 'gpt2api' },
      } }], providerSettings: [], credentials: [], bookSettings: [],
    }
    mocks.get.mockImplementation(async () => structuredClone(doc))
    mocks.prompts.mockResolvedValue(['batch_analysis', 'segment_summary', 'chapter_summary', 'qa_response'].map(type => ({
      id: `factory-${type}`, name: `audit-${type}`, content: `audit-${type}`, type, revision: 1, isFactoryDefault: true,
    })))
    const { getGlobalConfig, saveGlobalConfig } = await import('@/api/insight')
    const initial = await getGlobalConfig()
    const seeded = structuredClone(initial)
    for (const type of ['vlm', 'llm', 'embedding', 'reranker', 'imageGen'] as const) {
      seeded.config[type].model = 'seed-model'
    }
    await saveGlobalConfig(seeded, initial)
    const serialized = mocks.save.mock.calls[0]![0]
    doc.providerSettings = serialized.providerSettings.map((row: any) => ({ ...row, revision: 1, credentialVersionId: `version-${row.domain}` }))
    doc.credentials = doc.providerSettings.map((row: any) => ({
      domain: row.domain, provider: row.provider, credentialId: `id-${row.domain}`, credentialVersionId: row.credentialVersionId,
      revision: 1, currentVersion: 1, hasKey: true, secret: { api_key: `dummy-${row.domain}` },
    }))
    const snapshot = await getGlobalConfig()
    const baseline = structuredClone(snapshot)
    expect(snapshot.config[kind].apiKey).not.toBe('')
    snapshot.config[kind].apiKey = ''
    const saved = await saveGlobalConfig(snapshot, baseline)
    expect(saved.config[kind].apiKey).toBe('')
    const domains = { vlm: 'insight_vlm', llm: 'insight_chat', embedding: 'insight_embedding', reranker: 'insight_reranker', imageGen: 'insight_image_gen' }
    const tx = mocks.save.mock.calls[1]![0]
    expect(tx.credentialEdits).toEqual([expect.objectContaining({ domain: domains[kind], secret: { api_key: '' } })])
  })
})
