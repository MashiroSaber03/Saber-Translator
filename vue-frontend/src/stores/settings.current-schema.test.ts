import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const settingsApiMocks = vi.hoisted(() => ({
  getV2Settings: vi.fn(),
  saveV2SettingsTransaction: vi.fn(),
}))

vi.mock('@/api/v2/settings', () => ({
  getV2Settings: settingsApiMocks.getV2Settings,
  saveV2SettingsTransaction: settingsApiMocks.saveV2SettingsTransaction,
}))

import { useSettingsStore } from './settings'
import { createDefaultSettings } from './settings/defaults'

function workflowPreferencesEntry(revision = 1) {
  return {
    domain: 'workflow_preferences',
    revision,
    schemaVersion: 1,
    payload: {
      rememberWorkflowModeEnabled: false,
      lastWorkflowMode: 'translate-current',
    },
  }
}

describe('useSettingsStore backend-first loading', () => {
  beforeEach(() => {
    localStorage.clear()
    setActivePinia(createPinia())
    settingsApiMocks.getV2Settings.mockReset()
    settingsApiMocks.saveV2SettingsTransaction.mockReset()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('blocks backend writes while authoritative settings are unavailable', async () => {
    settingsApiMocks.getV2Settings.mockRejectedValue(new Error('offline'))

    const store = useSettingsStore()
    expect(await store.loadFromBackend()).toBe(false)
    expect(await store.saveToBackend()).toBe(false)
    expect(store.backendError).toContain('后端设置尚未加载')
  })

  it('rejects a backend document without the authoritative text-style defaults domain', async () => {
    const settings = createDefaultSettings()
    settingsApiMocks.getV2Settings.mockResolvedValue({
      settings: [
        {
          domain: 'translation',
          revision: 1,
          schemaVersion: 6,
          payload: {
            ...settings,
            textStyle: {
              ...settings.textStyle,
              textColor: '#FA0000',
              useAutoTextColor: true,
            },
          },
        },
        workflowPreferencesEntry(),
      ],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
    })

    const store = useSettingsStore()
    expect(await store.loadFromBackend()).toBe(false)
    expect(store.isBackendReady).toBe(false)
    expect(store.backendError).toContain('文字样式默认设置缺失')
    expect(store.settings.textStyle).toEqual(createDefaultSettings().textStyle)
  })

  it('rejects legacy text-style defaults instead of adapting them in the browser', async () => {
    const settings = createDefaultSettings()
    settingsApiMocks.getV2Settings.mockResolvedValue({
      settings: [
        {
          domain: 'translation',
          revision: 1,
          schemaVersion: 6,
          payload: settings,
        },
        {
          domain: 'text_style_defaults',
          revision: 1,
          schemaVersion: 1,
          payload: settings.textStyle,
        },
        workflowPreferencesEntry(),
      ],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
    })

    const store = useSettingsStore()
    expect(await store.loadFromBackend()).toBe(false)
    expect(store.backendError).toContain('文字样式默认设置版本无效')
  })

  it('rejects a partial text-style defaults fact instead of filling missing fields in the browser', async () => {
    const settings = createDefaultSettings()
    const { useAutoTextColor: _missing, ...partialTextStyle } = settings.textStyle
    settingsApiMocks.getV2Settings.mockResolvedValue({
      settings: [
        {
          domain: 'translation',
          revision: 1,
          schemaVersion: 6,
          payload: settings,
        },
        {
          domain: 'text_style_defaults',
          revision: 1,
          schemaVersion: 2,
          payload: partialTextStyle,
        },
        workflowPreferencesEntry(),
      ],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
    })

    const store = useSettingsStore()
    expect(await store.loadFromBackend()).toBe(false)
    expect(store.backendError).toContain('fields are incomplete')
    expect(store.settings.textStyle.useAutoTextColor).toBe(false)
  })

  it('loads provider memory without hydrating stored secrets into the browser', async () => {
    const settings = createDefaultSettings()
    settings.translation = {
      ...settings.translation,
      provider: 'custom',
      apiKey: '',
      modelName: 'translation-model',
      customBaseUrl: 'https://translation.example.com/v1',
    }
    settingsApiMocks.getV2Settings.mockResolvedValue({
      settings: [
        {
          domain: 'translation',
          revision: 4,
          schemaVersion: 6,
          payload: settings,
        },
        {
          domain: 'text_style_defaults',
          revision: 1,
          schemaVersion: 2,
          payload: settings.textStyle,
        },
        workflowPreferencesEntry(),
      ],
      bookSettings: [],
      providerSettings: [{
        domain: 'translation',
        provider: 'custom',
        revision: 2,
        schemaVersion: 1,
        credentialVersionId: 'version-1',
        payload: {
          modelName: 'cached-model',
          customBaseUrl: 'https://cached.example.com/v1',
          openaiOptions: settings.translation.openaiOptions,
        },
      }],
      credentials: [{
        credentialId: 'credential-1',
        credentialVersionId: 'version-1',
        currentVersion: 1,
        domain: 'translation',
        hasKey: true,
        provider: 'custom',
        revision: 1,
      }],
    })

    const store = useSettingsStore()
    store.initSettings()
    expect(await store.loadFromBackend()).toBe(true)

    expect(store.settings.translation.provider).toBe('custom')
    expect(store.settings.translation.apiKey).toBe('')
    expect(store.providerConfigs.translation.custom?.modelName).toBe('cached-model')
    expect(store.credentialSummaries[0]?.hasKey).toBe(true)
  })

  it('saves a new API key, applies its returned summary, and keeps it usable', async () => {
    const initialSettings = createDefaultSettings()
    initialSettings.translation = {
      ...initialSettings.translation,
      provider: 'deepseek',
      apiKey: '',
      modelName: 'deepseek-chat',
    }
    const initialDocument = {
      settings: [
        {
          domain: 'translation',
          revision: 1,
          schemaVersion: 6,
          payload: initialSettings,
        },
        {
          domain: 'text_style_defaults',
          revision: 1,
          schemaVersion: 2,
          payload: initialSettings.textStyle,
        },
        workflowPreferencesEntry(),
      ],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
    }
    settingsApiMocks.getV2Settings.mockResolvedValueOnce(initialDocument)
    settingsApiMocks.saveV2SettingsTransaction.mockResolvedValue({
      settings: [
        { domain: 'translation', revision: 2 },
        { domain: 'text_style_defaults', revision: 2 },
      ],
      bookSettings: [],
      providerSettings: [{
        domain: 'translation',
        provider: 'deepseek',
        revision: 1,
      }],
      credentials: [{
        credentialId: 'credential-1',
        credentialVersionId: 'version-1',
        currentVersion: 1,
        domain: 'translation',
        hasKey: true,
        provider: 'deepseek',
        revision: 1,
      }],
      prompts: [],
    })

    const store = useSettingsStore()
    expect(await store.loadFromBackend()).toBe(true)
    store.updateTranslationService({
      apiKey: '  sk-new-secret  ',
      modelName: 'deepseek-chat',
    })

    expect(await store.saveToBackend()).toBe(true)
    const transaction = settingsApiMocks.saveV2SettingsTransaction.mock.calls[0]?.[0]
    expect(transaction.credentialEdits).toContainEqual(expect.objectContaining({
      domain: 'translation',
      provider: 'deepseek',
      secret: { api_key: 'sk-new-secret' },
    }))
    expect(transaction.providerSettings).toContainEqual(expect.objectContaining({
      domain: 'translation',
      provider: 'deepseek',
      credentialEditRef: 'credential:translation:deepseek',
    }))
    expect(store.settings.translation.apiKey).toBe('')
    expect(store.hasCredential('translation', 'deepseek')).toBe(true)
    expect(settingsApiMocks.getV2Settings).toHaveBeenCalledTimes(1)
  })

  it('does not restore a stale chapter snapshot after saving parallel mode', async () => {
    const initialSettings = createDefaultSettings()
    settingsApiMocks.getV2Settings.mockResolvedValueOnce({
        settings: [
          {
            domain: 'translation',
            revision: 1,
            schemaVersion: 6,
            payload: initialSettings,
          },
          {
            domain: 'text_style_defaults',
            revision: 1,
            schemaVersion: 2,
            payload: initialSettings.textStyle,
          },
          workflowPreferencesEntry(),
        ],
        bookSettings: [],
        providerSettings: [],
        credentials: [],
      })
    settingsApiMocks.saveV2SettingsTransaction.mockResolvedValue({
      settings: [
        { domain: 'translation', revision: 2 },
        { domain: 'text_style_defaults', revision: 2 },
      ],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
      prompts: [],
    })

    const store = useSettingsStore()
    expect(await store.loadFromBackend()).toBe(true)
    expect(store.hydrateChapterWorkState('chapter-1', {
      parallel: { enabled: false, deepLearningLockSize: 1 },
    })).toBe(true)

    store.updateSettings({
      parallel: { enabled: true, deepLearningLockSize: 2 },
    })

    expect(await store.saveToBackend()).toBe(true)
    expect(store.settings.parallel).toEqual({
      enabled: true,
      deepLearningLockSize: 2,
    })
  })

  it('keeps current-page text style separate from reloaded global defaults', async () => {
    const settings = createDefaultSettings()
    const globalTextDefaults = {
      ...settings.textStyle,
      inpaintMethod: 'solid' as const,
      layoutDirection: 'auto' as const,
    }
    const document = {
      settings: [
        {
          domain: 'translation',
          revision: 1,
          schemaVersion: 6,
          payload: settings,
        },
        {
          domain: 'text_style_defaults',
          revision: 2,
          schemaVersion: 2,
          payload: globalTextDefaults,
        },
        workflowPreferencesEntry(),
      ],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
    }
    settingsApiMocks.getV2Settings.mockResolvedValue(document)

    const store = useSettingsStore()
    expect(await store.loadFromBackend()).toBe(true)
    expect(store.hydrateChapterWorkState('chapter-1', {})).toBe(true)

    const currentPageStyle = {
      autoFontSize: false,
      fillColor: '#102030',
      fontFamily: 'page-font-id',
      fontSize: 41,
      inpaintMethod: 'lama_mpe' as const,
      layoutDirection: 'horizontal' as const,
      lineSpacing: 1.7,
      strokeColor: '#405060',
      strokeEnabled: false,
      strokeWidth: 5,
      inlineAlign: 'end' as const,
      blockAlign: 'center' as const,
      textColor: '#708090',
      useAutoTextColor: false,
    }
    store.updateTextStyle(currentPageStyle)
    expect(await store.loadFromBackend()).toBe(true)

    expect(store.settings.textStyle).toEqual(currentPageStyle)
    expect(store.textStyleDefaults.inpaintMethod).toBe('solid')
    expect(store.textStyleDefaults.layoutDirection).toBe('auto')
  })

  it('saves global text defaults without replacing them with the current-page style', async () => {
    const settings = createDefaultSettings()
    settingsApiMocks.getV2Settings.mockResolvedValue({
      settings: [
        {
          domain: 'translation',
          revision: 1,
          schemaVersion: 6,
          payload: settings,
        },
        {
          domain: 'text_style_defaults',
          revision: 1,
          schemaVersion: 2,
          payload: settings.textStyle,
        },
        workflowPreferencesEntry(),
      ],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
    })
    settingsApiMocks.saveV2SettingsTransaction.mockResolvedValue({
      settings: [
        { domain: 'translation', revision: 2 },
        { domain: 'text_style_defaults', revision: 2 },
      ],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
      prompts: [],
    })

    const store = useSettingsStore()
    expect(await store.loadFromBackend()).toBe(true)
    expect(store.hydrateChapterWorkState('chapter-1', {})).toBe(true)
    store.updateTextStyle({ inpaintMethod: 'lama_mpe' })
    store.textStyleDefaults = {
      ...store.textStyleDefaults,
      inpaintMethod: 'litelama',
    }

    expect(await store.saveToBackend()).toBe(true)

    const transaction = settingsApiMocks.saveV2SettingsTransaction.mock.calls[0]?.[0]
    const textDefaultsMutation = transaction.settings.find(
      (entry: { domain: string }) => entry.domain === 'text_style_defaults',
    )
    const translationMutation = transaction.settings.find(
      (entry: { domain: string }) => entry.domain === 'translation',
    )
    expect(textDefaultsMutation.payload.inpaintMethod).toBe('litelama')
    expect(translationMutation.payload).not.toHaveProperty('textStyle')
  })

  it('does not submit a partial Baidu OCR credential replacement', async () => {
    const settings = createDefaultSettings()
    settingsApiMocks.getV2Settings.mockResolvedValue({
      settings: [
        {
          domain: 'translation',
          revision: 1,
          schemaVersion: 6,
          payload: settings,
        },
        {
          domain: 'text_style_defaults',
          revision: 1,
          schemaVersion: 2,
          payload: settings.textStyle,
        },
        workflowPreferencesEntry(),
      ],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
    })

    const store = useSettingsStore()
    expect(await store.loadFromBackend()).toBe(true)
    store.updateBaiduOcr({ apiKey: 'replacement-only', secretKey: '' })

    expect(await store.saveToBackend()).toBe(false)
    expect(store.backendError).toContain('必须同时填写')
    expect(settingsApiMocks.saveV2SettingsTransaction).not.toHaveBeenCalled()
  })
})
