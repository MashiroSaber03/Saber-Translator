import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

import { STORAGE_KEY_TRANSLATION_SETTINGS } from '@/constants'
import { useSettingsStore } from '@/stores/settings'
import { createDefaultSettings } from '@/stores/settings/defaults'

const { getUserSettingsMock, saveUserSettingsMock } = vi.hoisted(() => ({
  getUserSettingsMock: vi.fn(),
  saveUserSettingsMock: vi.fn(),
}))

vi.mock('@/api/config', () => ({
  getUserSettings: getUserSettingsMock,
  saveUserSettings: saveUserSettingsMock,
}))

interface SavedSettingsPayload {
  hqTranslation: {
    openaiOptions: {
      execution: {
        rpmLimit: number
      }
    }
  }
  proofreading: {
    rounds: Array<{
      openaiOptions: {
        execution: {
          rpmLimit: number
        }
      }
    }>
  }
}

describe('settings store current schema boundaries', () => {
  let localStorageMock: Record<string, string> = {}

  beforeEach(() => {
    localStorageMock = {}
    setActivePinia(createPinia())

    getUserSettingsMock.mockReset()
    saveUserSettingsMock.mockReset()
    saveUserSettingsMock.mockResolvedValue({ success: true })

    vi.spyOn(Storage.prototype, 'getItem').mockImplementation((key: string) => {
      return localStorageMock[key] || null
    })

    vi.spyOn(Storage.prototype, 'setItem').mockImplementation((key: string, value: string) => {
      localStorageMock[key] = value
    })

    vi.spyOn(Storage.prototype, 'removeItem').mockImplementation((key: string) => {
      delete localStorageMock[key]
    })
    vi.spyOn(console, 'warn').mockImplementation(() => undefined)
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('keeps settings store cloning on shared helpers', () => {
    const genericCloneFiles = [
      'src/stores/settings/defaults.ts',
      'src/stores/settings/index.ts',
    ]
    const providerModuleFiles = [
      'src/stores/settings/modules/translation.ts',
      'src/stores/settings/modules/hqTranslation.ts',
      'src/stores/settings/modules/ocr.ts',
      'src/stores/settings/modules/pluginAgent.ts',
    ]

    for (const file of genericCloneFiles) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')
      expect(source, file).toContain("import { deepClone } from '@/utils/deepClone'")
      expect(source, file).not.toContain('function cloneJson')
      expect(source, file).not.toContain('JSON.parse(JSON.stringify')
    }

    for (const file of providerModuleFiles) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')
      expect(source, file).toContain('cloneOpenAiOptions')
      expect(source, file).not.toContain('JSON.parse(JSON.stringify')
    }
  })

  it('keeps provider cache switching on shared settings helpers', () => {
    const providerModuleFiles = [
      'src/stores/settings/modules/translation.ts',
      'src/stores/settings/modules/hqTranslation.ts',
      'src/stores/settings/modules/ocr.ts',
      'src/stores/settings/modules/pluginAgent.ts',
      'src/stores/settings/modules/webImport.ts',
    ]

    for (const file of providerModuleFiles) {
      const source = readFileSync(resolve(process.cwd(), file), 'utf8')
      expect(source, file).toContain("from '../providerConfigCache'")
      expect(source, file).toContain('saveProviderCacheEntry')
      expect(source, file).toContain('restoreProviderCacheEntry')
      expect(source, file).toContain('snapshotProviderCredentials')
      expect(source, file).toContain('applyProviderCredentials')
      expect(source, file).toContain('clearProviderCredentials')
      expect(source, file).not.toMatch(/providerConfigs\.value\.\w+\[[^\]]+\]\s*=/)
    }
  })

  it('keeps removed-field probes typed without broad any escapes', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'tests/unit/settingsStoreCurrentSchemaBoundaries.spec.ts'),
      'utf8',
    )

    expect(source).not.toMatch(/\bas any\b|:\s*any\b|any\[\]/)
  })

  it('does not send removed session reset fields when saving backend settings', async () => {
    const store = useSettingsStore()
    store.settings.hqTranslation.openaiOptions.execution.rpmLimit = 9
    store.settings.proofreading.rounds = [
      {
        name: '第1轮',
        provider: 'siliconflow',
        apiKey: 'proof-key',
        modelName: 'proof-model',
        customBaseUrl: '',
        prompt: 'proof',
        batchSize: 2,
        openaiOptions: {
          request: {
            forceJsonOutput: false,
          },
          execution: {
            useStream: true,
            rpmLimit: 7,
            transportRetries: 1,
            businessRetries: 1,
          },
        },
      },
    ]

    const success = await store.saveToBackend()

    expect(success).toBe(true)
    expect(saveUserSettingsMock).toHaveBeenCalledTimes(1)
    const payload = saveUserSettingsMock.mock.calls[0]?.[0] as SavedSettingsPayload
    expect(payload).not.toHaveProperty('hqSessionReset')
    expect(payload.hqTranslation.openaiOptions.execution.rpmLimit).toBe(9)
    expect(payload.proofreading.rounds[0]).not.toHaveProperty('sessionReset')
    expect(payload.proofreading.rounds[0].openaiOptions.execution.rpmLimit).toBe(7)
  })

  it('keeps proofreading OpenAI UI patch fields out of the round schema', () => {
    const store = useSettingsStore()
    store.setProofreadingEnabled(true)
    store.addProofreadingRound({
      name: '第1轮',
      provider: 'siliconflow',
      apiKey: 'proof-key',
      modelName: 'proof-model',
      customBaseUrl: '',
      openaiOptions: {
        request: {
          forceJsonOutput: false,
        },
        execution: {
          useStream: false,
          rpmLimit: 0,
          transportRetries: 1,
          businessRetries: 0,
        },
      },
      batchSize: 2,
      prompt: 'proof',
    })

    store.updateProofreadingRound(0, {
      rpmLimit: 9,
      transportRetries: 4,
      businessRetries: 2,
      forceJsonOutput: true,
      useStream: true,
      extraBody: { top_p: 0.9 },
    })

    const round = store.settings.proofreading.rounds[0] as Record<string, unknown>
    expect(round).not.toHaveProperty('rpmLimit')
    expect(round).not.toHaveProperty('transportRetries')
    expect(round).not.toHaveProperty('businessRetries')
    expect(round).not.toHaveProperty('forceJsonOutput')
    expect(round).not.toHaveProperty('useStream')
    expect(round).not.toHaveProperty('extraBody')
    expect(store.settings.proofreading.rounds[0]?.openaiOptions).toEqual({
      request: {
        forceJsonOutput: true,
        extraBody: { top_p: 0.9 },
      },
      execution: {
        useStream: true,
        rpmLimit: 9,
        transportRetries: 4,
        businessRetries: 2,
      },
    })
  })

  it('ignores backend settings without the current schema version', async () => {
    getUserSettingsMock.mockResolvedValue({
      success: true,
      settings: {
        hqRpmLimit: '12',
        hqSessionReset: '5',
        proofreading: {
          enabled: true,
          rounds: [
            {
              name: '第1轮',
              provider: 'siliconflow',
              apiKey: 'proof-key',
              modelName: 'proof-model',
              rpmLimit: 4,
              useStream: true,
            },
          ],
        },
      },
    })

    const store = useSettingsStore()
    const loaded = await store.loadFromBackend()

    expect(loaded).toBe(false)
    expect(store.settings.hqTranslation.openaiOptions.execution.rpmLimit).not.toBe(12)
    expect('sessionReset' in (store.settings.hqTranslation as Record<string, unknown>)).toBe(false)
  })

  it('ignores local settings without the current schema version', () => {
    localStorageMock[STORAGE_KEY_TRANSLATION_SETTINGS] = JSON.stringify({
      hqTranslation: {
        lowReasoning: true,
        noThinkingMethod: 'volcano',
        forceJsonOutput: true,
      },
      proofreading: {
        enabled: true,
        maxRetries: 2,
        rounds: [
          {
            name: '第1轮',
            provider: 'siliconflow',
            apiKey: 'proof-key',
            modelName: 'proof-model',
            customBaseUrl: '',
            prompt: 'proof',
            batchSize: 2,
            rpmLimit: 7,
            maxRetries: 1,
            lowReasoning: true,
            noThinkingMethod: 'gemini',
            forceJsonOutput: false,
            useStream: true,
          },
        ],
      },
    })

    const store = useSettingsStore()
    store.loadFromStorage()

    expect('lowReasoning' in (store.settings.hqTranslation as Record<string, unknown>)).toBe(false)
    expect('noThinkingMethod' in (store.settings.hqTranslation as Record<string, unknown>)).toBe(false)
    expect(store.settings.hqTranslation.openaiOptions.request.forceJsonOutput).toBe(false)
  })

  it('ignores local settings with the current schema version when required current sections are missing', () => {
    localStorageMock[STORAGE_KEY_TRANSLATION_SETTINGS] = JSON.stringify({
      settingsSchemaVersion: 3,
      translation: {
        provider: 'custom',
        apiKey: 'partial-key',
      },
    })

    const store = useSettingsStore()
    store.loadFromStorage()

    expect(store.settings.translation.provider).toBe(createDefaultSettings().translation.provider)
    expect(store.settings.translation.apiKey).toBe('')
  })

  it('ignores local current-schema settings with invalid OCR prompt mode', () => {
    const invalidSettings = createDefaultSettings()
    invalidSettings.translation.apiKey = 'should-not-load'
    invalidSettings.aiVisionOcr.prompt = '对图中的日语进行OCR:'
    ;(invalidSettings.aiVisionOcr as Record<string, unknown>).promptMode = 'legacy-inferred'
    localStorageMock[STORAGE_KEY_TRANSLATION_SETTINGS] = JSON.stringify(invalidSettings)

    const store = useSettingsStore()
    store.loadFromStorage()

    expect(store.settings.translation.apiKey).toBe('')
    expect(store.settings.aiVisionOcr.promptMode).toBe(createDefaultSettings().aiVisionOcr.promptMode)
  })

  it('ignores backend settings with the current schema version when required current sections are missing', async () => {
    getUserSettingsMock.mockResolvedValue({
      success: true,
      settings: {
        settingsSchemaVersion: 3,
        translation: {
          provider: 'custom',
          apiKey: 'partial-key',
        },
      },
    })

    const store = useSettingsStore()
    const loaded = await store.loadFromBackend()

    expect(loaded).toBe(false)
    expect(store.settings.translation.provider).toBe(createDefaultSettings().translation.provider)
    expect(store.settings.translation.apiKey).toBe('')
  })

  it('ignores backend current-schema settings with invalid OCR prompt mode', async () => {
    const invalidSettings = createDefaultSettings()
    invalidSettings.targetLanguage = 'en'
    invalidSettings.aiVisionOcr.prompt = '对图中的日语进行OCR:'
    ;(invalidSettings.aiVisionOcr as Record<string, unknown>).promptMode = 'legacy-inferred'
    getUserSettingsMock.mockResolvedValue({
      success: true,
      settings: invalidSettings,
    })

    const store = useSettingsStore()
    const loaded = await store.loadFromBackend()

    expect(loaded).toBe(false)
    expect(store.settings.targetLanguage).toBe(createDefaultSettings().targetLanguage)
    expect(store.settings.aiVisionOcr.promptMode).toBe(createDefaultSettings().aiVisionOcr.promptMode)
  })

  it('leaves settings unchanged when backend provider configs fail current-schema validation', async () => {
    const backendSettings = createDefaultSettings()
    backendSettings.targetLanguage = 'en'
    backendSettings.translation.apiKey = 'backend-key'
    getUserSettingsMock.mockResolvedValue({
      success: true,
      settings: {
        ...backendSettings,
        providerConfigs: {
          translation: {},
          hqTranslation: {},
          pluginAgent: {},
        },
      },
    })

    const store = useSettingsStore()
    const loaded = await store.loadFromBackend()

    expect(loaded).toBe(false)
    expect(store.settings.targetLanguage).toBe(createDefaultSettings().targetLanguage)
    expect(store.settings.translation.apiKey).toBe('')
  })
})
