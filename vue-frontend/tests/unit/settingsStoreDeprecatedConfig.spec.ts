import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

import { STORAGE_KEY_TRANSLATION_SETTINGS } from '@/constants'
import { useSettingsStore } from '@/stores/settings'

const { getUserSettingsMock, saveUserSettingsMock } = vi.hoisted(() => ({
  getUserSettingsMock: vi.fn(),
  saveUserSettingsMock: vi.fn(),
}))

vi.mock('@/api/config', () => ({
  getUserSettings: getUserSettingsMock,
  saveUserSettings: saveUserSettingsMock,
}))

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
  })

  afterEach(() => {
    vi.restoreAllMocks()
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
    const payload = saveUserSettingsMock.mock.calls[0]?.[0] as Record<string, any>
    expect(payload).not.toHaveProperty('hqSessionReset')
    expect(payload.hqTranslation.openaiOptions.execution.rpmLimit).toBe(9)
    expect(payload.proofreading.rounds[0]).not.toHaveProperty('sessionReset')
    expect(payload.proofreading.rounds[0].openaiOptions.execution.rpmLimit).toBe(7)
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
    expect('sessionReset' in (store.settings.hqTranslation as any)).toBe(false)
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

    expect('lowReasoning' in (store.settings.hqTranslation as any)).toBe(false)
    expect('noThinkingMethod' in (store.settings.hqTranslation as any)).toBe(false)
    expect(store.settings.hqTranslation.openaiOptions.request.forceJsonOutput).toBe(false)
  })
})
