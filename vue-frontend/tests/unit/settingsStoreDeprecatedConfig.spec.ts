import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

import { useSettingsStore } from '@/stores/settingsStore'

const { getUserSettingsMock, saveUserSettingsMock } = vi.hoisted(() => ({
  getUserSettingsMock: vi.fn(),
  saveUserSettingsMock: vi.fn(),
}))

vi.mock('@/api/config', () => ({
  getUserSettings: getUserSettingsMock,
  saveUserSettings: saveUserSettingsMock,
}))

describe('settings store deprecated HQ/proofreading fields', () => {
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

  it('does not send deprecated session reset fields when saving backend settings', async () => {
    const store = useSettingsStore()
    store.settings.hqTranslation.rpmLimit = 9
    store.settings.proofreading.rounds = [
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
        lowReasoning: false,
        noThinkingMethod: 'gemini',
        forceJsonOutput: false,
        useStream: true,
      },
    ]

    const success = await store.saveToBackend()

    expect(success).toBe(true)
    expect(saveUserSettingsMock).toHaveBeenCalledTimes(1)
    const payload = saveUserSettingsMock.mock.calls[0]?.[0] as Record<string, any>
    expect(payload).not.toHaveProperty('hqSessionReset')
    expect(payload.hqRpmLimit).toBe('9')
    expect(payload.proofreading.rounds[0]).not.toHaveProperty('sessionReset')
    expect(payload.proofreading.rounds[0].rpmLimit).toBe(7)
  })

  it('ignores deprecated session reset fields when loading backend settings', async () => {
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

    expect(loaded).toBe(true)
    expect(store.settings.hqTranslation.rpmLimit).toBe(12)
    expect('sessionReset' in (store.settings.hqTranslation as any)).toBe(false)
    expect('sessionReset' in (store.settings.proofreading.rounds[0] as any)).toBe(false)
    expect(store.settings.proofreading.rounds[0]?.rpmLimit).toBe(4)
  })
})
