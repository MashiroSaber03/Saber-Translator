import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const configMocks = vi.hoisted(() => ({
  getUserSettings: vi.fn(),
}))

vi.mock('@/api/config', () => ({
  getUserSettings: configMocks.getUserSettings,
}))

import { useSettingsStore } from './settings'

describe('useSettingsStore legacy backend fallback defaults', () => {
  beforeEach(() => {
    localStorage.clear()
    setActivePinia(createPinia())
    configMocks.getUserSettings.mockReset()
  })

  it('uses updated defaults when migrating legacy providerSettings payloads', async () => {
    configMocks.getUserSettings.mockResolvedValue({
      success: true,
      settings: {
        modelProvider: 'siliconflow',
        hqTranslateProvider: 'siliconflow',
        providerSettings: {
          modelProvider: {
            siliconflow: {},
          },
          hqTranslateProvider: {
            siliconflow: {},
          },
        },
      },
    })

    const store = useSettingsStore()
    const loaded = await store.loadFromBackend()

    expect(loaded).toBe(true)
    expect(store.providerConfigs.translation.siliconflow?.openaiOptions?.execution?.useStream).toBe(true)
    expect(store.providerConfigs.hqTranslation.siliconflow?.openaiOptions?.execution?.transportRetries).toBe(3)
    expect(store.providerConfigs.hqTranslation.siliconflow?.openaiOptions?.execution?.businessRetries).toBe(3)
  })
})
