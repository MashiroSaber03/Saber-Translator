import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

import { STORAGE_KEY_TRANSLATION_SETTINGS } from '@/constants'
import { useSettingsStore } from '@/stores/settingsStore'

const { getUserSettingsMock, saveUserSettingsMock } = vi.hoisted(() => ({
  getUserSettingsMock: vi.fn(),
  saveUserSettingsMock: vi.fn()
}))

vi.mock('@/api/config', () => ({
  getUserSettings: getUserSettingsMock,
  saveUserSettings: saveUserSettingsMock
}))

describe('settings store min text block area percent', () => {
  let localStorageMock: Record<string, string> = {}

  beforeEach(() => {
    localStorageMock = {}
    setActivePinia(createPinia())

    vi.spyOn(Storage.prototype, 'getItem').mockImplementation((key: string) => {
      return localStorageMock[key] || null
    })

    vi.spyOn(Storage.prototype, 'setItem').mockImplementation((key: string, value: string) => {
      localStorageMock[key] = value
    })

    vi.spyOn(Storage.prototype, 'removeItem').mockImplementation((key: string) => {
      delete localStorageMock[key]
    })

    getUserSettingsMock.mockReset()
    saveUserSettingsMock.mockReset()
    saveUserSettingsMock.mockResolvedValue({ success: true })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('defaults minTextBlockAreaPercent to 0.05', () => {
    const store = useSettingsStore()

    expect(store.settings.minTextBlockAreaPercent).toBe(0.05)
  })

  it('hydrates minTextBlockAreaPercent from localStorage and preserves zero', () => {
    localStorageMock[STORAGE_KEY_TRANSLATION_SETTINGS] = JSON.stringify({
      minTextBlockAreaPercent: 0
    })

    const store = useSettingsStore()
    store.loadFromStorage()

    expect(store.settings.minTextBlockAreaPercent).toBe(0)
  })

  it('saves minTextBlockAreaPercent to backend settings', async () => {
    const store = useSettingsStore()
    store.settings.minTextBlockAreaPercent = 2.5

    const saved = await store.saveToBackend()

    expect(saved).toBe(true)
    expect(saveUserSettingsMock).toHaveBeenCalledWith(expect.objectContaining({
      minTextBlockAreaPercent: 2.5,
      settingsSchemaVersion: 3
    }))
  })
})
