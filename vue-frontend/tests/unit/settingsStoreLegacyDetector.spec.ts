import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

import { STORAGE_KEY_TRANSLATION_SETTINGS } from '@/constants'
import { useSettingsStore } from '@/stores/settingsStore'

const { getUserSettingsMock } = vi.hoisted(() => ({
  getUserSettingsMock: vi.fn()
}))

vi.mock('@/api/config', () => ({
  getUserSettings: getUserSettingsMock
}))

describe('settings store legacy detector migration', () => {
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
  })

  afterEach(() => {
    vi.restoreAllMocks()
    getUserSettingsMock.mockReset()
  })

  it('should migrate legacy yolov5 detector from localStorage to default', () => {
    localStorageMock[STORAGE_KEY_TRANSLATION_SETTINGS] = JSON.stringify({
      textDetector: 'yolov5'
    })

    const store = useSettingsStore()
    store.loadFromStorage()

    expect(store.settings.textDetector).toBe('default')
  })

  it('should migrate legacy yolov5 detector from backend settings to default', async () => {
    getUserSettingsMock.mockResolvedValue({
      success: true,
      settings: {
        textDetector: 'yolov5'
      }
    })

    const store = useSettingsStore()
    const loaded = await store.loadFromBackend()

    expect(loaded).toBe(true)
    expect(store.settings.textDetector).toBe('default')
  })
})
