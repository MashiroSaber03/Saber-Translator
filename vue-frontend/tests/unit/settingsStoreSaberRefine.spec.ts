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

describe('settings store saber yolo refine', () => {
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

  it('defaults enableSaberYoloRefine to true', () => {
    const store = useSettingsStore()

    expect(typeof store.setEnableAuxYoloDetection).toBe('function')
    expect(typeof store.setAuxYoloConfThreshold).toBe('function')
    expect(typeof store.setAuxYoloOverlapThreshold).toBe('function')
    expect(store.settings.enableSaberYoloRefine).toBe(true)
    expect(store.settings.saberYoloRefineOverlapThreshold).toBe(50)
    expect(store.settings.enableAuxYoloDetection).toBe(false)
    expect(store.settings.auxYoloConfThreshold).toBe(0.4)
    expect(store.settings.auxYoloOverlapThreshold).toBe(0.1)
  })

  it('loads enableSaberYoloRefine from localStorage', () => {
    localStorageMock[STORAGE_KEY_TRANSLATION_SETTINGS] = JSON.stringify({
      enableSaberYoloRefine: false,
      saberYoloRefineOverlapThreshold: 35,
      enableAuxYoloDetection: true,
      auxYoloConfThreshold: 0.55,
      auxYoloOverlapThreshold: 0.2
    })

    const store = useSettingsStore()
    store.loadFromStorage()

    expect(store.settings.enableSaberYoloRefine).toBe(false)
    expect(store.settings.saberYoloRefineOverlapThreshold).toBe(35)
    expect(store.settings.enableAuxYoloDetection).toBe(true)
    expect(store.settings.auxYoloConfThreshold).toBe(0.55)
    expect(store.settings.auxYoloOverlapThreshold).toBe(0.2)
  })

  it('loads enableSaberYoloRefine from backend settings', async () => {
    getUserSettingsMock.mockResolvedValue({
      success: true,
      settings: {
        enableSaberYoloRefine: false,
        saberYoloRefineOverlapThreshold: '35',
        enableAuxYoloDetection: true,
        auxYoloConfThreshold: '0.55',
        auxYoloOverlapThreshold: '0.2'
      }
    })

    const store = useSettingsStore()
    const loaded = await store.loadFromBackend()

    expect(loaded).toBe(true)
    expect(store.settings.enableSaberYoloRefine).toBe(false)
    expect(store.settings.saberYoloRefineOverlapThreshold).toBe(35)
    expect(store.settings.enableAuxYoloDetection).toBe(true)
    expect(store.settings.auxYoloConfThreshold).toBe(0.55)
    expect(store.settings.auxYoloOverlapThreshold).toBe(0.2)
  })

  it('saves enableSaberYoloRefine to backend settings', async () => {
    const store = useSettingsStore()
    store.settings.enableSaberYoloRefine = false
    store.settings.saberYoloRefineOverlapThreshold = 35
    store.settings.enableAuxYoloDetection = true
    store.settings.auxYoloConfThreshold = 0.55
    store.settings.auxYoloOverlapThreshold = 0.2

    const saved = await store.saveToBackend()

    expect(saved).toBe(true)
    expect(saveUserSettingsMock).toHaveBeenCalledWith(expect.objectContaining({
      enableSaberYoloRefine: false,
      saberYoloRefineOverlapThreshold: '35',
      enableAuxYoloDetection: true,
      auxYoloConfThreshold: '0.55',
      auxYoloOverlapThreshold: '0.2'
    }))
  })
})
