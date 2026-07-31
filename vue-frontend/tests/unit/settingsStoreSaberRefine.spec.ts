import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

import type { V2SettingsDocument, V2SettingsTransaction } from '@/api/v2/settings'
import { useSettingsStore } from '@/stores/settings'
import { createDefaultSettings } from '@/stores/settings/defaults'

const settingsApiMocks = vi.hoisted(() => ({
  getV2Settings: vi.fn(),
  saveV2SettingsTransaction: vi.fn(),
}))

vi.mock('@/api/v2/settings', () => ({
  getV2Settings: settingsApiMocks.getV2Settings,
  saveV2SettingsTransaction: settingsApiMocks.saveV2SettingsTransaction,
}))

function settingsDocument(
  settings = createDefaultSettings(),
  revision = 6,
): V2SettingsDocument {
  return {
    settings: [
      {
        domain: 'translation',
        payload: settings as unknown as Record<string, unknown>,
        revision,
        schemaVersion: 3,
      },
      {
        domain: 'text_style_defaults',
        payload: settings.textStyle as unknown as Record<string, unknown>,
        revision,
        schemaVersion: 1,
      },
    ],
    bookSettings: [],
    providerSettings: [],
    credentials: [],
  }
}

describe('settings store saber yolo refine', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    settingsApiMocks.getV2Settings.mockReset()
    settingsApiMocks.saveV2SettingsTransaction.mockReset()
    settingsApiMocks.getV2Settings.mockResolvedValue(settingsDocument())
    settingsApiMocks.saveV2SettingsTransaction.mockResolvedValue({
      settings: [{ domain: 'translation', revision: 7 }],
      bookSettings: [],
      providerSettings: [],
      credentials: [],
    })
  })

  it('uses the current detection defaults', () => {
    const store = useSettingsStore()

    expect(store.settings.enableSaberYoloRefine).toBe(true)
    expect(store.settings.saberYoloRefineOverlapThreshold).toBe(50)
    expect(store.settings.enableAuxYoloDetection).toBe(false)
    expect(store.settings.auxYoloConfThreshold).toBe(0.4)
    expect(store.settings.auxYoloOverlapThreshold).toBe(0.1)
  })

  it('loads saber refine and auxiliary detector settings from backend', async () => {
    const settings = createDefaultSettings()
    settings.enableSaberYoloRefine = false
    settings.saberYoloRefineOverlapThreshold = 35
    settings.enableAuxYoloDetection = true
    settings.auxYoloConfThreshold = 0.55
    settings.auxYoloOverlapThreshold = 0.2
    settingsApiMocks.getV2Settings.mockResolvedValue(settingsDocument(settings))

    const store = useSettingsStore()

    expect(await store.loadFromBackend()).toBe(true)
    expect(store.settings).toMatchObject({
      enableSaberYoloRefine: false,
      saberYoloRefineOverlapThreshold: 35,
      enableAuxYoloDetection: true,
      auxYoloConfThreshold: 0.55,
      auxYoloOverlapThreshold: 0.2,
    })
  })

  it('saves detector settings through the v2 transaction', async () => {
    const store = useSettingsStore()
    expect(await store.loadFromBackend()).toBe(true)
    store.setEnableSaberYoloRefine(false)
    store.setSaberYoloRefineOverlapThreshold(35)
    store.setEnableAuxYoloDetection(true)
    store.setAuxYoloConfThreshold(0.55)
    store.setAuxYoloOverlapThreshold(0.2)

    expect(await store.saveToBackend()).toBe(true)

    const transaction = (
      settingsApiMocks.saveV2SettingsTransaction.mock.calls[0]?.[0]
    ) as V2SettingsTransaction
    expect(transaction.settings?.[0]?.payload).toMatchObject({
      enableSaberYoloRefine: false,
      saberYoloRefineOverlapThreshold: 35,
      enableAuxYoloDetection: true,
      auxYoloConfThreshold: 0.55,
      auxYoloOverlapThreshold: 0.2,
      settingsSchemaVersion: 3,
    })
  })
})
