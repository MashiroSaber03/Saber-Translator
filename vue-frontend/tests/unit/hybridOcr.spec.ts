import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

import { STORAGE_KEY_TRANSLATION_SETTINGS } from '@/constants'
import { useSettingsStore } from '@/stores/settings'
import { createDefaultSettings } from '@/stores/settings/defaults'

describe('hybrid OCR settings', () => {
  const storageState: Record<string, string> = {}

  beforeEach(() => {
    setActivePinia(createPinia())
    for (const key of Object.keys(storageState)) {
      delete storageState[key]
    }

    vi.spyOn(Storage.prototype, 'getItem').mockImplementation((key: string) => storageState[key] ?? null)
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation((key: string, value: string) => {
      storageState[key] = value
    })
    vi.spyOn(Storage.prototype, 'removeItem').mockImplementation((key: string) => {
      delete storageState[key]
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('enabling hybrid OCR normalizes to the recommended 48px -> MangaOCR combo', () => {
    const store = useSettingsStore()

    expect(store.settings.ocrEngine).toBe('manga_ocr')

    store.updateHybridOcr({ enabled: true })

    expect(store.settings.ocrEngine).toBe('48px_ocr')
    expect(store.settings.hybridOcr.secondaryEngine).toBe('manga_ocr')
    expect(store.settings.hybridOcr.confidenceThreshold).toBe(0.2)
  })

  it('loadFromStorage keeps current-schema confidence threshold and normalizes unsupported combos', () => {
    const settings = createDefaultSettings()
    settings.ocrEngine = 'ai_vision'
    settings.hybridOcr = {
      enabled: true,
      secondaryEngine: 'paddle_ocr' as never,
      confidenceThreshold: 0.35
    }
    storageState[STORAGE_KEY_TRANSLATION_SETTINGS] = JSON.stringify(settings)

    const store = useSettingsStore()
    store.loadFromStorage()

    expect(store.settings.ocrEngine).toBe('48px_ocr')
    expect(store.settings.hybridOcr.secondaryEngine).toBe('manga_ocr')
    expect(store.settings.hybridOcr.confidenceThreshold).toBe(0.35)
  })

  it('loadFromStorage preserves current custom provider ids and zero-valued AI options', () => {
    const settings = createDefaultSettings()
    settings.translation.provider = 'custom'
    settings.translation.openaiOptions.execution.rpmLimit = 0
    settings.translation.openaiOptions.execution.businessRetries = 0
    settings.hqTranslation.provider = 'custom'
    settings.hqTranslation.openaiOptions.execution.rpmLimit = 0
    settings.hqTranslation.openaiOptions.execution.businessRetries = 0
    settings.aiVisionOcr.provider = 'custom'
    settings.aiVisionOcr.openaiOptions.execution.rpmLimit = 0
    settings.aiVisionOcr.minImageSize = 0
    settings.proofreading.maxRetries = 0
    storageState[STORAGE_KEY_TRANSLATION_SETTINGS] = JSON.stringify(settings)

    const store = useSettingsStore()
    store.loadFromStorage()

    expect(store.settings.translation.provider).toBe('custom')
    expect(store.settings.translation.openaiOptions.execution.rpmLimit).toBe(0)
    expect(store.settings.translation.openaiOptions.execution.businessRetries).toBe(0)
    expect(store.settings.hqTranslation.provider).toBe('custom')
    expect(store.settings.hqTranslation.openaiOptions.execution.rpmLimit).toBe(0)
    expect(store.settings.hqTranslation.openaiOptions.execution.businessRetries).toBe(0)
    expect(store.settings.aiVisionOcr.provider).toBe('custom')
    expect(store.settings.aiVisionOcr.openaiOptions.execution.rpmLimit).toBe(0)
    expect(store.settings.aiVisionOcr.minImageSize).toBe(0)
    expect(store.settings.proofreading.maxRetries).toBe(0)
  })
})
