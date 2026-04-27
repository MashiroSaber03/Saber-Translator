import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

import { useSettingsStore } from '@/stores/settingsStore'

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

  it('loadFromStorage migrates legacy threshold fields and unsupported combos', () => {
    storageState['translationSettings'] = JSON.stringify({
      ocrEngine: 'ai_vision',
      hybridOcr: {
        enabled: true,
        secondaryEngine: 'paddle_ocr',
        threshold48px: 0.35
      }
    })

    const store = useSettingsStore()
    store.loadFromStorage()

    expect(store.settings.ocrEngine).toBe('48px_ocr')
    expect(store.settings.hybridOcr.secondaryEngine).toBe('manga_ocr')
    expect(store.settings.hybridOcr.confidenceThreshold).toBe(0.35)
  })

  it('loadFromStorage migrates legacy custom provider ids and preserves zero-valued AI settings', () => {
    storageState['translationSettings'] = JSON.stringify({
      translation: {
        provider: 'custom_openai',
        rpmLimit: 0,
        maxRetries: 0
      },
      hqTranslation: {
        provider: 'custom_openai',
        rpmLimit: 0,
        maxRetries: 0
      },
      aiVisionOcr: {
        provider: 'custom_openai_vision',
        rpmLimit: 0,
        minImageSize: 0
      },
      proofreading: {
        maxRetries: 0
      }
    })

    const store = useSettingsStore()
    store.loadFromStorage()

    expect(store.settings.translation.provider).toBe('custom')
    expect(store.settings.translation.rpmLimit).toBe(0)
    expect(store.settings.translation.maxRetries).toBe(0)
    expect(store.settings.hqTranslation.provider).toBe('custom')
    expect(store.settings.hqTranslation.rpmLimit).toBe(0)
    expect(store.settings.hqTranslation.maxRetries).toBe(0)
    expect(store.settings.aiVisionOcr.provider).toBe('custom')
    expect(store.settings.aiVisionOcr.rpmLimit).toBe(0)
    expect(store.settings.aiVisionOcr.minImageSize).toBe(0)
    expect(store.settings.proofreading.maxRetries).toBe(0)
  })
})
