import { beforeEach, describe, expect, it } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

import { useSettingsStore } from '@/stores/settings'

describe('hybrid OCR settings', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('enabling hybrid OCR normalizes to the recommended 48px -> MangaOCR combo', () => {
    const store = useSettingsStore()

    expect(store.settings.ocrEngine).toBe('manga_ocr')

    store.updateHybridOcr({ enabled: true })

    expect(store.settings.ocrEngine).toBe('48px_ocr')
    expect(store.settings.hybridOcr.secondaryEngine).toBe('manga_ocr')
    expect(store.settings.hybridOcr.confidenceThreshold).toBe(0.2)
  })

})
