import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useSettingsStore } from '@/stores/settings'

describe('settingsStore routine logging', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localStorage.clear()
  })

  afterEach(() => {
    vi.restoreAllMocks()
    localStorage.clear()
  })

  it('does not write routine console logs for normal settings updates', () => {
    const consoleLog = vi.spyOn(console, 'log').mockImplementation(() => {})
    const store = useSettingsStore()

    store.setTextDetector('ctd')
    store.setMinTextBlockAreaPercent(0.2)
    store.setEnableAuxYoloDetection(true)
    store.setAuxYoloConfThreshold(0.5)
    store.setAuxYoloOverlapThreshold(0.2)
    store.setEnableSaberYoloRefine(true)
    store.setSaberYoloRefineOverlapThreshold(60)
    store.setOcrEngine('manga_ocr')
    store.setSourceLanguage('japanese')
    store.setAiVisionOcrPromptMode('json')
    store.setTranslatePromptMode(false)
    store.saveTranslationProviderConfig('custom')
    store.restoreTranslationProviderConfig('custom')
    store.setPdfProcessingMethod('frontend')
    store.setAutoSaveInBookshelfMode(true)
    store.setRemoveTextWithOcr(true)
    store.setEnableVerboseLogs(true)
    store.setLamaDisableResize(true)
    store.resetToDefaults()

    expect(consoleLog).not.toHaveBeenCalled()
  })
})
