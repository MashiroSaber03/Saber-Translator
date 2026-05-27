import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useSettingsStore } from './settings'

describe('useSettingsStore factory defaults', () => {
  beforeEach(() => {
    localStorage.clear()
    setActivePinia(createPinia())
  })

  it('uses the expected source defaults for translation settings', () => {
    const store = useSettingsStore()

    expect(store.settings.textStyle.autoFontSize).toBe(true)
    expect(store.settings.hybridOcr.secondaryEngine).toBe('48px_ocr')
    expect(store.settings.translation.openaiOptions.execution.useStream).toBe(true)
    expect(store.settings.hqTranslation.openaiOptions.execution.transportRetries).toBe(3)
    expect(store.settings.hqTranslation.openaiOptions.execution.businessRetries).toBe(3)
    expect(store.settings.pluginAgent.openaiOptions.execution.rpmLimit).toBe(0)
    expect(store.settings.autoSaveInBookshelfMode).toBe(true)
  })
})
