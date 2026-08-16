import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useSettingsStore } from '@/stores/settings'
import { createDefaultSettings } from '@/stores/settings/defaults'

describe('settings chapter work state', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('accepts an empty chapter override for current settings', () => {
    const store = useSettingsStore()
    store.settings = createDefaultSettings()

    expect(store.hydrateChapterWorkState('chapter-1', {})).toBe(true)
  })

  it('round-trips the PaddleOCR-VL prompt language in chapter work state', () => {
    const store = useSettingsStore()
    store.updatePaddleOcrVl({ sourceLanguage: 'thai' })

    expect(store.chapterWorkStatePayload()).toMatchObject({
      paddleOcrVl: { sourceLanguage: 'thai' },
    })
    expect(store.hydrateChapterWorkState('chapter-1', {
      paddleOcrVl: { sourceLanguage: 'korean' },
    })).toBe(true)
    expect(store.settings.paddleOcrVl.sourceLanguage).toBe('korean')
  })

  it('accepts an empty override with current HQ and proofreading settings', () => {
    const store = useSettingsStore()
    const normalized = createDefaultSettings()
    normalized.hqTranslation.provider = 'siliconflow'
    normalized.hqTranslation.modelName = 'vision-model'
    normalized.proofreading.enabled = true
    normalized.proofreading.rounds = [{
      ...normalized.hqTranslation,
      id: '11111111-1111-4111-8111-111111111111',
      name: '第1轮校对',
      modelName: 'text-model',
    }]
    store.settings = normalized

    const scrubbedWorkState = store.chapterWorkStatePayload()
    const rounds = (scrubbedWorkState.proofreading as {
      rounds: Array<Record<string, unknown>>
    }).rounds
    expect(rounds[0]).not.toHaveProperty('apiKey')
    expect(store.hydrateChapterWorkState('chapter-1', scrubbedWorkState)).toBe(true)
  })
})
