import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useSettingsStore } from '@/stores/settings'
import { createDefaultSettings } from '@/stores/settings/defaults'
import { normalizeSettings } from '@/stores/settings/normalizeSettings'

describe('settings chapter work state', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('accepts an empty chapter override after normalization', () => {
    const store = useSettingsStore()
    const normalized = createDefaultSettings()
    normalizeSettings(normalized)
    store.settings = normalized

    expect(store.hydrateChapterWorkState('chapter-1', {})).toBe(true)
  })

  it('accepts an empty override with normalized HQ and proofreading settings', () => {
    const store = useSettingsStore()
    const normalized = createDefaultSettings()
    normalized.hqTranslation.provider = 'siliconflow'
    normalized.hqTranslation.modelName = 'vision-model'
    normalized.proofreading.enabled = true
    normalized.proofreading.rounds = [{
      ...normalized.hqTranslation,
      name: '第1轮校对',
      modelName: 'text-model',
    }]
    normalizeSettings(normalized)
    store.settings = normalized

    const scrubbedWorkState = store.chapterWorkStatePayload()
    const rounds = (scrubbedWorkState.proofreading as {
      rounds: Array<Record<string, unknown>>
    }).rounds
    expect(rounds[0]).not.toHaveProperty('apiKey')
    expect(store.hydrateChapterWorkState('chapter-1', scrubbedWorkState)).toBe(true)
  })
})
