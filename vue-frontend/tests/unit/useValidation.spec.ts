import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useSettingsStore } from '@/stores/settings'

describe('useValidation', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    localStorage.clear()
  })

  it('exposes settings highlight state without querying the header DOM', async () => {
    vi.useFakeTimers()
    const { useValidation } = await import('@/composables/useValidation')
    const validation = useValidation()

    try {
      const store = useSettingsStore()
      store.setTranslationProvider('deepseek')
      store.updateTranslationService({
        apiKey: '',
        modelName: '',
      })
      expect(validation.validateBeforeTranslation()).toBe(false)

      expect(validation.isSettingsButtonHighlighted.value).toBe(true)

      vi.advanceTimersByTime(3000)

      expect(validation.isSettingsButtonHighlighted.value).toBe(false)
    } finally {
      vi.useRealTimers()
    }
  })

  it('requires a Base URL for every custom proofreading round', async () => {
    const { useValidation } = await import('@/composables/useValidation')
    const settingsStore = useSettingsStore()
    settingsStore.settings.proofreading.rounds = [{
      ...settingsStore.settings.hqTranslation,
      id: '11111111-1111-4111-8111-111111111111',
      name: '第一轮',
      provider: 'custom',
      apiKey: 'secret',
      modelName: 'custom-model',
      customBaseUrl: '',
    }]

    expect(useValidation().validateBeforeTranslation('proofread')).toBe(false)
  })

  it('rejects a provider that does not support normal translation', async () => {
    const { useValidation } = await import('@/composables/useValidation')
    const settingsStore = useSettingsStore()
    settingsStore.settings.translation.provider = 'openai' as never
    settingsStore.settings.translation.apiKey = 'secret'
    settingsStore.settings.translation.modelName = 'gpt-4o'

    expect(useValidation().validateBeforeTranslation('normal')).toBe(false)
  })
})
