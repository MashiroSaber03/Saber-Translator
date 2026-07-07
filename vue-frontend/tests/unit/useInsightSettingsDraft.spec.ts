import { nextTick, ref } from 'vue'
import { describe, expect, it, vi } from 'vitest'

import { useInsightSettingsDraft } from '@/components/insight/settings/useInsightSettingsDraft'

type DraftConfig = {
  provider: string
  model: string
  extra: {
    temperature: number
  }
}

describe('useInsightSettingsDraft', () => {
  it('emits the current draft immediately and when watched fields change', async () => {
    const provider = ref('openai')
    const model = ref('gpt-4o-mini')
    const temperature = ref(0.4)
    const emitDraft = vi.fn()

    useInsightSettingsDraft<DraftConfig>({
      sources: [provider, model, temperature],
      buildDraft: () => ({
        provider: provider.value,
        model: model.value,
        extra: {
          temperature: temperature.value,
        },
      }),
      applyDraft: (config) => {
        provider.value = config.provider
        model.value = config.model
        temperature.value = config.extra.temperature
      },
      loadDraft: () => ({
        provider: 'openai',
        model: 'gpt-4o-mini',
        extra: {
          temperature: 0.4,
        },
      }),
      emitDraft,
    })

    expect(emitDraft).toHaveBeenLastCalledWith({
      provider: 'openai',
      model: 'gpt-4o-mini',
      extra: {
        temperature: 0.4,
      },
    })

    model.value = 'gpt-4.1-mini'
    await nextTick()

    expect(emitDraft).toHaveBeenLastCalledWith({
      provider: 'openai',
      model: 'gpt-4.1-mini',
      extra: {
        temperature: 0.4,
      },
    })
  })

  it('reloads from the source draft when a parent sync request changes', async () => {
    const provider = ref('openai')
    const model = ref('gpt-4o-mini')
    const syncRequestId = ref(0)
    const nextStoreDraft = ref<DraftConfig>({
      provider: 'ollama',
      model: 'qwen2.5vl',
      extra: {
        temperature: 0.2,
      },
    })
    const emitDraft = vi.fn()

    useInsightSettingsDraft<DraftConfig>({
      sources: [provider, model],
      buildDraft: () => ({
        provider: provider.value,
        model: model.value,
        extra: {
          temperature: nextStoreDraft.value.extra.temperature,
        },
      }),
      applyDraft: (config) => {
        provider.value = config.provider
        model.value = config.model
      },
      loadDraft: () => nextStoreDraft.value,
      emitDraft,
      syncRequestId: () => syncRequestId.value,
    })

    syncRequestId.value += 1
    await nextTick()

    expect(provider.value).toBe('ollama')
    expect(model.value).toBe('qwen2.5vl')
    expect(emitDraft).toHaveBeenLastCalledWith({
      provider: 'ollama',
      model: 'qwen2.5vl',
      extra: {
        temperature: 0.2,
      },
    })
  })
})
