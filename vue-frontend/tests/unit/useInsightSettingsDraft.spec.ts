import { nextTick, ref } from 'vue'
import { describe, expect, it, vi } from 'vitest'
import { useInsightSettingsDraft } from '@/components/insight/settings/useInsightSettingsDraft'

describe('useInsightSettingsDraft', () => {
  it('emits initial fields and preserves subsequent empty and zero values', async () => {
    const model = ref('stored-model')
    const temperature = ref(0.4)
    const emitDraft = vi.fn()
    useInsightSettingsDraft({
      sources: [model, temperature],
      buildDraft: () => ({ model: model.value, temperature: temperature.value }),
      emitDraft,
    })
    expect(emitDraft).toHaveBeenLastCalledWith({ model: 'stored-model', temperature: 0.4 })
    model.value = ''
    temperature.value = 0
    await nextTick()
    expect(emitDraft).toHaveBeenLastCalledWith({ model: '', temperature: 0 })
  })
})
