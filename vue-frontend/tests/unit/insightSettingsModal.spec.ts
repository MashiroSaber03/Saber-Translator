import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import InsightSettingsModal from '@/components/insight/InsightSettingsModal.vue'

const apiMocks = vi.hoisted(() => ({
  getGlobalConfig: vi.fn(),
  saveGlobalConfig: vi.fn(),
}))

vi.mock('@/api/insight', () => ({
  getGlobalConfig: apiMocks.getGlobalConfig,
  saveGlobalConfig: apiMocks.saveGlobalConfig,
}))

const baseModalStub = defineComponent({
  emits: ['close'],
  template: '<div><slot /><slot name="footer" /></div>',
})

function createSettingsTabStub() {
  return defineComponent({
    setup(_, { expose }) {
      expose({
        getConfig: () => ({}),
        getCustomPrompts: () => ({}),
        syncFromStore: vi.fn(),
      })
      return {}
    },
    template: '<div />',
  })
}

describe('InsightSettingsModal', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    setActivePinia(createPinia())
    apiMocks.getGlobalConfig.mockReset().mockResolvedValue({ success: true, config: {} })
    apiMocks.saveGlobalConfig.mockReset().mockResolvedValue({ success: true })
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('clears delayed message and save-success timers when the modal unmounts', async () => {
    const wrapper = mount(InsightSettingsModal, {
      global: {
        plugins: [createPinia()],
        stubs: {
          BaseModal: baseModalStub,
          VlmSettingsTab: createSettingsTabStub(),
          LlmSettingsTab: createSettingsTabStub(),
          BatchSettingsTab: createSettingsTabStub(),
          EmbeddingSettingsTab: createSettingsTabStub(),
          RerankerSettingsTab: createSettingsTabStub(),
          PromptsSettingsTab: createSettingsTabStub(),
          ImageGenSettingsTab: createSettingsTabStub(),
        },
      },
    })

    await flushPromises()

    const saveButton = wrapper.findAll('button').find(button => button.text() === '保存')
    expect(saveButton).toBeTruthy()

    await saveButton!.trigger('click')
    await flushPromises()
    expect(apiMocks.saveGlobalConfig).toHaveBeenCalled()

    const clearTimeoutSpy = vi.spyOn(globalThis, 'clearTimeout')
    wrapper.unmount()

    expect(clearTimeoutSpy.mock.calls.length).toBeGreaterThanOrEqual(2)
  })
})
