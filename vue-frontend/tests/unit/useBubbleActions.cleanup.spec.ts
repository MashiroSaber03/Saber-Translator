import { flushPromises, mount } from '@vue/test-utils'
import { defineComponent, h } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useBubbleStore } from '@/stores/bubbleStore'
import { createBubbleState } from '@/utils/bubbleFactory'
import { useBubbleActions } from '@/composables/useBubbleActions'

describe('useBubbleActions cleanup', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('clears pending preview timers when the owning component unmounts', async () => {
    const onDelayedPreview = vi.fn()

    const Harness = defineComponent({
      setup() {
        const bubbleStore = useBubbleStore()
        bubbleStore.setBubbles([
          createBubbleState({
            coords: [0, 0, 120, 80],
            polygon: [],
          }),
        ])
        bubbleStore.selectBubble(0)

        return useBubbleActions({ onDelayedPreview })
      },
      render() {
        return h('button', {
          class: 'trigger-update',
          onClick: () => this.handleBubbleUpdate({ translatedText: 'cleanup-check' }),
        })
      },
    })

    const wrapper = mount(Harness)
    await wrapper.find('.trigger-update').trigger('click')
    wrapper.unmount()

    await vi.runAllTimersAsync()
    await flushPromises()

    expect(onDelayedPreview).not.toHaveBeenCalled()
  })
})
