import { flushPromises, mount } from '@vue/test-utils'
import { defineComponent, h } from 'vue'
import { beforeEach, describe, expect, it, vi, afterEach } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useBubbleStore } from '@/stores/bubbleStore'
import { createBubbleState } from '@/utils/bubbleFactory'
import { useBubbleActions } from '@/composables/useBubbleActions'

describe('useBubbleActions', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('runs one trailing preview render after updates arrive during an in-flight render', async () => {
    let resolveFirstRender!: () => void
    const onDelayedPreview = vi
      .fn<() => Promise<void>>()
      .mockImplementationOnce(
        () =>
          new Promise<void>((resolve) => {
            resolveFirstRender = resolve
          })
      )
      .mockResolvedValueOnce()

    const Harness = defineComponent({
      setup() {
        const bubbleStore = useBubbleStore()
        bubbleStore.setBubbles([
          createBubbleState({
            coords: [0, 0, 100, 60],
            polygon: [],
          }),
        ])
        bubbleStore.selectBubble(0)

        const actions = useBubbleActions({ onDelayedPreview })

        return {
          ...actions,
          triggerUpdate(text: string) {
            actions.handleBubbleUpdate({ translatedText: text })
          },
        }
      },
      render() {
        return h('div')
      },
    })

    const wrapper = mount(Harness)

    ;(wrapper.vm as unknown as { triggerUpdate: (text: string) => void }).triggerUpdate('第一次更新')
    await vi.advanceTimersByTimeAsync(150)
    expect(onDelayedPreview).toHaveBeenCalledTimes(1)

    ;(wrapper.vm as unknown as { triggerUpdate: (text: string) => void }).triggerUpdate('第二次更新')
    await vi.advanceTimersByTimeAsync(150)
    expect(onDelayedPreview).toHaveBeenCalledTimes(1)

    resolveFirstRender()
    await flushPromises()
    await vi.runOnlyPendingTimersAsync()
    await flushPromises()

    expect(onDelayedPreview).toHaveBeenCalledTimes(2)
  })
})
