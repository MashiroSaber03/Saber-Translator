import { beforeEach, describe, expect, it } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useBubbleStore } from '@/stores/bubbleStore'
import { createBubbleState } from '@/utils/bubbleFactory'

describe('bubbleStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('recomputes autoTextDirection when bubble coords change', () => {
    const bubbleStore = useBubbleStore()

    bubbleStore.setBubbles([
      createBubbleState({
        coords: [0, 0, 200, 100],
        polygon: [],
        textDirection: 'auto',
        autoTextDirection: 'horizontal',
      }),
    ])

    bubbleStore.updateBubble(0, {
      coords: [0, 0, 100, 220],
    })

    expect(bubbleStore.bubbles[0]?.autoTextDirection).toBe('vertical')
  })
})
