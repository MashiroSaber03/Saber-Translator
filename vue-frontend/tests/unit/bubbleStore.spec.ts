import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { createBubbleState } from '@/utils/bubbleFactory'
import { addTestImage } from '../helpers/imageFixtures'

describe('bubbleStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  afterEach(() => {
    vi.restoreAllMocks()
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

  it('keeps the selected bubble when backend state replaces the same stable bubbles', () => {
    const bubbleStore = useBubbleStore()
    bubbleStore.setBubbles([
      createBubbleState({
        backendBubbleId: 'bubble-1',
        coords: [0, 0, 200, 100],
        translatedText: 'before',
      }),
      createBubbleState({
        backendBubbleId: 'bubble-2',
        coords: [20, 20, 120, 220],
      }),
    ])
    bubbleStore.selectBubble(0)

    bubbleStore.setBubbles([
      createBubbleState({
        backendBubbleId: 'bubble-1',
        coords: [0, 0, 200, 100],
        translatedText: 'after',
      }),
      createBubbleState({
        backendBubbleId: 'bubble-2',
        coords: [20, 20, 120, 220],
      }),
    ], true)

    expect(bubbleStore.selectedIndex).toBe(0)
    expect(bubbleStore.selectedBubble?.backendBubbleId).toBe('bubble-1')
    expect(bubbleStore.selectedBubble?.translatedText).toBe('after')
  })

  it('does not write routine console logs for normal bubble state transitions', () => {
    const consoleLog = vi.spyOn(console, 'log').mockImplementation(() => {})
    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()

    addTestImage(imageStore, 'page.png', '/api/v2/assets/source-1')
    bubbleStore.setBubbles([
      createBubbleState({ coords: [0, 0, 200, 100] }),
      createBubbleState({ coords: [20, 20, 120, 220] }),
    ])
    bubbleStore.addBubble([40, 40, 120, 180])
    bubbleStore.selectBubble(0)
    bubbleStore.toggleMultiSelect(1)
    bubbleStore.updateBubble(0, { coords: [0, 0, 100, 220] })
    bubbleStore.updateSelectedBubble({ translatedText: 'updated selected' })
    bubbleStore.updateAllBubbles({ textColor: '#111111' })
    bubbleStore.deleteSelected()
    bubbleStore.clearBubbles()
    bubbleStore.clearBubblesLocal()
    bubbleStore.saveAsInitial()

    expect(consoleLog).not.toHaveBeenCalled()
  })
})
