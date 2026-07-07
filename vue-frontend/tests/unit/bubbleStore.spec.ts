import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { createBubbleState } from '@/utils/bubbleFactory'

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

  it('syncs the current image mirror when resetting to the initial bubbles', () => {
    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()

    imageStore.addImage('page.png', 'data:image/png;base64,page')
    bubbleStore.setBubbles([
      createBubbleState({
        coords: [0, 0, 200, 100],
        translatedText: 'initial translation',
      }),
    ])
    bubbleStore.updateBubble(0, { translatedText: 'edited translation' })

    expect(imageStore.currentImage?.bubbleTexts).toEqual(['edited translation'])

    bubbleStore.resetToInitial()

    expect(bubbleStore.bubbles[0]?.translatedText).toBe('initial translation')
    expect(imageStore.currentImage?.bubbleTexts).toEqual(['initial translation'])
    expect(imageStore.currentImage?.bubbleStates?.[0]?.translatedText).toBe('initial translation')
  })

  it('does not write routine console logs for normal bubble state transitions', () => {
    const consoleLog = vi.spyOn(console, 'log').mockImplementation(() => {})
    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()

    imageStore.addImage('page.png', 'data:image/png;base64,page')
    bubbleStore.setBubbles([
      createBubbleState({ coords: [0, 0, 200, 100] }),
      createBubbleState({ coords: [20, 20, 120, 220] }),
    ])
    bubbleStore.addBubble([40, 40, 120, 180])
    bubbleStore.selectBubble(0)
    bubbleStore.toggleMultiSelect(1)
    bubbleStore.updateBubble(0, { coords: [0, 0, 100, 220] })
    bubbleStore.updateSelectedBubble({ translatedText: 'updated selected' })
    bubbleStore.updateAllSelected({ fillColor: '#ffffff' })
    bubbleStore.updateAllBubbles({ textColor: '#111111' })
    bubbleStore.deleteBubble(2)
    bubbleStore.deleteSelected()
    bubbleStore.clearBubbles()
    bubbleStore.clearBubblesLocal()
    bubbleStore.resetToInitial()
    bubbleStore.saveAsInitial()

    expect(consoleLog).not.toHaveBeenCalled()
  })
})
