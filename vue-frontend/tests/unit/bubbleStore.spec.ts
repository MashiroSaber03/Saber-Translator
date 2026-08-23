import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import { createBubbleState } from '@/utils/bubbleFactory'
import { addTestImage } from '../helpers/imageFixtures'

describe('bubbleStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('leaves automatic direction to the backend when bubble coords change', () => {
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

    expect(bubbleStore.bubbles[0]?.autoTextDirection).toBe('horizontal')
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

  it('owns its bubble state instead of mutating the caller snapshot', () => {
    const bubbleStore = useBubbleStore()
    const external = [createBubbleState({
      backendBubbleId: 'bubble-1',
      coords: [0, 0, 200, 100],
      translatedText: 'backend snapshot',
    })]

    bubbleStore.setBubbles(external, true)
    bubbleStore.updateBubble(0, { translatedText: 'local edit' })

    expect(bubbleStore.bubbles[0]?.translatedText).toBe('local edit')
    expect(external[0]?.translatedText).toBe('backend snapshot')
  })

  it('stores newly drawn bubble coordinates as backend-safe integers', () => {
    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()
    addTestImage(imageStore, 'page.png', '/api/v2/assets/source-1')

    const bubble = bubbleStore.addBubble([10.4, 20.6, 110.7, 220.2])

    expect(bubble.coords).toEqual([10, 21, 111, 220])
  })

  it('inherits new bubble styles from the authoritative current page', () => {
    const imageStore = useImageStore()
    const settingsStore = useSettingsStore()
    const bubbleStore = useBubbleStore()
    settingsStore.settings.textStyle.fontFamily = 'stale-global-font'
    settingsStore.settings.textStyle.fontSize = 12
    addTestImage(imageStore, 'custom-font-page.png', '/api/v2/assets/source-2', {
      documentRevision: 3,
      fontFamily: 'uploaded-font-id',
      fontSize: 41,
      inlineAlign: 'center',
      blockAlign: 'end',
    })

    const bubble = bubbleStore.addBubble([10, 20, 110, 220])

    expect(bubble).toMatchObject({
      fontFamily: 'uploaded-font-id',
      fontSize: 41,
      inlineAlign: 'center',
      blockAlign: 'end',
    })
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
