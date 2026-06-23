import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useEditMode } from '@/composables/useEditMode'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { createBubbleState } from '@/utils/bubbleFactory'

describe('useEditMode', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('syncs an empty bubble array on no-render exit without routine logs', () => {
    const consoleLog = vi.spyOn(console, 'log').mockImplementation(() => {})
    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()
    const editMode = useEditMode()

    imageStore.addImage('page.png', 'data:image/png;base64,page', {
      bubbleStates: [
        createBubbleState({ coords: [0, 0, 120, 120] }),
      ],
    })
    bubbleStore.clearBubblesLocal()
    editMode.isActive.value = true

    editMode.exitEditModeWithoutRender()

    expect(imageStore.currentImage?.bubbleStates).toEqual([])
    expect(editMode.isActive.value).toBe(false)
    expect(consoleLog).not.toHaveBeenCalled()
  })
})
