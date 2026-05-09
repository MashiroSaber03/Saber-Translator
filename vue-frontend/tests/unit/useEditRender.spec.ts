import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useImageStore } from '@/stores/imageStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useSettingsStore } from '@/stores/settingsStore'
import { createBubbleState } from '@/utils/bubbleFactory'

const { executeRenderMock } = vi.hoisted(() => ({
  executeRenderMock: vi.fn(),
}))

vi.mock('@/composables/translation/core/steps', () => ({
  executeRender: executeRenderMock,
}))

describe('useEditRender', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    executeRenderMock.mockReset()
  })

  it('ignores render results when the user switches to another image before the request completes', async () => {
    let resolveRender!: (value: { finalImage: string; bubbleStates: any[] }) => void
    const pendingRender = new Promise<{ finalImage: string; bubbleStates: any[] }>((resolve) => {
      resolveRender = resolve
    })
    executeRenderMock.mockReturnValueOnce(pendingRender)

    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()

    imageStore.addImage('page-1.png', 'data:image/png;base64,page1')
    imageStore.addImage('page-2.png', 'data:image/png;base64,page2')
    bubbleStore.setBubbles([
      createBubbleState({
        coords: [0, 0, 120, 80],
        polygon: [],
        translatedText: '第一页译文',
      }),
    ])

    const { useEditRender } = await import('@/composables/useEditRender')
    const { reRenderFullImage } = useEditRender()

    const renderPromise = reRenderFullImage()

    imageStore.setCurrentImageIndex(1)
    resolveRender({ finalImage: 'rendered-page-1', bubbleStates: bubbleStore.bubbles })

    await expect(renderPromise).resolves.toBe(false)
    expect(imageStore.images[0]?.translatedDataURL).toBeNull()
    expect(imageStore.images[1]?.translatedDataURL).toBeNull()
  })

  it('uses preserve render policy for ordinary edit rerenders even if page-level auto options remain enabled', async () => {
    executeRenderMock.mockResolvedValue({
      finalImage: 'rendered-page-1',
      bubbleStates: [
        createBubbleState({
          coords: [0, 0, 120, 80],
          polygon: [],
          translatedText: '第一页译文',
          fontSize: 27,
          textColor: '#123456',
          fillColor: '#abcdef',
          autoFgColor: [1, 2, 3],
          autoBgColor: [10, 11, 12],
        }),
      ],
    })

    const imageStore = useImageStore()
    const bubbleStore = useBubbleStore()
    const settingsStore = useSettingsStore()

    settingsStore.updateTextStyle({
      autoFontSize: true,
      useAutoTextColor: true,
    })

    imageStore.addImage('page-1.png', 'data:image/png;base64,page1')
    imageStore.updateCurrentImage({
      translatedDataURL: 'data:image/png;base64,old-render',
      cleanImageData: 'clean-image',
    })

    bubbleStore.setBubbles([
      createBubbleState({
        coords: [0, 0, 120, 80],
        polygon: [],
        translatedText: '第一页译文',
        fontSize: 27,
        textColor: '#123456',
        fillColor: '#abcdef',
        autoFgColor: [1, 2, 3],
        autoBgColor: [10, 11, 12],
      }),
    ])

    const { useEditRender } = await import('@/composables/useEditRender')
    const { reRenderFullImage } = useEditRender()

    await expect(reRenderFullImage()).resolves.toBe(true)
    expect(executeRenderMock).toHaveBeenCalledWith(expect.objectContaining({
      renderStylePolicy: {
        fontSize: 'preserve',
        color: 'preserve',
      },
    }))
  })
})
