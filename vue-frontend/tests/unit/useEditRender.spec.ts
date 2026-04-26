import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useImageStore } from '@/stores/imageStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { createBubbleState } from '@/utils/bubbleFactory'

const { reRenderImageMock } = vi.hoisted(() => ({
  reRenderImageMock: vi.fn(),
}))

vi.mock('@/api/translate', () => ({
  reRenderImage: reRenderImageMock,
}))

describe('useEditRender', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    reRenderImageMock.mockReset()
  })

  it('ignores render results when the user switches to another image before the request completes', async () => {
    let resolveRender!: (value: { rendered_image: string }) => void
    const pendingRender = new Promise<{ rendered_image: string }>((resolve) => {
      resolveRender = resolve
    })
    reRenderImageMock.mockReturnValueOnce(pendingRender)

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
    resolveRender({ rendered_image: 'rendered-page-1' })

    await expect(renderPromise).resolves.toBe(false)
    expect(imageStore.images[0]?.translatedDataURL).toBeNull()
    expect(imageStore.images[1]?.translatedDataURL).toBeNull()
  })
})
