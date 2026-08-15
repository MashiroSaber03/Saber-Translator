import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent, h } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { createBubbleState } from '@/utils/bubbleFactory'

const {
  flushPageDocumentMock,
  getPageRenderStatusMock,
  queuePageDocumentSaveMock,
} = vi.hoisted(() => ({
  flushPageDocumentMock: vi.fn(),
  getPageRenderStatusMock: vi.fn(),
  queuePageDocumentSaveMock: vi.fn(),
}))

vi.mock('@/api/v2/content', () => ({
  getPageRenderStatus: getPageRenderStatusMock,
}))

vi.mock('@/services/pageDocumentPersistence', () => ({
  flushPageDocument: flushPageDocumentMock,
  queuePageDocumentSave: queuePageDocumentSaveMock,
}))

function seedEditor(): void {
  const imageStore = useImageStore()
  imageStore.setImages([{
    id: 'page-1',
    chapterId: 'chapter-1',
    documentRevision: 3,
    fileName: 'page-1.png',
    sourceAssetUrl: '/api/v2/assets/source-1',
    translatedAssetUrl: null,
  }])
  useBubbleStore().setBubbles([
    createBubbleState({
      backendBubbleId: 'bubble-1',
      coords: [0, 0, 100, 60],
      translatedText: '译文',
    }),
  ], true)
}

function readyStatus() {
  return {
    pageId: 'page-1',
    documentRevision: 4,
    renderedRevision: 4,
    renderStatus: 'ready',
    translatedUrl: '/api/v2/assets/translated-1',
  }
}

describe('useEditRender backend-first orchestration', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    flushPageDocumentMock.mockReset().mockResolvedValue(undefined)
    queuePageDocumentSaveMock.mockReset().mockResolvedValue(undefined)
    getPageRenderStatusMock.mockReset().mockResolvedValue(readyStatus())
    seedEditor()
  })

  it('persists the page and polls only that page until backend rendering is ready', async () => {
    const onRenderSuccess = vi.fn()
    const { useEditRender } = await import('@/composables/useEditRender')
    const { reRenderFullImage } = useEditRender({ onRenderSuccess })

    await expect(reRenderFullImage()).resolves.toBe(true)
    expect(queuePageDocumentSaveMock).toHaveBeenCalledWith(
      'page-1',
      3,
      useBubbleStore().bubbles,
    )
    expect(flushPageDocumentMock).toHaveBeenCalledWith('page-1')
    expect(getPageRenderStatusMock).toHaveBeenCalledTimes(1)
    expect(getPageRenderStatusMock).toHaveBeenCalledWith('page-1', expect.any(AbortSignal))
    expect(onRenderSuccess).toHaveBeenCalledWith('/api/v2/assets/translated-1')
    expect(useImageStore().currentImage).toMatchObject({
      documentRevision: 4,
      renderedRevision: 4,
      translatedAssetUrl: '/api/v2/assets/translated-1',
    })
  })

  it('surfaces backend render failure and stays on the current image', async () => {
    getPageRenderStatusMock.mockResolvedValue({
      ...readyStatus(),
      renderStatus: 'render_failed',
      renderedRevision: null,
      translatedUrl: null,
    })
    const onRenderError = vi.fn()
    const { useEditRender } = await import('@/composables/useEditRender')
    const { reRenderFullImage } = useEditRender({ onRenderError })

    await expect(reRenderFullImage()).resolves.toBe(false)
    expect(onRenderError).toHaveBeenCalledWith('后端渲染失败')
  })

  it('rejects a render status response for another page', async () => {
    getPageRenderStatusMock.mockResolvedValue({
      ...readyStatus(),
      pageId: 'other-page',
    })
    const onRenderError = vi.fn()
    const { useEditRender } = await import('@/composables/useEditRender')

    await expect(useEditRender({ onRenderError }).reRenderFullImage()).resolves.toBe(false)

    expect(onRenderError).toHaveBeenCalledWith(
      '页面 page-1 的渲染状态身份不匹配',
    )
  })

  it('ignores a completed poll after the owning component unmounts', async () => {
    let resolveStatus!: (value: ReturnType<typeof readyStatus>) => void
    getPageRenderStatusMock.mockReturnValue(new Promise(resolve => {
      resolveStatus = resolve
    }))
    const onRenderSuccess = vi.fn()
    const { useEditRender } = await import('@/composables/useEditRender')
    const Harness = defineComponent({
      setup() {
        const render = useEditRender({ onRenderSuccess })
        return { start: render.reRenderFullImage }
      },
      render: () => h('div'),
    })
    const wrapper = mount(Harness)
    const pending = (wrapper.vm as unknown as { start: () => Promise<boolean> }).start()
    await flushPromises()
    wrapper.unmount()
    resolveStatus(readyStatus())

    await expect(pending).resolves.toBe(false)
    expect(onRenderSuccess).not.toHaveBeenCalled()
    expect(useImageStore().currentImage).toMatchObject({
      documentRevision: 3,
      translatedAssetUrl: null,
    })
    expect(useImageStore().currentImage?.renderedRevision).toBeUndefined()
  })

  it('does not let a superseded render response overwrite the latest render state', async () => {
    let resolveFirst!: (value: ReturnType<typeof readyStatus>) => void
    getPageRenderStatusMock
      .mockImplementationOnce(() => new Promise(resolve => {
        resolveFirst = resolve
      }))
      .mockResolvedValueOnce({
        ...readyStatus(),
        documentRevision: 5,
        renderedRevision: 5,
        translatedUrl: '/api/v2/assets/translated-latest',
      })
    const onRenderSuccess = vi.fn()
    const { useEditRender } = await import('@/composables/useEditRender')
    const render = useEditRender({ onRenderSuccess })

    const first = render.reRenderFullImage()
    await vi.waitFor(() => expect(getPageRenderStatusMock).toHaveBeenCalledTimes(1))
    const second = render.reRenderFullImage()

    await expect(second).resolves.toBe(true)
    resolveFirst({
      ...readyStatus(),
      translatedUrl: '/api/v2/assets/translated-stale',
    })
    await expect(first).resolves.toBe(false)

    expect(useImageStore().currentImage).toMatchObject({
      documentRevision: 5,
      renderedRevision: 5,
      translatedAssetUrl: '/api/v2/assets/translated-latest',
    })
    expect(onRenderSuccess).toHaveBeenCalledTimes(1)
    expect(onRenderSuccess).toHaveBeenCalledWith('/api/v2/assets/translated-latest')
  })
})
