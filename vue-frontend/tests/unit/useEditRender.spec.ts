import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent, h } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { createBubbleState } from '@/utils/bubbleFactory'

const {
  flushPageDocumentMock,
  getPageSummaryMock,
  queuePageDocumentSaveMock,
} = vi.hoisted(() => ({
  flushPageDocumentMock: vi.fn(),
  getPageSummaryMock: vi.fn(),
  queuePageDocumentSaveMock: vi.fn(),
}))

vi.mock('@/api/v2/content', () => ({
  getPageSummary: getPageSummaryMock,
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

function readySummary() {
  return {
    id: 'page-1',
    chapterId: 'chapter-1',
    ordinal: 1,
    logicalSourcePath: 'page-1.png',
    sourceRevision: 1,
    documentRevision: 4,
    renderedRevision: 4,
    renderStatus: 'ready',
    sourceUrl: '/api/v2/assets/source-1',
    thumbnailSourceUrl: '/api/v2/assets/source-thumb-1',
    translatedUrl: '/api/v2/assets/translated-1',
    thumbnailTranslatedUrl: '/api/v2/assets/translated-thumb-1',
  }
}

describe('useEditRender backend-first orchestration', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    flushPageDocumentMock.mockReset().mockResolvedValue(undefined)
    queuePageDocumentSaveMock.mockReset().mockResolvedValue(undefined)
    getPageSummaryMock.mockReset().mockResolvedValue(readySummary())
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
    expect(getPageSummaryMock).toHaveBeenCalledTimes(1)
    expect(getPageSummaryMock).toHaveBeenCalledWith('page-1')
    expect(onRenderSuccess).toHaveBeenCalledWith('/api/v2/assets/translated-1')
    expect(useImageStore().currentImage).toMatchObject({
      documentRevision: 4,
      renderedRevision: 4,
      translatedAssetUrl: '/api/v2/assets/translated-1',
      thumbnailTranslatedUrl: '/api/v2/assets/translated-thumb-1',
    })
  })

  it('surfaces backend render failure and stays on the current image', async () => {
    getPageSummaryMock.mockResolvedValue({
      ...readySummary(),
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

  it('ignores a completed poll after the owning component unmounts', async () => {
    let resolveSummary!: (value: ReturnType<typeof readySummary>) => void
    getPageSummaryMock.mockReturnValue(new Promise(resolve => {
      resolveSummary = resolve
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
    resolveSummary(readySummary())

    await expect(pending).resolves.toBe(false)
    expect(onRenderSuccess).not.toHaveBeenCalled()
  })
})
