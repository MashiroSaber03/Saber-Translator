import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent, h, ref } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useEditWorkspace } from '@/components/edit/useEditWorkspace'
import { useImageStore } from '@/stores/imageStore'
import { useBubbleStore } from '@/stores/bubbleStore'
import { TEXT_STYLE_DEFAULTS } from '@/defaults/textStyleDefaults'
import { addTestImage } from '../helpers/imageFixtures'

const mocks = vi.hoisted(() => ({ getPageDocument: vi.fn(), mutation: vi.fn(), toast: vi.fn() }))
vi.mock('@/api/v2/content', () => ({ getPageDocument: mocks.getPageDocument }))
vi.mock('@/services/pageDocumentPersistence', () => ({
  queuePageDocumentMutation: mocks.mutation,
  queuePageDocumentSave: vi.fn(), flushPageDocument: vi.fn(),
  registerPageDocument: () => [],
}))
vi.mock('@/utils/toast', async importOriginal => ({
  ...await importOriginal<typeof import('@/utils/toast')>(),
  showToast: mocks.toast,
}))

beforeEach(() => {
  vi.clearAllMocks()
  setActivePinia(createPinia())
  addTestImage(useImageStore(), 'empty.jpg', '/source', {
    id: 'page', chapterId: 'chapter', documentRevision: 4, bubbleStates: [], width: 200, height: 200,
  })
  mocks.getPageDocument.mockResolvedValue({
    pageId: 'page', chapterId: 'chapter', documentRevision: 4,
    bubbles: [], defaultFontId: null, pageStyleDefaults: { ...TEXT_STYLE_DEFAULTS },
  })
  mocks.mutation.mockResolvedValue(undefined)
})

async function harness(ordinaryActive = ref(true)) {
  let editor!: ReturnType<typeof useEditWorkspace>
  const wrapper = mount(defineComponent({
    setup() { editor = useEditWorkspace(vi.fn(), ordinaryActive); return () => h('div') },
  }))
  await flushPromises()
  return { editor, wrapper }
}

describe('repair choices without a selected text box', () => {
  it('brings the next off-screen bubble into view without changing its geometry or zoom', async () => {
    const { editor, wrapper } = await harness()
    const store = useBubbleStore()
    store.addBubble([0, 0, 50, 50])
    store.addBubble([1000, 2000, 1100, 2100])
    store.selectBubble(0)
    const viewport = document.createElement('div')
    vi.spyOn(viewport, 'getBoundingClientRect').mockReturnValue({ width: 300, height: 400 } as DOMRect)
    editor.imageComparisonRef.value = { originalViewportRef: viewport, translatedViewportRef: viewport } as never
    editor.selectNextBubble()
    expect(editor.toolbarScale.value).toBe(1)
    expect(editor.translatedTransformStyle.value.transform).toBe('translate(-900px, -1850px) scale(1)')
    expect(editor.originalTransformStyle.value.transform).toBe(editor.translatedTransformStyle.value.transform)
    expect(store.bubbles[1]!.coords).toEqual([1000, 2000, 1100, 2100])
    wrapper.unmount()
  })
  it('fits an image loaded while hidden, without resetting a completed fit on mode switches', async () => {
    const active = ref(false)
    const { editor, wrapper } = await harness(active)
    const image = document.createElement('img')
    Object.defineProperties(image, { naturalWidth: { value: 3065 }, naturalHeight: { value: 12288 } })
    const viewport = document.createElement('div')
    const bounds = vi.spyOn(viewport, 'getBoundingClientRect').mockReturnValue({ width: 0, height: 0 } as DOMRect)
    editor.imageComparisonRef.value = {
      originalViewportRef: viewport, translatedViewportRef: viewport,
      originalImageRef: image, translatedImageRef: image,
    } as never
    editor.fitToScreen()
    expect(editor.toolbarScale.value).toBe(1)
    bounds.mockReturnValue({ width: 800, height: 700 } as DOMRect)
    active.value = true
    await flushPromises()
    expect(editor.toolbarScale.value).toBeCloseTo(700 / 12288 * 0.95)
    editor.zoomIn()
    const zoomed = editor.toolbarScale.value
    active.value = false
    await flushPromises()
    active.value = true
    await flushPromises()
    expect(editor.toolbarScale.value).toBe(zoomed)
    wrapper.unmount()
  })
  it('recovers ordinary controls after fine editing reloads the current document', async () => {
    mocks.getPageDocument.mockRejectedValueOnce(new Error('offline'))
    const { editor, wrapper } = await harness()
    expect(editor.isBusy.value).toBe(true)
    editor.handleDocumentLoaded('another-page')
    expect(editor.isBusy.value).toBe(true)
    editor.handleDocumentLoaded('page')
    await flushPromises()
    expect(editor.isBusy.value).toBe(false)
    wrapper.unmount()
  })
  it('keeps the selected text box when the same page is loaded again', async () => {
    const { editor, wrapper } = await harness()
    const store = useBubbleStore()
    store.addBubble([10, 10, 40, 60])
    store.addBubble([80, 80, 120, 160])
    store.selectBubble(1)

    editor.handleDocumentLoaded('page')
    expect(store.selectedIndex).toBe(1)

    store.clearSelection()
    editor.handleDocumentLoaded('page')
    expect(store.selectedIndex).toBe(0)
    wrapper.unmount()
  })
  it('saves the chosen method and color as page defaults through existing persistence', async () => {
    const { editor, wrapper } = await harness()
    editor.handleBubbleUpdateWithSync({ inpaintMethod: 'solid' })
    editor.handleBubbleUpdateWithSync({ fillColor: '#33aa66' })
    expect(mocks.mutation).toHaveBeenLastCalledWith('page', 4, [], {
      pageStyleDefaultsPatch: { fillColor: '#33aa66' },
    })
    expect(editor.currentInpaintMethod.value).toBe('solid')
    expect(editor.currentFillColor.value).toBe('#33aa66')
    wrapper.unmount()
  })

  it('keeps selected text-box repair settings on the existing bubble save path', async () => {
    const { editor, wrapper } = await harness()
    const bubbles = useBubbleStore()
    bubbles.addBubble([10, 10, 80, 80])
    bubbles.selectBubble(0)
    editor.handleBubbleUpdateWithSync({ inpaintMethod: 'solid', fillColor: '#33aa66' })
    expect(bubbles.bubbles[0]?.inpaintMethod).toBe('solid')
    expect(bubbles.bubbles[0]?.fillColor).toBe('#33aa66')
    expect(mocks.mutation).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('keeps fine-editor repair choices on the page when the ordinary selection reloads', async () => {
    const { editor, wrapper } = await harness(ref(false))
    const bubbles = useBubbleStore()
    bubbles.addBubble([10, 10, 80, 80])
    bubbles.updateBubble(0, { inpaintMethod: 'solid', fillColor: '#abcdef' })
    bubbles.selectBubble(0)
    editor.handleBubbleUpdateWithSync({ inpaintMethod: 'litelama' }, true)
    editor.handleBubbleUpdateWithSync({ fillColor: '#33aa66' }, true)
    bubbles.setBubbles([...bubbles.bubbles], true)
    await flushPromises()
    expect(useImageStore().currentImage).toMatchObject({ inpaintMethod: 'litelama', fillColor: '#33aa66' })
    expect(bubbles.bubbles[0]).toMatchObject({ inpaintMethod: 'solid', fillColor: '#abcdef' })
    expect(mocks.mutation).toHaveBeenLastCalledWith('page', 4, expect.any(Array), {
      pageStyleDefaultsPatch: { fillColor: '#33aa66' },
    })
    wrapper.unmount()
  })

  it('reports a save failure while retaining the local choice', async () => {
    const { editor, wrapper } = await harness()
    mocks.mutation.mockRejectedValueOnce(new Error('disk full'))
    editor.handleBubbleUpdateWithSync({ fillColor: '#33aa66' })
    await flushPromises()
    expect(mocks.toast).toHaveBeenLastCalledWith('disk full', 'error')
    expect(editor.currentFillColor.value).toBe('#33aa66')
    wrapper.unmount()
  })
})
