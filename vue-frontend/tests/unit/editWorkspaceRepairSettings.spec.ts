import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent, h } from 'vue'
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

async function harness() {
  let editor!: ReturnType<typeof useEditWorkspace>
  const wrapper = mount(defineComponent({
    setup() { editor = useEditWorkspace(vi.fn()); return () => h('div') },
  }))
  await flushPromises()
  return { editor, wrapper }
}

describe('repair choices without a selected text box', () => {
  it('saves the chosen method and color as page defaults through existing persistence', async () => {
    const { editor, wrapper } = await harness()
    editor.handleBubbleUpdateWithSync({ inpaintMethod: 'solid' })
    editor.handleBubbleUpdateWithSync({ fillColor: '#33aa66' })
    expect(mocks.mutation).toHaveBeenLastCalledWith('page', 4, [], {
      pageStyleDefaultsPatch: { inpaintMethod: 'solid', fillColor: '#33aa66' },
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
