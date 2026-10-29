import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent, h } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useBubbleActions } from '@/composables/useBubbleActions'
import { useBubbleStore } from '@/stores/bubbleStore'
import { useImageStore } from '@/stores/imageStore'
import { createDefaultSettings } from '@/stores/settings/defaults'
import { createBubbleState } from '@/utils/bubbleFactory'

const {
  getPageDocumentMock,
  queuePageDocumentSaveMock,
  queuePageDocumentMutationMock,
  registerPageDocumentMock,
  runBubbleRepairMock,
  runPageOperationMock,
  showToastMock,
} = vi.hoisted(() => ({
  getPageDocumentMock: vi.fn(),
  queuePageDocumentSaveMock: vi.fn(),
  queuePageDocumentMutationMock: vi.fn().mockResolvedValue(undefined),
  registerPageDocumentMock: vi.fn(),
  runBubbleRepairMock: vi.fn(),
  runPageOperationMock: vi.fn(),
  showToastMock: vi.fn(),
}))

vi.mock('@/api/v2/content', () => ({
  getPageDocument: getPageDocumentMock,
}))
vi.mock('@/api/v2/operations', () => ({
  runBubbleRepair: runBubbleRepairMock,
  runPageOperation: runPageOperationMock,
}))
vi.mock('@/services/pageDocumentPersistence', () => ({
  queuePageDocumentSave: queuePageDocumentSaveMock,
  queuePageDocumentMutation: queuePageDocumentMutationMock,
  registerPageDocument: registerPageDocumentMock,
}))
vi.mock('@/utils/toast', () => ({
  showToast: showToastMock,
}))

function seedEditor(): void {
  useImageStore().setImages([{
    id: 'page-1',
    chapterId: 'chapter-1',
    documentRevision: 7,
    fileName: 'page.png',
    sourceAssetUrl: '/api/v2/assets/source-1',
    translatedAssetUrl: null,
  }])
  const bubbleStore = useBubbleStore()
  bubbleStore.setBubbles([
    createBubbleState({
      backendBubbleId: 'bubble-1',
      coords: [0, 0, 100, 60],
      originalText: '旧文本',
    }),
  ], true)
  bubbleStore.selectBubble(0)
}

describe('useBubbleActions backend-first operations', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.useFakeTimers()
    showToastMock.mockReset()
    queuePageDocumentSaveMock.mockReset().mockResolvedValue(undefined)
    queuePageDocumentMutationMock.mockReset().mockResolvedValue(undefined)
    runPageOperationMock.mockReset().mockResolvedValue(undefined)
    runBubbleRepairMock.mockReset().mockResolvedValue(undefined)
    const { fontFamily, ...pageStyleDefaults } = createDefaultSettings().textStyle
    getPageDocumentMock.mockReset().mockResolvedValue({
      pageId: 'page-1',
      chapterId: 'chapter-1',
      documentRevision: 8,
      defaultFontId: fontFamily,
      pageStyleDefaults,
      bubbles: [],
    })
    registerPageDocumentMock.mockReset().mockImplementation(document => {
      const bubbles = [
      createBubbleState({
        backendBubbleId: 'bubble-1',
        coords: [0, 0, 100, 60],
        originalText: '后端识别文本',
      }),
      ]
      useImageStore().updateCurrentImage({ ...document.pageStyleDefaults, documentRevision: document.documentRevision, hasUnsavedChanges: false })
      useBubbleStore().setBubbles(bubbles, true)
      return bubbles
    })
    seedEditor()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('queues OCR as a durable backend page operation without browser image data', async () => {
    useImageStore().updateCurrentImage({ renderedRevision: 7 })
    const onReRender = vi.fn(async () => {
      useImageStore().updateCurrentImage({ renderedRevision: 8, translatedAssetUrl: '/new.png' })
    })
    const actions = useBubbleActions({ onReRender })
    await actions.handleOcrRecognize(0)

    expect(queuePageDocumentSaveMock).toHaveBeenCalledWith(
      'page-1',
      7,
      [expect.objectContaining({
        backendBubbleId: 'bubble-1',
        originalText: '旧文本',
      })],
    )
    expect(runPageOperationMock).toHaveBeenCalledWith('page-1', {
      baseRevision: 7,
      bubbleId: 'bubble-1',
      kind: 'bubble_ocr',
    })
    expect(JSON.stringify(runPageOperationMock.mock.calls)).not.toContain('data:image')
    expect(getPageDocumentMock).toHaveBeenCalledWith('page-1')
    expect(useBubbleStore().bubbles[0]?.originalText).toBe('后端识别文本')
    expect(useImageStore().currentImage).toMatchObject({
      documentRevision: 8, renderedRevision: 8, translatedAssetUrl: '/new.png',
    })
    expect(onReRender).toHaveBeenCalledOnce()
  })

  it('queues selected-bubble repair in the backend and requests authoritative rerender', async () => {
    const onReRender = vi.fn().mockResolvedValue(undefined)
    const actions = useBubbleActions({ onReRender })
    await actions.repairSelectedBubble()

    expect(runBubbleRepairMock).toHaveBeenCalledWith(
      'page-1',
      'bubble-1',
      7,
    )
    expect(onReRender).toHaveBeenCalledTimes(1)
    expect(JSON.stringify(runBubbleRepairMock.mock.calls)).not.toContain('data:image')
  })

  it('rejects an operation reload that belongs to another chapter', async () => {
    getPageDocumentMock.mockResolvedValueOnce({
      pageId: 'page-1',
      chapterId: 'other-chapter',
      documentRevision: 8,
      bubbles: [],
    })
    const actions = useBubbleActions()

    await actions.handleOcrRecognize(0)

    expect(registerPageDocumentMock).not.toHaveBeenCalled()
    expect(useImageStore().currentImage?.documentRevision).toBe(7)
    expect(showToastMock).toHaveBeenCalledWith(
      '页面 page-1 的后端文档身份不匹配',
      'error',
    )
  })

  it('refreshes the repair revision after failure without rerendering a success', async () => {
    runBubbleRepairMock.mockRejectedValueOnce(new Error('model failed'))
    const onReRender = vi.fn()
    const actions = useBubbleActions({ onReRender })
    await actions.repairSelectedBubble()
    expect(showToastMock).toHaveBeenCalledWith('model failed', 'error')
    expect(onReRender).not.toHaveBeenCalled()
    expect(useImageStore().currentImage?.documentRevision).toBe(8)
    await actions.repairSelectedBubble()
    expect(runBubbleRepairMock).toHaveBeenLastCalledWith('page-1', 'bubble-1', 8)
    expect(onReRender).toHaveBeenCalledOnce()
  })

  it('delegates preview coalescing to the shared save and render path without a second timer', async () => {
    let resolveFirst!: () => void
    const onDelayedPreview = vi
      .fn<() => Promise<void>>()
      .mockImplementationOnce(() => new Promise<void>(resolve => {
        resolveFirst = resolve
      }))
      .mockResolvedValueOnce(undefined)
    const Harness = defineComponent({
      setup() {
        const actions = useBubbleActions({ onReRender: onDelayedPreview })
        return {
          update: (text: string) => actions.handleBubbleUpdate({ translatedText: text }),
        }
      },
      render: () => h('div'),
    })
    const wrapper = mount(Harness)
    const update = (wrapper.vm as unknown as { update: (text: string) => void }).update

    update('第一次')
    await vi.advanceTimersByTimeAsync(150)
    update('第二次')
    await vi.advanceTimersByTimeAsync(150)
    expect(onDelayedPreview).toHaveBeenCalledTimes(2)
    resolveFirst()
    await flushPromises()
    await vi.runOnlyPendingTimersAsync()
    expect(onDelayedPreview).toHaveBeenCalledTimes(2)
  })

  it('keeps drawing rectangle output limited to interaction geometry', () => {
    const actions = useBubbleActions()
    actions.currentDrawingRect.value = [10, 20, 40, 70]
    expect(actions.getDrawingRectStyle()).toEqual({
      position: 'absolute',
      left: '10px',
      top: '20px',
      width: '30px',
      height: '50px',
    })
  })

  it('duplicates the selected text with independent rich styles and fresh identities', () => {
    const store = useBubbleStore()
    store.updateBubble(0, { translatedText: '中文！', position: { x: 3, y: 4 }, richText: {
      runs: [{ len: 9, style: { size_pt: 40, tracking: 150, horizontal_scale: 0.8, language: 'ja' } }],
      paragraphs: [{ len: 9, style: { align: 'Center' } }],
    } })
    store.addBubble([10, 20, 80, 90], { backendBubbleId: 'second', translatedText: '第二框' })
    store.toggleMultiSelect(1)
    const onReRender = vi.fn()
    useBubbleActions({ onReRender }).duplicateSelectedBubbles()
    expect(store.bubbles).toHaveLength(4)
    expect(store.selectedIndices).toEqual([2, 3])
    expect(store.bubbles[2]).toMatchObject({ coords: [20, 20, 120, 80], translatedText: '中文！', originalText: '',
      position: { x: 3, y: 4 }, richText: store.bubbles[0]!.richText })
    expect(store.bubbles[2]!.backendBubbleId).toBeUndefined()
    expect(store.bubbles[2]!.duplicateSourceId).toBe('bubble-1')
    expect(store.bubbles[2]!.clientMutationId).toBeUndefined()
    store.bubbles[2]!.richText!.runs[0]!.style.tracking = 900
    expect(store.bubbles[0]!.richText!.runs[0]!.style.tracking).toBe(150)
    expect(onReRender).toHaveBeenCalledOnce()
  })

  it('nudges multiple frames through the existing geometry commit without accumulating normalized offsets', async () => {
    const store = useBubbleStore()
    store.updateBubble(0, { position: { x: 2, y: 3 } })
    store.addBubble([10, 20, 80, 90], { backendBubbleId: 'second', position: { x: -4, y: 5 } })
    store.toggleMultiSelect(1)
    const onDelayedPreview = vi.fn()
    useBubbleActions({ onReRender: onDelayedPreview }).nudgeSelectedText(1, -10)
    expect(store.bubbles.map(bubble => bubble.position)).toEqual([{ x: 2, y: 3 }, { x: -4, y: 5 }])
    expect(store.bubbles.map(bubble => bubble.coords)).toEqual([[1, -10, 101, 50], [11, 10, 81, 80]])
    expect(queuePageDocumentMutationMock).toHaveBeenCalledOnce()
    expect(queuePageDocumentMutationMock).toHaveBeenCalledWith('page-1', 7, store.bubbles, {
      bubbleFields: { 'bubble-1': ['coords'], second: ['coords'] },
    })
    await vi.advanceTimersByTimeAsync(150)
    expect(onDelayedPreview).toHaveBeenCalledOnce()
  })
})
