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
  registerPageDocumentMock,
  runBubbleRepairMock,
  runPageOperationMock,
  showToastMock,
} = vi.hoisted(() => ({
  getPageDocumentMock: vi.fn(),
  queuePageDocumentSaveMock: vi.fn(),
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
    registerPageDocumentMock.mockReset().mockReturnValue([
      createBubbleState({
        backendBubbleId: 'bubble-1',
        coords: [0, 0, 100, 60],
        originalText: '后端识别文本',
      }),
    ])
    seedEditor()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('queues OCR as a durable backend page operation without browser image data', async () => {
    const actions = useBubbleActions()
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

  it('runs one trailing preview after edits arrive during an in-flight preview', async () => {
    let resolveFirst!: () => void
    const onDelayedPreview = vi
      .fn<() => Promise<void>>()
      .mockImplementationOnce(() => new Promise<void>(resolve => {
        resolveFirst = resolve
      }))
      .mockResolvedValueOnce(undefined)
    const Harness = defineComponent({
      setup() {
        const actions = useBubbleActions({ onDelayedPreview })
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
    expect(onDelayedPreview).toHaveBeenCalledTimes(1)
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
})
