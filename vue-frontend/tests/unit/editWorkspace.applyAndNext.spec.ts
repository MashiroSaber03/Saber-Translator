import { flushPromises, mount } from '@vue/test-utils'
import { defineComponent, h, ref } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { useImageStore } from '@/stores/imageStore'

const { reRenderFullImageMock, showToastMock } = vi.hoisted(() => ({
  reRenderFullImageMock: vi.fn(),
  showToastMock: vi.fn(),
}))

vi.mock('@/composables/useImageViewer', () => ({
  useImageViewer: () => ({
    scale: ref(1),
    translateX: ref(0),
    translateY: ref(0),
    zoomAt: vi.fn(),
    zoomIn: vi.fn(),
    zoomOut: vi.fn(),
    resetZoom: vi.fn(),
    startDrag: vi.fn(),
    drag: vi.fn(),
    endDrag: vi.fn(),
    getTransform: vi.fn(() => ({ scale: 1, translateX: 0, translateY: 0 })),
    setTransform: vi.fn(),
  }),
}))

vi.mock('@/composables/useEditRender', () => ({
  useEditRender: () => ({
    reRenderFullImage: reRenderFullImageMock,
  }),
}))

vi.mock('@/composables/useBubbleActions', () => ({
  useBubbleActions: () => ({
    isDrawingMode: ref(false),
    isDrawingBox: ref(false),
    currentDrawingRect: ref(null),
    isMiddleButtonDown: ref(false),
    handleBubbleSelect: vi.fn(),
    handleBubbleMultiSelect: vi.fn(),
    handleClearMultiSelect: vi.fn(),
    handleBubbleDragStart: vi.fn(),
    handleBubbleDragging: vi.fn(),
    handleBubbleDragEnd: vi.fn(),
    handleBubbleResizeStart: vi.fn(),
    handleBubbleResizing: vi.fn(),
    handleBubbleResizeEnd: vi.fn(),
    handleBubbleRotateStart: vi.fn(),
    handleBubbleRotating: vi.fn(),
    handleBubbleRotateEnd: vi.fn(),
    toggleDrawingMode: vi.fn(),
    handleDrawBubble: vi.fn(),
    getDrawingRectStyle: vi.fn(() => ({})),
    handleBubbleUpdate: vi.fn(),
    deleteSelectedBubbles: vi.fn(),
    repairSelectedBubble: vi.fn(),
    handleOcrRecognize: vi.fn(),
  }),
}))

vi.mock('@/composables/useBrush', () => ({
  useBrush: () => ({
    brushMode: ref(null),
    brushSize: ref(20),
    mouseX: ref(0),
    mouseY: ref(0),
    isBrushKeyDown: ref(false),
    toggleBrushMode: vi.fn(),
    exitBrushMode: vi.fn(),
    startBrushPainting: vi.fn(),
    continueBrushPainting: vi.fn(),
    finishBrushPainting: vi.fn(),
    adjustBrushSize: vi.fn(),
  }),
}))

vi.mock('@/composables/useTranslationPipeline', () => ({
  useTranslation: () => ({
    translateWithCurrentBubbles: vi.fn(),
  }),
}))

vi.mock('@/composables/translation/core/steps', () => ({
  executeDetection: vi.fn(),
  saveDetectionResultToImage: vi.fn(),
}))

vi.mock('@/utils/toast', () => ({
  showToast: showToastMock,
}))

import EditWorkspace from '@/components/edit/EditWorkspace.vue'

let pinia: ReturnType<typeof createPinia>

const EditToolbarStub = defineComponent({
  name: 'EditToolbar',
  emits: ['apply-and-next'],
  setup(_props, { emit }) {
    return () =>
      h(
        'button',
        {
          class: 'apply-and-next-trigger',
          onClick: () => emit('apply-and-next'),
        },
        'apply-and-next'
      )
  },
})

describe('EditWorkspace applyAndNext', () => {
  beforeEach(() => {
    pinia = createPinia()
    setActivePinia(pinia)

    const imageStore = useImageStore()
    imageStore.addImage('page-1.png', 'data:image/png;base64,page1')
    imageStore.addImage('page-2.png', 'data:image/png;base64,page2')

    reRenderFullImageMock.mockReset()
    showToastMock.mockReset()
  })

  it('does not navigate to the next image when re-rendering fails', async () => {
    reRenderFullImageMock.mockResolvedValue(false)

    const imageStore = useImageStore()
    const goToNextSpy = vi.spyOn(imageStore, 'goToNext')

    const wrapper = mount(EditWorkspace, {
      props: {
        isEditModeActive: true,
      },
      global: {
        plugins: [pinia],
        stubs: {
          EditToolbar: EditToolbarStub,
          EditThumbnailPanel: true,
          BubbleOverlay: true,
          BubbleEditor: true,
        },
      },
    })

    await wrapper.find('.apply-and-next-trigger').trigger('click')
    await flushPromises()

    expect(reRenderFullImageMock).toHaveBeenCalledTimes(1)
    expect(goToNextSpy).not.toHaveBeenCalled()
  })
})
