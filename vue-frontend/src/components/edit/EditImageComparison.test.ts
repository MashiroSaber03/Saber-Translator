import { mount } from '@vue/test-utils'
import { defineComponent, h } from 'vue'
import { describe, expect, it } from 'vitest'

import EditImageComparison from './EditImageComparison.vue'
import type { BubbleCoords, BubbleState } from '@/types/bubble'

const BubbleOverlayStub = defineComponent({
  name: 'BubbleOverlay',
  emits: [
    'dragStart',
    'dragEnd',
    'resizeStart',
    'resizeEnd',
    'rotateStart',
    'rotateEnd',
  ],
  setup() {
    return () => h('div', { class: 'bubble-overlay-stub' })
  },
})

const baseBubble: BubbleState = {
  originalText: '',
  translatedText: '',
  textboxText: '',
  coords: [10, 20, 110, 220],
  polygon: [],
  fontSize: 24,
  fontFamily: 'Arial',
  textDirection: 'vertical',
  autoTextDirection: 'vertical',
  textColor: '#000000',
  fillColor: '#ffffff',
  rotationAngle: 0,
  position: { x: 0, y: 0 },
  strokeEnabled: false,
  strokeColor: '#ffffff',
  strokeWidth: 0,
  lineSpacing: 1.2,
  textAlign: 'center',
  inpaintMethod: 'solid',
  autoFgColor: null,
  autoBgColor: null,
  colorConfidence: 0,
  textlines: [],
  ocrResult: null,
}

function mountComparison() {
  return mount(EditImageComparison, {
    props: {
      viewMode: 'translated',
      layoutMode: 'horizontal',
      currentImage: {
        id: 'page-1',
        name: 'page-1.png',
        originalDataURL: 'data:image/png;base64,page1',
        translatedDataURL: 'data:image/png;base64,page1-translated',
      },
      bubbles: [baseBubble],
      selectedBubble: baseBubble,
      selectedBubbleIndex: 0,
      selectedIndices: [0],
      scale: 1,
      originalScale: 1,
      isDrawingMode: false,
      brushMode: null,
      currentImageWidth: 1000,
      currentImageHeight: 1200,
      currentDrawingRect: null,
      drawingRectStyle: {},
      originalTransformStyle: {},
      translatedTransformStyle: {},
      isOcrLoading: false,
      isTranslateLoading: false,
    },
    global: {
      stubs: {
        BubbleOverlay: BubbleOverlayStub,
        BubbleEditor: true,
      },
    },
  })
}

describe('EditImageComparison event forwarding', () => {
  it('forwards BubbleOverlay multi-argument edit events without dropping coordinates', () => {
    const wrapper = mountComparison()
    const overlay = wrapper.getComponent(BubbleOverlayStub)
    const startEvent = new MouseEvent('mousedown')
    const resizedCoords: BubbleCoords = [15, 25, 120, 240]
    const draggedCoords: BubbleCoords = [30, 40, 130, 260]

    overlay.vm.$emit('dragStart', 0, startEvent)
    overlay.vm.$emit('dragEnd', 0, draggedCoords)
    overlay.vm.$emit('resizeStart', 0, 'se', startEvent)
    overlay.vm.$emit('resizeEnd', 0, resizedCoords)
    overlay.vm.$emit('rotateStart', 0, startEvent)
    overlay.vm.$emit('rotateEnd', 0, 15)

    expect(wrapper.emitted('bubbleDragStart')?.[0]).toEqual([0, startEvent])
    expect(wrapper.emitted('bubbleDragEnd')?.[0]).toEqual([0, draggedCoords])
    expect(wrapper.emitted('bubbleResizeStart')?.[0]).toEqual([0, 'se', startEvent])
    expect(wrapper.emitted('bubbleResizeEnd')?.[0]).toEqual([0, resizedCoords])
    expect(wrapper.emitted('bubbleRotateStart')?.[0]).toEqual([0, startEvent])
    expect(wrapper.emitted('bubbleRotateEnd')?.[0]).toEqual([0, 15])
  })
})
