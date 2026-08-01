import { mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { defineComponent, h } from 'vue'
import { describe, expect, it } from 'vitest'

import EditImageComparison from './EditImageComparison.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import type { BubbleCoords, BubbleState } from '@/types/bubble'

const BubbleOverlayStub = defineComponent({
  name: 'BubbleOverlay',
  emits: [
    'dragEnd',
    'resizeEnd',
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
        sourceAssetUrl: '/api/v2/assets/page1',
        translatedAssetUrl: '/api/v2/assets/page1-translated',
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

function mountCleanComparison() {
  return mount(EditImageComparison, {
    props: {
      viewMode: 'translated',
      layoutMode: 'horizontal',
      currentImage: {
        id: 'page-1',
        name: 'page-1.png',
        sourceAssetUrl: '/api/v2/assets/page1',
        translatedAssetUrl: null,
        cleanAssetUrl: '/api/v2/assets/page1-clean',
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
  it('uses the clean asset as the editable result when no translated asset exists', () => {
    const wrapper = mountCleanComparison()
    const image = wrapper.get(
      '.edit-image-comparison__image-panel--translated .edit-image-comparison__image',
    )

    expect(image.attributes('src')).toBe('/api/v2/assets/page1-clean')
    expect(image.attributes('alt')).toBe('消字图')
    expect(wrapper.text()).toContain('消字图')
  })

  it('renders panel collapse controls through shared plus and minus icons', async () => {
    const wrapper = mountComparison()
    const toggle = wrapper.findAllComponents(UiIconButton)[0]
    expect(toggle).toBeTruthy()
    expect(toggle!.props('label')).toBe('折叠原图面板')
    expect(toggle!.props('title')).toBe('折叠/展开')

    expect(toggle!.text()).not.toMatch(/[+−]/)
    expect(toggle!.findComponent(UiIcon).props('name')).toBe('minus')

    await toggle!.trigger('click')

    expect(toggle!.text()).not.toMatch(/[+−]/)
    expect(toggle!.findComponent(UiIcon).props('name')).toBe('plus')
  })

  it('forwards BubbleOverlay multi-argument edit events without dropping coordinates', () => {
    const wrapper = mountComparison()
    const overlay = wrapper.getComponent(BubbleOverlayStub)
    const resizedCoords: BubbleCoords = [15, 25, 120, 240]
    const draggedCoords: BubbleCoords = [30, 40, 130, 260]

    overlay.vm.$emit('dragEnd', 0, draggedCoords)
    overlay.vm.$emit('resizeEnd', 0, resizedCoords)
    overlay.vm.$emit('rotateEnd', 0, 15)

    expect(wrapper.emitted('bubbleDragEnd')?.[0]).toEqual([0, draggedCoords])
    expect(wrapper.emitted('bubbleResizeEnd')?.[0]).toEqual([0, resizedCoords])
    expect(wrapper.emitted('bubbleRotateEnd')?.[0]).toEqual([0, 15])
  })

  it('forwards BubbleEditor bulk style payloads to the workspace owner', () => {
    const wrapper = mountComparison()
    const payload: Partial<BubbleState> = {
      fontSize: 28,
      fontFamily: 'fonts/STXIHEI.TTF',
      textDirection: 'horizontal',
      textColor: '#111111',
      fillColor: '#ffffff',
      strokeEnabled: true,
      strokeColor: '#222222',
      strokeWidth: 2,
      inpaintMethod: 'solid',
      lineSpacing: 1.4,
      textAlign: 'center',
    }

    wrapper.getComponent({ name: 'BubbleEditor' }).vm.$emit('applyToAllStyle', payload)

    expect(wrapper.emitted('applyToAllStyle')?.[0]).toEqual([payload])
  })

  it('maps comparison owner colors through semantic tokens', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/EditImageComparison.vue'),
      'utf8',
    )

    expect(source).not.toMatch(/#[0-9A-Fa-f]{3,8}\b|rgba?\(/)
    expect(source).not.toMatch(/var\(--color-[a-z0-9-]+,\s*var\(--[a-z0-9-]+\)\)/)
    expect(source).not.toContain('--edit-image-comparison-resize-handle-background')
    expect(source).toContain('--edit-image-comparison-panel-background: var(--color-surface-inverse-panel)')
    expect(source).toContain('--edit-image-comparison-translated-title-text: var(--color-action-success-bright)')
  })

  it('exposes image element refs for workspace geometry without wrapper queries', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/EditImageComparison.vue'),
      'utf8',
    )
    const workspaceSource = readFileSync(
      resolve(process.cwd(), 'src/components/edit/useEditWorkspace.ts'),
      'utf8',
    )

    expect(source).toContain('originalImageRef')
    expect(source).toContain('translatedImageRef')
    expect(workspaceSource).not.toContain("querySelector('img')")
    expect(workspaceSource).not.toContain('querySelector("img")')
  })

  it('fully wires vertical comparison layout styles', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/EditImageComparison.vue'),
      'utf8',
    )

    expect(source).toContain(":class=\"{ 'edit-image-comparison__divider--vertical': layoutMode === 'vertical' }\"")
    expect(source).toMatch(/\.edit-image-comparison--layout-vertical \.edit-image-comparison__canvas-region\s*\{[^}]*flex-direction:\s*column;/)
    expect(source).toMatch(/\.edit-image-comparison--layout-vertical \.edit-image-comparison__image-panel\s*\{[^}]*min-height:\s*150px;/)
    expect(source).toMatch(/\.edit-image-comparison__divider--vertical\s*\{[^}]*height:\s*8px;[^}]*cursor:\s*ns-resize;/)
    expect(source).toMatch(/\.edit-image-comparison__divider--vertical \.edit-image-comparison__divider-handle\s*\{[^}]*writing-mode:\s*horizontal-tb;/)
  })

  it('uses edit-image-comparison owner hooks instead of generic local layout classes', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/EditImageComparison.vue'),
      'utf8',
    )

    expect(source).toContain('class="edit-image-comparison"')
    expect(source).toContain('edit-image-comparison__canvas-region')
    expect(source).toContain('edit-image-comparison__image-panel')
    expect(source).toContain('edit-image-comparison__panel-header')
    expect(source).toContain('edit-image-comparison__panel-title')
    expect(source).toContain('edit-image-comparison__viewport')
    expect(source).toContain('edit-image-comparison__canvas-wrapper')
    expect(source).toContain('edit-image-comparison__image')
    expect(source).toContain('edit-image-comparison__editor-panel')
    expect(source).not.toMatch(/class="[^"]*\b(?:edit-main-layout|image-comparison-container|image-panel|original-panel|translated-panel|panel-header|panel-title|panel-toggle|image-viewport|image-canvas-wrapper|panel-divider|divider-handle|edit-panel-container|panel-resize-handle|drawing-rect-edit|translated-drawing-rect|vertical-divider)\b/)
    expect(source).not.toMatch(/\.(?:edit-main-layout|image-comparison-container|image-panel|original-panel|translated-panel|panel-header|panel-title|panel-toggle|image-viewport|image-canvas-wrapper|panel-divider|divider-handle|edit-panel-container|panel-resize-handle|drawing-rect-edit|translated-drawing-rect|vertical-divider)\b/)
  })
})
