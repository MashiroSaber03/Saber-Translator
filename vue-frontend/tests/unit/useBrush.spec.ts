import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { defineComponent, h } from 'vue'
import { createPinia, setActivePinia } from 'pinia'
import { useBrush } from '@/composables/useBrush'
import { useImageStore } from '@/stores/imageStore'

vi.mock('@/utils/maskMerger', () => ({
  addErasureToUserMask: vi.fn(async () => 'updated-user-mask'),
  addRestorationToUserMask: vi.fn(async () => 'updated-user-mask'),
}))

vi.mock('@/utils/toast', () => ({
  showToast: vi.fn(),
}))

const originalImage = globalThis.Image

class InstantImage {
  naturalWidth = 200
  naturalHeight = 120
  onload: (() => void) | null = null
  onerror: (() => void) | null = null

  set src(_value: string) {
    queueMicrotask(() => this.onload?.())
  }
}

function installCanvasMocks() {
  const canvasContext = {
    beginPath: vi.fn(),
    arc: vi.fn(),
    fill: vi.fn(),
    fillRect: vi.fn(),
    drawImage: vi.fn(),
    fillStyle: '',
    globalCompositeOperation: 'source-over',
  }

  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(canvasContext as unknown as CanvasRenderingContext2D)
  vi.spyOn(HTMLCanvasElement.prototype, 'toDataURL').mockReturnValue('data:image/png;base64,clean-after-brush')
}

function createViewport() {
  const viewport = document.createElement('div')
  const wrapper = document.createElement('div')
  wrapper.className = 'image-canvas-wrapper'
  wrapper.getBoundingClientRect = () => ({
    x: 0,
    y: 0,
    left: 0,
    top: 0,
    right: 200,
    bottom: 120,
    width: 200,
    height: 120,
    toJSON: () => ({}),
  })

  const image = document.createElement('img')
  Object.defineProperty(image, 'naturalWidth', { value: 200, configurable: true })
  Object.defineProperty(image, 'naturalHeight', { value: 120, configurable: true })
  wrapper.appendChild(image)
  viewport.appendChild(wrapper)

  return viewport
}

describe('useBrush', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    globalThis.Image = InstantImage as unknown as typeof Image
    installCanvasMocks()
  })

  afterEach(() => {
    globalThis.Image = originalImage
    vi.restoreAllMocks()
  })

  it('applies a repair brush stroke without routine console logs', async () => {
    const consoleLog = vi.spyOn(console, 'log').mockImplementation(() => {})
    const onBrushComplete = vi.fn()
    const imageStore = useImageStore()
    const viewport = createViewport()

    imageStore.addImage('page.png', 'data:image/png;base64,page')

    const Harness = defineComponent({
      setup() {
        return {
          ...useBrush({
            onBrushComplete,
            getCurrentRepairSettings: () => ({
              inpaintMethod: 'solid',
              fillColor: '#ffffff',
            }),
          }),
        }
      },
      render() {
        return h('div')
      },
    })
    const wrapper = mount(Harness)
    const brush = wrapper.vm as unknown as ReturnType<typeof useBrush>

    brush.enterBrushMode('repair')
    brush.startBrushPainting(
      new MouseEvent('mousedown', {
        button: 0,
        clientX: 40,
        clientY: 40,
      }),
      viewport,
    )
    brush.finishBrushPainting()

    await vi.waitFor(() => {
      expect(onBrushComplete).toHaveBeenCalled()
    })

    expect(imageStore.currentImage?.cleanImageData).toBe('clean-after-brush')
    expect(imageStore.currentImage?.userMask).toBe('updated-user-mask')
    expect(consoleLog).not.toHaveBeenCalled()
    wrapper.unmount()
  })
})
