import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { defineComponent, h } from 'vue'
import { createPinia, setActivePinia } from 'pinia'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { useBrush } from '@/composables/useBrush'
import { useImageStore } from '@/stores/imageStore'
import { addErasureToUserMask } from '@/utils/maskMerger'
import { showToast } from '@/utils/toast'

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

function createBrushSurface() {
  const viewport = document.createElement('div')
  const wrapper = document.createElement('div')
  wrapper.className = 'edit-image-comparison__canvas-wrapper'
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

  return {
    viewport,
    wrapper,
    image,
  }
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
    const surface = createBrushSurface()

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
      surface,
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

  it('does not write a brush result after the owner unmounts', async () => {
    const onBrushComplete = vi.fn()
    const imageStore = useImageStore()
    const surface = createBrushSurface()

    imageStore.addImage('page.png', 'data:image/png;base64,page')
    const initialCleanImageData = imageStore.currentImage?.cleanImageData
    const initialUserMask = imageStore.currentImage?.userMask

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
      surface,
    )
    brush.finishBrushPainting()
    wrapper.unmount()

    await flushPromises()
    await flushPromises()

    expect(imageStore.currentImage?.cleanImageData).toBe(initialCleanImageData)
    expect(imageStore.currentImage?.userMask).toBe(initialUserMask)
    expect(onBrushComplete).not.toHaveBeenCalled()
  })

  it('reports repair brush failures without completing the stroke', async () => {
    vi.mocked(addErasureToUserMask).mockRejectedValueOnce(new Error('mask merge failed'))
    const onBrushComplete = vi.fn()
    const imageStore = useImageStore()
    const surface = createBrushSurface()
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
      surface,
    )
    brush.finishBrushPainting()

    await flushPromises()
    await flushPromises()

    expect(showToast).toHaveBeenCalledWith('画笔修复失败', 'error')
    expect(onBrushComplete).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('uses explicit surface refs instead of querying viewport internals', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/composables/useBrush.ts'),
      'utf8',
    )

    expect(source).not.toContain('querySelector')
  })

  it('creates an inert brush overlay canvas through explicit canvas style ownership', () => {
    const imageStore = useImageStore()
    const surface = createBrushSurface()
    imageStore.addImage('page.png', 'data:image/png;base64,page')

    const Harness = defineComponent({
      setup() {
        return useBrush()
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
      surface,
    )

    const canvas = surface.wrapper.querySelector('canvas[aria-hidden="true"]') as HTMLCanvasElement | null
    const source = readFileSync(
      resolve(process.cwd(), 'src/composables/useBrush.ts'),
      'utf8',
    )

    expect(source).not.toContain('brushOverlayCanvas')
    expect(canvas).toBeTruthy()
    expect(canvas?.getAttribute('aria-hidden')).toBe('true')
    expect(canvas?.style.pointerEvents).toBe('none')
    expect(canvas?.style.zIndex).toBe('var(--z-canvas)')
    expect(source).not.toContain('style.cssText')

    wrapper.unmount()
  })

  it('keeps the brush composable free of scaffold-style section narration', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/composables/useBrush.ts'),
      'utf8',
    )

    expect(source).not.toMatch(/={6,}/)
    expect(source).not.toContain('状态定义')
    expect(source).not.toContain('返回接口')
    expect(source).not.toContain('创建临时画布')
  })

  it('keeps brush property tests on the composable boundary', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'tests/property/brush.property.ts'),
      'utf8',
    )

    expect(source).toContain("from '@/composables/useBrush'")
    for (const shadowHelper of [
      'function setBrush' + 'Size',
      'function adjustBrush' + 'Size',
      'function handleBrush' + 'Wheel',
      'function calculateRotation' + 'Center',
      'function normalize' + 'Angle',
      'function calculateAngle' + 'FromCenter',
      'function calculateRotated' + 'Angle',
    ]) {
      expect(source).not.toContain(shadowHelper)
    }
  })
})
