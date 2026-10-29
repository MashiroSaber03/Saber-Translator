import { flushPromises, mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent, h } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useBrush, type BrushCallbacks } from '@/composables/useBrush'
import { paintMaskCircles } from '@/utils/binaryMaskPng'
import { useImageStore } from '@/stores/imageStore'
import { addTestImage } from '../helpers/imageFixtures'

const mocks = vi.hoisted(() => ({
  createMaskRepair: vi.fn(),
  getPageDocument: vi.fn(),
  queuePageDocumentSave: vi.fn(),
  registerPageDocument: vi.fn(),
  toast: vi.fn(),
  waitForOperation: vi.fn(),
}))

vi.mock('@/api/v2/operations', () => ({
  createMaskRepair: mocks.createMaskRepair,
  waitForOperation: mocks.waitForOperation,
}))

vi.mock('@/api/v2/content', () => ({
  getPageDocument: mocks.getPageDocument,
}))

vi.mock('@/services/pageDocumentPersistence', () => ({
  queuePageDocumentSave: mocks.queuePageDocumentSave,
  registerPageDocument: mocks.registerPageDocument,
}))

vi.mock('@/utils/toast', () => ({
  showToast: mocks.toast,
}))

function installCanvasMocks() {
  const context = {
    arc: vi.fn(),
    beginPath: vi.fn(),
    ellipse: vi.fn(),
    fill: vi.fn(),
    fillRect: vi.fn(),
    fillStyle: '',
    clearRect: vi.fn(), closePath: vi.fn(), moveTo: vi.fn(), lineTo: vi.fn(), stroke: vi.fn(), strokeStyle: '',
  }
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(
    context as unknown as CanvasRenderingContext2D,
  )
}

function createBrushSurface({
  naturalHeight = 120,
  naturalWidth = 200,
  viewportHeight = 120,
  viewportWidth = 200,
} = {}) {
  const viewport = document.createElement('div')
  Object.defineProperty(viewport, 'clientWidth', { value: viewportWidth })
  Object.defineProperty(viewport, 'clientHeight', { value: viewportHeight })
  const wrapper = document.createElement('div')
  wrapper.getBoundingClientRect = () => ({
    bottom: 120,
    height: 120,
    left: 0,
    right: 200,
    toJSON: () => ({}),
    top: 0,
    width: 200,
    x: 0,
    y: 0,
  })
  const image = document.createElement('img')
  Object.defineProperty(image, 'naturalWidth', { value: naturalWidth })
  Object.defineProperty(image, 'naturalHeight', { value: naturalHeight })
  return { image, viewport, wrapper }
}

function mountBrush(onBrushComplete = vi.fn(), onMaskStroke?: BrushCallbacks['onMaskStroke']) {
  let brushApi: ReturnType<typeof useBrush> | null = null
  const Harness = defineComponent({
    setup() {
      brushApi = useBrush({
        getCurrentRepairSettings: () => ({
          fillColor: '#ffffff',
          inpaintMethod: 'solid',
        }),
        onBrushComplete,
        onMaskStroke,
      })
      return brushApi
    },
    render: () => h('div'),
  })
  const wrapper = mount(Harness)
  if (!brushApi) throw new Error('useBrush test harness did not initialize')
  return {
    brush: brushApi,
    onBrushComplete,
    wrapper,
  }
}

function drawStroke(
  brush: ReturnType<typeof useBrush>,
  mode: 'repair' | 'restore' = 'repair',
) {
  brush.toggleBrushMode(mode)
  brush.startBrushPainting(
    new MouseEvent('mousedown', { button: 0, clientX: 40, clientY: 40 }),
    createBrushSurface(),
  )
  brush.finishBrushPainting()
}

function readBlobBytes(blob: Blob): Promise<Uint8Array> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onerror = () => reject(reader.error ?? new Error('读取掩膜失败'))
    reader.onload = () => resolve(new Uint8Array(reader.result as ArrayBuffer))
    reader.readAsArrayBuffer(blob)
  })
}

describe('useBrush', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
    installCanvasMocks()
    mocks.queuePageDocumentSave.mockResolvedValue(undefined)
    mocks.createMaskRepair.mockResolvedValue({ operationId: 'operation-1' })
    mocks.waitForOperation.mockResolvedValue({ id: 'operation-1', status: 'completed' })
    mocks.getPageDocument.mockResolvedValue({
      bubbles: [],
      chapterId: 'chapter-1',
      defaultFontId: null,
      documentRevision: 4,
      pageId: 'page-1',
      pageStyleDefaults: {},
      renderStatus: 'ready',
    })
    mocks.registerPageDocument.mockImplementation(document => {
      useImageStore().updateCurrentImage({ documentRevision: document.documentRevision, bubbleStates: [], hasUnsavedChanges: false })
      return []
    })
    addTestImage(useImageStore(), 'page.png', '/api/v2/assets/source-1', {
      chapterId: 'chapter-1',
      documentRevision: 3,
      height: 120,
      id: 'page-1',
      width: 200,
    })
  })

  it.each(['rectangle', 'lasso'] as const)('submits a closed %s selection through the existing repair operation', async shape => {
    const { brush, wrapper, onBrushComplete } = mountBrush()
    brush.brushShape.value = shape
    brush.toggleBrushMode('repair')
    brush.startBrushPainting(new MouseEvent('mousedown', { button: 0, clientX: 40, clientY: 30 }), createBrushSurface())
    for (const [clientX, clientY] of (shape === 'rectangle' ? [[80, 30], [80, 80]] : [[80, 30], [80, 80], [40, 80]]))
      brush.continueBrushPainting(new MouseEvent('mousemove', { clientX, clientY }))
    brush.finishBrushPainting()
    await vi.waitFor(() => expect(onBrushComplete).toHaveBeenCalledOnce())
    expect(mocks.createMaskRepair).toHaveBeenCalledWith('page-1', expect.any(Blob), { baseRevision: 3, method: 'solid', fillColor: '#ffffff' })
    const preview = HTMLCanvasElement.prototype.getContext.call(document.createElement('canvas'), '2d')!
    expect(preview.fill).toHaveBeenLastCalledWith('evenodd')
    wrapper.unmount()
  })

  it('requires a clone source, freezes its aligned offset and clears it when switching pages', async () => {
    const { brush, wrapper, onBrushComplete } = mountBrush()
    brush.toggleBrushMode('clone')
    const surface = createBrushSurface()
    brush.startBrushPainting(new MouseEvent('mousedown', { button: 0, clientX: 40, clientY: 40 }), surface)
    expect(brush.isBrushPainting.value).toBe(false)
    expect(mocks.createMaskRepair).not.toHaveBeenCalled()
    brush.startBrushPainting(new MouseEvent('mousedown', { button: 0, altKey: true, clientX: 60, clientY: 20 }), surface)
    expect(brush.cloneSource.value).toEqual({ x: 60, y: 20 })
    brush.startBrushPainting(new MouseEvent('mousedown', { button: 0, clientX: 40, clientY: 40 }), surface)
    brush.finishBrushPainting()
    await vi.waitFor(() => expect(onBrushComplete).toHaveBeenCalledOnce())
    expect(mocks.createMaskRepair).toHaveBeenCalledWith('page-1', expect.any(Blob), { baseRevision: 3, method: 'clone', cloneOffset: [20, -20] })
    addTestImage(useImageStore(), 'next.png', '/api/v2/assets/source-2', { id: 'page-2', chapterId: 'chapter-1', documentRevision: 1 })
    useImageStore().setCurrentImageIndex(1)
    await flushPromises()
    expect(brush.cloneSource.value).toBeNull()
    expect(brush.brushMode.value).toBeNull()
    wrapper.unmount()
  })

  it('cancels an unfinished region on exit without changing the image', async () => {
    const { brush, wrapper } = mountBrush()
    brush.brushShape.value = 'rectangle'
    brush.toggleBrushMode('repair')
    brush.startBrushPainting(new MouseEvent('mousedown', { button: 0, clientX: 40, clientY: 30 }), createBrushSurface())
    brush.continueBrushPainting(new MouseEvent('mousemove', { clientX: 80, clientY: 80 }))
    brush.exitBrushMode()
    brush.finishBrushPainting()
    await flushPromises()
    expect(mocks.createMaskRepair).not.toHaveBeenCalled()
    expect(brush.isBrushPainting.value).toBe(false)
    expect(brush.brushMode.value).toBe(null)
    wrapper.unmount()
  })

  it('keeps fast strokes continuous in the preview and submitted mask', async () => {
    const save = vi.fn().mockResolvedValue(undefined)
    const { brush, wrapper } = mountBrush(undefined, save)
    brush.toggleBrushMode('repair')
    brush.startBrushPainting(new MouseEvent('mousedown', { button: 0, clientX: 40, clientY: 40 }), createBrushSurface())
    const initialSize = brush.brushSize.value
    brush.adjustBrushSize(5)
    expect(brush.brushSize.value).toBe(initialSize)
    brush.continueBrushPainting(new MouseEvent('mousemove', { clientX: 160, clientY: 40 }))
    brush.finishBrushPainting()
    await flushPromises()
    const bounds = save.mock.calls[0]![0]
    const pixels = new Uint8Array(200 * 120)
    paintMaskCircles(pixels, 200, 120, bounds.path.map((p: {x: number; y: number}) => ({...p, radius: bounds.radius})), 255)
    for (let x = 40; x <= 160; x++) expect(pixels[40 * 200 + x]).toBe(255)
    const preview = HTMLCanvasElement.prototype.getContext.call(document.createElement('canvas'), '2d')!
    expect(preview.ellipse).toHaveBeenCalledTimes(bounds.path.length)
    brush.adjustBrushSize(5)
    expect(brush.brushSize.value).toBe(initialSize + 5)
    expect(mocks.createMaskRepair).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('follows an outside-image drag without connecting across untouched image content', async () => {
    const save = vi.fn().mockResolvedValue(undefined)
    const { brush, wrapper } = mountBrush(undefined, save)
    brush.toggleBrushMode('repair')
    brush.startBrushPainting(new MouseEvent('mousedown', { button: 0, clientX: 40, clientY: 40 }), createBrushSurface())
    for (const [clientX, clientY] of [[40, -40], [160, -40], [160, 40]]) {
      brush.continueBrushPainting(new MouseEvent('mousemove', { clientX, clientY }))
    }
    brush.finishBrushPainting()
    await flushPromises()
    const bounds = save.mock.calls[0]![0]
    const pixels = new Uint8Array(200 * 120)
    paintMaskCircles(pixels, 200, 120, bounds.path.map((p: {x: number; y: number}) => ({...p, radius: bounds.radius})), 255)
    expect(pixels[40 * 200 + 40]).toBe(255)
    expect(pixels[40 * 200 + 160]).toBe(255)
    expect(pixels[40 * 200 + 100]).toBe(0)
    wrapper.unmount()
  })

  it('submits the interaction mask as a backend repair operation', async () => {
    const { brush, onBrushComplete } = mountBrush()

    drawStroke(brush)
    await vi.waitFor(() => expect(onBrushComplete).toHaveBeenCalled())

    expect(mocks.queuePageDocumentSave).toHaveBeenCalledWith('page-1', 3, [])
    expect(mocks.createMaskRepair).toHaveBeenCalledWith(
      'page-1',
      expect.any(Blob),
      {
        baseRevision: 3,
        fillColor: '#ffffff',
        method: 'solid',
      },
    )
    expect(mocks.waitForOperation).toHaveBeenCalledWith(
      'operation-1',
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    )
    const mask = mocks.createMaskRepair.mock.calls[0]?.[1] as Blob
    const bytes = await readBlobBytes(mask)
    expect(bytes[24]).toBe(8)
    expect(bytes[25]).toBe(0)
  })

  it('submits the same grayscale binary mask contract for restore strokes', async () => {
    const { brush, onBrushComplete } = mountBrush()

    drawStroke(brush, 'restore')
    await vi.waitFor(() => expect(onBrushComplete).toHaveBeenCalled())

    expect(mocks.createMaskRepair).toHaveBeenCalledWith(
      'page-1',
      expect.any(Blob),
      {
        baseRevision: 3,
        method: 'restore_source',
      },
    )
    const mask = mocks.createMaskRepair.mock.calls[0]?.[1] as Blob
    const bytes = await readBlobBytes(mask)
    expect(bytes[24]).toBe(8)
    expect(bytes[25]).toBe(0)
  })

  it('reports backend repair failures without completing the stroke', async () => {
    mocks.createMaskRepair.mockRejectedValueOnce(new Error('画笔修复失败'))
    const { brush, onBrushComplete } = mountBrush()

    drawStroke(brush)
    await vi.waitFor(() => expect(mocks.toast).toHaveBeenCalled())

    expect(mocks.toast).toHaveBeenCalledWith('画笔修复失败', 'error')
    expect(onBrushComplete).not.toHaveBeenCalled()
  })

  it.each(['submission', 'execution'])('refreshes the revision after %s failure before a new stroke', async phase => {
    const message = phase === 'submission' ? 'revision changed' : 'model failed'
    const failedRequest = phase === 'submission' ? mocks.createMaskRepair : mocks.waitForOperation
    failedRequest.mockRejectedValueOnce(new Error(message))
    const { brush, onBrushComplete } = mountBrush()
    drawStroke(brush)
    await vi.waitFor(() => expect(brush.isBrushSubmitting.value).toBe(false))
    expect(mocks.toast).toHaveBeenCalledWith(message, 'error')
    expect(onBrushComplete).not.toHaveBeenCalled()
    expect(useImageStore().currentImage?.documentRevision).toBe(4)
    brush.exitBrushMode()
    drawStroke(brush)
    await vi.waitFor(() => expect(onBrushComplete).toHaveBeenCalledOnce())
    expect(mocks.createMaskRepair).toHaveBeenLastCalledWith('page-1', expect.any(Blob), {
      baseRevision: 4, fillColor: '#ffffff', method: 'solid',
    })
  })

  it('does not publish a result after the owner unmounts', async () => {
    let resolveOperation!: () => void
    mocks.waitForOperation.mockImplementationOnce(() => new Promise(resolve => {
      resolveOperation = () => resolve({ id: 'operation-1', status: 'completed' })
    }))
    const { brush, onBrushComplete, wrapper } = mountBrush()

    drawStroke(brush)
    await vi.waitFor(() => expect(mocks.waitForOperation).toHaveBeenCalled())
    wrapper.unmount()
    resolveOperation()
    await flushPromises()

    expect(onBrushComplete).not.toHaveBeenCalled()
  })

  it('keeps the preview canvas at viewport resolution for a large source image', () => {
    const { brush, wrapper } = mountBrush()
    const surface = createBrushSurface({
      naturalHeight: 12_000,
      naturalWidth: 8_000,
      viewportHeight: 600,
      viewportWidth: 800,
    })
    brush.toggleBrushMode('repair')

    brush.startBrushPainting(
      new MouseEvent('mousedown', { button: 0, clientX: 40, clientY: 40 }),
      surface,
    )

    const canvas = surface.wrapper.querySelector('canvas')
    expect(canvas).toBeInstanceOf(HTMLCanvasElement)
    expect(canvas!.width).toBeLessThanOrEqual(800 * Math.max(1, window.devicePixelRatio || 1))
    expect(canvas!.height).toBeLessThanOrEqual(600 * Math.max(1, window.devicePixelRatio || 1))
    expect(canvas!.width * canvas!.height).toBeLessThan(8_000 * 12_000)
    wrapper.unmount()
  })

  it('does not accept another stroke while a repair operation is in flight', async () => {
    let resolveOperation!: () => void
    mocks.waitForOperation.mockImplementationOnce(() => new Promise(resolve => {
      resolveOperation = () => resolve({ id: 'operation-1', status: 'completed' })
    }))
    const { brush } = mountBrush()

    drawStroke(brush)
    await vi.waitFor(() => expect(mocks.waitForOperation).toHaveBeenCalledOnce())
    expect(brush.isBrushSubmitting.value).toBe(true)

    const secondSurface = createBrushSurface()
    brush.startBrushPainting(
      new MouseEvent('mousedown', { button: 0, clientX: 50, clientY: 50 }),
      secondSurface,
    )
    brush.finishBrushPainting()
    expect(mocks.createMaskRepair).toHaveBeenCalledOnce()
    expect(secondSurface.wrapper.querySelector('canvas')).toBeNull()

    resolveOperation()
    await vi.waitFor(() => expect(brush.isBrushSubmitting.value).toBe(false))
  })

  it('rejects a repaired document that belongs to another chapter', async () => {
    mocks.getPageDocument.mockResolvedValueOnce({
      bubbles: [],
      chapterId: 'other-chapter',
      defaultFontId: null,
      documentRevision: 4,
      pageId: 'page-1',
      pageStyleDefaults: {},
      renderStatus: 'ready',
    })
    const { brush, onBrushComplete } = mountBrush()

    drawStroke(brush)
    await vi.waitFor(() => expect(mocks.toast).toHaveBeenCalledWith(
      '页面 page-1 的后端文档身份不匹配',
      'error',
    ))

    expect(mocks.registerPageDocument).not.toHaveBeenCalled()
    expect(onBrushComplete).not.toHaveBeenCalled()
  })

  it('contains no Base64 mask or browser-side image repair path', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/composables/useBrush.ts'),
      'utf8',
    )

    expect(source).not.toContain('toDataURL')
    expect(source).not.toContain('toBlob')
    expect(source).not.toContain('base64')
    expect(source).not.toContain('cleanAssetUrl')
  })
})
