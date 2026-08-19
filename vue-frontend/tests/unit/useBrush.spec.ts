import { flushPromises, mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent, h } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useBrush } from '@/composables/useBrush'
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

function mountBrush(onBrushComplete = vi.fn()) {
  let brushApi: ReturnType<typeof useBrush> | null = null
  const Harness = defineComponent({
    setup() {
      brushApi = useBrush({
        getCurrentRepairSettings: () => ({
          fillColor: '#ffffff',
          inpaintMethod: 'solid',
        }),
        onBrushComplete,
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
      pageStyleSchemaVersion: 2,
      renderStatus: 'ready',
    })
    mocks.registerPageDocument.mockReturnValue([])
    addTestImage(useImageStore(), 'page.png', '/api/v2/assets/source-1', {
      chapterId: 'chapter-1',
      documentRevision: 3,
      height: 120,
      id: 'page-1',
      width: 200,
    })
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
      pageStyleSchemaVersion: 2,
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
