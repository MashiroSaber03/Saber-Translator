import { flushPromises, mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent, h } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useBrush } from '@/composables/useBrush'
import { useImageStore } from '@/stores/imageStore'

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
    fill: vi.fn(),
    fillRect: vi.fn(),
    fillStyle: '',
  }
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(
    context as unknown as CanvasRenderingContext2D,
  )
  vi.spyOn(HTMLCanvasElement.prototype, 'toBlob').mockImplementation(callback => {
    callback(new Blob(['mask'], { type: 'image/png' }))
  })
}

function createBrushSurface() {
  const viewport = document.createElement('div')
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
  Object.defineProperty(image, 'naturalWidth', { value: 200 })
  Object.defineProperty(image, 'naturalHeight', { value: 120 })
  return { image, viewport, wrapper }
}

function mountBrush(onBrushComplete = vi.fn()) {
  const Harness = defineComponent({
    setup() {
      return useBrush({
        getCurrentRepairSettings: () => ({
          fillColor: '#ffffff',
          inpaintMethod: 'solid',
        }),
        onBrushComplete,
      })
    },
    render: () => h('div'),
  })
  const wrapper = mount(Harness)
  return {
    brush: wrapper.vm as unknown as ReturnType<typeof useBrush>,
    onBrushComplete,
    wrapper,
  }
}

function drawStroke(brush: ReturnType<typeof useBrush>) {
  brush.enterBrushMode('repair')
  brush.startBrushPainting(
    new MouseEvent('mousedown', { button: 0, clientX: 40, clientY: 40 }),
    createBrushSurface(),
  )
  brush.finishBrushPainting()
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
      pageStyleSchemaVersion: 1,
      renderedRevision: 4,
      sourceRevision: 1,
    })
    mocks.registerPageDocument.mockReturnValue([])
    useImageStore().addImage('page.png', '/api/v2/assets/source-1', {
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
  })

  it('reports backend repair failures without completing the stroke', async () => {
    mocks.createMaskRepair.mockRejectedValueOnce(new Error('画笔修复失败'))
    const { brush, onBrushComplete } = mountBrush()

    drawStroke(brush)
    await flushPromises()
    await flushPromises()

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
    await flushPromises()
    wrapper.unmount()
    resolveOperation()
    await flushPromises()

    expect(onBrushComplete).not.toHaveBeenCalled()
  })

  it('contains no Base64 mask or browser-side image repair path', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/composables/useBrush.ts'),
      'utf8',
    )

    expect(source).not.toContain('toDataURL')
    expect(source).not.toContain('base64')
    expect(source).not.toContain('cleanImageData')
  })
})
