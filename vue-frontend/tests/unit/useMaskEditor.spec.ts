import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent, h, ref } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ApiClientError } from '@/api/client'
import { useMaskEditor } from '@/composables/edit/useMaskEditor'
import { useImageStore } from '@/stores/imageStore'
import { addTestImage } from '../helpers/imageFixtures'
import type { InpaintMethod } from '@/types/bubble'

const mocks = vi.hoisted(() => ({
  getPageMasks: vi.fn(),
  savePageUserMask: vi.fn(),
  getPageDocument: vi.fn(),
  getPageRenderStatus: vi.fn(),
  createMaskRepair: vi.fn(),
  waitForOperation: vi.fn(),
  queuePageDocumentSave: vi.fn(),
  flushPageDocument: vi.fn(),
  registerPageDocument: vi.fn(),
  showToast: vi.fn(),
  onApplied: vi.fn(),
}))
vi.mock('@/api/v2/masks', () => mocks)
vi.mock('@/api/v2/content', () => mocks)
vi.mock('@/api/v2/operations', () => mocks)
vi.mock('@/services/pageDocumentPersistence', () => mocks)
vi.mock('@/utils/toast', () => mocks)

let editor: ReturnType<typeof useMaskEditor>
let wrapper: ReturnType<typeof mount>
const repairMethod = ref<InpaintMethod>('solid')
const encoded = (value: number) => btoa(String.fromCharCode(...new Array(16).fill(value)))
const documentAt = (revision: number) => ({
  pageId: 'page',
  documentRevision: revision,
  renderStatus: 'ready',
  bubbles: [],
})
const renderAt = (revision: number, pageId = 'page') => ({
  pageId,
  documentRevision: revision,
  renderedRevision: revision,
  renderStatus: 'ready',
  cleanUrl: `/clean-${revision}.png`,
  translatedUrl: `/translated-${revision}.png`,
})
const previewAt = (revision: number) => ({
  pageId: 'page',
  sourceRevision: 1,
  documentRevision: revision,
  width: 4,
  height: 4,
  preciseMask: encoded(255),
  boxMask: encoded(255),
  processing: { boxExpandRatio: 2, dilateSize: 3, closeGaps: false },
  automaticMask: encoded(255),
  userMask: encoded(127),
  finalMask: encoded(255),
})
const stroke = { path: [{ x: 2, y: 2 }], radius: 1, x: 1, y: 1, width: 2, height: 2 }

beforeEach(() => {
  vi.clearAllMocks()
  repairMethod.value = 'solid'
  setActivePinia(createPinia())
  addTestImage(useImageStore(), 'mask.png', '/source.png', {
    id: 'page',
    documentRevision: 1,
    width: 4,
    height: 4,
    bubbleStates: [],
  })
  mocks.getPageMasks.mockResolvedValue(previewAt(1))
  mocks.getPageRenderStatus.mockImplementation(async (pageId: string) => {
    const preview = await mocks.getPageMasks.mock.results.at(-1)!.value
    return renderAt(preview.documentRevision, pageId)
  })
  mocks.savePageUserMask.mockResolvedValue({ document: documentAt(2) })
  mocks.registerPageDocument.mockReturnValue([])
  mocks.createMaskRepair.mockResolvedValue({ operationId: 'operation' })
  mocks.getPageDocument.mockResolvedValue(documentAt(2))
  vi.stubGlobal(
    'Image',
    class {
      naturalWidth = 4
      naturalHeight = 4
      onload?: () => void
      srcValue = ''
      set src(value: string) {
        this.srcValue = value
        queueMicrotask(() => this.onload?.())
      }
    }
  )
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(() => {
    let raw = new Uint8Array(16)
    return {
      drawImage: (image: { srcValue: string }) => {
        raw = Uint8Array.from(atob(image.srcValue.split(',')[1]!), char => char.charCodeAt(0))
      },
      getImageData: () => ({
        data: Uint8ClampedArray.from({ length: 64 }, (_, i) => raw[Math.floor(i / 4)]!),
      }),
      createImageData: () => ({ data: new Uint8ClampedArray(64) }),
      putImageData: vi.fn(),
    } as unknown as CanvasRenderingContext2D
  })
  vi.spyOn(HTMLCanvasElement.prototype, 'toDataURL').mockReturnValue(
    'data:image/png;base64,overlay'
  )
  wrapper = mount(
    defineComponent({
      setup() {
        editor = useMaskEditor({
          getRepairSettings: () => ({ inpaintMethod: repairMethod.value, fillColor: '#ffeeaa' }),
          onApplied: mocks.onApplied,
        })
        return () => h('div')
      },
    })
  )
})
afterEach(() => {
  wrapper.unmount()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('mask editor persistence and execution', () => {
  it.each(['ready', 'stale'])('keeps the displayed render revision accurate after a %s mask save', async renderStatus => {
    useImageStore().updateCurrentImage({ renderedRevision: 1 })
    mocks.savePageUserMask.mockResolvedValue({ document: { ...documentAt(2), renderStatus } })
    mocks.getPageMasks.mockResolvedValueOnce(previewAt(1)).mockResolvedValue(previewAt(2))
    mocks.getPageRenderStatus.mockResolvedValueOnce(renderAt(1)).mockResolvedValue({
      ...renderAt(2), renderStatus, renderedRevision: renderStatus === 'ready' ? 2 : 1,
      cleanUrl: '/clean-1.png', translatedUrl: '/translated-1.png',
    })
    await editor.open()
    await editor.stroke(stroke)
    expect(useImageStore().currentImage?.documentRevision).toBe(2)
    expect(useImageStore().currentImage?.renderedRevision).toBe(renderStatus === 'ready' ? 2 : 1)
    expect(useImageStore().currentImage?.cleanAssetUrl).toBe('/clean-1.png')
    expect(useImageStore().currentImage?.translatedAssetUrl).toBe('/translated-1.png')
    expect(mocks.createMaskRepair).not.toHaveBeenCalled()
    expect(mocks.onApplied).not.toHaveBeenCalled()
  })

  it('adopts externally edited bubbles with their matching clean and translated images', async () => {
    await editor.open()
    expect(editor.close()).toBe(true)
    mocks.getPageMasks.mockResolvedValue(previewAt(7))
    mocks.getPageDocument.mockResolvedValue(documentAt(7))
    await editor.open()
    expect(useImageStore().currentImage).toMatchObject({
      documentRevision: 7, renderedRevision: 7,
      cleanAssetUrl: '/clean-7.png', translatedAssetUrl: '/translated-7.png',
    })
    expect(mocks.registerPageDocument).toHaveBeenCalledWith(documentAt(7))
    expect(editor.state.value?.documentRevision).toBe(7)
    expect(mocks.getPageMasks).toHaveBeenCalledTimes(2)
    expect(mocks.createMaskRepair).not.toHaveBeenCalled()
    expect(mocks.onApplied).not.toHaveBeenCalled()
  })

  it('refreshes finished rendering even when the document revision has not changed', async () => {
    mocks.getPageRenderStatus.mockResolvedValueOnce({ ...renderAt(1), renderStatus: 'stale', renderedRevision: null })
    await editor.open()
    expect(useImageStore().currentImage?.renderedRevision).toBeNull()
    await editor.retry()
    expect(useImageStore().currentImage).toMatchObject({
      renderedRevision: 1, translatedAssetUrl: '/translated-1.png',
    })
    expect(mocks.getPageDocument).not.toHaveBeenCalled()
  })

  it('does not resave an accepted stroke when reading render status fails', async () => {
    await editor.open()
    mocks.getPageMasks.mockResolvedValue(previewAt(2))
    mocks.getPageRenderStatus.mockRejectedValueOnce(new Error('offline'))
    await editor.stroke(stroke)
    expect(editor.dirty.value).toBe(false)
    expect(editor.error.value).toBe('offline')
    expect(useImageStore().currentImage).toMatchObject({ documentRevision: 2, renderedRevision: 1 })
    await editor.apply()
    expect(mocks.createMaskRepair).not.toHaveBeenCalled()
    await editor.retry()
    expect(editor.error.value).toBe('')
    expect(useImageStore().currentImage).toMatchObject({
      documentRevision: 2, renderedRevision: 2, translatedAssetUrl: '/translated-2.png',
    })
    expect(mocks.savePageUserMask).toHaveBeenCalledOnce()
  })

  it.each([
    { ...renderAt(3) },
    { ...renderAt(2), pageId: 'another-page' },
  ])('rejects render status that does not match the mask snapshot', async status => {
    mocks.getPageMasks.mockResolvedValue(previewAt(2))
    mocks.getPageRenderStatus.mockResolvedValue(status)
    await editor.open()
    expect(editor.overlay.value).toBeNull()
    expect(editor.error.value).toContain('页面已更新')
    expect(useImageStore().currentImage?.documentRevision).toBe(1)
    expect(useImageStore().currentImage?.translatedAssetUrl).toBeFalsy()
    expect(mocks.registerPageDocument).not.toHaveBeenCalled()
  })

  it('ignores a late render status after a newer preview has loaded', async () => {
    await editor.open()
    let finishStatus!: (value: ReturnType<typeof renderAt>) => void
    mocks.getPageMasks.mockResolvedValueOnce(previewAt(2))
    mocks.getPageRenderStatus.mockImplementationOnce(() => new Promise(resolve => { finishStatus = resolve }))
    repairMethod.value = 'lama_manga'
    await flushPromises()
    mocks.getPageMasks.mockResolvedValue(previewAt(3))
    mocks.getPageDocument.mockResolvedValue(documentAt(3))
    repairMethod.value = 'solid'
    await flushPromises()
    finishStatus(renderAt(2))
    await flushPromises()
    expect(useImageStore().currentImage).toMatchObject({
      documentRevision: 3, renderedRevision: 3, translatedAssetUrl: '/translated-3.png',
    })
    expect(editor.state.value?.documentRevision).toBe(3)
  })

  it('does not adopt an external document if its render status cannot be read', async () => {
    mocks.getPageMasks.mockResolvedValue(previewAt(7))
    mocks.getPageDocument.mockResolvedValue(documentAt(7))
    mocks.getPageRenderStatus.mockRejectedValueOnce(new Error('offline'))
    await editor.open()
    expect(mocks.registerPageDocument).not.toHaveBeenCalled()
    expect(useImageStore().currentImage?.documentRevision).toBe(1)
    expect(editor.error.value).toBe('offline')
    expect(editor.overlay.value).toBeNull()
    await editor.retry()
    expect(useImageStore().currentImage).toMatchObject({
      documentRevision: 7, renderedRevision: 7, translatedAssetUrl: '/translated-7.png',
    })
  })

  it('ignores a pending render status when navigating to another page', async () => {
    await editor.open()
    let finishStatus!: (value: ReturnType<typeof renderAt>) => void
    mocks.getPageRenderStatus.mockImplementationOnce(() => new Promise(resolve => { finishStatus = resolve }))
    const pending = editor.retry()
    await flushPromises()
    addTestImage(useImageStore(), 'next.png', '/next.png', {
      id: 'next-page', documentRevision: 3, width: 4, height: 4, bubbleStates: [],
    })
    mocks.getPageMasks.mockResolvedValue({ ...previewAt(3), pageId: 'next-page' })
    useImageStore().setCurrentImageIndex(1)
    await flushPromises()
    finishStatus(renderAt(1))
    await pending
    expect(useImageStore().currentImage).toMatchObject({
      id: 'next-page', documentRevision: 3, renderedRevision: 3, translatedAssetUrl: '/translated-3.png',
    })
    expect(editor.state.value?.pageId).toBe('next-page')
  })

  it('keeps the new page overlay when an older mask response arrives with a different source revision', async () => {
    let finishOld!: (value: ReturnType<typeof previewAt>) => void
    mocks.getPageMasks.mockImplementationOnce(() => new Promise(resolve => { finishOld = resolve }))
    const pending = editor.open()
    await flushPromises()
    addTestImage(useImageStore(), 'next.png', '/next.png', {
      id: 'next-page', documentRevision: 3, width: 4, height: 4, bubbleStates: [],
    })
    mocks.getPageMasks.mockResolvedValue({ ...previewAt(3), pageId: 'next-page', sourceRevision: 9 })
    useImageStore().setCurrentImageIndex(1)
    await flushPromises()
    const overlay = editor.overlay.value
    expect(overlay).toBeTruthy()
    finishOld(previewAt(1))
    await pending
    expect(editor.state.value?.pageId).toBe('next-page')
    expect(editor.overlay.value).toBe(overlay)
    expect(editor.error.value).toBe('')
    expect(mocks.getPageRenderStatus).toHaveBeenCalledOnce()
    expect(mocks.showToast).not.toHaveBeenCalled()
  })

  it('reloads the current revision when repair submission conflicts, allowing a retry', async () => {
    await editor.open()
    mocks.createMaskRepair.mockRejectedValueOnce(
      new ApiClientError({ code: 'operation_conflict', status: 409, message: 'revision changed' })
    )
    mocks.getPageDocument.mockResolvedValue(documentAt(7))
    mocks.getPageMasks.mockResolvedValue(previewAt(7))
    await editor.apply()
    expect(mocks.waitForOperation).not.toHaveBeenCalled()
    expect(mocks.onApplied).not.toHaveBeenCalled()
    expect(editor.state.value?.documentRevision).toBe(7)
    expect(useImageStore().currentImage?.documentRevision).toBe(7)
    expect(mocks.showToast).toHaveBeenLastCalledWith('revision changed', 'error')
    await editor.apply()
    expect(mocks.createMaskRepair).toHaveBeenLastCalledWith('page', expect.any(Blob), {
      method: 'solid', fillColor: '#ffeeaa', baseRevision: 7, fromSource: true,
    })
    expect(mocks.onApplied).toHaveBeenCalledOnce()
  })

  it('rejects a preview older than the document fetched during the same refresh', async () => {
    mocks.getPageMasks.mockResolvedValue(previewAt(2))
    mocks.getPageDocument.mockResolvedValue(documentAt(3))
    await editor.open()
    expect(editor.overlay.value).toBeNull()
    expect(editor.error.value).toContain('页面已更新')
    expect(useImageStore().currentImage?.documentRevision).toBe(1)
    await editor.apply()
    expect(mocks.createMaskRepair).not.toHaveBeenCalled()
    mocks.getPageMasks.mockResolvedValue(previewAt(3))
    await editor.retry()
    expect(editor.state.value?.documentRevision).toBe(3)
    expect(useImageStore().currentImage?.documentRevision).toBe(3)
    expect(editor.error.value).toBe('')
    expect(editor.overlay.value).toBeTruthy()
  })

  it('ignores an obsolete document response after a newer method preview has loaded', async () => {
    await editor.open()
    let finishDocument!: (value: ReturnType<typeof documentAt>) => void
    mocks.getPageMasks.mockResolvedValueOnce(previewAt(2))
    mocks.getPageDocument.mockImplementationOnce(() => new Promise(resolve => { finishDocument = resolve }))
    repairMethod.value = 'lama_manga'
    await flushPromises()
    mocks.getPageMasks.mockResolvedValue(previewAt(3))
    mocks.getPageDocument.mockResolvedValue(documentAt(3))
    repairMethod.value = 'solid'
    await flushPromises()
    expect(editor.state.value?.documentRevision).toBe(3)
    finishDocument(documentAt(2))
    await flushPromises()
    expect(mocks.registerPageDocument).not.toHaveBeenCalledWith(documentAt(2))
    expect(editor.state.value?.documentRevision).toBe(3)
    expect(useImageStore().currentImage?.documentRevision).toBe(3)
  })

  it('keeps the latest repair-method preview when an earlier request is still loading', async () => {
    await editor.open()
    let finishOld!: (value: ReturnType<typeof previewAt>) => void
    mocks.getPageMasks.mockImplementationOnce(() => new Promise(resolve => { finishOld = resolve }))
    repairMethod.value = 'lama_manga'
    await flushPromises()
    expect(editor.loading.value).toBe(true)
    repairMethod.value = 'solid'
    await flushPromises()
    expect(mocks.getPageMasks).toHaveBeenLastCalledWith('page', 'solid', expect.any(AbortSignal))
    finishOld({ ...previewAt(1), finalMask: encoded(0) })
    await flushPromises()
    expect(editor.state.value?.finalMask).toBe(encoded(255))
    expect(editor.loading.value).toBe(false)
  })

  it.each([
    ['clear', 127], ['repair', 255], ['preserve', 0],
  ] as const)('does not save an unchanged %s stroke', async (tool, value) => {
    mocks.getPageMasks.mockResolvedValue({ ...previewAt(1), userMask: encoded(value) })
    await editor.open()
    const preview = editor.overlay.value
    editor.tool.value = tool
    await editor.stroke(stroke)
    expect(editor.dirty.value).toBe(false)
    expect(editor.overlay.value).toBe(preview)
    expect(mocks.savePageUserMask).not.toHaveBeenCalled()
    expect(mocks.getPageMasks).toHaveBeenCalledOnce()
    expect(mocks.createMaskRepair).not.toHaveBeenCalled()
  })

  it('still saves a retained dirty stroke when the same region is drawn again', async () => {
    await editor.open()
    mocks.savePageUserMask.mockRejectedValueOnce(new Error('disk full'))
    await expect(editor.stroke(stroke)).rejects.toThrow('disk full')
    mocks.getPageMasks.mockResolvedValue(previewAt(2))
    await editor.stroke(stroke)
    expect(mocks.savePageUserMask).toHaveBeenCalledTimes(2)
    expect(editor.dirty.value).toBe(false)
  })

  it('preserves independent display switches when returning from text-box editing', async () => {
    await editor.open()
    editor.visibility.value.preserve = true
    editor.visibility.value.final = false
    expect(editor.close()).toBe(true)
    await editor.open()
    expect(editor.visibility.value.preserve).toBe(true)
    expect(editor.visibility.value.final).toBe(false)
    expect(editor.overlay.value).toBeTruthy()
  })
  it('loads the next page without treating its source revision as an original-image replacement', async () => {
    await editor.open()
    mocks.getPageMasks.mockResolvedValue({ ...previewAt(3), pageId: 'next-page', sourceRevision: 9 })
    addTestImage(useImageStore(), 'next.png', '/next.png', {
      id: 'next-page', documentRevision: 3, width: 4, height: 4, bubbleStates: [],
    })
    useImageStore().setCurrentImageIndex(1)
    await flushPromises()
    expect(editor.error.value).toBe('')
    expect(editor.overlay.value).toBeTruthy()
    expect(mocks.getPageMasks).toHaveBeenLastCalledWith('next-page', 'solid', expect.any(AbortSignal))
  })
  it('saves strokes without triggering repair, then refreshes the backend preview', async () => {
    await editor.open()
    mocks.getPageMasks.mockResolvedValue(previewAt(2))
    editor.tool.value = 'preserve'
    await editor.stroke(stroke)
    expect(mocks.savePageUserMask).toHaveBeenCalledWith(
      'page',
      1,
      expect.any(Blob),
      expect.any(AbortSignal)
    )
    expect(editor.dirty.value).toBe(false)
    expect(editor.overlay.value).toBeTruthy()
    expect(editor.busy.value).toBe(false)
    expect(mocks.createMaskRepair).not.toHaveBeenCalled()
    expect(mocks.onApplied).not.toHaveBeenCalled()
  })

  it('keeps an unsaved stroke and blocks apply/exit until retry or discard', async () => {
    await editor.open()
    mocks.savePageUserMask.mockRejectedValueOnce(new Error('disk full'))
    await expect(editor.stroke(stroke)).rejects.toThrow('disk full')
    expect(editor.dirty.value).toBe(true)
    expect(editor.error.value).toBe('disk full')
    expect(editor.close()).toBe(false)
    await editor.apply()
    expect(mocks.createMaskRepair).not.toHaveBeenCalled()
    mocks.getPageMasks.mockResolvedValue(previewAt(2))
    await editor.retry()
    expect(editor.dirty.value).toBe(false)
    expect(editor.error.value).toBe('')
  })

  it('keeps the local mask on conflict and uses the latest revision on explicit retry', async () => {
    await editor.open()
    mocks.savePageUserMask.mockRejectedValueOnce(
      new ApiClientError({ code: 'content_conflict', status: 409, message: 'revision changed' })
    )
    mocks.getPageDocument.mockResolvedValue(documentAt(7))
    await expect(editor.stroke(stroke)).rejects.toThrow('页面已被其他操作更新，笔迹已保留')
    expect(editor.dirty.value).toBe(true)
    expect(useImageStore().currentImage?.documentRevision).toBe(7)
    mocks.savePageUserMask.mockResolvedValue({ document: documentAt(8) })
    mocks.getPageMasks.mockResolvedValue(previewAt(8))
    await editor.retry()
    expect(mocks.savePageUserMask).toHaveBeenLastCalledWith(
      'page',
      7,
      expect.any(Blob),
      expect.any(AbortSignal)
    )
    expect(editor.dirty.value).toBe(false)
  })

  it('applies the confirmed final mask from source with current repair settings', async () => {
    await editor.open()
    for (const key of Object.keys(editor.visibility.value) as (keyof typeof editor.visibility.value)[])
      editor.visibility.value[key] = false
    await flushPromises()
    expect(mocks.savePageUserMask).not.toHaveBeenCalled()
    await editor.apply()
    expect(mocks.createMaskRepair).toHaveBeenCalledWith('page', expect.any(Blob), {
      method: 'solid',
      fillColor: '#ffeeaa',
      baseRevision: 1,
      fromSource: true,
    })
    expect(mocks.waitForOperation).toHaveBeenCalledWith('operation', {
      signal: expect.any(AbortSignal),
    })
    expect(mocks.onApplied).toHaveBeenCalledOnce()
    const mask = mocks.createMaskRepair.mock.calls[0]![1] as Blob
    const data = await new Promise<ArrayBuffer>(resolve => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result as ArrayBuffer)
      reader.readAsArrayBuffer(mask)
    })
    expect(Array.from(new Uint8Array(data))).toEqual(new Array(16).fill(255))
  })

  it('keeps retry available when reloading a conflicted page also fails', async () => {
    await editor.open()
    mocks.savePageUserMask.mockRejectedValueOnce(
      new ApiClientError({ code: 'content_conflict', status: 409, message: 'revision changed' })
    )
    mocks.getPageMasks.mockRejectedValueOnce(new Error('offline during reload'))
    await expect(editor.stroke(stroke)).rejects.toThrow('offline during reload')
    expect(editor.dirty.value).toBe(true)
    expect(editor.error.value).toContain('页面已被其他操作更新，笔迹已保留')
    expect(editor.busy.value).toBe(false)
    expect(editor.overlay.value).toBeTruthy()
    await editor.apply()
    expect(mocks.createMaskRepair).not.toHaveBeenCalled()

    mocks.getPageMasks.mockResolvedValue(previewAt(2))
    await editor.retry()
    expect(editor.dirty.value).toBe(false)
    expect(editor.error.value).toBe('')
  })

  it('never rebases old strokes onto a replacement source image', async () => {
    await editor.open()
    mocks.savePageUserMask.mockRejectedValue(
      new ApiClientError({ code: 'content_conflict', status: 409, message: 'source changed' })
    )
    mocks.getPageMasks.mockResolvedValue({ ...previewAt(2), sourceRevision: 2 })
    await expect(editor.stroke(stroke)).rejects.toThrow('原图已更换')
    expect(editor.dirty.value).toBe(true)
    expect(editor.overlay.value).toBeNull()
    expect(useImageStore().currentImage?.documentRevision).toBe(1)
    expect(mocks.getPageDocument).not.toHaveBeenCalled()
    await editor.apply()
    expect(mocks.createMaskRepair).not.toHaveBeenCalled()
  })

  it('refreshes the reserved revision after failure so apply can be retried without reload', async () => {
    await editor.open()
    mocks.waitForOperation.mockRejectedValueOnce(new Error('model failed'))
    mocks.getPageMasks.mockResolvedValue(previewAt(2))
    await editor.apply()
    expect(mocks.onApplied).not.toHaveBeenCalled()
    expect(editor.applying.value).toBe(false)
    expect(mocks.showToast).toHaveBeenLastCalledWith('model failed', 'error')
    expect(useImageStore().currentImage?.documentRevision).toBe(2)
    await editor.apply()
    expect(mocks.createMaskRepair).toHaveBeenLastCalledWith('page', expect.any(Blob), {
      method: 'solid', fillColor: '#ffeeaa', baseRevision: 2, fromSource: true,
    })
    expect(mocks.onApplied).toHaveBeenCalledOnce()
  })

  it('cancels pending preview requests when the editor unmounts', async () => {
    await editor.open()
    const signal = mocks.getPageMasks.mock.calls[0]![2] as AbortSignal
    wrapper.unmount()
    await flushPromises()
    expect(signal.aborted).toBe(true)
  })
})
