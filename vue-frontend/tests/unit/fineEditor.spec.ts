import { mount, flushPromises } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import FineEditor from '@/components/edit/FineEditor.vue'
import { useImageStore } from '@/stores/imageStore'
import { useSettingsStore } from '@/stores/settings'
import { addTestImage } from '../helpers/imageFixtures'

const mocks = vi.hoisted(() => ({
  get: vi.fn(), document: vi.fn(), register: vi.fn(), queue: vi.fn(), flush: vi.fn(),
  render: vi.fn(), repair: vi.fn(), wait: vi.fn(), port: vi.fn(), download: vi.fn(),
}))
vi.mock('@/api/client', () => ({ apiClient: { get: mocks.get } }))
vi.mock('@/api/v2/content', () => ({ getPageDocument: mocks.document }))
vi.mock('@/api/v2/operations', () => ({ createMaskRepair: mocks.repair, waitForOperation: mocks.wait }))
vi.mock('@/utils/browserDownload', () => ({ triggerBlobDownload: mocks.download }))
vi.mock('@/composables/useEditRender', () => ({ useEditRender: (callbacks?: { onRenderError: (message: string) => void }) => ({
  refreshRenderedImage: async () => { const result = await mocks.render(); if (result === false) callbacks?.onRenderError('后端渲染失败'); return result },
}) }))
vi.mock('@/composables/usePublicUserAccess', () => ({ usePublicUserAccess: () => ({ modelOptions: (options: unknown) => options }) }))
vi.mock('@/services/pageDocumentPersistence', () => ({
  flushPageDocument: mocks.flush, hasPendingPageDocument: () => false,
  queueEditorDocumentSave: mocks.queue, registerPageDocument: mocks.register,
  registerPageEditor: mocks.port, saveEditorDocument: vi.fn(),
}))

let wrappers: ReturnType<typeof mount>[] = []
beforeEach(() => {
  vi.resetAllMocks()
  setActivePinia(createPinia())
  addTestImage(useImageStore(), 'manga.png', '/source', {
    id: 'page', chapterId: 'chapter', documentRevision: 4, bubbleStates: [], width: 200, height: 200,
  })
  mocks.get.mockResolvedValue({ items: [] })
  mocks.document.mockResolvedValue({ pageId: 'page', chapterId: 'chapter', documentRevision: 4 })
  mocks.register.mockImplementation(document => {
    useImageStore().updateCurrentImage({ documentRevision: document.documentRevision })
    return []
  })
  mocks.port.mockReturnValue(vi.fn())
  mocks.queue.mockResolvedValue(undefined)
  mocks.flush.mockResolvedValue(undefined)
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
    ok: true, headers: new Headers({ 'X-Document-Revision': '4' }),
    arrayBuffer: async () => new ArrayBuffer(1),
  }))
})

afterEach(() => {
  for (const wrapper of wrappers) wrapper.unmount()
  wrappers = []
  vi.unstubAllGlobals()
})

async function editor(ready = true) {
  const wrapper = mount(FineEditor, {
    props: { pageId: 'page', active: true, inpaintMethod: 'solid', fillColor: '#ffffff' },
    attachTo: document.body,
  })
  wrappers.push(wrapper)
  const frame = wrapper.get('iframe').element as HTMLIFrameElement
  let revision = 4
  let selectionError: string | undefined
  let replaceError: string | undefined
  const send = vi.spyOn(frame.contentWindow!, 'postMessage').mockImplementation(message => {
    const request = message.payload
    queueMicrotask(() => {
      window.dispatchEvent(new MessageEvent('message', {
        origin: location.origin, source: frame.contentWindow,
        data: { saberEditor: true, requestId: request.requestId, revision,
          ...(request.op === 'font' ? { font: { family: 'Inter', fontStyle: 'SemiBold', postscriptName: 'Inter-SemiBold', weight: 600, italic: false } } : {}),
          ...(request.op === 'selection' ? { error: selectionError, bytes: new Uint8Array([1]) } : {}),
          ...(request.op === 'replace' ? { error: replaceError } : {}),
        },
      }))
    })
  })
  if (ready) window.dispatchEvent(new MessageEvent('message', {
    origin: location.origin, source: frame.contentWindow, data: { saberEditor: true, event: 'ready' },
  }))
  await flushPromises()
  return { wrapper, frame, send, revise: (value: number) => { revision = value },
    failReplace: (error?: string) => { replaceError = error },
    failSelection: (error?: string) => { selectionError = error } }
}

describe('fine editor page lifecycle', () => {
  it('keeps a render failure visible after a manual save and retries output without saving twice', async () => {
    const { wrapper, revise } = await editor()
    revise(5)
    mocks.render.mockResolvedValueOnce(false).mockResolvedValueOnce(true)
    await (wrapper.vm as unknown as { saveNow: () => Promise<void> }).saveNow()
    expect(wrapper.get('[role="status"]').text()).toContain('后端渲染失败')
    // The engine normally acknowledges the new revision through its saved reply.
    revise(4)
    await (wrapper.vm as unknown as { saveNow: () => Promise<void> }).saveNow()
    expect(mocks.queue).toHaveBeenCalledOnce()
    expect(mocks.render).toHaveBeenCalledTimes(2)
    expect(wrapper.get('[role="status"]').text()).not.toContain('后端渲染失败')
  })
  it('shows a canvas startup failure and reloads the canvas when retried', async () => {
    const { wrapper, frame } = await editor(false)
    window.dispatchEvent(new MessageEvent('message', {
      origin: location.origin, source: frame.contentWindow,
      data: { saberEditor: true, event: 'error', error: 'WebGL2 unavailable' },
    }))
    await flushPromises()
    expect(wrapper.text()).toContain('WebGL2 unavailable')
    expect(wrapper.text()).not.toContain('正在准备精细编辑')
    expect(wrapper.emitted('busy')?.at(-1)).toEqual([false])
    expect(frame.hasAttribute('inert')).toBe(true)
    const reload = vi.spyOn(frame, 'src', 'set').mockImplementation(() => {})
    await wrapper.findAll('button').find(button => button.text() === '重试')!.trigger('click')
    expect(reload).toHaveBeenCalledWith('/photocraft/index.html')
    window.dispatchEvent(new MessageEvent('message', {
      origin: location.origin, source: frame.contentWindow,
      data: { saberEditor: true, event: 'ready' },
    }))
    await flushPromises()
    expect(wrapper.text()).not.toContain('WebGL2 unavailable')
    expect(frame.hasAttribute('inert')).toBe(false)
    expect(wrapper.emitted('document-loaded')).toEqual([['page']])
    expect(mocks.queue).not.toHaveBeenCalled()
  })

  it('uses the project downloader only for exports from its own canvas', async () => {
    const { frame } = await editor()
    const exported = { saberEditor: true, event: 'download', filename: 'page.png', bytes: new Uint8Array([1,2,3]) }
    for (const [origin, source] of [['https://example.invalid', frame.contentWindow], [location.origin, window], [location.origin, frame.contentWindow]] as const) {
      window.dispatchEvent(new MessageEvent('message', { origin, source, data: exported }))
    }
    expect(mocks.download).toHaveBeenCalledOnce()
    const [blob, name] = mocks.download.mock.calls[0]!
    expect(name).toBe('page.png')
    expect(blob.size).toBe(3)
    expect(mocks.queue).not.toHaveBeenCalled()
  })

  it('refreshes added fonts on loading another page and reuses previously registered files', async () => {
    let added = false
    mocks.get.mockImplementation(async (url: string) => url === '/api/v2/editor/fonts'
      ? { items: [{ id: 'font-bold', url: '/font-bold', isDefault: true },
          ...(added ? [{ id: 'font-new', url: '/font-new', isDefault: false }] : [])] } : new ArrayBuffer(1))
    mocks.document.mockResolvedValue({ pageId: 'page', chapterId: 'chapter', documentRevision: 4, defaultFontId: 'font-bold' })
    const { wrapper, send } = await editor()
    added = true
    mocks.document.mockResolvedValue({ pageId: 'next-page', chapterId: 'chapter', documentRevision: 4 })
    await wrapper.setProps({ pageId: 'next-page' })
    await flushPromises()
    expect(mocks.get.mock.calls.filter(([url]) => url === '/font-bold')).toHaveLength(1)
    expect(mocks.get.mock.calls.filter(([url]) => url === '/font-new')).toHaveLength(1)
    expect(send.mock.calls.filter(([message]) => message.payload.op === 'font')).toHaveLength(2)
    expect(send.mock.calls.some(([message]) => message.payload.op === 'bubbles')).toBe(false)
  })

  it('uses one fixed dark palette and ignores site theme changes without saving or reopening', async () => {
    const root = document.documentElement
    const before = root.style.cssText
    root.style.setProperty('--color-surface-inverse', '#1a1a2e')
    root.style.setProperty('--color-surface-inverse-canvas', '#0d1b2a')
    root.style.setProperty('--color-surface-inverse-depth', '#0f0f23')
    root.style.setProperty('--color-surface-inverse-raised', '#2d2d44')
    root.style.setProperty('--color-text-inverse', '#fff')
    root.style.setProperty('--color-surface-brand', '#6366f1')
    root.style.setProperty('--color-status-error-bright', '#ff6b6b')
    root.style.setProperty('--color-status-warning-bright', '#ffc800')
    root.style.setProperty('--radius-md', '8px')
    root.style.setProperty('--radius-lg', '12px')
    try {
      const { wrapper, send } = await editor()
      expect(wrapper.emitted('document-loaded')).toEqual([['page']])
      const themes = () => send.mock.calls.map(([message]) => message.payload).filter(payload => payload.op === 'theme')
      expect(themes()).toHaveLength(1)
      expect(themes()[0]).toMatchObject({ colors: {
        surface: '#1a1a2e', canvas: '#0d1b2a', background: '#0f0f23', field: '#2d2d44',
        text: '#ffffff', accent: '#6366f1', danger: '#ff6b6b', warning: '#ffc800',
      }, radius: 8, radiusLarge: 12 })
      expect(themes()[0]).not.toHaveProperty('dark')
      useSettingsStore().setTheme('dark')
      await flushPromises()
      expect(themes()).toHaveLength(1)
      await wrapper.setProps({ active: false })
      useSettingsStore().setTheme('light')
      await flushPromises()
      expect(themes()).toHaveLength(1)
      await wrapper.setProps({ active: true })
      await flushPromises()
      expect(themes()).toHaveLength(1)
      expect(send.mock.calls.filter(([message]) => message.payload.op === 'open')).toHaveLength(1)
      expect(mocks.queue).not.toHaveBeenCalled()
    } finally {
      root.style.cssText = before
      useSettingsStore().setTheme('light')
    }
  })

  it('locks the previous canvas after a failed load and retries loading instead of saving it', async () => {
    vi.mocked(fetch).mockRejectedValueOnce(new Error('load offline'))
    const { wrapper, frame, send } = await editor()
    expect(wrapper.text()).toContain('load offline')
    expect(frame.hasAttribute('inert')).toBe(true)
    await wrapper.findAll('button').find(button => button.text() === '重试')!.trigger('click')
    await flushPromises()
    expect(frame.hasAttribute('inert')).toBe(false)
    expect(send.mock.calls.some(([message]) => message.payload.op === 'open')).toBe(true)
    expect(mocks.port).toHaveBeenCalledWith('page', expect.any(Object))
    expect(mocks.queue).not.toHaveBeenCalled()
  })

  it('does not write a new page revision for a save with no edits', async () => {
    const { wrapper } = await editor()
    await wrapper.findAll('button').find(button => button.text() === '保存')!.trigger('click')
    await flushPromises()
    expect(mocks.queue).not.toHaveBeenCalled()
    expect(mocks.flush).toHaveBeenCalledWith('page')
  })

  it('ignores a failed obsolete load after another page is ready', async () => {
    let fail!: (reason: Error) => void
    mocks.document.mockImplementationOnce(() => new Promise((_resolve, reject) => { fail = reject }))
    const { wrapper, frame } = await editor()
    mocks.document.mockResolvedValue({ pageId: 'next-page', chapterId: 'chapter', documentRevision: 4 })
    await wrapper.setProps({ pageId: 'next-page' })
    await flushPromises()
    expect(frame.hasAttribute('inert')).toBe(false)
    fail(new Error('obsolete page failed'))
    await flushPromises()
    expect(frame.hasAttribute('inert')).toBe(false)
    expect(wrapper.text()).not.toContain('obsolete page failed')
    expect(wrapper.emitted('document-loaded')).toEqual([['next-page']])
  })

  it.each(['button', 'shortcut'])('keeps a failed %s save in the canvas and retries the same editor state', async source => {
    const { wrapper, frame, send, revise } = await editor()
    revise(5)
    mocks.queue.mockRejectedValueOnce(new Error('disk full'))
    mocks.flush.mockRejectedValueOnce(new Error('disk full'))
    if (source === 'button') {
      await wrapper.findAll('button').find(button => button.text() === '保存')!.trigger('click')
    } else {
      window.dispatchEvent(new MessageEvent('message', {
        origin: location.origin, source: frame.contentWindow,
        data: { saberEditor: true, event: 'save' },
      }))
    }
    await flushPromises()
    expect(wrapper.text()).toContain('disk full')
    await wrapper.findAll('button').find(button => button.text() === '重试')!.trigger('click')
    await flushPromises()
    expect(mocks.queue).toHaveBeenCalledTimes(2)
    expect(send.mock.calls.filter(([message]) => message.payload.op === 'open')).toHaveLength(1)
    expect(wrapper.text()).not.toContain('disk full')
  })

  it('allows repair during autosave but waits for the existing save queue before executing', async () => {
    const { wrapper, frame, revise } = await editor()
    revise(5)
    let completeSave!: () => void
    mocks.queue.mockReturnValue(new Promise<void>(resolve => { completeSave = resolve }))
    mocks.repair.mockResolvedValue({ operationId: 'repair' })
    mocks.get.mockResolvedValue(new ArrayBuffer(1))
    window.dispatchEvent(new MessageEvent('message', {
      origin: location.origin, source: frame.contentWindow,
      data: { saberEditor: true, event: 'changed', pageId: 'page', revision: 5 },
    }))
    await flushPromises()
    expect(wrapper.text()).toContain('正在保存')
    const repair = wrapper.findAll('button').find(button => button.text() === '修复选区')!
    expect(repair.attributes('disabled')).toBeUndefined()
    await repair.trigger('click')
    await flushPromises()
    expect(mocks.repair).not.toHaveBeenCalled()
    completeSave()
    await flushPromises()
    expect(mocks.flush).toHaveBeenCalledWith('page')
    expect(mocks.repair).toHaveBeenCalledOnce()
  })

  it('adopts the repaired revision before the existing render path runs', async () => {
    const { wrapper } = await editor()
    mocks.repair.mockResolvedValue({ operationId: 'repair' })
    mocks.get.mockResolvedValue(new ArrayBuffer(1))
    mocks.document.mockResolvedValue({ pageId: 'page', chapterId: 'chapter', documentRevision: 5 })
    mocks.render.mockImplementation(async () => {
      expect(useImageStore().currentImage?.documentRevision).toBe(5)
    })
    await wrapper.findAll('button').find(button => button.text() === '修复选区')!.trigger('click')
    await flushPromises()
    expect(mocks.render).toHaveBeenCalledOnce()
    expect(mocks.repair).toHaveBeenCalledWith('page', expect.any(Blob), { method: 'solid', fillColor: '#ffffff', baseRevision: 4 })
    expect(wrapper.get('iframe').attributes()).not.toHaveProperty('inert')
  })

  it('does not autosave canvas notifications while a repair owns the document', async () => {
    const { wrapper, frame, revise } = await editor()
    let complete!: () => void
    mocks.repair.mockResolvedValue({ operationId: 'repair' })
    mocks.wait.mockReturnValue(new Promise<void>(resolve => { complete = resolve }))
    mocks.get.mockResolvedValue(new ArrayBuffer(1))
    const changed = (revision: number) => window.dispatchEvent(new MessageEvent('message', {
      origin: location.origin, source: frame.contentWindow,
      data: { saberEditor: true, event: 'changed', pageId: 'page', revision },
    }))
    await wrapper.findAll('button').find(button => button.text() === '修复选区')!.trigger('click')
    await flushPromises()
    expect(frame.hasAttribute('inert')).toBe(true)
    changed(5)
    window.dispatchEvent(new MessageEvent('message', {
      origin: location.origin, source: frame.contentWindow,
      data: { saberEditor: true, event: 'save' },
    }))
    await flushPromises()
    expect(mocks.queue).not.toHaveBeenCalled()
    mocks.document.mockResolvedValue({ pageId: 'page', chapterId: 'chapter', documentRevision: 5 })
    revise(5)
    complete()
    await flushPromises()
    expect(wrapper.text()).not.toContain('正在修复')
    expect(useImageStore().currentImage?.documentRevision).toBe(5)
    changed(6)
    await flushPromises()
    expect(mocks.queue).toHaveBeenCalledExactlyOnceWith('page', 6)
  })

  it('keeps a committed repair and reloads instead of saving a stale canvas after replacement fails', async () => {
    const { wrapper, frame, failReplace } = await editor()
    mocks.repair.mockResolvedValue({ operationId: 'repair' })
    mocks.get.mockImplementation(async (url: string) => url === '/api/v2/editor/fonts'
      ? { items: [] } : new ArrayBuffer(1))
    mocks.document.mockResolvedValue({ pageId: 'page', chapterId: 'chapter', documentRevision: 5 })
    failReplace('canvas response timed out')
    await wrapper.findAll('button').find(button => button.text() === '修复选区')!.trigger('click')
    await flushPromises()
    expect(wrapper.text()).toContain('canvas response timed out')
    expect(frame.hasAttribute('inert')).toBe(true)
    expect(useImageStore().currentImage?.documentRevision).toBe(5)
    expect(mocks.render).toHaveBeenCalledOnce()
    window.dispatchEvent(new MessageEvent('message', {
      origin: location.origin, source: frame.contentWindow,
      data: { saberEditor: true, event: 'changed', pageId: 'page', revision: 6 },
    }))
    await flushPromises()
    expect(mocks.queue).not.toHaveBeenCalled()
    failReplace()
    vi.mocked(fetch).mockResolvedValue({
      ok: true, headers: new Headers({ 'X-Document-Revision': '5' }),
      arrayBuffer: async () => new ArrayBuffer(1),
    } as Response)
    await wrapper.findAll('button').find(button => button.text() === '重试')!.trigger('click')
    await flushPromises()
    expect(frame.hasAttribute('inert')).toBe(false)
    expect(wrapper.text()).not.toContain('canvas response timed out')
    expect(mocks.queue).not.toHaveBeenCalled()
  })
})
