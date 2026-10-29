import { defineComponent, h, nextTick, ref } from 'vue'
import { mount, flushPromises } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { usePageHistory } from '@/composables/edit/usePageHistory'
import type { V2PageDocument } from '@/api/v2/content'

const mocks = vi.hoisted(() => ({ get: vi.fn(), save: vi.fn(), flush: vi.fn(), toast: vi.fn() }))
vi.mock('@/api/v2/content', () => ({ getPageDocument: mocks.get }))
vi.mock('@/services/pageDocumentPersistence', () => ({ saveEditorDocument: mocks.save, flushPageDocument: mocks.flush }))
vi.mock('@/utils/toast', () => ({ showToast: mocks.toast }))
const wrappers: ReturnType<typeof mount>[] = []
beforeEach(() => { vi.resetAllMocks(); vi.useFakeTimers() })
afterEach(() => { wrappers.splice(0).forEach(w => w.unmount()); vi.unstubAllGlobals(); vi.useRealTimers() })

async function harness(bytes = 10) {
  const pageId = ref<string | undefined>('page'), revision = ref<number | undefined>(1), enabled = ref(true), ready = ref(true)
  const dirty = ref(false)
  const doc = () => ({ pageId: pageId.value, chapterId: 'chapter', documentRevision: revision.value, bubbles: [], defaultFontId: null, pageStyleDefaults: {} }) as V2PageDocument
  mocks.get.mockImplementation(async () => doc())
  mocks.flush.mockResolvedValue(undefined)
  mocks.save.mockImplementation(async (_page, _revision, _file, snapshot: V2PageDocument) => ({ document: { ...snapshot, documentRevision: revision.value! + 1 } }))
  vi.stubGlobal('fetch', vi.fn().mockImplementation(async () => ({ ok: true,
    headers: new Headers({ 'X-Document-Revision': String(revision.value) }), blob: async () => ({ size: bytes }) as Blob,
  })))
  let history!: ReturnType<typeof usePageHistory>
  const onRestore = vi.fn(async (document: V2PageDocument) => { revision.value = document.documentRevision; dirty.value = false })
  const wrapper = mount(defineComponent({ setup() {
    history = usePageHistory({ pageId, revision, enabled, ready, dirty, onRestore }); return () => h('div')
  } }))
  wrappers.push(wrapper)
  await flushPromises()
  async function edit() { revision.value!++; await nextTick(); await vi.advanceTimersByTimeAsync(900); await flushPromises() }
  return { history, pageId, revision, enabled, dirty, edit, onRestore }
}

describe('bounded current-page history', () => {
  it('keeps ordinary history when switching modes and captures committed fine edits on return', async () => {
    const { history, enabled, revision, edit } = await harness()
    await edit()
    enabled.value = false
    await flushPromises()
    revision.value = 3
    await flushPromises()
    enabled.value = true
    await flushPromises()
    expect(history.canUndo.value).toBe(true)
    await history.undo()
    expect(mocks.save.mock.calls.at(-1)![3].documentRevision).toBe(2)
    await history.undo()
    expect(mocks.save.mock.calls.at(-1)![3].documentRevision).toBe(1)
    expect(history.canUndo.value).toBe(false)
    expect(history.canRedo.value).toBe(true)
  })
  it('blocks editing immediately when the initial history snapshot starts, before a user can queue a save', async () => {
    const pageId = ref('page'), revision = ref(1), ready = ref(false), enabled = ref(true)
    let history!: ReturnType<typeof usePageHistory>
    mocks.flush.mockImplementation(() => new Promise(() => {}))
    const wrapper = mount(defineComponent({ setup() {
      history = usePageHistory({ pageId, revision, enabled, ready, onRestore: async () => {} }); return () => h('div')
    } }))
    wrappers.push(wrapper)
    ready.value = true
    expect(history.initializing.value).toBe(true)
    wrapper.unmount()
  })
  it('restores the snapshot and business metadata in one existing save, then supports redo and branches', async () => {
    const { history, edit, onRestore } = await harness()
    await edit()
    expect(history.canUndo.value).toBe(true)
    await history.undo()
    expect(mocks.save).toHaveBeenCalledWith('page', 2, expect.objectContaining({ size: 10 }), expect.objectContaining({ documentRevision: 1 }))
    expect(onRestore).toHaveBeenCalledOnce()
    expect(history.canRedo.value).toBe(true)
    await history.redo()
    expect(mocks.save.mock.calls[1]![3].documentRevision).toBe(2)
    await history.undo()
    await edit()
    expect(history.canRedo.value).toBe(false)
  })
  it('keeps the history position if saving fails and releases it on page/mode changes', async () => {
    const { history, edit, pageId, enabled } = await harness()
    await edit()
    mocks.save.mockRejectedValueOnce(new Error('disk full'))
    await history.undo()
    expect(history.canUndo.value).toBe(true)
    expect(history.canRedo.value).toBe(false)
    expect(mocks.toast).toHaveBeenCalledWith('disk full', 'error')
    pageId.value = 'other'
    await flushPromises()
    expect(history.canUndo.value).toBe(false)
    enabled.value = false
    await flushPromises()
    expect(history.canRedo.value).toBe(false)
  })
  it('undoes the last edit once even when a settled save still has a stale dirty flag', async () => {
    const { history, edit, dirty } = await harness()
    await edit()
    dirty.value = true
    await history.undo()
    expect(mocks.save.mock.calls[0]![3].documentRevision).toBe(1)
    expect(history.canUndo.value).toBe(false)
    await history.redo()
    expect(mocks.save.mock.calls[1]![3].documentRevision).toBe(2)
  })
  it('bounds large-page snapshots by bytes as well as count', async () => {
    const { history, edit } = await harness(30 * 1024 * 1024)
    await edit(); await edit(); await edit()
    await history.undo()
    expect(history.canUndo.value).toBe(false)
    expect(history.canRedo.value).toBe(true)
  })
})
