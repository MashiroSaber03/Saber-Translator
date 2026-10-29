import { computed, onBeforeUnmount, ref, watch, type Ref } from 'vue'
import { getPageDocument, type V2PageDocument } from '@/api/v2/content'
import { flushPageDocument, saveEditorDocument } from '@/services/pageDocumentPersistence'
import { showToast } from '@/utils/toast'

interface Snapshot { file: Blob; document: V2PageDocument; cost: number }
const MAX_BYTES = 64 * 1024 * 1024
const MAX_ENTRIES = 10

/** Current-page undo only. Reuses the native document and its atomic save entry. */
export function usePageHistory(options: {
  pageId: Readonly<Ref<string | undefined>>
  revision: Readonly<Ref<number | undefined>>
  enabled: Readonly<Ref<boolean>>
  ready: Readonly<Ref<boolean>>
  dirty?: Readonly<Ref<boolean>>
  onRestore: (document: V2PageDocument) => Promise<void>
}) {
  const busy = ref(false)
  const initializing = ref(false)
  const error = ref('')
  const position = ref(-1)
  const length = ref(0)
  let entries: Snapshot[] = []
  let lastRevision: number | undefined
  let generation = 0
  let controller = new AbortController()
  let timer: ReturnType<typeof setTimeout> | undefined
  let capturing: Promise<void> | undefined
  const available = () => options.enabled.value && options.ready.value && !!options.pageId.value

  function capture(): Promise<void> {
    if (capturing) return capturing
    const page = options.pageId.value, token = generation, signal = controller.signal
    if (!page || !available() || (!options.dirty?.value && options.revision.value === lastRevision)) return Promise.resolve()
    const work = async () => {
      await flushPageDocument(page)
      if (signal.aborted) return
      const response = await fetch(`/api/v2/pages/${encodeURIComponent(page)}/editor-document`, {
        credentials: 'same-origin', cache: 'no-store', signal: AbortSignal.any([signal, AbortSignal.timeout(15_000)]),
      })
      if (!response.ok) throw new Error('撤销记录加载失败')
      const revision = Number(response.headers.get('X-Document-Revision'))
      const file = await response.blob()
      const document = await getPageDocument(page, signal)
      if (token !== generation) return
      if (document.pageId !== page || document.documentRevision !== revision) throw new Error('页面正在保存，撤销记录稍后更新')
      // A no-op flush can leave the UI dirty; only record new committed versions.
      if (revision === lastRevision) return
      const cost = file.size + JSON.stringify(document).length * 2
      if (cost > MAX_BYTES) throw new Error('本页文档较大，已超过 64 MiB 撤销缓存上限')
      entries.splice(position.value + 1)
      entries.push({ file, document, cost })
      let bytes = entries.reduce((sum, entry) => sum + entry.cost, 0)
      while (entries.length > MAX_ENTRIES || bytes > MAX_BYTES) bytes -= entries.shift()!.cost
      position.value = entries.length - 1
      length.value = entries.length
      lastRevision = revision
      error.value = ''
    }
    const promise = work().finally(() => { if (capturing === promise) capturing = undefined })
    capturing = promise
    return promise
  }

  function schedule(): void {
    clearTimeout(timer)
    if (!available() || busy.value) return
    // A burst of typing/dragging is one page operation; no full snapshots per keystroke.
    const token = generation
    timer = setTimeout(() => { void capture().catch(reason => { if (token === generation) error.value = String(reason instanceof Error ? reason.message : reason) }) }, 800)
  }

  watch([options.pageId, options.enabled, options.ready], async ([page], previous) => {
    generation++
    controller.abort(); controller = new AbortController()
    clearTimeout(timer)
    capturing = undefined
    if (page !== previous?.[0]) {
      entries = []; position.value = -1; length.value = 0; lastRevision = undefined
      error.value = ''
    }
    // A mode switch pauses observation, not the page's history. Returning from
    // fine editing captures its committed document through the same snapshot path.
    initializing.value = available() && entries.length === 0
    const token = generation
    if (!available()) return
    try { await capture() }
    catch (reason) { if (token === generation) error.value = reason instanceof Error ? reason.message : '撤销记录加载失败' }
    finally { if (token === generation) initializing.value = false }
  }, { immediate: true, flush: 'sync' })
  watch(options.revision, schedule)

  async function step(direction: -1 | 1): Promise<void> {
    if (!available() || busy.value || initializing.value) return
    busy.value = true
    clearTimeout(timer)
    const token = generation
    try {
      await capture()
      // A save may have completed while an earlier capture was in flight.
      if (options.revision.value !== lastRevision) await capture()
      if (token !== generation) return
      const next = position.value + direction, snapshot = entries[next]
      if (!snapshot) return
      const response = await saveEditorDocument(options.pageId.value!, lastRevision!, snapshot.file, snapshot.document)
      if (token !== generation) return
      lastRevision = response.document.documentRevision
      position.value = next
      await options.onRestore(response.document)
      error.value = ''
    } catch (reason) { showToast(reason instanceof Error ? reason.message : '撤销失败，请重试', 'error') }
    finally { busy.value = false }
  }

  onBeforeUnmount(() => { generation++; controller.abort(); clearTimeout(timer); entries = [] })
  return { busy, initializing, error, canUndo: computed(() => position.value > 0 || (length.value > 0 && (options.dirty?.value || options.revision.value !== lastRevision))),
    canRedo: computed(() => !options.dirty?.value && options.revision.value === lastRevision && position.value >= 0 && position.value < length.value - 1),
    undo: () => step(-1), redo: () => step(1) }
}
