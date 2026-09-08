// @vitest-environment jsdom
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { DEFAULT_PREFERENCE } from './storage'
import { ExtensionUi, type UiCallbacks } from './ui'
import type { BrowserSessionDto } from './types'
import type { StudioState } from './studio/protocol'
function callbacks(): UiCallbacks {
  return {
    onDiscover: vi.fn(),
    onConfirm: vi.fn(),
    onPreferenceChange: vi.fn(),
    onPanelOpenChange: vi.fn(),
    onFabPositionChange: vi.fn(),
    onToggleGlobal: vi.fn().mockResolvedValue(true),
    onTogglePage: vi.fn().mockResolvedValue(true),
    onRetryPage: vi.fn(),
    onRetryUploads: vi.fn(),
    onRetryStart: vi.fn(),
    onStopDiscovery: vi.fn(),
    onCancel: vi.fn(),
    onLoadLibraryBooks: vi.fn().mockResolvedValue([]),
    onImport: vi.fn().mockResolvedValue({
      destination: 'new',
      bookId: 'book',
      bookTitle: 'Example chapter',
      chapterId: 'chapter',
      chapterTitle: 'Example chapter',
      importedPages: 1,
      omittedPages: 0,
      termsAdded: 0,
    }),
    onDisableSite: vi.fn(),
    onDeleteAdaptation: vi.fn(),
    onCopyDiagnostics: vi.fn(),
  }
}

let ui: ExtensionUi
let handlers: UiCallbacks
let receive: EventListener
let post: ReturnType<typeof vi.spyOn>
const origin = 'chrome-extension://test-extension'
beforeEach(() => {
  vi.stubGlobal('chrome', {
    runtime: { getURL: (path: string) => `${origin}/${path}` },
  })
  const add = window.addEventListener.bind(window)
  vi.spyOn(window, 'addEventListener').mockImplementation((type, listener, options) => {
    if (type === 'message') receive = listener as EventListener
    add(type, listener, options)
  })
  handlers = callbacks()
  ui = new ExtensionUi(handlers, DEFAULT_PREFERENCE, 'Example chapter', true)
  const frameWindow = { postMessage: vi.fn() }
  Object.defineProperty(ui.shadow.querySelector('iframe'), 'contentWindow', {
    value: frameWindow,
  })
  post = frameWindow.postMessage
})
afterEach(() => {
  ui.remove()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})
async function send(action: string, payload?: unknown, overrides = {}) {
  receive({
    isTrusted: true,
    origin,
    source: ui.shadow.querySelector('iframe')!.contentWindow,
    data: { channel: 'saber:command', id: 1, action, payload },
    ...overrides,
  } as unknown as Event)
  await new Promise(resolve => setTimeout(resolve, 0))
}
function snapshot(): StudioState {
  return post.mock.calls.filter((call: any[]) => call[0].channel === 'saber:state').at(-1)![0].state
}
it('uses one closed host and one extension document for all three views', async () => {
  expect(ui.host.shadowRoot).toBeNull()
  ui.openManagement('settings')
  expect(snapshot().tab).toBe('settings')
  ui.openManagement('tasks')
  expect(snapshot().tab).toBe('tasks')
  expect(ui.shadow.querySelectorAll('iframe')).toHaveLength(1)
  await send('close')
  expect(handlers.onPanelOpenChange).toHaveBeenCalledWith(false)
})
it('rejects messages from webpage scripts, other frames and synthetic events', async () => {
  await send('confirm', ['page'], { source: window })
  await send('confirm', ['page'], { origin: 'https://example.com' })
  await send('confirm', ['page'], { isTrusted: false })
  expect(handlers.onConfirm).not.toHaveBeenCalled()
  await send('confirm', ['page'])
  expect(handlers.onConfirm).toHaveBeenCalledWith(['page'])
  expect(post).toHaveBeenCalledWith(
    { channel: 'saber:response', id: 1, ok: true, result: undefined },
    origin
  )
})
it('returns actionable RPC failures and does not invoke arbitrary methods', async () => {
  vi.mocked(handlers.onLoadLibraryBooks).mockRejectedValueOnce(new Error('连接已断开'))
  await send('books')
  expect(post).toHaveBeenLastCalledWith(
    { channel: 'saber:response', id: 1, ok: false, error: '连接已断开' },
    origin
  )
  await send('remove')
  expect(ui.host.isConnected).toBe(true)
})
it('retains upload and start recovery independently of polling updates', async () => {
  ui.showUploadError(2, { code: 'source', message: '读取失败' })
  ui.showSession({
    state: 'idle',
    pendingStart: true,
    pages: [],
    counts: { total: 0 },
  } as unknown as BrowserSessionDto)
  expect(snapshot()).toMatchObject({
    uploadError: { count: 2 },
    retryStart: true,
  })
  ui.clearUploadError()
  expect(snapshot().uploadError).toBeNull()
  await send('retry-start')
  await send('retry-uploads')
  expect(handlers.onRetryStart).toHaveBeenCalledOnce()
  expect(handlers.onRetryUploads).toHaveBeenCalledOnce()
})
it('updates original/result state and imports through the controller', async () => {
  vi.mocked(handlers.onToggleGlobal).mockResolvedValue(false)
  await send('toggle-global')
  expect(snapshot().translated).toBe(false)
  vi.mocked(handlers.onTogglePage).mockResolvedValue(false)
  await send('toggle-page', 'page')
  expect(snapshot().originalPageIds).toEqual(['page'])
  vi.mocked(handlers.onTogglePage).mockResolvedValue(true)
  await send('toggle-page', 'page')
  expect(snapshot().originalPageIds).toEqual([])
  await send('import', {
    destination: 'new',
    bookTitle: 'Example chapter',
    chapterTitle: 'Example chapter',
  })
  expect(snapshot().imported?.bookId).toBe('book')
})
it('hides the frame during image picking and restores it on cancel', () => {
  ui.setOpen(true)
  ui.setStatus('正在识别', '等待选图', 'busy')
  ui.startPicking()
  expect(ui.pickingMask().dataset.open).toBe('true')
  expect(ui.shadow.querySelector<HTMLElement>('.saber-panel')!.style.visibility).toBe('hidden')
  ui.stopPicking()
  expect(snapshot().notice.tone).toBe('ready')
  expect(ui.shadow.querySelector<HTMLElement>('.saber-panel')!.style.visibility).toBe('')
})
it('removes bridge listeners when the page controller is disposed', async () => {
  const remove = vi.spyOn(window, 'removeEventListener')
  ui.remove()
  expect(remove).toHaveBeenCalledWith('message', receive)
  expect(ui.host.isConnected).toBe(false)
})

it('keeps learned rules in the shared view state', () => {
  ui.setAdaptation({ selector: 'main img', kind: 'image', confirmedAt: 1 })
  expect(snapshot().preference.rule?.selector).toBe('main img')
  ui.setAdaptation(null)
  expect(snapshot().preference.rule).toBeUndefined()
})

it('anchors each opening to the FAB but preserves manual placement while open', () => {
  vi.stubGlobal('innerWidth', 1280)
  vi.stubGlobal('innerHeight', 960)
  const fab = ui.shadow.querySelector<HTMLElement>('.saber-fab')!
  const panel = ui.shadow.querySelector<HTMLElement>('.saber-panel')!
  let fabBox = new DOMRect(36, 136, 48, 48)
  vi.spyOn(fab, 'getBoundingClientRect').mockImplementation(() => fabBox)
  vi.spyOn(panel, 'getBoundingClientRect').mockReturnValue(new DOMRect(0, 0, 380, 680))
  ui.setOpen(true)
  expect(panel.style.left).toBe('8px')
  expect(panel.style.top).toBe('196px')
  panel.style.left = '250px'
  panel.style.top = '80px'
  ui.openManagement('tasks')
  expect(panel.style.left).toBe('250px')
  expect(panel.style.top).toBe('80px')
  ui.setOpen(false)
  fabBox = new DOMRect(616, 456, 48, 48)
  ui.setOpen(true)
  expect(panel.style.left).toBe('676px')
  expect(panel.style.top).toBe('140px')
})
it('shortens the panel when a narrow viewport has no full-height space beside the FAB', () => {
  vi.stubGlobal('innerWidth', 390)
  vi.stubGlobal('innerHeight', 740)
  const fab = ui.shadow.querySelector<HTMLElement>('.saber-fab')!
  const panel = ui.shadow.querySelector<HTMLElement>('.saber-panel')!
  vi.spyOn(fab, 'getBoundingClientRect').mockReturnValue(new DOMRect(170, 340, 48, 48))
  vi.spyOn(panel, 'getBoundingClientRect').mockImplementation(
    () => new DOMRect(0, 0, 374, Number.parseFloat(panel.style.maxHeight) || 640)
  )
  ui.setOpen(true)
  expect(panel.style.maxHeight).toBe('332px')
  expect(panel.style.top).toBe('400px')
  expect(panel.style.left).toBe('8px')
})
