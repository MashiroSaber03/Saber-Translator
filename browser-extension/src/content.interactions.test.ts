// @vitest-environment jsdom
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { PageController } from './content'
import { DEFAULT_PREFERENCE } from './storage'
import { elementCandidate } from './discovery'

let controller: any
let send: ReturnType<typeof vi.fn>
const session = () => ({ id: 'old-session', state: 'completed', pages: [], pendingStart: false, counts: { total: 0, completed: 0, queued: 0, translating: 0, failed: 0, cancelled: 0 } })
beforeEach(async () => {
  document.body.innerHTML = '<img class="comic" src="https://example.test/page.png" width="600" height="900">'
  vi.stubGlobal('CSS', { escape: (value: string) => value })
  send = vi.fn(async (request: any) => ({ ok: true, data: request.type === 'get-preference' ? { ...DEFAULT_PREFERENCE } : request.type === 'create-session' ? { ...session(), id: 'new-session' } : {} }))
  vi.stubGlobal('chrome', { runtime: { id: '', getURL: (path: string) => `chrome-extension://audit/${path}`, sendMessage: send } })
  controller = new PageController()
  await controller.initialize()
})
afterEach(async () => { await controller.dispose(); vi.restoreAllMocks(); vi.unstubAllGlobals() })

it('appends a context image without discarding existing results or changing discovery', async () => {
  controller.session = session()
  send.mockImplementation(async (request: any) => {
    if (request.type === 'hash-source') return { ok: true, data: 'image-key' }
    if (request.type === 'upload-page') return { ok: true, data: { id: 'added-page' } }
    if (request.type === 'start-session') return { ok: true, data: session() }
    return { ok: true, data: {} }
  })
  await controller.translateContextImage('https://example.test/page.png')
  expect(send).not.toHaveBeenCalledWith(expect.objectContaining({ type: 'discard-session' }))
  expect(send).not.toHaveBeenCalledWith(expect.objectContaining({ type: 'create-session' }))
  expect(controller.session.id).toBe('old-session')
  expect(controller.discoveryStopped).toBe(false)
  expect(send).toHaveBeenCalledWith(expect.objectContaining({ type: 'upload-page', payload: expect.objectContaining({ sessionId: 'old-session' }) }))
  expect(send).toHaveBeenCalledWith({ type: 'start-session', sessionId: 'old-session' })
  await controller.translateContextImage('https://example.test/page.png')
  expect(send.mock.calls.filter(([request]) => request.type === 'upload-page')).toHaveLength(1)
})

it('keeps the recognition method stable until the pending detection finishes', async () => {
  await controller.updatePreference({ method: 'dom-agent' })
  let resolve!: (result: any) => void
  const response = new Promise(done => { resolve = done })
  send.mockImplementation(async (request: any) => request.type === 'dom-detection' ? response : ({ ok: true, data: {} }))
  const pending = controller.discover('dom-agent')
  await controller.updatePreference({ method: 'similar' })
  const nodes = send.mock.calls.find(([request]) => request.type === 'dom-detection')![0].payload.nodes
  resolve({ ok: true, data: { nodeIds: nodes.map((node: any) => node.id), selector: '' } })
  await pending
  expect(controller.preference.method).toBe('dom-agent')
  expect(controller.activeMethod).toBe('dom-agent')
  expect(controller.ui.state.view).toBe('candidates')
  expect(controller.ui.state.candidates).toHaveLength(1)
  await controller.updatePreference({ method: 'similar' })
  expect(controller.preference.method).toBe('similar')
})

it('retains stopped discovery state through refresh and can resume', async () => {
  controller.session = session()
  controller.stopDiscovery()
  expect(controller.ui.state.notice.title).toBe('已停止继续发现')
  controller.showSession(controller.session)
  expect(controller.discoveryStopped).toBe(true)
  expect(controller.ui.state.notice.title).toBe('当前图片已全部完成')
  expect(controller.ui.state.discoveryStopped).toBe(true)
  const scan = vi.spyOn(controller, 'discoverLazyImages').mockResolvedValue(undefined)
  await controller.resumeDiscovery()
  expect(controller.discoveryStopped).toBe(false)
  expect(controller.ui.state.discoveryStopped).toBe(false)
  expect(controller.observer).not.toBeNull()
  expect(scan).toHaveBeenCalledOnce()
})

it.each([false, true])('restores polling and the previous discovery state (%s) after failed import', async (stopped) => {
  controller.session = session()
  controller.startObserver()
  if (stopped) controller.stopDiscovery()
  controller.startPolling(60000)
  vi.spyOn(controller, 'poll').mockResolvedValue(undefined)
  send.mockImplementation(async (request: any) => request.type === 'import-session' ? ({ ok: false, error: { code: 'conflict', message: '目标书籍不可用', retryable: false } }) : ({ ok: true, data: {} }))
  await expect(controller.importToLibrary({ destination: 'existing', targetBookId: 'missing', chapterTitle: 'chapter' })).rejects.toThrow('目标书籍不可用')
  expect(controller.currentTask()).not.toBeNull()
  expect(controller.pollTimer).not.toBeNull()
  expect(Boolean(controller.observer)).toBe(!stopped)
  expect(controller.discoveryStopped).toBe(stopped)
})

it.each([false, true])('reselects without creating a task and preserves imported library data (%s)', async (imported) => {
  controller.session = session()
  controller.imported = imported
  controller.showSession(controller.session)
  await controller.reselect()
  expect(controller.session).toBeNull()
  expect(controller.ui.state.view).toBe('idle')
  expect(controller.ui.state.session).toBeNull()
  expect(controller.ui.state.candidates).toEqual([])
  expect(send.mock.calls.filter(([request]) => request.type === 'discard-session')).toHaveLength(imported ? 0 : 1)
  expect(send).not.toHaveBeenCalledWith(expect.objectContaining({ type: 'create-session' }))
})

it('removes the picker keyboard listener when the page is disposed', async () => {
  await controller.discover('similar')
  const stop = vi.spyOn(controller.ui, 'stopPicking')
  await controller.dispose()
  expect(stop).toHaveBeenCalledOnce()
  document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }))
  expect(stop).toHaveBeenCalledOnce()
  expect(controller.pickerCleanup).toBeNull()
})

it('ignores a DOM recognition failure after the page has exited', async () => {
  let fail!: (error: Error) => void
  send.mockImplementation((request: any) => request.type === 'dom-detection'
    ? new Promise((_resolve, reject) => { fail = reject }) : Promise.resolve({ ok: true, data: {} }))
  const pending = controller.discover('dom-agent')
  await controller.dispose()
  fail(new Error('network failure'))
  await expect(pending).resolves.toBeUndefined()
})

it('stops automatic discovery after task-center cancellation', async () => {
  controller.session = session()
  controller.startObserver()
  send.mockResolvedValue({ ok: true, data: { ...session(), state: 'cancelled' } })
  await controller.poll(controller.currentTask())
  expect(controller.discoveryStopped).toBe(true)
  expect(controller.observer).toBeNull()
  const upload = vi.spyOn(controller, 'uploadCandidates')
  await controller.discoverLazyImages()
  expect(upload).not.toHaveBeenCalled()
  expect(send).not.toHaveBeenCalledWith(expect.objectContaining({ type: 'start-session' }))
})

it('keeps polling if cancellation fails so the task can be cancelled again', async () => {
  controller.session = session()
  send.mockResolvedValue({ ok: false, error: { code: 'saber_unreachable', message: 'offline', retryable: true } })
  await controller.cancel()
  expect(controller.pollTimer).not.toBeNull()
  expect(controller.discoveryStopped).toBe(true)
})

it.each(['import', 'download', 'failure'])('collects only selected originals without starting translation (%s)', async (operation) => {
  const first = document.querySelector('img')!
  const second = first.cloneNode() as HTMLImageElement
  second.src = 'https://example.test/ignored.png'
  document.body.append(second)
  controller.candidates = [elementCandidate(first), elementCandidate(second)]
  controller.ui.showCandidates(controller.candidates)
  const upload = vi.fn(async (..._args: unknown[]) => operation === 'failure' ? 0 : 1)
  vi.spyOn(controller, 'uploadCandidates').mockImplementation(upload)
  send.mockImplementation(async (request: any) => ({ ok: true, data:
    request.type === 'create-session' ? { ...session(), id: 'collect' } :
    request.type === 'import-session' ? { importedPages: 1, bookTitle: 'Saved' } : {} }))
  if (operation === 'failure') {
    await expect(controller.prepareSelected([controller.candidates[0].id])).rejects.toThrow('部分图片读取失败')
  } else if (operation === 'import') {
    await controller.importSelected([controller.candidates[0].id], { destination: 'new', bookTitle: 'Saved', chapterTitle: 'One' })
    expect(send).toHaveBeenCalledWith(expect.objectContaining({ type: 'import-session', payload: expect.objectContaining({ originalsOnly: true }) }))
    expect(send).not.toHaveBeenCalledWith(expect.objectContaining({ type: 'discard-session', sessionId: 'collect' }))
  } else {
    expect(await controller.prepareSelected([controller.candidates[0].id])).toBe('collect')
    expect(controller.ui.state.view).toBe('candidates')
    await controller.finishSelected('下载已发起')
  }
  expect(upload.mock.calls[0]![0]).toEqual([controller.candidates[0]])
  expect(send).not.toHaveBeenCalledWith(expect.objectContaining({ type: 'start-session' }))
  expect(controller.pollTimer).toBeNull()
  expect(controller.observer).toBeNull()
  expect(controller.session).toBeNull()
  expect(controller.taskStarting).toBeNull()
  expect(controller.ui.state.view).toBe('candidates')
})
