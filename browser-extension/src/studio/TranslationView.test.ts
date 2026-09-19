// @vitest-environment jsdom
import { createApp, nextTick, reactive, type App } from 'vue'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import TranslationView from './TranslationView.vue'
import type { StudioState } from './protocol'
import { DEFAULT_PREFERENCE } from '../storage'
const download = vi.hoisted(() => vi.fn(async () => undefined))
vi.mock('./downloadOriginals', () => ({ downloadOriginals: download }))

let app: App
let state: StudioState
let request: ReturnType<typeof vi.fn>
const button = (label: string) => [...document.querySelectorAll('button')].find(node => node.textContent?.replace('✦', '').trim() === label)!
const flush = async () => { await new Promise(resolve => setTimeout(resolve, 0)); await nextTick() }
beforeEach(() => {
  download.mockReset().mockResolvedValue(undefined)
  document.body.innerHTML = '<div id="app"></div>'
  state = reactive({
    open: true, title: 'Test', preference: { ...DEFAULT_PREFERENCE }, tab: 'translate',
    view: 'idle', notice: { title: 'Ready', message: '', tone: 'ready' },
    candidates: [], session: null, originalPageIds: [], translated: true,
    preparation: null, uploadError: null, retryStart: false, discoveryStopped: false,
    terms: [], imported: null,
  })
  request = vi.fn(async () => undefined)
  app = createApp(TranslationView, { state, request })
  app.mount('#app')
})
afterEach(() => app.unmount())
function completedSession() {
  state.session = {
    id: 'session', pageUrl: '', pageTitle: '', bookId: '', chapterId: '', mode: 'standard',
    glossaryEnabled: false, autoTermsEnabled: false, state: 'completed', pendingStart: false, expiresAt: '',
    counts: { total: 1, completed: 1, queued: 0, translating: 0, failed: 0, cancelled: 0 },
    pages: [{ id: 'page', clientPageKey: 'key', ordinal: 1, pageId: 'page', state: 'completed', resultReady: false, retryCount: 0, error: null }],
  }
  state.view = 'progress'
}
it('locks method and mode until asynchronous recognition completes', async () => {
  let finish!: () => void
  request.mockImplementation(() => new Promise<void>(resolve => { finish = resolve }))
  button('识别漫画图片').click()
  await flush()
  expect((document.querySelector('[role="combobox"]') as HTMLButtonElement).disabled).toBe(true)
  expect(button('高质量翻译').disabled).toBe(true)
  finish()
  await flush()
  expect(button('高质量翻译').disabled).toBe(false)
})
it('can cancel preparation without waiting for its upload request to finish', async () => {
  state.view = 'candidates'
  state.candidates = [{ id: 'image', sourceUrl: null, width: 600, height: 900 }]
  await flush()
  let finish!: () => void
  request.mockImplementation((action: string) => action === 'confirm'
    ? new Promise<void>(resolve => { finish = resolve }) : Promise.resolve())
  button('开始翻译 · 1 张').click()
  completedSession()
  state.session!.state = 'idle'
  await flush()
  button('取消任务').click()
  await flush()
  expect(request).toHaveBeenCalledWith('cancel', undefined)
  expect(button('重新选图').disabled).toBe(false)
  finish()
  await flush()
})
it('shows independent discovery state, explicit retry configuration and reselect', async () => {
  completedSession()
  state.discoveryStopped = true
  await flush()
  expect(document.body.textContent).toContain('继续发现已停止')
  expect(button('按原配置重翻')).toBeDefined()
  button('继续发现').click()
  await flush()
  expect(request).toHaveBeenCalledWith('resume-discovery', undefined)
  button('重新选图').click()
  await flush()
  expect(request).toHaveBeenCalledWith('reselect', undefined)
})

it.each([false, true])('downloads the checked images and cleans up the temporary session (failure: %s)', async (fails) => {
  state.view = 'candidates'
  state.candidates = [{ id: 'one', sourceUrl: null, width: 600, height: 900 }, { id: 'two', sourceUrl: null, width: 600, height: 900 }]
  await flush()
  const checkbox = document.querySelector('input[value="two"]') as HTMLInputElement
  checkbox.click()
  await flush()
  request.mockResolvedValue('download-session')
  if (fails) download.mockRejectedValue(new Error('下载失败'))
  button('下载原图（ZIP）').click()
  await flush()
  expect(request).toHaveBeenCalledWith('prepare-download', ['one'])
  expect(download).toHaveBeenCalledWith('download-session', 'Test')
  expect(request).toHaveBeenCalledWith('finish-download', fails ? undefined : '已发起原图 ZIP 下载')
  expect(checkbox.checked).toBe(false)
  expect(button('下载原图（ZIP）').disabled).toBe(false)
  if (fails) expect(document.querySelector('[role="alert"]')?.textContent).toContain('下载失败')
})

it('opens the existing library dialog and imports only the checked originals', async () => {
  state.view = 'candidates'
  state.candidates = [{ id: 'one', sourceUrl: null, width: 600, height: 900 }]
  request.mockImplementation(async (action: string, payload: unknown) => {
    if (action === 'import-selected') structuredClone(payload)
    return action === 'books' ? [] : { importedPages: 1, bookTitle: 'Test' }
  })
  await flush()
  button('仅导入书架').click()
  await flush()
  button('确认导入').click()
  await flush()
  expect(request).toHaveBeenCalledWith('import-selected', { ids: ['one'], command: { destination: 'new', bookTitle: 'Test', chapterTitle: 'Test' } })
  expect(document.querySelector('[role="dialog"]')).toBeNull()
  expect(request.mock.calls.some(([action]) => action === 'confirm')).toBe(false)
})
