// @vitest-environment jsdom
import { createApp, h, ref, type App } from 'vue'
import { afterEach, expect, it, vi } from 'vitest'
import { usePageBridge } from './pageBridge'
let app: App
let bridge: ReturnType<typeof usePageBridge>
function mount() {
  document.body.innerHTML = '<div id="app"></div>'
  app = createApp({
    setup() {
      bridge = usePageBridge()
      return () => h('div')
    },
  })
  app.mount('#app')
}
afterEach(() => {
  app.unmount()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
  vi.useRealTimers()
})
it('sends selected reactive arrays as cloneable data and resolves matching replies', async () => {
  mount()
  const post = vi.spyOn(window.parent, 'postMessage').mockImplementation(message => {
    structuredClone(message)
  })
  const selected = ref(['page-1', 'page-2'])
  const promise = bridge.request('confirm', selected.value)
  expect(post).toHaveBeenCalledWith(
    { channel: 'saber:command', id: 1, action: 'confirm', payload: ['page-1', 'page-2'] },
    '*'
  )
  window.dispatchEvent(
    new MessageEvent('message', {
      source: window.parent,
      data: { channel: 'saber:response', id: 1, ok: true, result: 'accepted' },
    })
  )
  await expect(promise).resolves.toBe('accepted')
})
it('surfaces controller failures and ignores unrelated message sources', async () => {
  mount()
  vi.spyOn(window.parent, 'postMessage').mockImplementation(() => {})
  window.dispatchEvent(
    new MessageEvent('message', {
      source: null,
      data: { channel: 'saber:state', state: { title: 'unrelated' } },
    })
  )
  expect(bridge.state.value).toBeNull()
  const promise = bridge.request('import')
  window.dispatchEvent(
    new MessageEvent('message', {
      source: window.parent,
      data: { channel: 'saber:response', id: 1, ok: false, error: '书籍不存在' },
    })
  )
  await expect(promise).rejects.toThrow('书籍不存在')
})

it('runs a requested DOM detection in the frame without a shorter client deadline', async () => {
  vi.useFakeTimers()
  vi.stubGlobal('chrome', { storage: { local: { get: async () => ({
    'saber-extension-settings-v1': { token: 'test-token', serverPort: 5000, domains: {} },
  }) } } })
  let finish!: (response: Response) => void
  let signal!: AbortSignal
  const fetch = vi.fn((_url: string, init: RequestInit) => {
    signal = init.signal as AbortSignal
    return new Promise<Response>(resolve => { finish = resolve })
  })
  vi.stubGlobal('fetch', fetch)
  const timer = vi.spyOn(AbortSignal, 'timeout')
  mount()
  const post = vi.spyOn(window.parent, 'postMessage').mockImplementation(() => {})
  const receive = () => window.dispatchEvent(new MessageEvent('message', {
    source: window.parent, data: { channel: 'saber:dom-detection', id: 1,
      payload: { pageUrl: 'https://example.test/chapter', pageTitle: 'Chapter', nodes: [] } },
  }))
  receive()
  expect(fetch).not.toHaveBeenCalled()
  const pending = bridge.request('discover', 'dom-agent')
  receive()
  await vi.advanceTimersByTimeAsync(121_000)
  receive()
  expect(fetch).toHaveBeenCalledOnce()
  expect(timer).not.toHaveBeenCalled()
  expect(signal.aborted).toBe(false)
  finish(Response.json({ nodeIds: ['page'], selector: '' }))
  await vi.advanceTimersByTimeAsync(0)
  expect(post).toHaveBeenCalledWith({ channel: 'saber:dom-detection-result', id: 1,
    ok: true, result: { nodeIds: ['page'], selector: '' } }, '*')
  window.dispatchEvent(new MessageEvent('message', { source: window.parent,
    data: { channel: 'saber:response', id: 1, ok: true } }))
  await pending
})

it('aborts the frame HTTP request when the page exits', async () => {
  vi.stubGlobal('chrome', { storage: { local: { get: async () => ({
    'saber-extension-settings-v1': { token: 'test-token', serverPort: 5000, domains: {} },
  }) } } })
  let signal!: AbortSignal
  vi.stubGlobal('fetch', vi.fn((_url: string, init: RequestInit) => {
    signal = init.signal as AbortSignal
    return new Promise(() => {})
  }))
  mount()
  vi.spyOn(window.parent, 'postMessage').mockImplementation(() => {})
  void bridge.request('discover', 'dom-agent')
  window.dispatchEvent(new MessageEvent('message', { source: window.parent,
    data: { channel: 'saber:dom-detection', id: 1, payload: { nodes: [] } } }))
  await new Promise(resolve => setTimeout(resolve, 0))
  app.unmount()
  expect(signal.aborted).toBe(true)
})
