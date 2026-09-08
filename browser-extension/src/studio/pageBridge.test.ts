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
