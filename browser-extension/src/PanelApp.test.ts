// @vitest-environment jsdom
import { createApp, nextTick, type App } from 'vue'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import PanelApp from './PanelApp.vue'
import type { PluginSettingsApi } from '../../vue-frontend/src/types/browserExtensionSettings'

vi.mock('./studio/TranslationView.vue', () => ({
  default: { render: () => null },
}))
vi.mock('./studio/SettingsView.vue', () => ({
  default: { render: () => null },
}))
let app: App
const flush = async () => {
  await new Promise(resolve => setImmediate(resolve))
  await nextTick()
}
beforeEach(() => {
  vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
  location.hash = 'tasks'
  document.body.innerHTML = '<div id="app"></div>'
  Element.prototype.scrollTo = vi.fn()
})
afterEach(() => {
  app.unmount()
  vi.useRealTimers()
  vi.unstubAllGlobals()
})
const job = {
  jobId: 'test-job',
  kind: 'translation',
  status: 'running',
  target: { chapter: '漫画章节' },
  progress: {
    completedItems: 1,
    totalItems: 3,
    failedItems: 0,
    cancelledItems: 0,
    skippedItems: 0,
  },
}
function mount(api: ReturnType<typeof vi.fn>) {
  app = createApp(PanelApp, { api: api as PluginSettingsApi })
  app.mount('#app')
}
const list = () => ({ items: [job], queuePaused: false, workerOnline: true })
const button = (label: string) =>
  [...document.querySelectorAll('button')].find(
    node => (node.getAttribute('aria-label') || node.textContent?.trim()) === label
  )!
const observe = (open: boolean) =>
  window.dispatchEvent(
    new MessageEvent('message', {
      source: window.parent,
      data: {
        channel: 'saber:state',
        state: { open, tab: location.hash.slice(1) },
      },
    })
  )

it('pauses refresh while the floating window is hidden and refreshes when reopened', async () => {
  const api = vi.fn().mockResolvedValue(list())
  mount(api)
  await flush()
  expect(api).toHaveBeenCalledTimes(1)
  observe(false)
  await nextTick()
  await vi.advanceTimersByTimeAsync(9000)
  expect(api).toHaveBeenCalledTimes(1)
  observe(true)
  await flush()
  expect(api).toHaveBeenCalledTimes(2)
  location.hash = 'settings'
  window.dispatchEvent(new HashChangeEvent('hashchange'))
  await flush()
  await vi.advanceTimersByTimeAsync(9000)
  expect(api).toHaveBeenCalledTimes(2)
})

it('updates expand/collapse controls without rebuilding unchanged task cards', async () => {
  const api = vi.fn(async (path: string) =>
    path.includes('/jobs?') ? list() : { ...job, failedItems: [], error: null }
  )
  mount(api)
  await flush()
  const card = document.querySelector('.job-card')
  button('查看详情').click()
  await flush()
  expect(button('收起详情')).toBeDefined()
  expect(document.body.textContent).toContain('任务 test-job')
  await vi.advanceTimersByTimeAsync(3000)
  expect(document.querySelector('.job-card')).toBe(card)
  button('收起详情').click()
  await flush()
  expect(button('查看详情')).toBeDefined()
  expect(document.querySelector('.job-detail')).toBeNull()
})

it('honors the outer entry after internal tab navigation without reloading', async () => {
  mount(vi.fn().mockResolvedValue(list()))
  await flush()
  const open = (section: string) =>
    window.dispatchEvent(
      new MessageEvent('message', {
        source: window.parent,
        data: { channel: 'saber:state', state: { tab: section } },
      })
    )
  open('settings')
  await nextTick()
  expect(location.hash).toBe('#settings')
  button('任务中心').click()
  await nextTick()
  expect(location.hash).toBe('#tasks')
  open('settings')
  await nextTick()
  expect(button('翻译配置').getAttribute('aria-selected')).toBe('true')
})

it('clears recovered load errors and displays command failures until retried', async () => {
  const api = vi.fn().mockRejectedValueOnce(new Error('连接中断')).mockResolvedValue(list())
  mount(api)
  await flush()
  expect(document.body.textContent).toContain('连接中断')
  button('刷新').click()
  await flush()
  expect(document.body.textContent).not.toContain('连接中断')
  api.mockRejectedValueOnce(new Error('任务已结束'))
  button('暂停').click()
  await flush()
  expect(document.body.textContent).toContain('任务已结束')
  button('暂停').click()
  await flush()
  expect(document.body.textContent).not.toContain('任务已结束')
})

it('does not overlap task refreshes and stops scheduling after unmount', async () => {
  let finish!: (value: unknown) => void
  const api = vi.fn(
    () =>
      new Promise(resolve => {
        finish = resolve
      })
  )
  mount(api)
  await flush()
  observe(false)
  await nextTick()
  observe(true)
  await nextTick()
  await vi.advanceTimersByTimeAsync(6000)
  expect(api).toHaveBeenCalledTimes(1)
  app.unmount()
  finish(list())
  await flush()
  await vi.advanceTimersByTimeAsync(6000)
  expect(api).toHaveBeenCalledTimes(1)
})
