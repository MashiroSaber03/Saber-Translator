// @vitest-environment jsdom
import { createApp, h, nextTick, ref, type App } from 'vue'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import ConnectionView from './studio/ConnectionView.vue'

let app: App
let connection: { token: string; serverPort: number; hostname: string }
let active = ref(true)
const sendMessage = vi.fn()
const tokenInput = () => document.querySelector<HTMLInputElement>('input[type=password]')!
const portInput = () => document.querySelector<HTMLInputElement>('input[type=number]')!
beforeEach(() => {
  document.body.innerHTML = '<div id="app"></div>'
  active = ref(true)
  connection = { token: 'original-token-with-at-least-32-characters', serverPort: 5000, hostname: 'comic.example' }
  sendMessage.mockReset().mockImplementation(async request => {
    if (request.type === 'get-connection-state') return { ok: true, data: { ...connection } }
    if (request.type === 'save-connection') connection = { ...connection, token: request.token.trim(), serverPort: request.serverPort }
    return { ok: true, data: {} }
  })
  vi.stubGlobal('chrome', {
    runtime: { sendMessage },
    storage: { onChanged: { addListener: vi.fn(), removeListener: vi.fn() } },
  })
  app = createApp({ setup: () => () => h(ConnectionView, { active: active.value, state: null, request: async () => {} }) })
  app.mount('#app')
})
afterEach(() => { app.unmount(); vi.unstubAllGlobals() })
async function reopen() {
  active.value = false
  await nextTick()
  active.value = true
  await nextTick()
}
it('refreshes an unedited connection form after another tab saves', async () => {
  await vi.waitFor(() => expect(document.body.textContent).toContain('已连接 Saber'))
  connection = { ...connection, token: 'new-token-from-another-tab-with-32-characters', serverPort: 5100 }
  await reopen()
  await vi.waitFor(() => expect(tokenInput().value).toBe(connection.token))
  expect(portInput().value).toBe('5100')
})
it('retains an unsaved draft without claiming to have tested that draft', async () => {
  await vi.waitFor(() => expect(document.body.textContent).toContain('已连接 Saber'))
  tokenInput().value = 'unsaved-token-with-at-least-32-characters'
  tokenInput().dispatchEvent(new Event('input', { bubbles: true }))
  connection.token = 'new-token-from-another-tab-with-32-characters'
  sendMessage.mockClear()
  await reopen()
  await vi.waitFor(() => expect(document.body.textContent).toContain('连接信息尚未保存'))
  expect(tokenInput().value).toBe('unsaved-token-with-at-least-32-characters')
  expect(sendMessage).not.toHaveBeenCalledWith({ type: 'status' })
  document.querySelector('form')!.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }))
  await vi.waitFor(() => expect(document.body.textContent).toContain('已连接 Saber'))
  connection.token = 'updated-again-after-the-draft-was-saved'
  await reopen()
  await vi.waitFor(() => expect(tokenInput().value).toBe(connection.token))
})

it('updates a visible clean form on a connection storage change', async () => {
  await vi.waitFor(() => expect(document.body.textContent).toContain('已连接 Saber'))
  const previous = { ...connection }
  connection.token = 'a-token-updated-while-this-view-stays-open'
  const listener = vi.mocked(chrome.storage.onChanged.addListener).mock.calls[0]![0]
  listener({ 'saber-extension-settings-v1': { oldValue: previous, newValue: connection } }, 'local')
  await vi.waitFor(() => expect(tokenInput().value).toBe(connection.token))
})
