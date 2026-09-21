// @vitest-environment jsdom
import { createApp, h, nextTick, ref, type App } from 'vue'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import StatusBar from './studio/StatusBar.vue'
import { isStoreInstallation, registerUpdateNotifications, UPDATE_KEY } from './updates'

let app: App | undefined
let active = ref(true)
const sendMessage = vi.fn()
const requestUpdateCheck = vi.fn()
const getSelf = vi.fn()
const changed = { addListener: vi.fn(), removeListener: vi.fn() }
const available = { addListener: vi.fn() }
const installed = { addListener: vi.fn() }
const set = vi.fn()
const remove = vi.fn()
beforeEach(() => {
  vi.clearAllMocks()
  active = ref(true)
  document.body.innerHTML = '<div id="app"></div>'
  sendMessage.mockResolvedValue({ ok: true, data: {} })
  requestUpdateCheck.mockResolvedValue({ status: 'no_update' })
  getSelf.mockResolvedValue({ installType: 'normal', updateUrl: 'https://clients2.google.com/service/update2/crx' })
  vi.stubGlobal('chrome', {
    runtime: { getManifest: () => ({ version: '1.1.3' }), sendMessage, requestUpdateCheck,
      onUpdateAvailable: available, onInstalled: installed },
    management: { getSelf },
    storage: { onChanged: changed, session: { get: async () => ({}), set, remove } },
  })
})
afterEach(() => { app?.unmount(); app = undefined; vi.unstubAllGlobals() })
async function mount() {
  app = createApp({ setup: () => () => h(StatusBar, { active: active.value }) })
  app.mount('#app')
  await vi.waitFor(() => expect(document.body.textContent).toContain('已连接 Saber'))
}
function button(text: string) {
  return Array.from(document.querySelectorAll('button')).find(b => b.textContent?.trim() === text)!
}
it('shows shared connection and version state without automatically checking updates', async () => {
  await mount()
  expect(document.body.textContent).toContain('v1.1.3')
  expect(requestUpdateCheck).not.toHaveBeenCalled()
  active.value = false
  await nextTick()
  window.dispatchEvent(new Event('focus'))
  expect(sendMessage).toHaveBeenCalledTimes(1)
  sendMessage.mockResolvedValue({ ok: false, error: { message: '配对令牌无效' } })
  active.value = true
  await vi.waitFor(() => expect(document.body.textContent).toContain('未连接 Saber'))
  expect(document.body.textContent).toContain('配对令牌无效')
  expect(button('设置连接')).toBeTruthy()
})
it.each(['no_update', 'throttled', 'update_available'])('handles store result %s', async status => {
  requestUpdateCheck.mockResolvedValue({ status, version: '1.1.4' })
  await mount()
  button('v1.1.3').click()
  await nextTick()
  button('检查更新').click()
  await vi.waitFor(() => expect(requestUpdateCheck).toHaveBeenCalledOnce())
  await nextTick()
  expect(document.body.textContent).toContain(status === 'no_update' ? '暂未发现' : status === 'throttled' ? '过于频繁' : '等待浏览器安装')
  if (status === 'update_available') expect(set).toHaveBeenCalledWith({ [UPDATE_KEY]: '1.1.4' })
})
it('hides store checks for unpacked installations', async () => {
  getSelf.mockResolvedValue({ installType: 'development' })
  await mount()
  button('v1.1.3').click()
  await nextTick()
  expect(button('检查更新')).toBeUndefined()
  expect(document.body.textContent).toContain('非商店安装')
})
it('keeps the connection action mounted while a focus check is pending', async () => {
  await mount()
  sendMessage.mockResolvedValueOnce({ ok: false, error: { message: '未启动后端' } })
  window.dispatchEvent(new Event('focus'))
  await vi.waitFor(() => expect(button('设置连接')).toBeTruthy())
  const configure = button('设置连接')
  let finish!: (value: unknown) => void
  sendMessage.mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
  window.dispatchEvent(new Event('focus'))
  await nextTick()
  expect(button('设置连接')).toBe(configure)
  finish({ ok: true, data: {} })
  await vi.waitFor(() => expect(document.body.textContent).toContain('已连接 Saber'))
})
it('shows update check errors without changing connection status', async () => {
  requestUpdateCheck.mockRejectedValueOnce(new Error('network error'))
  await mount()
  button('v1.1.3').click()
  await nextTick()
  button('检查更新').click()
  await vi.waitFor(() => expect(document.body.textContent).toContain('检查更新失败'))
  expect(document.body.textContent).toContain('已连接 Saber')
})
it('recognizes Edge by update source and excludes unrelated CRX installations', async () => {
  getSelf.mockResolvedValue({ installType: 'normal', updateUrl: 'https://edge.microsoft.com/extensionwebstorebase/v1/crx' })
  expect(await isStoreInstallation()).toBe(true)
  getSelf.mockResolvedValue({ installType: 'normal', updateUrl: 'https://example.com/update.xml' })
  expect(await isStoreInstallation()).toBe(false)
})
it('persists a browser update notification and clears it after installation', async () => {
  registerUpdateNotifications()
  available.addListener.mock.calls[0]![0]({ version: '1.1.4' })
  await vi.waitFor(() => expect(set).toHaveBeenCalledWith({ [UPDATE_KEY]: '1.1.4' }))
  installed.addListener.mock.calls[0]![0]({ reason: 'update' })
  expect(remove).toHaveBeenCalledWith(UPDATE_KEY)
})
