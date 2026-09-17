import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent, h, onUnmounted } from 'vue'
import { createMemoryHistory, createRouter, onBeforeRouteLeave, RouterView } from 'vue-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import AccountDock from '@/components/common/AccountDock.vue'
import { useAuthStore } from '@/stores/authStore'
import { useRuntimeStore } from '@/stores/runtimeStore'

const mocks = vi.hoisted(() => ({ logout: vi.fn(), toast: vi.fn(), credentials: vi.fn() }))
vi.mock('@/api/v2/auth', () => ({
  logout: mocks.logout,
  getCurrentSession: vi.fn(), login: vi.fn(), register: vi.fn(), getCapabilities: vi.fn(),
}))
vi.mock('@/utils/toast', () => ({ showToast: mocks.toast }))
vi.mock('@/services/browserCredentials', () => ({ configureBrowserCredentials: mocks.credentials }))

const cleanups: Array<() => void> = []

async function setup(leave: () => Promise<boolean>, start = '/translate') {
  const pinia = createPinia()
  setActivePinia(pinia)
  const runtime = useRuntimeStore()
  runtime.assumeLocalForTests()
  runtime.capabilities = { ...runtime.capabilities!, profile: 'public', requiresAuth: true }
  const auth = useAuthStore()
  auth.user = { id: 'alice', username: 'Alice', role: 'user' }
  auth.assetUsageBytes = 100
  auth.assetQuotaBytes = 1000
  const unmounted = vi.fn()
  const Editor = defineComponent({
    setup() {
      onBeforeRouteLeave(leave)
      onUnmounted(unmounted)
      return () => h('div', 'editor')
    },
  })
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/translate', component: Editor },
      { path: '/', name: 'bookshelf', component: { render: () => h('div', 'bookshelf') } },
      { path: '/login', component: { render: () => h('div', 'login') } },
      { path: '/account', component: { render: () => h('div', 'account') } },
    ],
  })
  await router.push(start)
  const wrapper = mount(defineComponent({ render: () => h('div', [h(RouterView), h(AccountDock)]) }), {
    global: { plugins: [pinia, router], stubs: { BaseModal: true } },
  })
  cleanups.push(() => wrapper.unmount())
  await flushPromises()
  const button = () => wrapper.findAll('button').find(item => item.text().includes('退出'))!
  return { auth, router, wrapper, button, unmounted }
}

beforeEach(() => {
  mocks.logout.mockReset().mockResolvedValue(undefined)
  mocks.toast.mockReset()
  mocks.credentials.mockReset()
})
afterEach(() => { cleanups.splice(0).forEach(cleanup => cleanup()) })

describe('AccountDock saves before logging out', () => {
  it('keeps authentication until the leave guard finishes and ignores repeat clicks', async () => {
    let finish!: (value: boolean) => void
    const leave = vi.fn(() => new Promise<boolean>(resolve => { finish = resolve }))
    const app = await setup(leave)
    mocks.logout.mockImplementation(async () => {
      expect(app.unmounted).toHaveBeenCalledOnce()
      expect(app.auth.authenticated).toBe(true)
    })
    await app.button().trigger('click')
    await flushPromises()
    await app.button().trigger('click')
    expect(app.button().attributes('disabled')).toBeDefined()
    expect(leave).toHaveBeenCalledOnce()
    expect(mocks.logout).not.toHaveBeenCalled()
    expect(app.auth.authenticated).toBe(true)
    finish(true)
    await flushPromises()
    expect(mocks.logout).toHaveBeenCalledOnce()
    expect(app.router.currentRoute.value.path).toBe('/login')
    expect(app.auth.authenticated).toBe(false)
    expect(app.auth.assetUsageBytes).toBe(0)
    expect(mocks.toast).not.toHaveBeenCalled()
  })

  it('stays on the editor with the session intact after save rejection, then allows retry', async () => {
    const leave = vi.fn().mockResolvedValueOnce(false).mockResolvedValueOnce(true)
    const app = await setup(leave)
    await app.button().trigger('click')
    await flushPromises()
    expect(app.router.currentRoute.value.path).toBe('/translate')
    expect(app.auth.authenticated).toBe(true)
    expect(mocks.logout).not.toHaveBeenCalled()
    expect(app.button().attributes('disabled')).toBeUndefined()
    await app.button().trigger('click')
    await flushPromises()
    expect(mocks.logout).toHaveBeenCalledOnce()
    expect(app.router.currentRoute.value.path).toBe('/login')
  })

  it('does not log out if another navigation cancels leaving the editor', async () => {
    let finish!: (value: boolean) => void
    const leave = vi.fn().mockImplementationOnce(() => new Promise<boolean>(resolve => { finish = resolve }))
      .mockResolvedValue(true)
    const app = await setup(leave)
    await app.button().trigger('click')
    await flushPromises()
    await app.router.push('/account')
    finish(true)
    await flushPromises()
    expect(mocks.logout).not.toHaveBeenCalled()
    expect(app.auth.authenticated).toBe(true)
    expect(app.router.currentRoute.value.path).toBe('/account')
  })

  it('logs out directly from the bookshelf without treating duplicate navigation as failure', async () => {
    const app = await setup(vi.fn().mockResolvedValue(true), '/')
    await app.button().trigger('click')
    await flushPromises()
    expect(mocks.logout).toHaveBeenCalledOnce()
    expect(app.router.currentRoute.value.path).toBe('/login')
  })

  it('preserves the client session and reports a failed logout request', async () => {
    mocks.logout.mockRejectedValue(new Error('网络连接失败'))
    const app = await setup(vi.fn().mockResolvedValue(true))
    await app.button().trigger('click')
    await flushPromises()
    expect(app.auth.authenticated).toBe(true)
    expect(app.router.currentRoute.value.path).toBe('/')
    expect(app.auth.assetUsageBytes).toBe(100)
    expect(mocks.credentials).not.toHaveBeenCalled()
    expect(mocks.toast).toHaveBeenCalledWith('退出失败：网络连接失败', 'error')
    expect(app.button().attributes('disabled')).toBeUndefined()
  })
})
