import { flushPromises, mount } from '@vue/test-utils'
import { defineComponent } from 'vue'
import { createMemoryHistory, createRouter } from 'vue-router'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const { authStoreMock, runtimeStoreMock, settingsStoreMock, taskCenterStoreMock } = vi.hoisted(
  () => ({
    authStoreMock: {
      authenticated: false,
      user: null as { id: string } | null,
      markUnauthenticated: vi.fn(),
    },
    runtimeStoreMock: {
      capabilities: { requiresAuth: false } as { requiresAuth: boolean } | null,
    },
    settingsStoreMock: {
      backendError: '',
      isBackendReady: true,
      initSettings: vi.fn(),
      loadFromBackend: vi.fn(),
    },
    taskCenterStoreMock: {
      close: vi.fn(),
      disconnect: vi.fn(),
      initialize: vi.fn(),
    },
  })
)

vi.mock('@/stores/settings', () => ({
  useSettingsStore: () => settingsStoreMock,
}))

vi.mock('@/stores/taskCenterStore', () => ({
  useTaskCenterStore: () => taskCenterStoreMock,
}))

vi.mock('@/stores/runtimeStore', () => ({
  useRuntimeStore: () => runtimeStoreMock,
}))

vi.mock('@/stores/authStore', () => ({
  useAuthStore: () => authStoreMock,
}))

import App from '@/App.vue'

const RoutePage = defineComponent({ template: '<main data-testid="route-page">route</main>' })

function createTestRouter() {
  return createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/', name: 'bookshelf', component: RoutePage },
      { path: '/reader', name: 'reader', component: RoutePage },
      { path: '/admin', name: 'admin', component: RoutePage, meta: { standalone: true } },
    ],
  })
}

describe('App reader immersion', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    settingsStoreMock.backendError = ''
    settingsStoreMock.isBackendReady = true
    runtimeStoreMock.capabilities = { requiresAuth: false }
    authStoreMock.authenticated = false
    authStoreMock.user = null
    settingsStoreMock.loadFromBackend.mockResolvedValue(true)
    taskCenterStoreMock.initialize.mockResolvedValue(undefined)
  })

  it('does not label a recoverable save error as restricted mode', async () => {
    settingsStoreMock.backendError = 'translation.boxExpand.ratio must be a number'
    settingsStoreMock.isBackendReady = true
    const router = createTestRouter()
    await router.push('/')
    await router.isReady()
    const wrapper = mount(App, {
      global: {
        plugins: [router],
        stubs: {
          ProductConfirmProvider: true,
          ProductTextInputProvider: true,
          TaskCenterDrawer: true,
          TaskCenterLauncher: true,
          ToastNotification: true,
        },
      },
    })
    await flushPromises()

    expect(wrapper.find('.backend-restricted-banner').exists()).toBe(false)
    expect(wrapper.find('[data-testid="route-page"]').exists()).toBe(true)
    wrapper.unmount()
  })

  it('does not mount an application route until saved settings are ready', async () => {
    settingsStoreMock.isBackendReady = true
    let resolveSettings!: (value: boolean) => void
    settingsStoreMock.loadFromBackend.mockImplementationOnce(
      () =>
        new Promise<boolean>(resolve => {
          resolveSettings = value => {
            resolve(value)
          }
        })
    )
    const router = createTestRouter()
    await router.push('/')
    await router.isReady()
    const wrapper = mount(App, {
      global: {
        plugins: [router],
        stubs: {
          ProductConfirmProvider: true,
          ProductTextInputProvider: true,
          TaskCenterDrawer: { template: '<div data-testid="task-drawer" />' },
          TaskCenterLauncher: { template: '<div data-testid="task-launcher" />' },
          ToastNotification: true,
        },
      },
    })
    await flushPromises()

    expect(wrapper.text()).toContain('正在读取已保存的应用设置')
    expect(wrapper.find('[data-testid="route-page"]').exists()).toBe(false)
    expect(taskCenterStoreMock.initialize).not.toHaveBeenCalled()

    resolveSettings(true)
    await flushPromises()

    expect(wrapper.find('[data-testid="route-page"]').exists()).toBe(true)
    expect(taskCenterStoreMock.initialize).toHaveBeenCalledOnce()
    wrapper.unmount()
  })

  it('keeps an application route unmounted when settings loading fails', async () => {
    settingsStoreMock.isBackendReady = false
    settingsStoreMock.backendError = '后端设置不可用'
    settingsStoreMock.loadFromBackend.mockResolvedValueOnce(false)
    const router = createTestRouter()
    await router.push('/')
    await router.isReady()
    const wrapper = mount(App, {
      global: {
        plugins: [router],
        stubs: {
          ProductConfirmProvider: true,
          ProductTextInputProvider: true,
          TaskCenterDrawer: true,
          TaskCenterLauncher: true,
          ToastNotification: true,
        },
      },
    })
    await flushPromises()

    expect(wrapper.text()).toContain('应用设置加载失败：后端设置不可用')
    expect(wrapper.text()).toContain('重试')
    expect(wrapper.find('[data-testid="route-page"]').exists()).toBe(false)
    expect(taskCenterStoreMock.initialize).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('mounts standalone routes without loading application settings or task services', async () => {
    settingsStoreMock.isBackendReady = false
    const router = createTestRouter()
    await router.push('/admin')
    await router.isReady()
    const wrapper = mount(App, {
      global: {
        plugins: [router],
        stubs: {
          ProductConfirmProvider: true,
          ProductTextInputProvider: true,
          TaskCenterDrawer: true,
          TaskCenterLauncher: true,
          ToastNotification: true,
        },
      },
    })
    await flushPromises()

    expect(wrapper.find('[data-testid="route-page"]').exists()).toBe(true)
    expect(settingsStoreMock.loadFromBackend).not.toHaveBeenCalled()
    expect(taskCenterStoreMock.initialize).not.toHaveBeenCalled()
    wrapper.unmount()
  })

  it('waits for runtime capabilities before starting protected application services', async () => {
    runtimeStoreMock.capabilities = null
    const router = createTestRouter()
    await router.push('/')
    await router.isReady()
    const wrapper = mount(App, {
      global: {
        plugins: [router],
        stubs: {
          ProductConfirmProvider: true,
          ProductTextInputProvider: true,
          TaskCenterDrawer: { template: '<div data-testid="task-drawer" />' },
          TaskCenterLauncher: { template: '<div data-testid="task-launcher" />' },
          ToastNotification: true,
        },
      },
    })
    await flushPromises()

    expect(settingsStoreMock.loadFromBackend).not.toHaveBeenCalled()
    expect(taskCenterStoreMock.initialize).not.toHaveBeenCalled()
    expect(wrapper.find('[data-testid="task-launcher"]').exists()).toBe(false)
    expect(wrapper.find('[data-testid="task-drawer"]').exists()).toBe(false)
    wrapper.unmount()
  })

  it('does not mount or initialize the task center on the reader route', async () => {
    const router = createTestRouter()
    await router.push('/reader')
    await router.isReady()
    const wrapper = mount(App, {
      global: {
        plugins: [router],
        stubs: {
          ProductConfirmProvider: true,
          ProductTextInputProvider: true,
          TaskCenterDrawer: { template: '<div data-testid="task-drawer" />' },
          TaskCenterLauncher: { template: '<div data-testid="task-launcher" />' },
          ToastNotification: true,
        },
      },
    })
    await flushPromises()

    expect(taskCenterStoreMock.initialize).not.toHaveBeenCalled()
    expect(taskCenterStoreMock.close).toHaveBeenCalledOnce()
    expect(taskCenterStoreMock.disconnect).toHaveBeenCalledOnce()
    expect(wrapper.find('[data-testid="task-launcher"]').exists()).toBe(false)
    expect(wrapper.find('[data-testid="task-drawer"]').exists()).toBe(false)

    await router.push('/')
    await flushPromises()

    expect(taskCenterStoreMock.initialize).toHaveBeenCalledOnce()
    expect(wrapper.find('[data-testid="task-launcher"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="task-drawer"]').exists()).toBe(true)
    wrapper.unmount()
  })

  it('does not let an older initialization disconnect the current non-reader route', async () => {
    let resolveInitialInitialization: (() => void) | undefined
    taskCenterStoreMock.initialize.mockImplementationOnce(
      () =>
        new Promise<void>(resolve => {
          resolveInitialInitialization = resolve
        })
    )
    const router = createTestRouter()
    await router.push('/')
    await router.isReady()
    const wrapper = mount(App, {
      global: {
        plugins: [router],
        stubs: {
          ProductConfirmProvider: true,
          ProductTextInputProvider: true,
          TaskCenterDrawer: { template: '<div data-testid="task-drawer" />' },
          TaskCenterLauncher: { template: '<div data-testid="task-launcher" />' },
          ToastNotification: true,
        },
      },
    })
    await flushPromises()

    await router.push('/reader')
    await flushPromises()
    expect(taskCenterStoreMock.disconnect).toHaveBeenCalledOnce()

    await router.push('/')
    await flushPromises()
    expect(taskCenterStoreMock.initialize).toHaveBeenCalledTimes(2)
    expect(wrapper.find('[data-testid="task-launcher"]').exists()).toBe(true)

    resolveInitialInitialization?.()
    await flushPromises()

    expect(taskCenterStoreMock.disconnect).toHaveBeenCalledOnce()
    wrapper.unmount()
  })
})
