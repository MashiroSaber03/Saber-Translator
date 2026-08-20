import { flushPromises, mount } from '@vue/test-utils'
import { defineComponent } from 'vue'
import { createMemoryHistory, createRouter } from 'vue-router'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const { settingsStoreMock, taskCenterStoreMock } = vi.hoisted(() => ({
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
}))

vi.mock('@/stores/settings', () => ({
  useSettingsStore: () => settingsStoreMock,
}))

vi.mock('@/stores/taskCenterStore', () => ({
  useTaskCenterStore: () => taskCenterStoreMock,
}))

import App from '@/App.vue'

const RoutePage = defineComponent({ template: '<main>route</main>' })

function createTestRouter() {
  return createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/', name: 'bookshelf', component: RoutePage },
      { path: '/reader', name: 'reader', component: RoutePage },
    ],
  })
}

describe('App reader immersion', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    settingsStoreMock.backendError = ''
    settingsStoreMock.isBackendReady = true
    settingsStoreMock.loadFromBackend.mockResolvedValue(undefined)
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
    taskCenterStoreMock.initialize.mockImplementationOnce(() => new Promise<void>((resolve) => {
      resolveInitialInitialization = resolve
    }))
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
