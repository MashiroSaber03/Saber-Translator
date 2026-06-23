import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import BookshelfView from '@/views/BookshelfView.vue'

const { getBooksMock, getTagsMock, getServerInfoMock, routerPushMock } = vi.hoisted(() => ({
  getBooksMock: vi.fn(),
  getTagsMock: vi.fn(),
  getServerInfoMock: vi.fn(),
  routerPushMock: vi.fn(),
}))

vi.mock('vue-router', () => ({
  useRouter: () => ({ push: routerPushMock }),
}))

vi.mock('@/api', () => ({
  getServerInfo: getServerInfoMock,
}))

vi.mock('@/api/bookshelf', () => ({
  getBooks: getBooksMock,
  getTags: getTagsMock,
  getBookDetail: vi.fn(),
}))

const AppShellStub = defineComponent({
  template: '<div class="app-shell-stub"><slot /></div>',
})

const AppHeaderStub = defineComponent({
  template: '<header class="app-header-stub"><slot name="header-links" /></header>',
})

function mountView() {
  setActivePinia(createPinia())
  return mount(BookshelfView, {
    global: {
      stubs: {
        AppShell: AppShellStub,
        AppHeader: AppHeaderStub,
        BookSearch: true,
        BookCard: true,
        BookModal: true,
        BookDetailModal: true,
        TagManageModal: true,
      },
    },
  })
}

describe('BookshelfView', () => {
  beforeEach(() => {
    routerPushMock.mockReset()
    getBooksMock.mockReset()
    getTagsMock.mockReset()
    getServerInfoMock.mockReset()
    getBooksMock.mockResolvedValue({ success: true, books: [] })
    getTagsMock.mockResolvedValue({ success: true, tags: [] })
    getServerInfoMock.mockResolvedValue({ success: true, lan_url: 'http://localhost:5173' })
  })

  it('registers pageshow handling before async bookshelf loading settles', () => {
    const addEventListenerSpy = vi.spyOn(window, 'addEventListener')
    const removeEventListenerSpy = vi.spyOn(window, 'removeEventListener')
    const logSpy = vi.spyOn(console, 'log').mockImplementation(() => undefined)
    getBooksMock.mockReturnValue(new Promise(() => {}))

    const wrapper = mountView()

    expect(addEventListenerSpy).toHaveBeenCalledWith('pageshow', expect.any(Function))
    const pageShowEvent = new Event('pageshow') as PageTransitionEvent
    Object.defineProperty(pageShowEvent, 'persisted', { value: true })
    window.dispatchEvent(pageShowEvent)
    expect(logSpy).not.toHaveBeenCalled()

    wrapper.unmount()
    expect(removeEventListenerSpy).toHaveBeenCalledWith('pageshow', expect.any(Function))

    addEventListenerSpy.mockRestore()
    removeEventListenerSpy.mockRestore()
    logSpy.mockRestore()
  })

  it('exposes accessible header actions and safe external links', () => {
    const wrapper = mountView()

    expect(wrapper.get('.bookshelf-header__copy-button').attributes('aria-label')).toBe('复制局域网地址')
    expect(wrapper.get('.bookshelf-header__theme-toggle').attributes('aria-label')).toBe('功能开发中')
    expect(wrapper.get('.bookshelf-header__tutorial-link').attributes('rel')).toBe('noopener noreferrer')
    expect(wrapper.get('.bookshelf-header__github-link').attributes('rel')).toBe('noopener noreferrer')
  })
})
