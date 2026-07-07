import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { defineComponent } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import BookshelfView from '@/views/BookshelfView.vue'
import { useSettingsStore } from '@/stores/settings'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductCardGrid from '@/components/product/ProductCardGrid.vue'
import UiIcon from '@/components/ui/UiIcon.vue'

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

const ProductPageHeaderStub = defineComponent({
  props: {
    actionsLabel: {
      type: String,
      default: '页面操作',
    },
    navLabel: {
      type: String,
      default: '页面导航',
    },
    variant: {
      type: String,
      default: 'default',
    },
  },
  template: `
    <header class="product-page-header" :class="'product-page-header--' + variant">
      <slot name="meta" />
      <nav :aria-label="navLabel"><slot name="nav" /></nav>
      <div role="group" :aria-label="actionsLabel"><slot name="actions" /></div>
    </header>
  `,
})

function mountView() {
  setActivePinia(createPinia())
  return mount(BookshelfView, {
    global: {
      stubs: {
        AppShell: AppShellStub,
        ProductPageHeader: ProductPageHeaderStub,
        ProductCardGrid,
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

  it('exposes accessible header actions and safe external links', async () => {
    const wrapper = mountView()
    const settingsStore = useSettingsStore()

    expect(wrapper.get('.product-page-header--brand').exists()).toBe(true)
    expect(wrapper.get('nav[aria-label="书架外部链接"]').exists()).toBe(true)
    expect(wrapper.get('[role="group"][aria-label="书架偏好操作"]').exists()).toBe(true)
    expect(wrapper.get('[aria-label="复制局域网地址"]').text()).toContain('复制')
    const themeToggle = wrapper.get('.bookshelf-header__theme-toggle')
    expect(themeToggle.attributes('aria-label')).toBe('切换深色模式')
    await themeToggle.trigger('click')
    expect(settingsStore.theme).toBe('dark')
    expect(wrapper.get('.bookshelf-header__tutorial-link').attributes('rel')).toBe('noopener noreferrer')
    const githubLink = wrapper.get('.bookshelf-header__github-link')
    expect(githubLink.attributes('rel')).toBe('noopener noreferrer')
    expect(githubLink.getComponent(UiIcon).props('name')).toBe('github')
  })

  it('keeps header metadata free of DOM id hooks', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/BookshelfView.vue'), 'utf8')

    expect(source).not.toContain('id="lanUrl"')
    expect(source).not.toContain('github.jpg')
    expect(source).not.toContain('bookshelf-header__github-icon')
  })

  it('routes header metadata through the shared product meta pill', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/BookshelfView.vue'), 'utf8')

    expect(source).toContain("import ProductHeaderMetaPill from '@/components/product/ProductHeaderMetaPill.vue'")
    expect(source).toContain('<ProductHeaderMetaPill')
    expect(source).not.toContain('bookshelf-header__lan-access')
    expect(source).not.toContain('bookshelf-header__lan-icon')
  })

  it('renders create-book actions with the shared plus icon', () => {
    const wrapper = mountView()

    const createButtons = wrapper.findAll('button')
      .filter(button => button.text().includes('新建书籍') || button.text().includes('新建第一本书'))

    expect(createButtons).toHaveLength(2)
    for (const button of createButtons) {
      expect(button.getComponent(UiIcon).props('name')).toBe('plus')
      expect(button.text()).not.toContain('+')
    }
  })

  it('groups bookshelf toolbar commands through the product action row', () => {
    const wrapper = mountView()

    const actionRow = wrapper.getComponent(ProductActionRow)
    expect(actionRow.props('ariaLabel')).toBe('书架主要操作')
    expect(actionRow.props('justify')).toBe('end')

    const toolbar = wrapper.get('.bookshelf-toolbar__actions')
    expect(toolbar.attributes('role')).toBe('group')
    expect(toolbar.attributes('aria-label')).toBe('书架主要操作')
    expect(toolbar.findAllComponents(UiIcon).map(icon => icon.props('name'))).toEqual([
      'plus',
      'tags',
      'languages',
    ])

    const source = readFileSync(resolve(process.cwd(), 'src/views/BookshelfView.vue'), 'utf8')
    expect(source).not.toContain('class="toolbar-actions"')
    expect(source).not.toContain('.toolbar-actions')
    expect(source).not.toContain('class="page-title"')
    expect(source).not.toContain('.page-title')
    expect(source).not.toContain('class="books-container"')
    expect(source).not.toContain('.books-container')
    expect(source).toContain('class="bookshelf-toolbar__title"')
    expect(source).toContain('class="bookshelf-main__books"')
    expect(source).toContain('class="bookshelf-toolbar__actions"')
  })

  it('renders bookshelf empty states through the product empty-state component', () => {
    const wrapper = mountView()

    const emptyStates = wrapper.findAllComponents(ProductEmptyState)
    expect(emptyStates.map(state => state.props('iconName'))).toEqual(['book-open'])
    expect(emptyStates[0].props()).toMatchObject({
      title: '书架空空如也',
      description: '点击"新建书籍"开始你的翻译之旅',
    })
    expect(wrapper.find('.ui-empty-state').exists()).toBe(false)
  })

  it('delegates repeated book-card layout to the product card grid', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/BookshelfView.vue'), 'utf8')

    expect(source).toContain("import ProductCardGrid from '@/components/product/ProductCardGrid.vue'")
    expect(source).toContain('<ProductCardGrid')
    expect(source).not.toContain('class="books-grid"')
    expect(source).not.toContain('grid-template-columns: repeat(auto-fill, minmax(160px, 1fr))')
  })
})
