import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { defineComponent, nextTick } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import BookshelfView from '@/views/BookshelfView.vue'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import { useRuntimeStore } from '@/stores/runtimeStore'
import { useSettingsStore } from '@/stores/settings'
import ProductEmptyState from '@/components/product/ProductEmptyState.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductCardGrid from '@/components/product/ProductCardGrid.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiIcon from '@/components/ui/UiIcon.vue'

const { getBooksMock, getTagsMock, getBookDetailMock, getServerInfoMock, routerPushMock } = vi.hoisted(() => ({
  getBooksMock: vi.fn(),
  getTagsMock: vi.fn(),
  getBookDetailMock: vi.fn(),
  getServerInfoMock: vi.fn(),
  routerPushMock: vi.fn(),
}))

vi.mock('vue-router', () => ({
  useRoute: () => ({ query: {} }),
  useRouter: () => ({ push: routerPushMock }),
}))

vi.mock('@/api/v2/system', () => ({
  getV2ServerInfo: getServerInfoMock,
}))

vi.mock('@/api/bookshelf', () => ({
  getBooks: getBooksMock,
  getTags: getTagsMock,
  getBookDetail: getBookDetailMock,
}))

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise
  })
  return { promise, resolve }
}

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

const BookCardStub = defineComponent({
  props: {
    book: {
      type: Object,
      required: true,
    },
  },
  emits: ['click'],
  template: '<button class="book-card-stub" @click="$emit(\'click\')">{{ book.title }}</button>',
})

function mountView(profile: 'local' | 'public' = 'local') {
  setActivePinia(createPinia())
  const runtimeStore = useRuntimeStore()
  if (profile === 'local') {
    runtimeStore.assumeLocalForTests()
  } else {
    runtimeStore.capabilities = {
      profile: 'public',
      requiresAuth: true,
      browserCredentials: true,
      registrationRequiresInvite: true,
      publicUserPolicy: {
        features: {
          translation: true,
          insight: true,
          characterStudio: true,
          editMode: true,
        },
        models: {
          detector_default: true,
          detector_ctd: true,
          detector_yolo: true,
          aux_ysg_yolo: true,
          saber_yolo: true,
          manga_ocr: true,
          ocr_48px: true,
          paddle_ocr: true,
          paddleocr_vl: true,
          lama_mpe: true,
          litelama: true,
        },
        settings: {
          lamaDisableResize: { editable: false, value: false },
          parallel: { allowed: false, maxDeepLearningConcurrency: 1 },
        },
      },
      features: { plugins: false, webImport: false, localProviders: false },
    }
  }
  return mount(BookshelfView, {
    global: {
      stubs: {
        AppShell: AppShellStub,
        ProductPageHeader: ProductPageHeaderStub,
        ProductCardGrid,
        BookSearch: true,
        BookCard: BookCardStub,
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
    getBookDetailMock.mockReset()
    getServerInfoMock.mockReset()
    getBooksMock.mockResolvedValue([])
    getTagsMock.mockResolvedValue([])
    getServerInfoMock.mockResolvedValue({ lanUrl: 'http://localhost:5173' })
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

  it('hides local network metadata in the public profile', async () => {
    const wrapper = mountView('public')
    await flushPromises()

    expect(wrapper.find('[aria-label="复制局域网地址"]').exists()).toBe(false)
    expect(wrapper.text()).not.toContain('局域网访问')
    expect(getServerInfoMock).not.toHaveBeenCalled()
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
      'check',
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
    expect(wrapper.get('.product-empty-state__icon-text').text()).toBe('📚')
    expect(emptyStates[0].props()).toMatchObject({
      title: '书架空空如也',
      description: '点击"新建书籍"开始你的翻译之旅',
    })
    expect(wrapper.find('.ui-empty-state').exists()).toBe(false)
  })

  it('distinguishes a load failure from an empty bookshelf', async () => {
    getBooksMock.mockRejectedValueOnce(new Error('database unavailable'))
    const wrapper = mountView()
    await flushPromises()

    const status = wrapper.getComponent(ProductStatusBanner)
    expect(status.props()).toMatchObject({
      role: 'alert',
      title: '书架加载失败',
      tone: 'danger',
    })
    expect(status.text()).toContain('database unavailable')
    expect(wrapper.findComponent(ProductEmptyState).exists()).toBe(false)
  })

  it('reports tag loading failures separately and retries only the tag request', async () => {
    const wrapper = mountView()
    await flushPromises()
    const store = useBookshelfStore()
    store.tagsError = 'tag service unavailable'
    store.loadTags = vi.fn().mockResolvedValue(undefined)
    await nextTick()

    const status = wrapper.findAllComponents(ProductStatusBanner)
      .find(banner => banner.props('title') === '标签加载失败')!
    expect(status.props()).toMatchObject({
      role: 'alert',
      tone: 'warning',
    })
    expect(status.text()).toContain('tag service unavailable')

    await status.get('button').trigger('click')
    expect(store.loadTags).toHaveBeenCalledOnce()
  })

  it('delegates repeated book-card layout to the product card grid', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/BookshelfView.vue'), 'utf8')

    expect(source).toContain("import ProductCardGrid from '@/components/product/ProductCardGrid.vue'")
    expect(source).toContain('<ProductCardGrid')
    expect(source).not.toContain('class="books-grid"')
    expect(source).not.toContain('grid-template-columns: repeat(auto-fill, minmax(160px, 1fr))')
  })

  it('disables batch translation when selected books contain no chapters', async () => {
    const wrapper = mountView()
    await flushPromises()
    const store = useBookshelfStore()
    store.books = [{ id: 'empty-book', title: 'Empty', chapterCount: 0 }]
    store.enterBatchMode()
    store.toggleBookSelection('empty-book')
    await nextTick()

    const translateButton = wrapper.findAll('button')
      .find(button => button.text().includes('翻译全部章节'))
    expect(translateButton?.attributes('disabled')).toBeDefined()

    store.books = [{ id: 'empty-book', title: 'Empty', chapterCount: 1 }]
    await nextTick()
    expect(translateButton?.attributes('disabled')).toBeUndefined()
  })

  it('only opens the latest book when detail requests finish out of order', async () => {
    getBooksMock.mockResolvedValue([
      { id: 'book-1', title: 'First' },
      { id: 'book-2', title: 'Second' },
    ])
    const first = deferred<{ id: string; title: string; chapters: [] }>()
    const second = deferred<{ id: string; title: string; chapters: [] }>()
    getBookDetailMock
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise)
    const wrapper = mountView()
    await flushPromises()

    const cards = wrapper.findAll('.book-card-stub')
    await cards[0]!.trigger('click')
    await cards[1]!.trigger('click')
    second.resolve({ id: 'book-2', title: 'Second', chapters: [] })
    await flushPromises()
    expect(useBookshelfStore().currentBookId).toBe('book-2')

    first.resolve({ id: 'book-1', title: 'First', chapters: [] })
    await flushPromises()

    expect(useBookshelfStore().currentBookId).toBe('book-2')
    expect(getBookDetailMock).toHaveBeenNthCalledWith(1, 'book-1')
    expect(getBookDetailMock).toHaveBeenNthCalledWith(2, 'book-2')
  })
})
