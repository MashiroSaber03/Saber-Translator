import { flushPromises, mount } from '@vue/test-utils'
import { defineComponent } from 'vue'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ReaderView from '@/views/ReaderView.vue'

const { routerPushMock, getBookDetailMock, getChapterImagesMock, toastErrorMock } = vi.hoisted(() => ({
  routerPushMock: vi.fn(),
  getBookDetailMock: vi.fn(),
  getChapterImagesMock: vi.fn(),
  toastErrorMock: vi.fn(),
}))

vi.mock('vue-router', () => ({
  useRouter: () => ({ push: routerPushMock }),
}))

vi.mock('@/api/bookshelf', () => ({
  getBookDetail: getBookDetailMock,
  getChapterImages: getChapterImagesMock,
}))

vi.mock('@/utils/toast', () => ({
  useToast: () => ({
    error: toastErrorMock,
  }),
}))

const AppShellStub = defineComponent({
  template: '<section><slot name="header" /><slot /></section>',
})

const ProductPageHeaderStub = defineComponent({
  props: {
    variant: {
      type: String,
      default: 'default',
    },
  },
  template: `
    <header class="product-page-header" :class="'product-page-header--' + variant">
      <slot name="brand" />
      <slot name="meta" />
      <slot name="nav" />
      <slot name="actions" />
    </header>
  `,
})

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

const ReaderCanvasStub = defineComponent({
  props: {
    images: {
      type: Array,
      default: () => [],
    },
  },
  template: '<div class="reader-canvas-stub">{{ images.map(image => image.original).join(",") }}</div>',
})

const ReaderControlsContractStub = defineComponent({
  inheritAttrs: false,
  template: '<div class="reader-controls-contract-stub" />',
})

describe('ReaderView', () => {
  beforeEach(() => {
    vi.spyOn(window, 'scrollTo').mockImplementation(() => undefined)
    routerPushMock.mockReset()
    toastErrorMock.mockReset()
    getBookDetailMock.mockReset().mockResolvedValue({
      success: true,
      book: {
        id: 'book-1',
        title: 'Book',
        chapters: [{ id: 'chapter-1', title: 'Chapter', startPage: 1, endPage: 1 }],
      },
    })
    getChapterImagesMock.mockReset().mockResolvedValue({
      success: true,
      images: [],
    })
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('names the icon-only reader settings action', () => {
    const wrapper = mount(ReaderView, {
      props: {
        bookId: 'book-1',
        chapterId: 'chapter-1',
      },
      global: {
        stubs: {
          AppShell: AppShellStub,
          ProductPageHeader: ProductPageHeaderStub,
          ReaderCanvas: true,
          ReaderControls: true,
        },
      },
    })

    expect(wrapper.get('.product-page-header--reader').exists()).toBe(true)
    expect(wrapper.get('[aria-label="阅读设置"]').exists()).toBe(true)
  })

  it('keeps the reader header free of legacy DOM id hooks', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/ReaderView.vue'), 'utf8')

    for (const id of [
      'backBtn',
      'bookTitle',
      'chapterTitle',
      'pageInfo',
      'viewOriginalBtn',
      'viewTranslatedBtn',
      'settingsBtn',
      'translateBtn',
    ]) {
      expect(source).not.toContain(`id="${id}"`)
    }
  })

  it('cancels the delayed failure redirect when the view unmounts', async () => {
    vi.useFakeTimers()
    getBookDetailMock.mockRejectedValueOnce(new Error('network down'))

    const wrapper = mount(ReaderView, {
      props: {
        bookId: 'book-1',
        chapterId: 'chapter-1',
      },
      global: {
        stubs: {
          AppShell: AppShellStub,
          ProductPageHeader: ProductPageHeaderStub,
          ReaderCanvas: true,
          ReaderControls: true,
        },
      },
    })

    await flushPromises()
    wrapper.unmount()

    await vi.advanceTimersByTimeAsync(2000)

    expect(routerPushMock).not.toHaveBeenCalledWith('/')
    expect(toastErrorMock).toHaveBeenCalledWith('加载失败: network down')
  })

  it('keeps newer chapter images when an older load resolves later', async () => {
    const firstBook = deferred<unknown>()
    const firstImages = deferred<unknown>()
    const secondBook = deferred<unknown>()
    const secondImages = deferred<unknown>()
    const book = {
      id: 'book-1',
      title: 'Book',
      chapters: [
        { id: 'chapter-1', title: 'Chapter 1', startPage: 1, endPage: 1 },
        { id: 'chapter-2', title: 'Chapter 2', startPage: 2, endPage: 2 },
      ],
    }

    getBookDetailMock
      .mockReturnValueOnce(firstBook.promise)
      .mockReturnValueOnce(secondBook.promise)
    getChapterImagesMock
      .mockReturnValueOnce(firstImages.promise)
      .mockReturnValueOnce(secondImages.promise)

    const wrapper = mount(ReaderView, {
      props: {
        bookId: 'book-1',
        chapterId: 'chapter-1',
      },
      global: {
        stubs: {
          AppShell: AppShellStub,
          ProductPageHeader: ProductPageHeaderStub,
          ReaderCanvas: ReaderCanvasStub,
          ReaderControls: true,
        },
      },
    })

    await wrapper.setProps({ chapterId: 'chapter-2' })

    secondBook.resolve({ success: true, book })
    secondImages.resolve({
      success: true,
      images: [{ page_num: 2, original: 'chapter-2-page', translated: 'chapter-2-translated' }],
    })
    await flushPromises()

    expect(wrapper.text()).toContain('chapter-2-page')

    firstBook.resolve({ success: true, book })
    firstImages.resolve({
      success: true,
      images: [{ page_num: 1, original: 'chapter-1-page', translated: 'chapter-1-translated' }],
    })
    await flushPromises()

    expect(wrapper.text()).toContain('chapter-2-page')
    expect(wrapper.text()).not.toContain('chapter-1-page')
  })

  it('does not wire no-op reader settings events back to the page owner', () => {
    const wrapper = mount(ReaderView, {
      props: {
        bookId: 'book-1',
        chapterId: 'chapter-1',
      },
      global: {
        stubs: {
          AppShell: AppShellStub,
          ProductPageHeader: ProductPageHeaderStub,
          ReaderCanvas: true,
          ReaderControls: ReaderControlsContractStub,
        },
      },
    })

    const controls = wrapper.getComponent(ReaderControlsContractStub)
    expect(controls.vm.$attrs).not.toHaveProperty('onSettingsChange')
  })

  it('keeps page-count state out of the ReaderControls contract', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/ReaderView.vue'), 'utf8')

    expect(source).not.toContain(':current-page=')
    expect(source).not.toContain(':total-pages=')
  })

  it('uses ProductHeaderAction public props for responsive labels', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/ReaderView.vue'), 'utf8')

    expect(source).not.toContain('.product-header-action__label')
    expect(source).toContain('collapse-label-on-mobile')
  })

  it('exposes pressed state for original and translated mode header actions', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/ReaderView.vue'), 'utf8')

    expect(source).toContain(':active="currentViewMode === \'original\'"')
    expect(source).toContain(':pressed="currentViewMode === \'original\'"')
    expect(source).toContain(':active="currentViewMode === \'translated\'"')
    expect(source).toContain(':pressed="currentViewMode === \'translated\'"')
  })

  it('keeps reader header helper hooks under the reader-header owner', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/views/ReaderView.vue'), 'utf8')

    for (const currentHook of [
      'reader-header__book-info',
      'reader-header__separator',
      'reader-header__chapter-title',
      'reader-header__page-info',
      'reader-header__view-mode-toggle',
    ]) {
      expect(source).toContain(currentHook)
    }

    for (const oldHook of [
      'book-info',
      'separator',
      'chapter-title',
      'page-info',
      'view-mode-toggle',
    ]) {
      expect(source).not.toMatch(new RegExp(`class="[^"]*\\b${oldHook}\\b`))
      expect(source).not.toMatch(new RegExp(`\\.${oldHook}\\b`))
    }
  })
})
