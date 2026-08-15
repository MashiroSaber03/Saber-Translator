import { flushPromises, mount } from '@vue/test-utils'
import { defineComponent, nextTick } from 'vue'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ReaderView from '@/views/ReaderView.vue'
import type { V2BookDetail, V2PageSummary } from '@/api/v2/content'

const { routerPushMock, getBookMock, listChapterPagesMock, toastErrorMock } = vi.hoisted(() => ({
  routerPushMock: vi.fn(),
  getBookMock: vi.fn(),
  listChapterPagesMock: vi.fn(),
  toastErrorMock: vi.fn(),
}))

vi.mock('vue-router', () => ({
  useRouter: () => ({ push: routerPushMock }),
}))

vi.mock('@/api/v2/content', () => ({
  getBook: getBookMock,
  listChapterPages: listChapterPagesMock,
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
    backgroundColor: String,
    imageGap: Number,
    imageWidth: Number,
    images: {
      type: Array,
      default: () => [],
    },
  },
  template:
    '<div class="reader-canvas-stub">{{ images.map(image => image.sourceUrl).join(",") }}</div>',
})

const ReaderControlsContractStub = defineComponent({
  inheritAttrs: false,
  emits: ['settingsChange'],
  template: '<div class="reader-controls-contract-stub" />',
})

function pageSummary(overrides: Partial<V2PageSummary> = {}): V2PageSummary {
  return {
    id: 'page-1',
    chapterId: 'chapter-1',
    ordinal: 1,
    logicalSourcePath: '1.png',
    sourceRevision: 1,
    documentRevision: 1,
    renderedRevision: null,
    renderStatus: 'not_rendered',
    detectionState: 'unprocessed',
    sourceUrl: '/source/1',
    thumbnailSourceUrl: '/thumb/1',
    cleanUrl: null,
    translatedUrl: null,
    width: 800,
    height: 1200,
    ...overrides,
  }
}

function bookDetail(overrides: Partial<V2BookDetail> = {}): V2BookDetail {
  return {
    id: 'book-1',
    title: 'Book',
    chapterOrderRevision: 1,
    chapters: [
      {
        id: 'chapter-1',
        title: 'Chapter',
        ordinal: 1,
        pageOrderRevision: 1,
      },
    ],
    tags: [],
    ...overrides,
  }
}

describe('ReaderView', () => {
  beforeEach(() => {
    routerPushMock.mockReset()
    toastErrorMock.mockReset()
    getBookMock.mockReset().mockResolvedValue(bookDetail())
    listChapterPagesMock.mockReset().mockResolvedValue({
      items: [],
      nextCursor: null,
      pageOrderRevision: 1,
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

  it('shows the persisted translated-page count from chapter metadata', async () => {
    listChapterPagesMock.mockResolvedValueOnce({
      items: [
        pageSummary({ translatedUrl: '/translated/1' }),
        pageSummary({
          id: 'page-2',
          ordinal: 2,
          logicalSourcePath: '2.png',
          sourceUrl: '/source/2',
          thumbnailSourceUrl: '/thumb/2',
        }),
        pageSummary({
          id: 'page-3',
          ordinal: 3,
          logicalSourcePath: '3.png',
          sourceUrl: '/source/3',
          thumbnailSourceUrl: '/thumb/3',
          translatedUrl: '/translated/3',
        }),
      ],
      nextCursor: null,
      pageOrderRevision: 1,
    })
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

    expect(wrapper.get('.reader-header__translated-count').text()).toBe('已翻译 2/3')
  })

  it('navigates to the adjacent chapter from the current book snapshot', async () => {
    getBookMock.mockResolvedValueOnce(
      bookDetail({
        chapters: [
          { id: 'chapter-1', title: 'Chapter 1', ordinal: 1, pageOrderRevision: 1 },
          { id: 'chapter-2', title: 'Chapter 2', ordinal: 2, pageOrderRevision: 1 },
        ],
      })
    )
    listChapterPagesMock.mockResolvedValueOnce({
      items: [pageSummary()],
      nextCursor: null,
      pageOrderRevision: 1,
    })
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
        },
      },
    })
    await flushPromises()

    const nextChapter = wrapper.findAll('button').find(button => button.text().includes('下一章'))
    expect(nextChapter).toBeTruthy()
    await nextChapter!.trigger('click')

    expect(routerPushMock).toHaveBeenCalledWith('/reader?book=book-1&chapter=chapter-2')
    wrapper.unmount()
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
    getBookMock.mockRejectedValueOnce(new Error('network down'))

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
    const book = bookDetail({
      chapters: [
        { id: 'chapter-1', title: 'Chapter 1', ordinal: 1, pageOrderRevision: 1 },
        { id: 'chapter-2', title: 'Chapter 2', ordinal: 2, pageOrderRevision: 1 },
      ],
    })

    getBookMock.mockReturnValueOnce(firstBook.promise).mockReturnValueOnce(secondBook.promise)
    listChapterPagesMock
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

    secondBook.resolve(book)
    secondImages.resolve({
      items: [
        pageSummary({
          id: 'page-2',
          chapterId: 'chapter-2',
          ordinal: 1,
          logicalSourcePath: '2.png',
          sourceUrl: 'chapter-2-page',
          thumbnailSourceUrl: 'chapter-2-thumb',
          translatedUrl: 'chapter-2-translated',
        }),
      ],
      nextCursor: null,
      pageOrderRevision: 1,
    })
    await flushPromises()

    expect(wrapper.text()).toContain('chapter-2-page')

    firstBook.resolve(book)
    firstImages.resolve({
      items: [
        pageSummary({
          sourceUrl: 'chapter-1-page',
          thumbnailSourceUrl: 'chapter-1-thumb',
          translatedUrl: 'chapter-1-translated',
        }),
      ],
      nextCursor: null,
      pageOrderRevision: 1,
    })
    await flushPromises()

    expect(wrapper.text()).toContain('chapter-2-page')
    expect(wrapper.text()).not.toContain('chapter-1-page')
  })

  it('rejects a chapter that does not belong to the routed book', async () => {
    getBookMock.mockResolvedValueOnce(
      bookDetail({
        chapters: [
          {
            id: 'chapter-other',
            title: 'Other chapter',
            ordinal: 1,
            pageOrderRevision: 1,
          },
        ],
      })
    )
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

    await flushPromises()

    expect(toastErrorMock).toHaveBeenCalledWith('加载失败: 章节不属于当前书籍')
    expect(wrapper.getComponent(ReaderCanvasStub).props('images')).toEqual([])
    wrapper.unmount()
  })

  it('rejects a partial response from the all-pages reader request', async () => {
    listChapterPagesMock.mockResolvedValueOnce({
      items: [pageSummary()],
      nextCursor: 1,
      pageOrderRevision: 1,
    })
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

    await flushPromises()

    expect(toastErrorMock).toHaveBeenCalledWith('加载失败: 章节页面列表不完整')
    expect(wrapper.getComponent(ReaderCanvasStub).props('images')).toEqual([])
    wrapper.unmount()
  })

  it('rejects pages that do not belong to the routed chapter', async () => {
    listChapterPagesMock.mockResolvedValueOnce({
      items: [pageSummary({ chapterId: 'chapter-other' })],
      nextCursor: null,
      pageOrderRevision: 1,
    })
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

    await flushPromises()

    expect(toastErrorMock).toHaveBeenCalledWith('加载失败: 章节页面归属不一致')
    expect(wrapper.getComponent(ReaderCanvasStub).props('images')).toEqual([])
    wrapper.unmount()
  })

  it('does not redisplay the previous chapter after the next load fails', async () => {
    listChapterPagesMock.mockResolvedValueOnce({
      items: [pageSummary({ sourceUrl: 'chapter-1-page' })],
      nextCursor: null,
      pageOrderRevision: 1,
    })
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
    await flushPromises()
    expect(wrapper.text()).toContain('chapter-1-page')

    getBookMock.mockRejectedValueOnce(new Error('next chapter failed'))
    listChapterPagesMock.mockRejectedValueOnce(new Error('next chapter failed'))
    await wrapper.setProps({ chapterId: 'chapter-2' })
    await flushPromises()

    expect(wrapper.text()).not.toContain('chapter-1-page')
    expect(toastErrorMock).toHaveBeenCalledWith('加载失败: next chapter failed')
    wrapper.unmount()
  })

  it('forwards published reader settings to the virtual canvas', async () => {
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
          ReaderControls: ReaderControlsContractStub,
        },
      },
    })

    const controls = wrapper.getComponent(ReaderControlsContractStub)
    controls.vm.$emit('settingsChange', {
      imageWidth: 72,
      imageGap: 24,
      bgColor: '#ffffff',
    })
    await nextTick()

    expect(wrapper.getComponent(ReaderCanvasStub).props()).toMatchObject({
      imageWidth: 72,
      imageGap: 24,
      backgroundColor: '#ffffff',
    })
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
