import { flushPromises, mount } from '@vue/test-utils'
import { defineComponent } from 'vue'
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
          ReaderCanvas: true,
          ReaderControls: true,
        },
      },
    })

    expect(wrapper.get('#settingsBtn').attributes('aria-label')).toBe('阅读设置')
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
})
