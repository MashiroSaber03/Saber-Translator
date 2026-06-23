import { flushPromises, mount } from '@vue/test-utils'
import { defineComponent } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ReaderView from '@/views/ReaderView.vue'

const { routerPushMock, getBookDetailMock, getChapterImagesMock } = vi.hoisted(() => ({
  routerPushMock: vi.fn(),
  getBookDetailMock: vi.fn(),
  getChapterImagesMock: vi.fn(),
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
    error: vi.fn(),
  }),
}))

const AppShellStub = defineComponent({
  template: '<section><slot name="header" /><slot /></section>',
})

describe('ReaderView', () => {
  beforeEach(() => {
    routerPushMock.mockReset()
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
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    getBookDetailMock.mockRejectedValueOnce(new Error('network down'))

    try {
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
      expect(consoleError).toHaveBeenCalled()
    } finally {
      consoleError.mockRestore()
    }
  })
})
