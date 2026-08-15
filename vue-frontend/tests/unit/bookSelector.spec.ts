import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'

const { getBooksMock } = vi.hoisted(() => ({
  getBooksMock: vi.fn(),
}))

vi.mock('@/api/bookshelf', () => ({
  getBooks: getBooksMock,
}))

import BookSelector from '@/components/insight/BookSelector.vue'
import ProductBookSelector from '@/components/product/ProductBookSelector.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiCombobox from '@/components/ui/UiCombobox.vue'

enableAutoUnmount(afterEach)

function deferred<T>() {
  let resolve: (value: T) => void = () => {}
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise
  })
  return { promise, resolve }
}

describe('BookSelector', () => {
  beforeEach(() => {
    getBooksMock.mockReset()
    getBooksMock.mockResolvedValue([])
  })

  it('loads the complete backend book list for the product selector', async () => {
    const books = [
      { id: 'book-1', title: '第一本书' },
      { id: 'book-2', title: '第二本书' },
    ]
    getBooksMock.mockResolvedValueOnce(books)

    const wrapper = mount(BookSelector)
    await flushPromises()
    const selector = wrapper.getComponent(ProductBookSelector)

    expect(getBooksMock).toHaveBeenCalledWith()
    expect(selector.props('books')).toEqual(books)

    selector.vm.$emit('select', 'book-2')

    expect(wrapper.emitted('select')).toEqual([['book-2']])
  })

  it('shows list failures with a working retry action', async () => {
    getBooksMock.mockRejectedValueOnce(new Error('书籍接口不可用'))
    getBooksMock.mockResolvedValueOnce([{ id: 'book-1', title: '第一本书' }])

    const wrapper = mount(BookSelector)
    await flushPromises()

    expect(wrapper.getComponent(ProductStatusBanner).props('title')).toBe('书籍列表加载失败')
    expect(wrapper.text()).toContain('书籍接口不可用')
    await wrapper.get('button').trigger('click')
    await flushPromises()
    expect(getBooksMock).toHaveBeenCalledTimes(2)
    expect(wrapper.getComponent(ProductBookSelector).props('books')).toHaveLength(1)
  })

  it('distinguishes loading and an actually empty bookshelf', async () => {
    const load = deferred<[]>()
    getBooksMock.mockReturnValueOnce(load.promise)
    const wrapper = mount(BookSelector)
    await wrapper.vm.$nextTick()

    expect(wrapper.getComponent(ProductStatusBanner).props('title')).toBe('正在加载书籍')
    load.resolve([])
    await flushPromises()
    expect(wrapper.getComponent(ProductStatusBanner).props('title')).toBe('书架中暂无书籍')
  })

  it('ignores a completed list request after the selector is unmounted', async () => {
    const load = deferred<Array<{ id: string; title: string }>>()
    getBooksMock.mockReturnValueOnce(load.promise)
    const wrapper = mount(BookSelector)
    wrapper.unmount()

    load.resolve([{ id: 'book-1', title: '第一本书' }])
    await flushPromises()

    expect(getBooksMock).toHaveBeenCalledTimes(1)
  })

  it('rejects non-string selector values instead of coercing them into book IDs', () => {
    const wrapper = mount(ProductBookSelector, {
      props: {
        modelValue: '',
        books: [{ id: 'book-1', title: '第一本书' }],
      },
    })

    wrapper.getComponent(UiCombobox).vm.$emit('change', 1)

    expect(wrapper.emitted('update:modelValue')).toBeUndefined()
    expect(wrapper.emitted('select')).toBeUndefined()
  })
})
