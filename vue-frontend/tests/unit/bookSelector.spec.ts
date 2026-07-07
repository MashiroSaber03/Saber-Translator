import { beforeEach, describe, expect, it } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { mount } from '@vue/test-utils'
import BookSelector from '@/components/insight/BookSelector.vue'
import ProductBookSelector from '@/components/product/ProductBookSelector.vue'
import { useBookshelfStore } from '@/stores/bookshelfStore'

describe('BookSelector', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('adapts bookshelf store books to the product selector', () => {
    const bookshelfStore = useBookshelfStore()
    bookshelfStore.books = [
      { id: 'book-1', title: '第一本书' },
      { id: 'book-2', title: '第二本书' },
    ] as typeof bookshelfStore.books

    const wrapper = mount(BookSelector)
    const selector = wrapper.getComponent(ProductBookSelector)

    expect(selector.props('books')).toEqual(bookshelfStore.books)

    selector.vm.$emit('select', 'book-2')

    expect(wrapper.emitted('select')).toEqual([['book-2']])
  })
})
