import { mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { existsSync, readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import BookSearch from '@/components/bookshelf/BookSearch.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import ProductSearchField from '@/components/product/ProductSearchField.vue'
import UiButton from '@/components/ui/UiButton.vue'
import type { TagData } from '@/types'

const tags: TagData[] = [
  { id: 'tag-action', name: 'Action', color: '#4466aa' },
  { id: 'tag-drama', name: 'Drama', color: '#aa6644' },
]

function mountSearch(options: {
  listeners?: { onSearch?: (query: string) => void }
  query?: string
  selectedTagNames?: string[]
} = {}) {
  return mount(BookSearch, {
    props: {
      tags,
      query: options.query ?? '',
      selectedTagNames: options.selectedTagNames ?? [],
    },
    attrs: options.listeners,
  })
}

describe('BookSearch', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('clears pending debounced search when unmounted', async () => {
    const onSearch = vi.fn()
    const wrapper = mountSearch({ listeners: { onSearch } })

    await wrapper.get('input').setValue('demo')
    expect(vi.getTimerCount()).toBe(1)

    wrapper.unmount()

    expect(vi.getTimerCount()).toBe(0)
    expect(onSearch).not.toHaveBeenCalled()
  })

  it('clears pending debounced search when the user submits immediately', async () => {
    const onSearch = vi.fn()
    const wrapper = mountSearch({ listeners: { onSearch } })

    await wrapper.get('input').setValue('demo')
    await wrapper.get('.book-search__submit-action').trigger('click')

    expect(onSearch).toHaveBeenCalledTimes(1)
    expect(onSearch).toHaveBeenCalledWith('demo')
    expect(vi.getTimerCount()).toBe(0)

    vi.advanceTimersByTime(300)

    expect(onSearch).toHaveBeenCalledTimes(1)
  })

  it('keeps the input synchronized with the active store query', async () => {
    const wrapper = mountSearch({ query: 'persisted query' })

    expect((wrapper.get('input').element as HTMLInputElement).value).toBe('persisted query')

    await wrapper.get('input').setValue('pending query')
    expect(vi.getTimerCount()).toBe(1)

    await wrapper.setProps({ query: 'restored query' })

    expect((wrapper.get('input').element as HTMLInputElement).value).toBe('restored query')
    expect(vi.getTimerCount()).toBe(0)
  })

  it('renders tag filters through the shared product chip contract', async () => {
    const wrapper = mountSearch({ selectedTagNames: ['Drama'] })

    const chipList = wrapper.getComponent(ProductChipList)

    expect(chipList.props('ariaLabel')).toBe('标签筛选')
    expect(chipList.props('items')).toEqual([
      {
        id: 'Action',
        label: 'Action',
        ariaLabel: '筛选标签 Action',
        interactive: true,
        selected: false,
        tone: 'neutral',
      },
      {
        id: 'Drama',
        label: 'Drama',
        ariaLabel: '取消筛选标签 Drama',
        interactive: true,
        selected: true,
        tone: 'custom',
        backgroundColor: '#aa6644',
        borderColor: '#aa6644',
        textColor: 'var(--color-text-inverse)',
      },
    ])

    chipList.vm.$emit('select', 'Action')
    expect(wrapper.emitted('filterTag')?.[0]).toEqual(['Action'])
    expect(wrapper.find('.tag-chip').exists()).toBe(false)
  })

  it('uses product icon button variants for search actions', async () => {
    const wrapper = mountSearch()

    const searchField = wrapper.getComponent(ProductSearchField)
    expect(searchField.props()).toMatchObject({
      ariaLabel: '搜索书籍',
      placeholder: '搜索书籍名称或标签...',
    })
    const submitAction = wrapper.get('.book-search__submit-action').getComponent(UiButton)
    expect(submitAction.props('variant')).toBe('primary')

    await searchField.get('input').setValue('demo')
    await searchField.get('button[aria-label="清除搜索"]').trigger('click')

    expect(wrapper.emitted('search')?.at(-1)).toEqual([''])
    expect(wrapper.find('.clear-search-btn').exists()).toBe(false)
  })

  it('uses the product search toolbar shell instead of a local filter card', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/bookshelf/BookSearch.vue'), 'utf8')

    expect(existsSync(resolve(process.cwd(), 'src/components/product/ProductSearchToolbar.vue'))).toBe(true)
    expect(source).toContain('@/components/product/ProductSearchToolbar.vue')
    expect(source).not.toContain('class="filter-bar"')
    expect(source).not.toContain('class="search-box"')
  })

  it('uses typed search-field model updates for debounced searching', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/bookshelf/BookSearch.vue'), 'utf8')

    expect(source).not.toContain('v-model="searchQuery"')
    expect(source).not.toContain('@input="handleInput"')
    expect(source).toContain(':model-value="searchQuery"')
    expect(source).toContain('@update:model-value="handleSearchQueryUpdate"')
  })

  it('uses an owner action hook for the submit button', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/bookshelf/BookSearch.vue'), 'utf8')

    expect(source).toContain('book-search__submit-action')
    expect(source).not.toContain('search-btn')
  })

  it('keeps search presentation hooks under the book-search owner', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/bookshelf/BookSearch.vue'), 'utf8')

    for (const oldClass of [
      'book-search-field',
      'book-search-submit-action',
      'book-search-tags',
    ]) {
      expect(source).not.toMatch(new RegExp(`class="[^"]*\\b${oldClass}\\b`))
      expect(source).not.toMatch(new RegExp(`\\.${oldClass}\\b`))
    }

    for (const ownerClass of [
      'book-search',
      'book-search__field',
      'book-search__submit-action',
      'book-search__tags',
    ]) {
      expect(source).toContain(ownerClass)
    }
  })
})
