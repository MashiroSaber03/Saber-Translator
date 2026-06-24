import { mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import BookSearch from '@/components/bookshelf/BookSearch.vue'
import type { TagData } from '@/types'

const tags: TagData[] = [
  { name: 'Action', color: '#4466aa' },
  { name: 'Drama', color: '#aa6644' },
]

function mountSearch(options: {
  listeners?: { onSearch?: (query: string) => void }
  selectedTagNames?: string[]
} = {}) {
  return mount(BookSearch, {
    props: {
      tags,
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

  it('renders tag filters as pressed buttons', async () => {
    const wrapper = mountSearch({ selectedTagNames: ['Drama'] })

    const tagButtons = wrapper.findAll('.tag-chip')

    expect(tagButtons.map(button => button.element.tagName)).toEqual(['BUTTON', 'BUTTON'])
    expect(tagButtons[0].attributes('aria-pressed')).toBe('false')
    expect(tagButtons[1].attributes('aria-pressed')).toBe('true')
  })
})
