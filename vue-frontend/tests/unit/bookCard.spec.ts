import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import BookCard from '@/components/bookshelf/BookCard.vue'
import type { BookData } from '@/types'

function book(overrides: Partial<BookData> = {}): BookData {
  return {
    id: 'book-1',
    title: 'Demo Book',
    cover: 'broken-cover.png',
    tags: [],
    chapters: [],
    chapter_count: 0,
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:00Z',
    ...overrides,
  }
}

describe('BookCard', () => {
  it('uses a native button for opening the book card', async () => {
    const wrapper = mount(BookCard, {
      props: { book: book({ title: 'Saber' }) },
    })

    const card = wrapper.find('.book-card')
    expect(card.element.tagName).toBe('BUTTON')
    expect(card.attributes('aria-label')).toBe('打开书籍：Saber')

    await card.trigger('click')
    expect(wrapper.emitted('click')).toHaveLength(1)
  })

  it('uses Vue state to show one cover placeholder after image load fails', async () => {
    const wrapper = mount(BookCard, {
      props: { book: book() },
    })

    await wrapper.get('img').trigger('error')
    await wrapper.vm.$nextTick()

    expect(wrapper.find('img').exists()).toBe(false)
    expect(wrapper.findAll('.book-cover-placeholder')).toHaveLength(1)
    expect(wrapper.find('.book-cover-placeholder').text()).toBe('📖')
  })

  it('receives tag colors as explicit page-owned data', () => {
    const wrapper = mount(BookCard, {
      props: {
        book: book({ tags: ['Drama'] }),
        tags: [{ name: 'Drama', color: '#aa6644' }],
      },
    })

    expect(wrapper.get('.book-tag').attributes('style')).toContain('background: rgb(170, 102, 68);')
  })
})
