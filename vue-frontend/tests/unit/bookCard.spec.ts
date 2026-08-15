import { beforeEach, describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import BookCard from '@/components/bookshelf/BookCard.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import type { BookData } from '@/types'

function book(overrides: Partial<BookData> = {}): BookData {
  return {
    id: 'book-1',
    title: 'Demo Book',
    cover: 'broken-cover.png',
    tags: [],
    chapters: [],
    chapterCount: 0,
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:00Z',
    ...overrides,
  }
}

describe('BookCard', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

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

  it('uses the product record-card button contract instead of reskinning UiButton as a card', () => {
    const wrapper = mount(BookCard, {
      props: { book: book({ title: 'Saber' }) },
    })
    const source = readFileSync(resolve(process.cwd(), 'src/components/bookshelf/BookCard.vue'), 'utf8')

    const recordCard = wrapper.getComponent(ProductRecordCard)
    expect(recordCard.props('as')).toBe('button')
    expect(recordCard.props('ariaLabel')).toBe('打开书籍：Saber')
    expect(source).not.toContain('variant="toolbar"')
    expect(source).not.toContain('UiButton')
  })

  it('uses Vue state to show one cover placeholder after image load fails', async () => {
    const wrapper = mount(BookCard, {
      props: { book: book() },
    })

    await wrapper.get('img').trigger('error')
    await wrapper.vm.$nextTick()

    expect(wrapper.find('img').exists()).toBe(false)
    expect(wrapper.findAll('.book-card__cover-placeholder')).toHaveLength(1)
    expect(wrapper.find('.book-card__cover-placeholder').text()).toBe('📖')
    expect(wrapper.find('.book-card__cover-placeholder').attributes('aria-label')).toBe('无封面')
  })

  it('renders book tags through the shared product chip contract', () => {
    const wrapper = mount(BookCard, {
      props: {
        book: book({ tags: ['Drama'] }),
        tags: [{ id: 'tag-drama', name: 'Drama', color: '#aa6644' }],
      },
    })

    const chipList = wrapper.getComponent(ProductChipList)

    expect(chipList.props('ariaLabel')).toBe('书籍标签')
    expect(chipList.props('items')).toEqual([
      {
        id: 'Drama',
        label: 'Drama',
        tone: 'custom',
        backgroundColor: '#aa6644',
        textColor: 'var(--color-text-inverse)',
      },
    ])
    expect(wrapper.find('.book-tag').exists()).toBe(false)
  })

  it('reveals the cover action overlay on keyboard focus as well as hover', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/bookshelf/BookCard.vue'), 'utf8')

    expect(source).toContain('.book-card:focus-visible .book-card__cover::before')
  })

  it('keeps book card internals under the card owner', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/bookshelf/BookCard.vue'), 'utf8')

    for (const oldClass of [
      'book-cover',
      'book-cover-placeholder',
      'book-info',
      'book-title',
      'book-chapter-count',
      'book-tags',
    ]) {
      expect(source).not.toMatch(new RegExp(`class="[^"]*\\b${oldClass}\\b`))
      expect(source).not.toMatch(new RegExp(`\\.${oldClass}\\b`))
    }
    expect(source).not.toMatch(/\.book-card__[^{]+ img\b/)

    for (const ownerClass of [
      'book-card__cover',
      'book-card__cover-image',
      'book-card__cover-placeholder',
      'book-card__info',
      'book-card__title',
      'book-card__chapter-count',
      'book-card__tags',
    ]) {
      expect(source).toContain(ownerClass)
    }
  })

  it('maps card owner colors through semantic tokens', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/bookshelf/BookCard.vue'), 'utf8')
    const style = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''

    expect(style).not.toMatch(/#[0-9a-fA-F]{3,8}\b|rgba?\(/)
  })
})
