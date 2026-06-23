import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import BookDetailSummary from '@/components/bookshelf/book-detail/BookDetailSummary.vue'
import ChapterFormContent from '@/components/bookshelf/book-detail/ChapterFormContent.vue'
import QuickTagPicker from '@/components/bookshelf/book-detail/QuickTagPicker.vue'
import type { BookData, TagData } from '@/types/api'

const book: BookData = {
  id: 'book-1',
  title: 'Demo Book',
  cover: '',
  tags: ['Drama'],
  chapters: [],
  chapter_count: 0,
  createdAt: '2026-01-01T00:00:00Z',
  updatedAt: '2026-01-01T00:00:00Z',
}

const availableTags: TagData[] = [
  { name: 'Action', color: '#4466aa' },
]

describe('bookshelf detail child components', () => {
  it('uses button semantics for removable detail tags and add-tag action', () => {
    const wrapper = mount(BookDetailSummary, {
      props: {
        book,
        chapterCount: 0,
        formatDate: () => '2026-01-01',
        getTagColor: () => '#4466aa',
      },
    })

    expect(wrapper.get('.remove-detail-tag').element.tagName).toBe('BUTTON')
    expect(wrapper.get('.remove-detail-tag').attributes('aria-label')).toBe('移除标签 Drama')
    expect(wrapper.get('.btn-add-tag').attributes('aria-label')).toBe('添加标签')
  })

  it('emits save from the chapter form on Enter keydown', async () => {
    const wrapper = mount(ChapterFormContent, {
      props: { modelValue: 'Chapter 1' },
    })

    await wrapper.get('input').trigger('keydown.enter')

    expect(wrapper.emitted('save')).toHaveLength(1)
  })

  it('renders quick tag choices as buttons', () => {
    const wrapper = mount(QuickTagPicker, {
      props: {
        availableTags,
        filter: 'New',
        showCreateNewTagOption: true,
      },
    })

    const buttons = wrapper.findAll('.quick-tag-item')

    expect(buttons.map(button => button.element.tagName)).toEqual(['BUTTON', 'BUTTON'])
    expect(buttons[0].attributes('aria-label')).toBe('添加标签 Action')
    expect(buttons[1].attributes('aria-label')).toBe('创建并添加标签 New')
  })
})
