import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import BookDetailSummary from '@/components/bookshelf/book-detail/BookDetailSummary.vue'
import ChapterList from '@/components/bookshelf/book-detail/ChapterList.vue'
import ChapterFormContent from '@/components/bookshelf/book-detail/ChapterFormContent.vue'
import ChapterRow from '@/components/bookshelf/book-detail/ChapterRow.vue'
import QuickTagPicker from '@/components/bookshelf/book-detail/QuickTagPicker.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import ProductSearchField from '@/components/product/ProductSearchField.vue'
import ProductScrollStack from '@/components/product/ProductScrollStack.vue'
import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import type { BookData, TagData } from '@/types/api'

const book: BookData = {
  id: 'book-1',
  title: 'Demo Book',
  cover: '',
  tags: ['Drama'],
  chapters: [],
  chapterCount: 0,
  createdAt: '2026-01-01T00:00:00Z',
  updatedAt: '2026-01-01T00:00:00Z',
}

const availableTags: TagData[] = [
  { name: 'Action', color: '#4466aa' },
]

describe('bookshelf detail child components', () => {
  it('uses the shared product chip list for removable detail tags and add-tag action', () => {
    const wrapper = mount(BookDetailSummary, {
      props: {
        book,
        chapterCount: 0,
        formatDate: () => '2026-01-01',
        getTagColor: () => '#4466aa',
      },
    })

    const chipList = wrapper.getComponent(ProductChipList)
    expect(chipList.props('ariaLabel')).toBe('书籍详情标签')
    expect(chipList.props('items')).toEqual([
      {
        id: 'Drama',
        label: 'Drama',
        ariaLabel: '移除标签 Drama',
        iconName: 'x',
        interactive: true,
        tone: 'custom',
        backgroundColor: '#4466aa',
        borderColor: '#4466aa',
        textColor: 'var(--color-text-inverse)',
      },
    ])

    chipList.vm.$emit('select', 'Drama')
    expect(wrapper.emitted('removeTag')?.[0]).toEqual(['Drama'])
    expect(wrapper.find('.detail-tag').exists()).toBe(false)
    expect(wrapper.find('.remove-detail-tag').exists()).toBe(false)

    expect(wrapper.get('.book-detail-summary__add-tag').attributes('aria-label')).toBe('添加标签')
    expect(wrapper.get('.book-detail-summary__add-tag').getComponent(UiIcon).props('name')).toBe('plus')
    expect(wrapper.getComponent(UiIconButton).props()).toMatchObject({
      label: '添加标签',
      variant: 'soft',
      size: 'sm',
    })
    expect(wrapper.get('.book-detail-summary__add-tag').text()).not.toContain('+')

    const insightButton = wrapper.findAll('.book-detail-summary__actions button')
      .find(button => button.text().includes('漫画分析'))
    expect(insightButton).toBeTruthy()
    expect(insightButton!.getComponent(UiIcon).props('name')).toBe('bar-chart')
    expect(insightButton!.text()).not.toContain('●')
  })

  it('does not assert shared button primitives through internal class names', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/unit/bookDetailComponents.spec.ts'), 'utf8')
    const buttonClassPrefix = 'ui-' + 'button--'
    const iconButtonClassPrefix = 'ui-' + 'icon-button--'

    expect(source).not.toContain(buttonClassPrefix)
    expect(source).not.toContain(iconButtonClassPrefix)
  })

  it('falls back to the detail cover placeholder when the cover image fails', async () => {
    const wrapper = mount(BookDetailSummary, {
      props: {
        book: {
          ...book,
          cover: 'broken-cover.png',
        },
        chapterCount: 0,
        formatDate: () => '2026-01-01',
        getTagColor: () => '#4466aa',
      },
    })

    await wrapper.get('.book-detail-summary__cover-image').trigger('error')
    await wrapper.vm.$nextTick()

    expect(wrapper.find('.book-detail-summary__cover-image').exists()).toBe(false)
    expect(wrapper.get('.book-detail-summary__cover-placeholder').text()).toBe('无封面')

    await wrapper.setProps({
      book: {
        ...book,
        cover: 'replacement-cover.png',
      },
    })

    expect(wrapper.get('.book-detail-summary__cover-image').attributes('src')).toBe('replacement-cover.png')
  })

  it('emits save from the chapter form on Enter keydown', async () => {
    const wrapper = mount(ChapterFormContent, {
      props: { modelValue: 'Chapter 1' },
    })

    await wrapper.get('input').trigger('keydown.enter')

    expect(wrapper.emitted('save')).toHaveLength(1)
  })

  it('renders the chapter form field through the shared UI field primitive', () => {
    const wrapper = mount(ChapterFormContent, {
      props: { modelValue: 'Chapter 1' },
    })

    const field = wrapper.getComponent(UiField)

    expect(field.props()).toMatchObject({
      label: '章节名称',
      required: true,
      variant: 'dialog',
      controlId: 'chapterTitleInput',
    })
    expect(field.classes()).toContain('chapter-form-content__field')
  })

  it('renders existing quick tag choices through product chips and keeps creation as a command card', () => {
    const wrapper = mount(QuickTagPicker, {
      props: {
        availableTags,
        filter: 'New',
        showCreateNewTagOption: true,
      },
    })

    const chipList = wrapper.getComponent(ProductChipList)
    expect(chipList.props('ariaLabel')).toBe('可添加标签')
    expect(chipList.props('items')).toEqual([
      {
        id: 'Action',
        label: 'Action',
        ariaLabel: '添加标签 Action',
        iconName: 'plus',
        interactive: true,
        tone: 'custom',
        backgroundColor: '#4466aa',
        borderColor: '#4466aa',
        textColor: 'var(--color-text-inverse)',
      },
    ])
    chipList.vm.$emit('select', 'Action')
    expect(wrapper.emitted('add')?.[0]).toEqual(['Action'])

    const createCommand = wrapper.getComponent(ProductRecordCard)
    expect(createCommand.props('as')).toBe('button')
    expect(createCommand.attributes('aria-label')).toBe('创建并添加标签 New')
    expect(createCommand.getComponent(UiIcon).props('name')).toBe('plus')
    expect(createCommand.text()).not.toContain('+')
    expect(wrapper.find('.tag-color-dot').exists()).toBe(false)
    expect(wrapper.find('.quick-tag-name').exists()).toBe(false)
  })

  it('renders quick tag filtering through the product search field', () => {
    const wrapper = mount(QuickTagPicker, {
      props: {
        availableTags,
        filter: 'New',
        showCreateNewTagOption: true,
      },
    })

    const searchField = wrapper.getComponent(ProductSearchField)
    expect(searchField.props()).toMatchObject({
      modelValue: 'New',
      ariaLabel: '搜索或创建标签',
      placeholder: '输入标签名称进行搜索或创建...',
      autofocus: true,
    })

    searchField.vm.$emit('update:modelValue', 'Drama')
    expect(wrapper.emitted('update:filter')?.[0]).toEqual(['Drama'])

    searchField.vm.$emit('search', 'Drama')
    expect(wrapper.emitted('submit')).toBeTruthy()

    expect(wrapper.find('.quick-tag-input').exists()).toBe(false)
  })

  it('keeps the create-new quick tag command on the product record-card shell', () => {
    const wrapper = mount(QuickTagPicker, {
      props: {
        availableTags,
        filter: 'New',
        showCreateNewTagOption: true,
      },
    })

    const cards = wrapper.findAllComponents(ProductRecordCard)

    expect(cards).toHaveLength(1)
    expect(cards[0]?.props('as')).toBe('button')
    expect(cards[0]?.attributes('aria-label')).toBe('创建并添加标签 New')
  })

  it('renders the quick tag empty state through the product status banner', () => {
    const wrapper = mount(QuickTagPicker, {
      props: {
        availableTags: [],
        filter: '',
        showCreateNewTagOption: false,
      },
    })

    const emptyState = wrapper.getComponent(ProductStatusBanner)
    expect(emptyState.props()).toMatchObject({
      tone: 'neutral',
      iconName: 'tags',
      role: 'note',
    })
    expect(emptyState.text()).toContain('所有标签已添加或暂无标签')
    expect(wrapper.find('.quick-tags-empty').exists()).toBe(false)
  })

  it('renders chapter creation with the shared plus icon', () => {
    const wrapper = mount(ChapterList, {
      props: {
        chapters: [],
        draggedChapterIndex: null,
        dragOverChapterIndex: null,
      },
    })

    const createButton = wrapper.get('button')
    expect(createButton.text()).toContain('新建章节')
    expect(createButton.getComponent(UiIcon).props('name')).toBe('plus')
    expect(createButton.text()).not.toContain('+')

    const emptyState = wrapper.getComponent(ProductStatusBanner)
    expect(emptyState.props()).toMatchObject({
      tone: 'neutral',
      iconName: 'book-open',
      role: 'note',
    })
    expect(emptyState.text()).toContain('暂无章节，点击上方按钮创建')
    expect(wrapper.find('.empty-state-small').exists()).toBe(false)
  })

  it('uses the product section header for the chapter list heading', () => {
    const wrapper = mount(ChapterList, {
      props: {
        chapters: [],
        draggedChapterIndex: null,
        dragOverChapterIndex: null,
      },
    })
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/bookshelf/book-detail/ChapterList.vue'),
      'utf8'
    )
    const header = wrapper.getComponent(ProductSectionHeader)

    expect(header.props()).toMatchObject({
      title: '章节列表',
      iconName: 'book-open',
    })
    expect(header.text()).toContain('新建章节')
    expect(source).toContain("import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'")
    expect(source).not.toContain('class="section-header"')
    expect(source).not.toContain('.section-header {')
    expect(source).not.toContain('.section-header h3')
  })

  it('delegates chapter-list scrolling to the product scroll stack', () => {
    const wrapper = mount(ChapterList, {
      props: {
        chapters: [
          {
            id: 'chapter-1',
            title: 'Chapter 1',
            imageCount: 2,
          },
        ],
        draggedChapterIndex: null,
        dragOverChapterIndex: null,
      },
    })
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/bookshelf/book-detail/ChapterList.vue'),
      'utf8'
    )

    const scrollStack = wrapper.getComponent(ProductScrollStack)
    expect(scrollStack.classes()).toContain('chapter-list__list')
    expect(scrollStack.props()).toMatchObject({
      ariaLabel: '章节列表',
      gap: 'sm',
      padding: 'none',
      role: 'region',
    })
    expect(source).toContain("import ProductScrollStack from '@/components/product/ProductScrollStack.vue'")
    expect(source).not.toContain('::-webkit-scrollbar')
  })

  it('renders chapter rows through product record-card and action row primitives', () => {
    const wrapper = mount(ChapterRow, {
      props: {
        chapter: {
          id: 'chapter-1',
          title: 'Chapter 1',
          imageCount: 2,
        },
        index: 0,
        isDragging: false,
        isDragOver: false,
      },
    })

    const card = wrapper.getComponent(ProductRecordCard)
    expect(card.classes()).toContain('chapter-row')
    expect(card.props('ariaLabel')).toBe('章节 Chapter 1，拖拽可调整排序')
    expect(card.attributes('aria-grabbed')).toBe('false')
    expect(wrapper.getComponent(ProductActionRow).props('ariaLabel')).toBe('Chapter 1 章节操作')
    expect(wrapper.get('.chapter-row__drag-handle').getComponent(UiIcon).props('name')).toBe('grip-vertical')
    expect(wrapper.get('.chapter-row__drag-handle').text()).not.toContain('⋮')

    const buttonVariants = wrapper.getComponent(ProductActionRow).findAllComponents(UiButton)
      .map(button => button.props('variant'))
    expect(buttonVariants).toEqual([
      'primary',
      'secondary',
      'secondary',
      'danger',
    ])
    expect(wrapper.find('.chapter-action-btn').exists()).toBe(false)
    expect(wrapper.find('.chapter-enter-btn').exists()).toBe(false)
    expect(wrapper.find('.chapter-read-btn').exists()).toBe(false)
  })

  it('keeps chapter rows responsive inside narrow detail modals', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/bookshelf/book-detail/ChapterRow.vue'),
      'utf8'
    )
    const style = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''

    expect(style).toMatch(/\.chapter-row__content\s*{[\s\S]*flex-wrap:\s*wrap/)
    expect(style).toMatch(/\.chapter-row__content\s*{[\s\S]*min-width:\s*0/)
    expect(style).toMatch(/\.chapter-row__info\s*{[\s\S]*flex:\s*1 1 260px/)
    expect(style).toMatch(/\.chapter-row__title\s*{[\s\S]*overflow-wrap:\s*anywhere/)
    expect(style).toMatch(/\.chapter-row__title\s*{[\s\S]*white-space:\s*normal/)
    expect(style).toMatch(/\.chapter-row__actions\s*{[\s\S]*flex:\s*1 1 280px/)
    expect(style).toMatch(/\.chapter-row__actions\s*{[\s\S]*min-width:\s*0/)
  })

  it('keeps book-detail child visual tokens with their child owners', () => {
    const modalSource = readFileSync(resolve(process.cwd(), 'src/components/bookshelf/BookDetailModal.vue'), 'utf8')
    const modalStyle = modalSource.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''
    const summaryStyle = readFileSync(
      resolve(process.cwd(), 'src/components/bookshelf/book-detail/BookDetailSummary.vue'),
      'utf8'
    ).match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''
    const quickTagStyle = readFileSync(
      resolve(process.cwd(), 'src/components/bookshelf/book-detail/QuickTagPicker.vue'),
      'utf8'
    ).match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''

    expect(modalStyle).not.toMatch(/--book-detail-(accent|cover-shadow|new-tag)/)
    expect(summaryStyle).toContain('--book-detail-summary-cover-shadow: var(--shadow-medium)')
    expect(summaryStyle).not.toMatch(/--ui-button-/)
    expect(quickTagStyle).toContain('--quick-tag-picker-new-background-start: var(--color-focus-brand-soft)')
    expect(quickTagStyle).not.toContain('tag-color-dot')
    expect(quickTagStyle).not.toContain('quick-tag-name')

    for (const styleBlock of [modalStyle, summaryStyle, quickTagStyle]) {
      expect(styleBlock).not.toMatch(/#[0-9a-fA-F]{3,8}\b|rgba?\(/)
    }
  })

  it('renders nested detail dialogs with product dialog action rows', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/bookshelf/BookDetailModal.vue'), 'utf8')

    expect(source).toContain("import ProductActionRow from '@/components/product/ProductActionRow.vue'")
    expect(source).toContain('aria-label="章节表单操作"')
    expect(source).toContain('aria-label="快速标签操作"')
    expect(source).toContain('aria-label="书籍详情删除操作"')
    expect(source.match(/<ProductActionRow/g)).toHaveLength(3)
    expect(source).not.toContain('<template #footer>\r\n      <UiButton')
    expect(source).not.toContain('<template #footer>\n      <UiButton')
  })

  it('keeps bookshelf wire aliases out of detail child UI owners', () => {
    const cardSource = readFileSync(
      resolve(process.cwd(), 'src/components/bookshelf/BookCard.vue'),
      'utf8'
    )
    const summarySource = readFileSync(
      resolve(process.cwd(), 'src/components/bookshelf/book-detail/BookDetailSummary.vue'),
      'utf8'
    )
    const chapterRowSource = readFileSync(
      resolve(process.cwd(), 'src/components/bookshelf/book-detail/ChapterRow.vue'),
      'utf8'
    )
    const childSources = [cardSource, summarySource, chapterRowSource].join('\n')

    expect(childSources).not.toMatch(/\b(chapter_count|created_at|updated_at|image_count|has_session|session_path)\b/)
    expect(cardSource).toContain('book.chapterCount')
    expect(summarySource).toContain('formatDate(book.createdAt)')
    expect(summarySource).toContain('formatDate(book.updatedAt)')
    expect(chapterRowSource).toContain('chapter.imageCount')
  })

  it('keeps detail child hooks under their component owners', () => {
    const sources = {
      summary: readFileSync(resolve(process.cwd(), 'src/components/bookshelf/book-detail/BookDetailSummary.vue'), 'utf8'),
      chapterForm: readFileSync(resolve(process.cwd(), 'src/components/bookshelf/book-detail/ChapterFormContent.vue'), 'utf8'),
      chapterList: readFileSync(resolve(process.cwd(), 'src/components/bookshelf/book-detail/ChapterList.vue'), 'utf8'),
      chapterRow: readFileSync(resolve(process.cwd(), 'src/components/bookshelf/book-detail/ChapterRow.vue'), 'utf8'),
      quickTag: readFileSync(resolve(process.cwd(), 'src/components/bookshelf/book-detail/QuickTagPicker.vue'), 'utf8'),
    }

    for (const oldClass of [
      'book-info-section',
      'book-cover-large',
      'book-cover-placeholder',
      'book-meta',
      'meta-item',
      'meta-label',
      'detail-tags',
      'no-tags-hint',
      'book-actions',
      'chapter-form-field',
      'chapters-section',
      'chapters-list',
      'chapter-empty-state',
      'chapter-item',
      'chapter-row-content',
      'chapter-drag-handle',
      'chapter-info',
      'chapter-order',
      'chapter-title',
      'chapter-meta',
      'chapter-actions',
      'quick-tag-input-wrapper',
      'quick-tag-list',
      'quick-tag-item',
      'quick-tag-content',
      'tag-icon',
      'quick-tags-empty-state',
    ]) {
      for (const source of Object.values(sources)) {
        expect(source).not.toMatch(new RegExp(`class="[^"]*\\b${oldClass}\\b`))
        expect(source).not.toMatch(new RegExp(`\\.${oldClass}\\b`))
      }
    }

    expect(sources.chapterRow).not.toContain('dragging: isDragging')
    expect(sources.chapterRow).not.toContain("'drag-over': isDragOver")
    expect(sources.quickTag).not.toContain('class="quick-tag-item new-tag"')
    expect(sources.quickTag).not.toContain('.quick-tag-item.new-tag')

    expect(sources.summary).toContain('book-detail-summary__cover')
    expect(sources.chapterForm).toContain('chapter-form-content__field')
    expect(sources.chapterList).toContain('chapter-list__list')
    expect(sources.chapterRow).toContain('chapter-row__actions')
    expect(sources.quickTag).toContain('quick-tag-picker__item--new')
  })
})
