import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent, nextTick } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import BookModal from '@/components/bookshelf/BookModal.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductFileDropzone from '@/components/product/ProductFileDropzone.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import { setTestBooks, setTestTags } from '../helpers/bookshelfFixtures'

const BaseModalStub = defineComponent({
  template: '<section class="base-modal-stub"><slot /><footer><slot name="footer" /></footer></section>',
})

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise
  })
  return { promise, resolve }
}

describe('BookModal', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('renders cover upload through the shared product file dropzone', () => {
    const wrapper = mount(BookModal, {
      global: {
        stubs: {
          BaseModal: BaseModalStub,
        },
      },
    })

    const dropzone = wrapper.getComponent(ProductFileDropzone)

    expect(dropzone.props('inputId')).toBe('bookCoverInput')
    expect(dropzone.props('accept')).toBe('image/*')
    expect(dropzone.props('label')).toBe('上传书籍封面')
    expect(wrapper.find('.cover-upload-area').exists()).toBe(false)
  })

  it('uses the shared product chip list for selected tag removal', async () => {
    const store = useBookshelfStore()
    setTestBooks(store, [{
      id: 'book-1',
      title: 'Demo Book',
      cover: '',
      tags: ['Drama'],
      chapters: [],
      chapterCount: 0,
      createdAt: '2026-01-01T00:00:00Z',
      updatedAt: '2026-01-01T00:00:00Z',
    }])

    const wrapper = mount(BookModal, {
      props: {
        bookId: 'book-1',
      },
      global: {
        stubs: {
          BaseModal: BaseModalStub,
        },
      },
    })
    await nextTick()

    const chipList = wrapper.getComponent(ProductChipList)

    expect(chipList.props('items')).toEqual([
      {
        id: 'Drama',
        label: 'Drama',
        ariaLabel: '移除标签 Drama',
        iconName: 'x',
        interactive: true,
        tone: 'primary',
      },
    ])
    expect(wrapper.find('.selected-tag').exists()).toBe(false)
    expect(wrapper.find('.remove-tag').exists()).toBe(false)

    chipList.vm.$emit('select', 'Drama')
    await nextTick()

    expect(wrapper.findComponent(ProductChipList).exists()).toBe(false)
  })

  it('renders modal form sections through the shared field primitive', () => {
    const wrapper = mount(BookModal, {
      global: {
        stubs: {
          BaseModal: BaseModalStub,
        },
      },
    })

    expect(wrapper.findAllComponents(UiField).map(field => field.props('label'))).toEqual([
      '书籍名称',
      '封面图片',
      '标签',
    ])
    expect(wrapper.find('.book-modal__field').exists()).toBe(false)
    expect(wrapper.find('.book-modal__title-input').exists()).toBe(false)
  })

  it('binds every book form field to a stable primitive control id', () => {
    const wrapper = mount(BookModal, {
      global: {
        stubs: {
          BaseModal: BaseModalStub,
        },
      },
    })

    const controlIds = wrapper.findAllComponents(UiField)
      .map(field => field.props('controlId'))

    expect(controlIds).toEqual([
      'bookTitle',
      'bookCoverInput',
      'bookTagInput',
    ])

    const source = readFileSync(resolve(process.cwd(), 'src/components/bookshelf/BookModal.vue'), 'utf8')
    expect(source).toContain('input-id="bookCoverInput"')
    expect(source).toContain('id="bookTagInput"')
  })

  it('renders footer actions through the product dialog action row', () => {
    const wrapper = mount(BookModal, {
      global: {
        stubs: {
          BaseModal: BaseModalStub,
        },
      },
    })

    const actionRow = wrapper.getComponent(ProductActionRow)

    expect(actionRow.props('variant')).toBe('dialog')
    expect(actionRow.props('ariaLabel')).toBe('书籍表单操作')
  })

  it('submits a slow book creation only once', async () => {
    const store = useBookshelfStore()
    const pending = deferred<{ id: string; title: string }>()
    store.createBook = vi.fn().mockReturnValue(pending.promise)
    const wrapper = mount(BookModal, {
      global: {
        stubs: {
          BaseModal: BaseModalStub,
        },
      },
    })
    await wrapper.get('#bookTitle').setValue('Slow Book')
    const saveButton = wrapper.findAllComponents(UiButton)
      .find(button => button.text() === '保存')!

    await saveButton.trigger('click')
    await saveButton.trigger('click')

    expect(store.createBook).toHaveBeenCalledTimes(1)
    expect(saveButton.props('loading')).toBe(true)

    pending.resolve({ id: 'book-slow', title: 'Slow Book' })
    await flushPromises()
  })

  it('renders tag suggestions through product record-card buttons', async () => {
    const store = useBookshelfStore()
    setTestTags(store, [{ id: 'tag-drama', name: 'Drama', color: '#4466aa', bookCount: 1 }])

    const wrapper = mount(BookModal, {
      global: {
        stubs: {
          BaseModal: BaseModalStub,
        },
      },
    })

    await wrapper.get('input[placeholder="输入标签名称..."]').trigger('focus')
    await nextTick()

    const suggestions = wrapper.findAllComponents(ProductRecordCard)

    expect(suggestions).toHaveLength(1)
    expect(suggestions[0].props('as')).toBe('button')
    expect(suggestions[0].attributes('aria-label')).toBe('添加标签 Drama')
    expect(wrapper.find('.tag-suggestion.ui-button').exists()).toBe(false)
  })

  it('maps modal owner colors through semantic tokens', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/bookshelf/BookModal.vue'), 'utf8')
    const styleBlock = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''

    expect(styleBlock).not.toMatch(/#[0-9a-fA-F]{3,8}\b|rgba?\(/)
    expect(styleBlock).not.toMatch(/color-text-muted|color-border-subtle|color-surface-base/)
    expect(styleBlock).not.toContain('--book-modal-shadow-default')
    expect(styleBlock).toContain('--book-modal-tag-suggestions-shadow')
  })

  it('uses the embedded input primitive for tag entry instead of restyling UiInput locally', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/bookshelf/BookModal.vue'), 'utf8')
    const styleBlock = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''

    expect(source).toContain('variant="embedded"')
    expect(styleBlock).not.toMatch(/\.book-modal__tag-input\s*\{[\s\S]*border:\s*none/)
    expect(styleBlock).not.toMatch(/\.book-modal__tag-input\s*\{[\s\S]*background:\s*transparent/)
    expect(styleBlock).not.toMatch(/\.book-modal__tag-input\s*\{[\s\S]*outline:\s*none/)
  })

  it('keeps modal presentation hooks under the book-modal owner', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/bookshelf/BookModal.vue'), 'utf8')

    for (const oldClass of [
      'cover-preview',
      'cover-placeholder',
      'upload-icon',
      'tag-input-container',
      'tag-dropdown',
      'tag-suggestions',
      'tag-suggestion',
    ]) {
      expect(source).not.toMatch(new RegExp(`class="[^"]*\\b${oldClass}\\b`))
      expect(source).not.toMatch(new RegExp(`\\.${oldClass}\\b`))
    }
    expect(source).not.toMatch(/\.book-modal__[^{]+ img\b/)

    for (const ownerClass of [
      'book-modal__form',
      'book-modal__cover-preview',
      'book-modal__cover-image',
      'book-modal__cover-placeholder',
      'book-modal__upload-icon',
      'book-modal__tag-input-container',
      'book-modal__tag-dropdown',
      'book-modal__tag-suggestions',
      'book-modal__tag-suggestion',
    ]) {
      expect(source).toContain(ownerClass)
    }
  })
})
