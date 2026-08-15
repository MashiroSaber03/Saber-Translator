import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { defineComponent } from 'vue'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import TagManageModal from '@/components/bookshelf/TagManageModal.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiColorInput from '@/components/ui/UiColorInput.vue'
import UiField from '@/components/ui/UiField.vue'
import UiFormGrid from '@/components/ui/UiFormGrid.vue'
import { useBookshelfStore } from '@/stores/bookshelfStore'
import { setTestTags } from '../helpers/bookshelfFixtures'
import { showToast } from '@/utils/toast'

const { confirmProductActionMock } = vi.hoisted(() => ({
  confirmProductActionMock: vi.fn(),
}))

vi.mock('@/utils/toast', () => ({
  showToast: vi.fn(),
}))

vi.mock('@/composables/useProductConfirm', () => ({
  confirmProductAction: confirmProductActionMock,
}))

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

function mountModal() {
  setActivePinia(createPinia())
  return mount(TagManageModal, {
    global: {
      stubs: {
        BaseModal: BaseModalStub,
      },
    },
  })
}

describe('TagManageModal', () => {
  beforeEach(() => {
    vi.mocked(showToast).mockClear()
    confirmProductActionMock.mockReset()
    confirmProductActionMock.mockResolvedValue(true)
  })

  it('does not report success when tag creation fails', async () => {
    const wrapper = mountModal()
    const store = useBookshelfStore()
    store.createTag = vi.fn().mockRejectedValue(new Error('create failed'))

    await wrapper.get('input[type="text"]').setValue('New Tag')
    await wrapper.get('button').trigger('click')
    await flushPromises()

    expect(showToast).toHaveBeenCalledWith('create failed', 'error')
    expect(showToast).not.toHaveBeenCalledWith('标签创建成功', 'success')
  })

  it('submits a slow tag creation only once', async () => {
    const wrapper = mountModal()
    const store = useBookshelfStore()
    const pending = deferred<{ id: string; name: string; color: string }>()
    store.createTag = vi.fn().mockReturnValue(pending.promise)
    await wrapper.get('input[type="text"]').setValue('Slow Tag')
    const addButton = wrapper.findAllComponents(UiButton)
      .find(button => button.text() === '添加')!

    await addButton.trigger('click')
    await addButton.trigger('click')

    expect(store.createTag).toHaveBeenCalledTimes(1)
    expect(addButton.props('loading')).toBe(true)

    pending.resolve({ id: 'slow-tag', name: 'Slow Tag', color: '#667eea' })
    await flushPromises()
  })

  it('confirms before deleting a tag and skips deletion when cancelled', async () => {
    const wrapper = mountModal()
    const store = useBookshelfStore()
    setTestTags(store, [{ id: 'delete-tag', name: '待删标签', color: '#667eea', bookCount: 2 }])
    store.deleteTagApi = vi.fn().mockResolvedValue(undefined)
    confirmProductActionMock.mockResolvedValueOnce(false)
    await wrapper.vm.$nextTick()

    await wrapper.get('.tag-manage-modal__row-delete-action').trigger('click')
    await flushPromises()

    expect(confirmProductActionMock).toHaveBeenCalledWith({
      title: '删除标签',
      message: '确定要删除标签“待删标签”吗？此操作不会删除书籍，但会从相关书籍中移除该标签。',
      confirmText: '删除',
      cancelText: '取消',
      tone: 'danger',
    })
    expect(store.deleteTagApi).not.toHaveBeenCalled()
  })

  it('uses product button variants for tag row actions', async () => {
    const wrapper = mountModal()
    const store = useBookshelfStore()
    setTestTags(store, [{ id: 'editable-tag', name: '可编辑标签', color: '#667eea', bookCount: 1 }])
    await wrapper.vm.$nextTick()

    const viewButtons = wrapper.findAllComponents(UiButton)
      .filter(button => (
        button.classes().includes('tag-manage-modal__row-edit-action') ||
        button.classes().includes('tag-manage-modal__row-delete-action')
      ))
      .map(button => button.props('variant'))
    expect(viewButtons).toEqual(['secondary', 'danger'])

    await wrapper.get('.tag-manage-modal__row-edit-action').trigger('click')
    await wrapper.vm.$nextTick()

    const editButtons = wrapper.findAllComponents(UiButton)
      .filter(button => (
        button.classes().includes('tag-manage-modal__edit-save-action') ||
        button.classes().includes('tag-manage-modal__edit-cancel-action')
      ))
      .map(button => button.props('variant'))
    expect(editButtons).toEqual(['primary', 'secondary'])
  })

  it('uses owner action hooks instead of stale tag button class names', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/bookshelf/TagManageModal.vue'),
      'utf8',
    )

    expect(source).toContain('tag-manage-modal__row-edit-action')
    expect(source).toContain('tag-manage-modal__edit-cancel-action')
    expect(source).not.toMatch(/tag-(edit|delete|save|cancel)-btn/)
  })

  it('renders tag rows through the shared product record-card shell', async () => {
    const wrapper = mountModal()
    const store = useBookshelfStore()
    setTestTags(store, [{ id: 'record-tag', name: '记录标签', color: '#667eea', bookCount: 1 }])
    await wrapper.vm.$nextTick()

    const row = wrapper.getComponent(ProductRecordCard)

    expect(row.classes()).toContain('tag-manage-modal__item')
    expect(row.attributes('aria-label')).toBe('标签 记录标签')
    expect(wrapper.find('.tag-manage-modal__item > .tag-view-mode').exists()).toBe(false)
  })

  it('renders tag row metadata through shared product chips', async () => {
    const wrapper = mountModal()
    const store = useBookshelfStore()
    setTestTags(store, [{ id: 'metadata-tag', name: '元信息标签', color: '#667eea', bookCount: 3 }])
    await wrapper.vm.$nextTick()

    const chipList = wrapper.getComponent(ProductChipList)

    expect(chipList.props('ariaLabel')).toBe('元信息标签 标签信息')
    expect(chipList.props('items')).toEqual([
      {
        id: 'tag-元信息标签',
        label: '元信息标签',
        tone: 'custom',
        backgroundColor: '#667eea',
        borderColor: '#667eea',
        textColor: 'var(--color-text-inverse)',
      },
      {
        id: 'count-元信息标签',
        label: '3 本',
        tone: 'neutral',
      },
    ])
    expect(wrapper.find('.tag-color-dot').exists()).toBe(false)
    expect(wrapper.find('.tag-name').exists()).toBe(false)
    expect(wrapper.find('.tag-book-count').exists()).toBe(false)
  })

  it('renders the new-tag form through shared field and action primitives', () => {
    const wrapper = mountModal()

    expect(wrapper.getComponent(UiFormGrid).exists()).toBe(true)
    expect(wrapper.findAllComponents(UiField).map(field => field.props('label'))).toEqual([
      '标签名称',
      '标签颜色',
    ])
    expect(wrapper.getComponent(ProductActionRow).props('ariaLabel')).toBe('新建标签操作')
    expect(wrapper.find('.form-row').exists()).toBe(false)
    expect(wrapper.find('.tag-manage-modal__new-name-input').exists()).toBe(false)
  })

  it('renders tag color controls through the shared color input primitive', async () => {
    const wrapper = mountModal()
    const store = useBookshelfStore()

    const newColorInput = wrapper.getComponent(UiColorInput)
    expect(newColorInput.props()).toMatchObject({
      inputId: 'tag-manage-new-color',
      modelValue: '#667eea',
      title: '选择颜色',
    })

    newColorInput.vm.$emit('update:modelValue', '#334455')
    await wrapper.vm.$nextTick()

    setTestTags(store, [{ id: 'edit-tag', name: '待编辑标签', color: '#667eea', bookCount: 1 }])
    await wrapper.vm.$nextTick()
    await wrapper.get('.tag-manage-modal__row-edit-action').trigger('click')
    await wrapper.vm.$nextTick()

    const colorInputs = wrapper.findAllComponents(UiColorInput)
    expect(colorInputs).toHaveLength(2)
    expect(colorInputs[1]?.props()).toMatchObject({
      inputId: 'tag-edit-color-待编辑标签',
      modelValue: '#667eea',
      title: '选择颜色',
    })

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/bookshelf/TagManageModal.vue'),
      'utf8',
    )
    expect(source).toContain("import UiColorInput from '@/components/ui/UiColorInput.vue'")
    expect(source).not.toContain('type="color"')
    expect(source).not.toContain('tag-manage-color-input')
  })

  it('keeps the default tag color in the Bookshelf constants owner', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/bookshelf/TagManageModal.vue'),
      'utf8',
    )

    expect(source).toContain("import { BOOKSHELF_DEFAULT_TAG_COLOR } from '@/constants/bookshelf'")
    expect(source).toContain('newTagColor = ref(BOOKSHELF_DEFAULT_TAG_COLOR)')
    expect(source).not.toContain("ref('#667eea')")
    expect(source).not.toContain("= '#667eea'")
    expect(source).not.toContain("|| '#667eea'")
  })

  it('does not override input primitive focus tokens from the modal owner', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/bookshelf/TagManageModal.vue'),
      'utf8',
    )

    expect(source).not.toMatch(/--ui-input-/)
    expect(source).not.toContain('--tag-manage-modal-row-background')
    expect(source).toContain('--product-record-card-background: var(--color-surface-app)')
    expect(source).toContain('variant="settings"')
  })

  it('renders the empty tag list through the product status banner', () => {
    const wrapper = mountModal()

    const emptyState = wrapper.getComponent(ProductStatusBanner)
    expect(emptyState.props()).toMatchObject({
      tone: 'neutral',
      iconName: 'tags',
      role: 'note',
    })
    expect(emptyState.text()).toContain('暂无标签，请在上方添加')
    expect(wrapper.find('.empty-hint').exists()).toBe(false)
  })

  it('keeps tag loading failures visible in the modal and exposes a retry', async () => {
    const wrapper = mountModal()
    const store = useBookshelfStore()
    store.tagsError = 'tag refresh failed'
    store.loadTags = vi.fn().mockResolvedValue(undefined)
    await wrapper.vm.$nextTick()

    const status = wrapper.findAllComponents(ProductStatusBanner)
      .find(banner => banner.props('title') === '标签加载失败')!
    expect(status.props()).toMatchObject({
      role: 'alert',
      tone: 'warning',
    })
    expect(status.text()).toContain('tag refresh failed')

    await status.get('button').trigger('click')
    expect(store.loadTags).toHaveBeenCalledOnce()
  })

  it('renders modal footer actions through the product dialog action row', () => {
    const wrapper = mountModal()

    const footerRow = wrapper.findAllComponents(ProductActionRow)
      .find(row => row.props('ariaLabel') === '标签管理弹窗操作')

    expect(footerRow?.props('variant')).toBe('dialog')
  })

  it('renders the edit-tag form through shared field and action primitives', async () => {
    const wrapper = mountModal()
    const store = useBookshelfStore()
    setTestTags(store, [{ id: 'edit-tag', name: '待编辑标签', color: '#667eea', bookCount: 1 }])
    await wrapper.vm.$nextTick()

    await wrapper.get('.tag-manage-modal__row-edit-action').trigger('click')
    await wrapper.vm.$nextTick()

    expect(wrapper.findAllComponents(UiField).map(field => field.props('label'))).toContain('编辑标签名称')
    expect(wrapper.findAllComponents(UiField).map(field => field.props('label'))).toContain('编辑标签颜色')
    expect(wrapper.findAllComponents(ProductActionRow).some(row => row.props('ariaLabel') === '编辑标签操作')).toBe(true)
    expect(wrapper.find('.edit-name-input').exists()).toBe(false)
    expect(wrapper.find('.edit-color-input').exists()).toBe(false)
  })

  it('keeps tag management hooks under the modal owner', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/bookshelf/TagManageModal.vue'),
      'utf8',
    )

    for (const oldClass of [
      'tag-manage-form',
      'tag-list',
      'tag-list-empty-state',
      'tag-manage-item',
      'tag-view-mode',
      'tag-edit-mode',
      'tag-edit-fields',
      'tag-metadata',
      'tag-row-edit-action',
      'tag-row-delete-action',
      'tag-edit-actions',
      'tag-edit-save-action',
      'tag-edit-cancel-action',
    ]) {
      expect(source).not.toMatch(new RegExp(`class="[^"]*\\b${oldClass}\\b`))
      expect(source).not.toMatch(new RegExp(`\\.${oldClass}\\b`))
    }

    for (const ownerClass of [
      'tag-manage-modal__form',
      'tag-manage-modal__list',
      'tag-manage-modal__empty-state',
      'tag-manage-modal__item',
      'tag-manage-modal__view-mode',
      'tag-manage-modal__edit-mode',
      'tag-manage-modal__edit-fields',
      'tag-manage-modal__metadata',
      'tag-manage-modal__row-edit-action',
      'tag-manage-modal__row-delete-action',
      'tag-manage-modal__edit-actions',
      'tag-manage-modal__edit-save-action',
      'tag-manage-modal__edit-cancel-action',
    ]) {
      expect(source).toContain(ownerClass)
    }
  })
})
