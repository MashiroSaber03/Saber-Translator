import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { mount } from '@vue/test-utils'
import { defineComponent, h } from 'vue'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

vi.mock('@/components/common/BaseModal.vue', () => ({
  default: defineComponent({
    name: 'BaseModalStub',
    props: ['modelValue', 'title', 'size', 'customClass'],
    emits: ['update:modelValue', 'close'],
    setup(props, { slots }) {
      return () => props.modelValue
        ? h('div', { class: ['base-modal-stub', props.customClass] }, [
            h('div', { class: 'base-modal-title' }, props.title),
            h('div', { class: 'base-modal-body' }, slots.default ? slots.default() : []),
            h('div', { class: 'base-modal-footer' }, slots.footer ? slots.footer() : []),
          ])
        : null
    },
  }),
}))

import PageSelectionModal from '@/components/translate/PageSelectionModal.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductBreadcrumbTrail from '@/components/product/ProductBreadcrumbTrail.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import ProductFolderCard from '@/components/product/ProductFolderCard.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import VirtualThumbnailGrid from '@/components/virtual/VirtualThumbnailGrid.vue'
import UiButton from '@/components/ui/UiButton.vue'
import { useImageStore } from '@/stores/imageStore'
import { setTestImages } from '../helpers/imageFixtures'

describe('PageSelectionModal', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    const imageStore = useImageStore()
    imageStore.clearImages()
    setTestImages(imageStore, Array.from({ length: 4 }, (_, index) => ({
      fileName: `${String(index + 1).padStart(3, '0')}.png`,
      sourceAssetUrl: `data:image/png;base64,page-${index + 1}`,
    })))
    imageStore.updateImageByIndex(1, {
      translationStatus: 'completed',
      translatedAssetUrl: '/api/v2/assets/done',
    })
    imageStore.updateImageByIndex(2, {
      translationStatus: 'failed',
    })
  })

  it('toggles draft selection on thumbnail click and emits sorted pages on confirm', async () => {
    const wrapper = mount(PageSelectionModal, {
      props: {
        modelValue: true,
        selectedPages: [2],
      },
    })

    const thumbnailGrid = wrapper.getComponent(VirtualThumbnailGrid)
    expect(thumbnailGrid.props('ariaLabel')).toBe('选择翻译页码')
    expect(thumbnailGrid.props('items')).toMatchObject([
      {
        id: 0,
        label: '1',
        selected: false,
      },
      {
        id: 1,
        label: '2',
        selected: true,
        selectedBadge: '已选',
        marked: true,
      },
      {
        id: 2,
        label: '3',
        selected: false,
        cornerLabel: '!',
      },
      {
        id: 3,
        label: '4',
        selected: false,
      },
    ])
    expect(wrapper.find('.page-selection-thumbnail').exists()).toBe(false)

    thumbnailGrid.vm.$emit('select', 0)
    thumbnailGrid.vm.$emit('select', 2)
    thumbnailGrid.vm.$emit('select', 1)

    await wrapper.find('[data-testid="confirm-page-selection-button"]').trigger('click')

    expect(wrapper.emitted('confirm')?.[0]).toEqual([[1, 3]])
  })

  it('replaces draft selection with failed pages when filter shortcut is clicked', async () => {
    const wrapper = mount(PageSelectionModal, {
      props: {
        modelValue: true,
        selectedPages: [1, 2],
      },
    })

    const failedPagesButton = wrapper.findAllComponents(UiButton)
      .find(button => button.text() === '失败页')
    expect(failedPagesButton).toBeTruthy()

    await failedPagesButton?.trigger('click')
    await wrapper.find('[data-testid="confirm-page-selection-button"]').trigger('click')

    expect(wrapper.emitted('confirm')?.[0]).toEqual([[3]])
  })

  it('uses product summary chips and action rows for the modal chrome', () => {
    const wrapper = mount(PageSelectionModal, {
      props: {
        modelValue: true,
        selectedPages: [1, 2],
      },
    })

    const statusBanner = wrapper.getComponent(ProductStatusBanner)
    expect(statusBanner.props('tone')).toBe('neutral')
    expect(statusBanner.props('title')).toBe('页码选择')

    const chipList = wrapper.getComponent(ProductChipList)
    expect(chipList.props('ariaLabel')).toBe('页码选择统计')
    expect(chipList.props('items')).toEqual([
      { id: 'total', label: '共 4 张', tone: 'neutral' },
      { id: 'selected', label: '已选 2 张', tone: 'primary' },
    ])

    const actionRows = wrapper.findAllComponents(ProductActionRow)
    expect(actionRows.map(row => row.props('ariaLabel'))).toEqual([
      '页码选择快捷操作',
      '指定翻译页码操作',
    ])
    expect(actionRows.map(row => row.props('variant'))).toEqual(['default', 'dialog'])

    const shortcutButtons = wrapper
      .findAllComponents(UiButton)
      .filter(button => ['全选', '清空', '失败页', '未翻译页', '已翻译页', '手动标注页'].includes(button.text()))
    expect(shortcutButtons.map(button => button.props('variant'))).toEqual([
      'secondary',
      'secondary',
      'danger',
      'secondary',
      'secondary',
      'secondary',
    ])

    expect(wrapper.find('.page-selection-summary-card').exists()).toBe(false)
    expect(wrapper.find('.page-selection-toolbar-card').exists()).toBe(false)
    expect(wrapper.find('.page-selection-toolbar-btn').exists()).toBe(false)
    expect(wrapper.find('.page-selection-footer-btn').exists()).toBe(false)
  })

  it('keeps the page browser card styles in a single owner block', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/PageSelectionModal.vue'), 'utf8')
    const styleBlock = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''
    const browserCardBlocks = styleBlock.match(/^\.page-selection-browser-card\s*\{/gm) ?? []

    expect(browserCardBlocks).toHaveLength(1)
  })

  it('replaces draft selection from directly entered pages and ranges', async () => {
    const wrapper = mount(PageSelectionModal, {
      props: {
        modelValue: true,
        selectedPages: [2],
      },
    })

    await wrapper.get('#page-selection-input').setValue('1,3-4')
    await wrapper.get('[data-testid="apply-page-selection-input"]').trigger('click')
    await wrapper.get('[data-testid="confirm-page-selection-button"]').trigger('click')

    expect(wrapper.emitted('confirm')?.[0]).toEqual([[1, 3, 4]])
  })

  it('keeps the previous draft selection when direct page input is invalid', async () => {
    const wrapper = mount(PageSelectionModal, {
      props: {
        modelValue: true,
        selectedPages: [2],
      },
    })

    await wrapper.get('#page-selection-input').setValue('1,5')
    await wrapper.get('[data-testid="apply-page-selection-input"]').trigger('click')

    expect(wrapper.text()).toContain('页码 5 超出当前总页数 4')
    await wrapper.get('[data-testid="confirm-page-selection-button"]').trigger('click')
    expect(wrapper.emitted('confirm')?.[0]).toEqual([[2]])
  })

  it('emits one close update for one modal close request', () => {
    const wrapper = mount(PageSelectionModal, {
      props: { modelValue: true, selectedPages: [] },
    })

    wrapper.getComponent({ name: 'BaseModalStub' }).vm.$emit('close')

    expect(wrapper.emitted('update:modelValue')).toEqual([[false]])
  })

  it('uses button semantics for folder navigation controls', async () => {
    const imageStore = useImageStore()
    imageStore.updateImageByIndex(0, {
      folderPath: 'chapter-a',
    })
    imageStore.updateImageByIndex(1, {
      folderPath: 'chapter-a',
    })

    const wrapper = mount(PageSelectionModal, {
      props: {
        modelValue: true,
        selectedPages: [],
      },
    })

    const folderCard = wrapper.getComponent(ProductFolderCard)
    expect(folderCard.props()).toMatchObject({
      ariaLabel: '打开文件夹 chapter-a',
      count: 2,
      countId: 'chapter-a',
      folderName: 'chapter-a',
    })
    expect(wrapper.find('.page-selection-folder-card.ui-button').exists()).toBe(false)
    const folderCount = wrapper.findAllComponents(ProductChipList)
      .find(chipList => chipList.props('ariaLabel') === 'chapter-a 文件夹图片数量')
    expect(folderCount?.props('items')).toEqual([
      {
        id: 'chapter-a',
        label: '2 张',
        tone: 'neutral',
      },
    ])

    folderCard.vm.$emit('select')
    await wrapper.vm.$nextTick()

    expect(wrapper.get('.page-selection-folder-back-button').element.tagName).toBe('BUTTON')
    const backButton = wrapper.findAllComponents(UiButton)
      .find(button => button.text().includes('返回上级'))
    expect(backButton?.props('variant')).toBe('secondary')
    expect(backButton?.props('tone')).toBe('primary')
    expect(backButton?.props('size')).toBe('sm')
    expect(backButton?.props('block')).toBe(true)

    const source = readFileSync(resolve(process.cwd(), 'src/components/translate/PageSelectionModal.vue'), 'utf8')
    expect(source).toContain('ProductFolderCard')
    expect(source).toContain(':count-id="subfolder.path"')
    expect(source).not.toContain('ProductRecordCard')
    expect(source).not.toContain('folderImageCountChips')
    expect(source).not.toContain('folder-count')
    expect(source).not.toContain('class="folder-info"')
    expect(source).not.toContain('class="folder-name"')
    expect(source).toContain('class="page-selection-folder-back-button"')
    expect(source).toContain('class="page-selection-folder-back-icon"')
    expect(source).not.toContain('class="page-selection-folder-back-btn"')
    expect(source).not.toContain('class="back-icon"')
    expect(source).not.toContain('page-selection-filter-failed')
    const backButtonBlock = source.match(/\.page-selection-folder-back-button\s*\{([\s\S]*?)\n\}/)?.[1] ?? ''
    expect(backButtonBlock).not.toMatch(/\b(background|border|padding|width|font|color|cursor|transition):/)

    const breadcrumbTrail = wrapper.getComponent(ProductBreadcrumbTrail)
    expect(breadcrumbTrail.props('items')).toMatchObject([
      { path: '', name: '根目录' },
      { path: 'chapter-a', name: 'chapter-a' },
    ])
    breadcrumbTrail.vm.$emit('select', '')

    const thumbnailGrid = wrapper.getComponent(VirtualThumbnailGrid)
    expect(thumbnailGrid.props('items')).toMatchObject([
      {
        id: 0,
        label: '1',
      },
      {
        id: 1,
        label: '2',
      },
    ])
  })
})
