import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'
import { mount } from '@vue/test-utils'
import { defineComponent, h } from 'vue'

vi.mock('@/components/common/BaseModal.vue', () => ({
  default: defineComponent({
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
import { useImageStore } from '@/stores/imageStore'

describe('PageSelectionModal', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    const imageStore = useImageStore()
    imageStore.clearImages()
    imageStore.addImage('001.png', 'data:image/png;base64,aaa')
    imageStore.addImage('002.png', 'data:image/png;base64,bbb')
    imageStore.addImage('003.png', 'data:image/png;base64,ccc')
    imageStore.addImage('004.png', 'data:image/png;base64,ddd')
    imageStore.updateImageByIndex(1, {
      translationStatus: 'completed',
      translatedDataURL: 'data:image/png;base64,done',
    })
    imageStore.updateImageByIndex(2, {
      translationStatus: 'failed',
      translationFailed: true,
    })
  })

  it('toggles draft selection on thumbnail click and emits sorted pages on confirm', async () => {
    const wrapper = mount(PageSelectionModal, {
      props: {
        modelValue: true,
        selectedPages: [2],
      },
    })

    const items = wrapper.findAll('.page-selection-thumbnail')
    expect(items).toHaveLength(4)

    await items[0]?.trigger('click')
    await items[2]?.trigger('click')
    await items[1]?.trigger('click')

    await wrapper.find('.page-selection-confirm-btn').trigger('click')

    expect(wrapper.emitted('confirm')?.[0]).toEqual([[1, 3]])
  })

  it('replaces draft selection with failed pages when filter shortcut is clicked', async () => {
    const wrapper = mount(PageSelectionModal, {
      props: {
        modelValue: true,
        selectedPages: [1, 2],
      },
    })

    await wrapper.find('.page-selection-filter-failed').trigger('click')
    await wrapper.find('.page-selection-confirm-btn').trigger('click')

    expect(wrapper.emitted('confirm')?.[0]).toEqual([[3]])
  })

  it('uses button semantics for folder navigation controls', async () => {
    const imageStore = useImageStore()
    imageStore.updateImageByIndex(0, {
      relativePath: 'chapter-a/001.png',
      folderPath: 'chapter-a',
    })
    imageStore.updateImageByIndex(1, {
      relativePath: 'chapter-a/002.png',
      folderPath: 'chapter-a',
    })

    const wrapper = mount(PageSelectionModal, {
      props: {
        modelValue: true,
        selectedPages: [],
      },
    })

    const folderButton = wrapper.get('.page-selection-folder-item')
    expect(folderButton.element.tagName).toBe('BUTTON')

    await folderButton.trigger('click')

    expect(wrapper.get('.page-selection-folder-back-btn').element.tagName).toBe('BUTTON')
    expect(wrapper.get('.page-selection-breadcrumb-item').element.tagName).toBe('BUTTON')
    expect(wrapper.get('.page-selection-thumbnail').element.tagName).toBe('BUTTON')
  })
})
