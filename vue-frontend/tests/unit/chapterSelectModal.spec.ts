import { mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import ChapterSelectModal from '@/components/insight/ChapterSelectModal.vue'
import ProductActionRow from '@/components/product/ProductActionRow.vue'
import ProductRecordCard from '@/components/product/ProductRecordCard.vue'

function mountModal() {
  return mount(ChapterSelectModal, {
    props: {
      chapters: [
        { id: 'chapter-1', title: '第一章', startPage: 1, endPage: 12 },
        { id: 'chapter-2', title: '第二章', startPage: 13, endPage: 24 },
      ],
    },
    global: {
      stubs: {
        BaseModal: {
          props: {
            modelValue: {
              type: Boolean,
              default: undefined,
            },
          },
          template: '<section data-testid="base-modal" :data-model-value="String(modelValue)"><slot /><footer><slot name="footer" /></footer></section>',
        },
      },
    },
  })
}

describe('ChapterSelectModal', () => {
  it('renders chapter choices as buttons and emits the confirmed chapter', async () => {
    const wrapper = mountModal()

    const chapterItems = wrapper.findAllComponents(ProductRecordCard)
    expect(chapterItems).toHaveLength(2)
    expect(chapterItems.map(item => item.props('as'))).toEqual(['button', 'button'])

    await chapterItems[1]!.trigger('click')
    expect(wrapper.get('button[aria-pressed="true"]').text()).toContain('第二章')

    const confirmButton = wrapper.findAll('button').find(button => button.text() === '确定')
    expect(confirmButton).toBeTruthy()
    await confirmButton!.trigger('click')

    expect(wrapper.emitted('select')).toEqual([[ 'chapter-2' ]])
  })

  it('passes an explicit open state to the modal shell', () => {
    const wrapper = mountModal()

    expect(wrapper.get('[data-testid="base-modal"]').attributes('data-model-value')).toBe('true')
  })

  it('renders confirmation actions through the product dialog action row', () => {
    const wrapper = mountModal()

    const actionRow = wrapper.getComponent(ProductActionRow)

    expect(actionRow.props('variant')).toBe('dialog')
    expect(actionRow.props('ariaLabel')).toBe('章节选择操作')
  })

  it('does not confirm a chapter that disappeared from the current chapter list', async () => {
    const wrapper = mountModal()
    await wrapper.findAllComponents(ProductRecordCard)[1]!.trigger('click')
    await wrapper.setProps({
      chapters: [{ id: 'chapter-1', title: '第一章', startPage: 1, endPage: 12 }],
    })

    const confirmButton = wrapper.findAll('button').find(button => button.text() === '确定')
    expect(confirmButton?.attributes('disabled')).toBeDefined()
    await confirmButton!.trigger('click')
    expect(wrapper.emitted('select')).toBeUndefined()
  })

  it('renders explicit zero page boundaries without truthiness fallback', async () => {
    const wrapper = mountModal()
    await wrapper.setProps({
      chapters: [{ id: 'chapter-0', title: '空章节', startPage: 0, endPage: 0 }],
    })

    expect(wrapper.text()).toContain('第 0-0 页')
  })

  it('keeps chapter select modal hooks under the modal owner', () => {
    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/ChapterSelectModal.vue'), 'utf8')
    const oldHooks = [
      'chapter-select-body',
      'hint-text',
      'chapters-list',
      'chapter-choice-card',
      'chapter-title',
      'check-icon',
    ]

    for (const hook of oldHooks) {
      const escapedHook = hook.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
      expect(source).not.toMatch(new RegExp(`(?<![\\w-])${escapedHook}(?![\\w-])`))
    }
    expect(source).toContain('chapter-select-modal__body')
    expect(source).toContain('chapter-select-modal__choice-card')
    expect(source).toContain('chapter-select-modal__check-icon')
  })
})
