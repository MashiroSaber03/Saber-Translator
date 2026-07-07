import { mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

import NotesToolbar from '@/components/insight/notes/NotesToolbar.vue'
import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'
import UiSelect from '@/components/ui/UiSelect.vue'

describe('NotesToolbar', () => {
  it('uses the fixed select primitive for note filtering', () => {
    const wrapper = mount(NotesToolbar, {
      props: {
        filter: 'all',
        filterOptions: [
          { label: '全部笔记', value: 'all' },
          { label: '文本笔记', value: 'text' },
          { label: '问答笔记', value: 'qa' },
        ],
      },
    })

    const filterSelect = wrapper.getComponent(UiSelect)
    expect(filterSelect.props('modelValue')).toBe('all')
    expect(filterSelect.attributes('aria-label')).toBe('筛选笔记类型')
    expect(filterSelect.props('options')).toEqual(expect.arrayContaining([
      expect.objectContaining({ value: 'text' }),
      expect.objectContaining({ value: 'qa' }),
    ]))

    filterSelect.vm.$emit('change', 'qa')

    expect(wrapper.emitted('update:filter')?.[0]).toEqual(['qa'])
  })

  it('uses the product section header for the notes filter toolbar', () => {
    const wrapper = mount(NotesToolbar, {
      props: {
        filter: 'all',
        filterOptions: [
          { label: '全部笔记', value: 'all' },
          { label: '文本笔记', value: 'text' },
        ],
      },
    })
    const source = readFileSync(resolve(process.cwd(), 'src/components/insight/notes/NotesToolbar.vue'), 'utf8')
    const header = wrapper.getComponent(ProductSectionHeader)

    expect(header.props()).toMatchObject({
      title: '笔记',
      iconName: 'file-text',
      size: 'sm',
    })
    expect(header.findComponent(UiSelect).exists()).toBe(true)
    expect(source).toContain("import ProductSectionHeader from '@/components/product/ProductSectionHeader.vue'")
    expect(source).not.toContain('section-header-with-actions')
    expect(source).not.toContain('class="section-title"')
    expect(source).not.toContain('.section-title')
  })
})
