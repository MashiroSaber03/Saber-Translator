import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import ProductSearchField from '@/components/product/ProductSearchField.vue'
import ProductSearchToolbar from '@/components/product/ProductSearchToolbar.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'

describe('ProductSearchToolbar', () => {
  it('renders search and optional filter slots under one product shell', () => {
    const wrapper = mount(ProductSearchToolbar, {
      props: {
        ariaLabel: '书籍搜索和筛选',
      },
      slots: {
        search: '<div data-test="search">搜索书籍</div>',
        filters: '<div data-test="filters">标签筛选</div>',
      },
    })

    expect(wrapper.attributes('aria-label')).toBe('书籍搜索和筛选')
    expect(wrapper.find('.product-search-toolbar__search [data-test="search"]').exists()).toBe(true)
    expect(wrapper.find('.product-search-toolbar__filters [data-test="filters"]').exists()).toBe(true)
  })

  it('omits the filters region when no filters are provided', () => {
    const wrapper = mount(ProductSearchToolbar, {
      slots: {
        search: '<div data-test="search">搜索书籍</div>',
      },
    })

    expect(wrapper.find('.product-search-toolbar__filters').exists()).toBe(false)
  })

  it('renders the search clear action through the shared icon-button primitive', async () => {
    const wrapper = mount(ProductSearchField, {
      props: {
        modelValue: 'Saber',
        clearLabel: '清空书籍搜索',
      },
    })

    const clearAction = wrapper.getComponent(UiIconButton)
    expect(clearAction.props('label')).toBe('清空书籍搜索')
    expect(clearAction.props('title')).toBe('清空书籍搜索')

    await clearAction.trigger('click')

    expect(wrapper.emitted('update:modelValue')).toEqual([['']])
    expect(wrapper.emitted('clear')).toEqual([[]])
  })
})
