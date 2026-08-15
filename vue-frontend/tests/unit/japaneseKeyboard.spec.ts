import { mount } from '@vue/test-utils'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'

import JapaneseKeyboard from '@/components/edit/JapaneseKeyboard.vue'
import ProductSegmentedTabs from '@/components/product/ProductSegmentedTabs.vue'
import UiIcon from '@/components/ui/UiIcon.vue'
import UiButton from '@/components/ui/UiButton.vue'
import UiField from '@/components/ui/UiField.vue'
import UiIconButton from '@/components/ui/UiIconButton.vue'
import UiSelect from '@/components/ui/UiSelect.vue'

describe('JapaneseKeyboard', () => {
  it('uses product segmented tabs for kana mode and inserts the selected script', async () => {
    const wrapper = mount(JapaneseKeyboard, {
      props: {
        visible: true,
      },
    })

    const modeTabs = wrapper
      .findAllComponents(ProductSegmentedTabs)
      .find((tabs) => tabs.props('ariaLabel') === '假名字符类型')
    expect(modeTabs).toBeTruthy()
    expect(modeTabs!.props('ariaLabel')).toBe('假名字符类型')
    expect(modeTabs!.props('activeTab')).toBe('hiragana')
    expect(modeTabs!.props('tabs')).toEqual([
      { id: 'hiragana', label: '平假名' },
      { id: 'katakana', label: '片假名' },
    ])

    modeTabs!.vm.$emit('select', 'katakana')
    await wrapper.vm.$nextTick()

    const firstKanaKey = wrapper.findAll('.kana-keyboard__key').find(button => button.text().includes('あ'))
    expect(firstKanaKey).toBeTruthy()

    await firstKanaKey!.trigger('click')

    expect(wrapper.emitted('insert')?.at(-1)).toEqual(['ア', 'original'])
  })

  it('uses a named icon-only close action', async () => {
    const wrapper = mount(JapaneseKeyboard, {
      props: {
        visible: true,
      },
    })

    const closeButton = wrapper.getComponent(UiIconButton)
    expect(closeButton.props('label')).toBe('关闭50音键盘')
    expect(closeButton.props('title')).toBe('关闭')
    expect(closeButton.props('variant')).toBe('inverse')
    expect(closeButton.props('size')).toBe('xs')
    expect(closeButton.props('shape')).toBe('circle')
    expect(closeButton.getComponent(UiIcon).props('name')).toBe('x')

    await closeButton.get('button').trigger('click')

    expect(wrapper.emitted('close')).toHaveLength(1)

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/JapaneseKeyboard.vue'),
      'utf8',
    )
    expect(source).toContain('UiIconButton')
    expect(source).not.toContain('<UiButton\n        variant="toolbar"\n        class="kana-keyboard-close"')
    expect(source).not.toContain('.kana-keyboard-close:hover')
  })

  it('uses the fixed select primitive for insertion target', () => {
    const wrapper = mount(JapaneseKeyboard, {
      props: {
        visible: true,
        defaultTarget: 'translated',
      },
    })

    const targetField = wrapper.getComponent(UiField)
    expect(targetField.props('label')).toBe('输入到：')
    expect(targetField.props('controlId')).toBe('kanaTargetField')
    expect(targetField.props('layout')).toBe('inline')

    const targetSelect = wrapper.getComponent(UiSelect)
    expect(targetSelect.get('button').attributes('id')).toBe('kanaTargetField')
    expect(targetSelect.props('modelValue')).toBe('translated')
    expect(targetSelect.props('options')).toEqual([
      { label: '原文', value: 'original' },
      { label: '译文', value: 'translated' },
    ])

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/JapaneseKeyboard.vue'),
      'utf8',
    )
    expect(source).not.toMatch(/<label\b/)
    expect(source).not.toContain('.kana-target-select label')
  })

  it('uses product category tabs and semantic owner tokens', () => {
    const wrapper = mount(JapaneseKeyboard, {
      props: {
        visible: true,
      },
    })

    const tabGroups = wrapper.findAllComponents(ProductSegmentedTabs)
    expect(tabGroups).toHaveLength(2)
    expect(tabGroups[0]!.props('ariaLabel')).toBe('假名分类')
    expect(tabGroups[0]!.props('tabs')).toEqual([
      { id: 'basic', label: '基本' },
      { id: 'dakuten', label: '浊/半浊音' },
      { id: 'combo', label: '拗音' },
      { id: 'special', label: '特殊' },
    ])

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/JapaneseKeyboard.vue'),
      'utf8',
    )

    expect(source).not.toMatch(/#[0-9A-Fa-f]{3,8}\b|rgba?\(/)
    expect(source).not.toContain('class="kana-tab"')
  })

  it('renders only the selected category content', async () => {
    const wrapper = mount(JapaneseKeyboard, {
      props: { visible: true },
    })
    const categoryTabs = wrapper.findAllComponents(ProductSegmentedTabs)[0]!

    expect(wrapper.findAll('.kana-keyboard__tab-content')).toHaveLength(1)
    expect(wrapper.text()).toContain('あ行')
    expect(wrapper.text()).not.toContain('が行')

    categoryTabs.vm.$emit('select', 'dakuten')
    await wrapper.vm.$nextTick()

    expect(wrapper.findAll('.kana-keyboard__tab-content')).toHaveLength(1)
    expect(wrapper.text()).toContain('が行')
    expect(wrapper.text()).not.toContain('あ行')
  })

  it('keeps keyboard DOM hooks under the kana keyboard owner contract', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/JapaneseKeyboard.vue'),
      'utf8',
    )

    expect(source).toContain('class="kana-keyboard__header"')
    expect(source).toContain('class="kana-keyboard__title"')
    expect(source).toContain('class="kana-keyboard__tabs"')
    expect(source).toContain('class="kana-keyboard__options"')
    expect(source).toContain('class="kana-keyboard__mode-select"')
    expect(source).toContain('class="kana-keyboard__target-select"')
    expect(source).toContain('class="kana-keyboard__tab-content kana-keyboard__tab-content--active"')
    expect(source).toContain('kana-keyboard__tab-content--active')
    expect(source).toContain('class="kana-keyboard__table"')
    expect(source).toContain('kana-keyboard__table--combo')
    expect(source).toContain('class="kana-keyboard__row-label"')
    expect(source).toContain('class="kana-keyboard__key"')
    expect(source).toContain('kana-keyboard__key--pressed')
    expect(source).toContain('kana-keyboard__key--special')
    expect(source).toContain('class="kana-keyboard__hiragana"')
    expect(source).toContain('class="kana-keyboard__katakana"')
    expect(source).toContain('class="kana-keyboard__special-grid"')
    expect(source).toContain('class="kana-keyboard__char-label"')
    expect(source).toContain('class="kana-keyboard__footer"')

    expect(source).not.toContain('class="kana-keyboard-header"')
    expect(source).not.toContain('class="kana-keyboard-title"')
    expect(source).not.toContain('class="kana-keyboard-tabs"')
    expect(source).not.toContain('class="kana-keyboard-options"')
    expect(source).not.toContain('class="kana-mode-select"')
    expect(source).not.toContain('class="kana-target-select"')
    expect(source).not.toContain('class="kana-tab-content"')
    expect(source).not.toContain('class="kana-table"')
    expect(source).not.toContain('class="row-label"')
    expect(source).not.toContain('class="kana-key"')
    expect(source).not.toContain('class="special-chars-grid"')
    expect(source).not.toContain('class="kana-keyboard-footer"')
    expect(source).not.toMatch(/:\s*class="\{\s*active:/)
    expect(source).not.toContain('.kana-key.pressed')
  })

  it('uses the shared danger button variant for backspace instead of local secondary overrides', () => {
    const wrapper = mount(JapaneseKeyboard, {
      props: {
        visible: true,
      },
    })

    const backspaceButton = wrapper
      .findAllComponents(UiButton)
      .find(button => button.text() === '退格')
    expect(backspaceButton).toBeTruthy()
    expect(backspaceButton!.props('variant')).toBe('danger')
    expect(backspaceButton!.props('size')).toBe('sm')

    const source = readFileSync(
      resolve(process.cwd(), 'src/components/edit/JapaneseKeyboard.vue'),
      'utf8',
    )
    expect(source).not.toContain('--japanese-keyboard-backspace')
    expect(source).not.toContain('--ui-button-secondary-')
  })
})
