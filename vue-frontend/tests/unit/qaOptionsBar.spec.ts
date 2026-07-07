import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import QAOptionsBar from '@/components/insight/qa/QAOptionsBar.vue'
import ProductChipList from '@/components/product/ProductChipList.vue'
import ProductSegmentedTabs from '@/components/product/ProductSegmentedTabs.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'
import UiField from '@/components/ui/UiField.vue'
import UiNumberField from '@/components/ui/UiNumberField.vue'
import UiSpinner from '@/components/ui/UiSpinner.vue'

const globalModeProps = {
  globalModeExamples: ['故事的主题是什么？'],
  isRebuildingEmbeddings: false,
  progressLabel: '',
  qaMode: 'global' as const,
  threshold: 0,
  topK: 5,
  useParentChild: true,
  useReasoning: true,
  useReranker: true,
}

const preciseModeProps = {
  ...globalModeProps,
  qaMode: 'precise' as const,
}

function cssRule(style: string, selector: string): string {
  const escapedSelector = selector.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  return style.match(new RegExp(`${escapedSelector}\\s*{([\\s\\S]*?)}`))?.[1] ?? ''
}

describe('QAOptionsBar', () => {
  it('uses explicit button semantics for global-mode example questions', async () => {
    const wrapper = mount(QAOptionsBar, {
      props: globalModeProps,
    })

    const examples = wrapper.getComponent(ProductChipList)
    expect(examples.props('ariaLabel')).toBe('全局模式示例问题')
    expect(examples.props('items')).toEqual([
      {
        id: '故事的主题是什么？',
        label: '故事的主题是什么？',
        ariaLabel: '提问示例：故事的主题是什么？',
        interactive: true,
        tone: 'neutral',
      },
    ])

    examples.vm.$emit('select', '故事的主题是什么？')

    expect(wrapper.emitted('askExample')?.[0]?.[0]).toBe('故事的主题是什么？')
  })

  it('uses the current checkbox primitive for precise-mode boolean options', () => {
    const wrapper = mount(QAOptionsBar, {
      props: preciseModeProps,
    })

    expect(wrapper.findAllComponents(UiCheckbox)).toHaveLength(3)
  })

  it('uses the shared number-field primitive for precise-mode numeric options', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/qa/QAOptionsBar.vue'),
      'utf8',
    )
    expect(source).not.toMatch(/<label\b/)
    expect(source).not.toContain('input-label compact')

    const wrapper = mount(QAOptionsBar, {
      props: preciseModeProps,
    })

    const numericFields = wrapper.findAllComponents(UiField).filter(field => (
      field.props('label') === 'Top K:' || field.props('label') === '阈值:'
    ))
    expect(numericFields.map(field => field.props())).toEqual([
      expect.objectContaining({
        controlId: 'qaTopK',
        label: 'Top K:',
        layout: 'inline',
        variant: 'settings',
      }),
      expect.objectContaining({
        controlId: 'qaThreshold',
        label: '阈值:',
        layout: 'inline',
        variant: 'settings',
      }),
    ])

    const numberFields = wrapper.findAllComponents(UiNumberField)
    expect(numberFields).toHaveLength(2)
    expect(numberFields[0]?.props()).toMatchObject({
      inputId: 'qaTopK',
      min: 1,
      max: 20,
      modelValue: 5,
      size: 'xs',
    })
    expect(numberFields[1]?.props()).toMatchObject({
      inputId: 'qaThreshold',
      min: 0,
      max: 1,
      modelValue: 0,
      step: 0.1,
      size: 'xs',
    })

    numberFields[0]?.vm.$emit('update:modelValue', 8)
    numberFields[1]?.vm.$emit('update:modelValue', 0.4)

    expect(wrapper.emitted('update:topK')?.[0]).toEqual([8])
    expect(wrapper.emitted('update:threshold')?.[0]).toEqual([0.4])
    expect(wrapper.find('.input-small').exists()).toBe(false)
  })

  it('uses shared loading status feedback while rebuilding embeddings', () => {
    const wrapper = mount(QAOptionsBar, {
      props: {
        ...preciseModeProps,
        isRebuildingEmbeddings: true,
        progressLabel: '重建 3/10',
      },
    })

    const spinner = wrapper.getComponent(UiSpinner)
    expect(spinner.props()).toMatchObject({
      decorative: false,
      label: '向量索引重建中',
    })
    const rebuildButton = wrapper.find('button[title="重建向量索引"]')
    expect(rebuildButton.attributes('disabled')).toBeDefined()
    expect(rebuildButton.text()).toContain('重建 3/10')
  })

  it('uses product segmented tabs for the QA mode switch', () => {
    const wrapper = mount(QAOptionsBar, {
      props: preciseModeProps,
    })

    const modeTabs = wrapper.getComponent(ProductSegmentedTabs)
    expect(modeTabs.props('activeTab')).toBe('precise')
    expect(modeTabs.props('ariaLabel')).toBe('问答模式')
    expect(modeTabs.props('tabs')).toEqual([
      { id: 'precise', label: '精确模式', iconName: 'target' },
      { id: 'global', label: '全局模式', iconName: 'globe' },
    ])

    modeTabs.vm.$emit('update:activeTab', 'global')

    expect(wrapper.emitted('update:qaMode')?.[0]?.[0]).toBe('global')
  })

  it('hides decorative dividers from assistive technologies', () => {
    const wrapper = mount(QAOptionsBar, {
      props: preciseModeProps,
    })

    const dividers = wrapper.findAll('.qa-options-bar__divider')
    expect(dividers).toHaveLength(3)
    expect(dividers.every(divider => divider.attributes('aria-hidden') === 'true')).toBe(true)
  })

  it('keeps the compact option bar responsive inside narrow insight panes', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/qa/QAOptionsBar.vue'),
      'utf8',
    )
    const style = source.match(/<style scoped>([\s\S]*)<\/style>/)?.[1] ?? ''

    expect(source).toContain('class="qa-options-bar"')
    expect(source).not.toMatch(/\.(?:chat-options|chat-option-divider|precise-mode-options|global-mode-hint|hint-text|welcome-examples)\b/)
    expect(cssRule(style, '.qa-options-bar')).toContain('flex-wrap: wrap')
    expect(cssRule(style, '.qa-options-bar')).toContain('min-width: 0')
    expect(cssRule(style, '.qa-options-bar__precise-options')).toContain('min-width: 0')
    expect(cssRule(style, '.qa-options-bar__global-hint')).toContain('min-width: 0')
    expect(cssRule(style, '.qa-options-bar__example-list')).toContain('max-width: 100%')
  })
})
