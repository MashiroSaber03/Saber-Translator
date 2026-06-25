import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import QAOptionsBar from '@/components/insight/qa/QAOptionsBar.vue'
import UiCheckbox from '@/components/ui/UiCheckbox.vue'

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

describe('QAOptionsBar', () => {
  it('uses explicit button semantics for global-mode example questions', async () => {
    const wrapper = mount(QAOptionsBar, {
      props: globalModeProps,
    })

    const exampleButton = wrapper.find('.example-tag')
    expect(exampleButton.element.tagName).toBe('BUTTON')
    expect(exampleButton.attributes('aria-label')).toBe('提问示例：故事的主题是什么？')

    await exampleButton.trigger('click')

    expect(wrapper.emitted('askExample')?.[0]?.[0]).toBe('故事的主题是什么？')
  })

  it('uses the current checkbox primitive for precise-mode boolean options', () => {
    const wrapper = mount(QAOptionsBar, {
      props: preciseModeProps,
    })

    expect(wrapper.findAllComponents(UiCheckbox)).toHaveLength(3)
  })
})
