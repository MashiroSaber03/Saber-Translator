import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import QAMessageItem from '@/components/insight/qa/QAMessageItem.vue'
import type { QAMessage } from '@/stores/insightStore'

const assistantMessage: QAMessage = {
  id: 'qa-message-1',
  role: 'assistant',
  content: '第 5 页揭示了关键线索。',
  timestamp: '2026-05-21T10:00:00Z',
  citations: [{ page: 5 }],
}

describe('QAMessageItem', () => {
  it('uses explicit button semantics for citation navigation', async () => {
    const wrapper = mount(QAMessageItem, {
      props: {
        message: assistantMessage,
        renderMarkdown: content => `<p>${content}</p>`,
      },
    })

    const citationButton = wrapper.find('.citation-item')
    expect(citationButton.element.tagName).toBe('BUTTON')
    expect(citationButton.attributes('aria-label')).toBe('查看第 5 页')

    await citationButton.trigger('click')

    expect(wrapper.emitted('selectPage')?.[0]?.[0]).toBe(5)
  })
})
