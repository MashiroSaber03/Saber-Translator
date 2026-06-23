import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useInsightStore } from '@/stores/insightStore'

const { sendChatMock } = vi.hoisted(() => ({
  sendChatMock: vi.fn(),
}))

vi.mock('@/api/insight', () => ({
  sendChat: sendChatMock,
  rebuildEmbeddings: vi.fn(),
  getRebuildEmbeddingsStatus: vi.fn(),
}))

import QAPanel from '@/components/insight/QAPanel.vue'

describe('QAPanel Markdown rendering', () => {
  beforeEach(() => {
    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    sendChatMock.mockReset()
  })

  it('sanitizes API-provided Markdown HTML before rendering assistant answers', async () => {
    sendChatMock.mockResolvedValue({
      success: true,
      answer: '<img src="x" onerror="alert(1)">[bad](javascript:alert(2))<script>alert(3)</script>**安全文本**',
      citations: [],
      mode: 'precise',
    })

    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'

    const wrapper = mount(QAPanel, {
      global: {
        plugins: [pinia],
      },
    })

    await wrapper.find('textarea').setValue('这页发生了什么？')
    await wrapper.find('.send-btn').trigger('click')
    await flushPromises()

    const answerHtml = wrapper.get('.answer-text').html()
    expect(answerHtml).toContain('<strong>安全文本</strong>')
    expect(answerHtml).not.toContain('onerror')
    expect(answerHtml).not.toContain('javascript:')
    expect(answerHtml).not.toContain('<script')
  })
})
