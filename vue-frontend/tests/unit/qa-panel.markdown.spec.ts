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

function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(res => {
    resolve = res
  })
  return { promise, resolve }
}

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

  it('ignores stale chat responses after switching books', async () => {
    const staleResponse = deferred<{
      success: true
      answer: string
      citations: never[]
      mode: 'precise'
    }>()
    sendChatMock.mockReturnValueOnce(staleResponse.promise)

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
    expect(sendChatMock).toHaveBeenCalledWith(
      'book-1',
      '这页发生了什么？',
      expect.any(Object)
    )

    store.currentBookId = 'book-2'
    staleResponse.resolve({
      success: true,
      answer: 'book-1 stale answer',
      citations: [],
      mode: 'precise',
    })
    await flushPromises()

    expect(wrapper.text()).not.toContain('book-1 stale answer')
    expect(store.qaHistory.some(message => message.isLoading)).toBe(false)
    expect(store.isStreaming).toBe(false)
  })
})
