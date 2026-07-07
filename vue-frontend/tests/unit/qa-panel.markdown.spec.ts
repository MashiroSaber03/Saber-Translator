import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useInsightStore } from '@/stores/insightStore'
import ProductComposer from '@/components/product/ProductComposer.vue'
import ProductScrollStack from '@/components/product/ProductScrollStack.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'

const { sendChatMock } = vi.hoisted(() => ({
  sendChatMock: vi.fn(),
}))

vi.mock('@/api/insight', () => ({
  sendChat: sendChatMock,
  rebuildEmbeddings: vi.fn(),
  getRebuildEmbeddingsStatus: vi.fn(),
}))

import QAPanel from '@/components/insight/QAPanel.vue'
import QAMessageList from '@/components/insight/qa/QAMessageList.vue'

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

    const composer = wrapper.getComponent(ProductComposer)
    await composer.get('textarea').setValue('这页发生了什么？')
    await composer.get('button').trigger('click')
    await flushPromises()

    const answerHtml = wrapper.get('.qa-message-item__answer-text').html()
    expect(answerHtml).toContain('<strong>安全文本</strong>')
    expect(answerHtml).not.toContain('onerror')
    expect(answerHtml).not.toContain('javascript:')
    expect(answerHtml).not.toContain('<script')
  })

  it('renders messages through the product scroll stack contract', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/qa/QAMessageList.vue'),
      'utf8',
    )
    expect(source).toContain('ProductStatusBanner')
    expect(source).not.toContain('welcome-message')
    expect(source).not.toContain('welcome-icon')

    const wrapper = mount(QAMessageList, {
      props: {
        messages: [],
        renderMarkdown: (content: string) => content,
      },
    })

    const stack = wrapper.getComponent(ProductScrollStack)
    expect(stack.props('role')).toBe('log')
    expect(stack.props('ariaLabel')).toBe('问答消息')
    expect(stack.props('ariaLive')).toBe('polite')
    const emptyState = wrapper.getComponent(ProductStatusBanner)
    expect(emptyState.props()).toMatchObject({
      iconName: 'message',
      role: 'note',
      title: '智能问答',
      tone: 'neutral',
    })
    expect(wrapper.text()).toContain('针对已分析的漫画内容提问，获取精准回答')
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

    const composer = wrapper.getComponent(ProductComposer)
    await composer.get('textarea').setValue('这页发生了什么？')
    await composer.get('button').trigger('click')
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

  it('uses owner-prefixed hooks for the QA panel and message list shells', () => {
    const panelSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/QAPanel.vue'),
      'utf8',
    )
    const listSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/qa/QAMessageList.vue'),
      'utf8',
    )

    expect(panelSource).toContain('class="qa-panel"')
    expect(panelSource).toContain('qa-panel__input-shell')
    expect(panelSource).not.toMatch(/\.(?:qa-container|chat-input-container)\b/)
    expect(listSource).toContain('class="qa-message-list"')
    expect(listSource).not.toContain('class="chat-messages"')
  })

  it('requests message-list scrolling through a typed prop instead of a child expose', () => {
    const panelSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/QAPanel.vue'),
      'utf8',
    )
    const listSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/qa/QAMessageList.vue'),
      'utf8',
    )

    expect(panelSource).toContain('messageScrollRequestId')
    expect(panelSource).toContain(':scroll-request-id="messageScrollRequestId"')
    expect(panelSource).not.toContain('ref="messageList"')
    expect(panelSource).not.toContain('InstanceType<typeof QAMessageList>')

    expect(listSource).toContain('scrollRequestId')
    expect(listSource).not.toContain('defineExpose')
  })
})
