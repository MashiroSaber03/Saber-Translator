import { flushPromises, mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'

import { useInsightStore } from '@/stores/insightStore'
import ProductComposer from '@/components/product/ProductComposer.vue'
import ProductScrollStack from '@/components/product/ProductScrollStack.vue'
import ProductStatusBanner from '@/components/product/ProductStatusBanner.vue'
import QAOptionsBar from '@/components/insight/qa/QAOptionsBar.vue'

const { getQAStatusMock, sendChatMock } = vi.hoisted(() => ({
  getQAStatusMock: vi.fn(),
  sendChatMock: vi.fn(),
}))

vi.mock('@/api/insight', () => ({
  getQAStatus: getQAStatusMock,
  sendChat: sendChatMock,
  rebuildEmbeddings: vi.fn(),
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
    getQAStatusMock.mockReset()
    getQAStatusMock.mockResolvedValue({
      available: true,
      reason: null,
    })
    sendChatMock.mockReset()
  })

  it('sanitizes API-provided Markdown HTML before rendering assistant answers', async () => {
    sendChatMock.mockResolvedValue({
      success: true,
      answer:
        '<img src="x" onerror="alert(1)">[bad](javascript:alert(2))<script>alert(3)</script>**安全文本**',
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
    await flushPromises()

    const composer = wrapper.getComponent(ProductComposer)
    await composer.get('textarea').setValue('这页发生了什么？')
    expect(composer.get('button').attributes('disabled')).toBeUndefined()
    await composer.get('button').trigger('click')
    await flushPromises()

    const answerHtml = wrapper.get('.qa-message-item__answer-text').html()
    expect(answerHtml).toContain('<strong>安全文本</strong>')
    expect(answerHtml).not.toContain('onerror')
    expect(answerHtml).not.toContain('javascript:')
    expect(answerHtml).not.toContain('<script')
  })

  it('projects coalesced SSE chunks before the final answer arrives', async () => {
    const finalResponse = deferred<{
      answer: string
      citations: never[]
      mode: 'precise'
    }>()
    sendChatMock.mockImplementationOnce(
      (_bookId: string, _question: string, options: { onChunk: (content: string) => void }) => {
        options.onChunk('正在流式返回的片段')
        return finalResponse.promise
      }
    )

    const pinia = createPinia()
    setActivePinia(pinia)
    const store = useInsightStore()
    store.currentBookId = 'book-1'
    const wrapper = mount(QAPanel, {
      global: { plugins: [pinia] },
    })
    await flushPromises()

    const composer = wrapper.getComponent(ProductComposer)
    await composer.get('textarea').setValue('流式回答测试')
    await composer.get('button').trigger('click')
    await new Promise(resolve => setTimeout(resolve, 25))
    await flushPromises()

    expect(wrapper.text()).toContain('正在流式返回的片段')
    expect(store.isStreaming).toBe(true)

    finalResponse.resolve({
      answer: '最终回答',
      citations: [],
      mode: 'precise',
    })
    await flushPromises()
    expect(wrapper.text()).toContain('最终回答')
    expect(store.isStreaming).toBe(false)
  })

  it('renders messages through the product scroll stack contract', () => {
    const source = readFileSync(
      resolve(process.cwd(), 'src/components/insight/qa/QAMessageList.vue'),
      'utf8'
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
    expect(emptyState.get('.product-status-banner__icon-text').text()).toBe('💬')
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
    expect(wrapper.getComponent(ProductComposer).props('showSubmitIcon')).toBe(false)
    await flushPromises()

    const composer = wrapper.getComponent(ProductComposer)
    await composer.get('textarea').setValue('这页发生了什么？')
    expect(composer.get('button').attributes('disabled')).toBeUndefined()
    await composer.get('button').trigger('click')
    await flushPromises()
    expect(sendChatMock).toHaveBeenCalledWith('book-1', '这页发生了什么？', expect.any(Object))
    const signal = sendChatMock.mock.calls[0]?.[2]?.signal as AbortSignal
    expect(signal.aborted).toBe(false)

    store.currentBookId = 'book-2'
    await nextTick()
    expect(signal.aborted).toBe(true)
    expect(store.qaHistory).toEqual([])
    staleResponse.resolve({
      success: true,
      answer: 'book-1 stale answer',
      citations: [],
      mode: 'precise',
    })
    await flushPromises()

    expect(wrapper.text()).not.toContain('book-1 stale answer')
    expect(store.qaHistory).toEqual([])
    expect(store.isStreaming).toBe(false)
  })

  it('aborts the current answer when the QA mode changes', async () => {
    const staleResponse = deferred<{
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
      global: { plugins: [pinia] },
    })
    await flushPromises()

    const composer = wrapper.getComponent(ProductComposer)
    await composer.get('textarea').setValue('精确模式问题')
    await composer.get('button').trigger('click')
    await flushPromises()
    const signal = sendChatMock.mock.calls[0]?.[2]?.signal as AbortSignal
    expect(signal.aborted).toBe(false)

    wrapper.getComponent(QAOptionsBar).vm.$emit('update:qaMode', 'global')
    await nextTick()
    expect(signal.aborted).toBe(true)
    expect(store.qaHistory).toEqual([])

    staleResponse.resolve({ answer: '过期回答', citations: [], mode: 'precise' })
    await flushPromises()
    expect(wrapper.text()).not.toContain('过期回答')
    expect(store.isStreaming).toBe(false)
  })

  it('uses owner-prefixed hooks for the QA panel and message list shells', () => {
    const panelSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/QAPanel.vue'),
      'utf8'
    )
    const listSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/qa/QAMessageList.vue'),
      'utf8'
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
      'utf8'
    )
    const listSource = readFileSync(
      resolve(process.cwd(), 'src/components/insight/qa/QAMessageList.vue'),
      'utf8'
    )

    expect(panelSource).toContain('messageScrollRequestId')
    expect(panelSource).toContain(':scroll-request-id="messageScrollRequestId"')
    expect(panelSource).not.toContain('ref="messageList"')
    expect(panelSource).not.toContain('InstanceType<typeof QAMessageList>')

    expect(listSource).toContain('scrollRequestId')
    expect(listSource).not.toContain('defineExpose')
  })
})
