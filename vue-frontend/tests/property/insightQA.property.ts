import { describe, expect, it } from 'vitest'
import * as fc from 'fast-check'
import { setActivePinia, createPinia } from 'pinia'
import { useInsightStore, type QAMessage } from '@/stores/insightStore'

function createStore(): ReturnType<typeof useInsightStore> {
  setActivePinia(createPinia())
  return useInsightStore()
}

function addMessages(store: ReturnType<typeof useInsightStore>, messages: QAMessage[]): void {
  for (const message of messages) {
    store.addQAMessage(message)
  }
}

const messageIdArbitrary = fc.stringOf(fc.constantFrom(...'0123456789'.split('')), {
  minLength: 1,
  maxLength: 20,
})
const messageRoleArbitrary = fc.constantFrom<'user' | 'assistant'>('user', 'assistant')
const messageContentArbitrary = fc.string({ minLength: 0, maxLength: 500 })
const isoDateArbitrary = fc.date().map(date => date.toISOString())

const qaMessageArbitrary: fc.Arbitrary<QAMessage> = fc.record({
  id: messageIdArbitrary,
  role: messageRoleArbitrary,
  content: messageContentArbitrary,
  timestamp: isoDateArbitrary,
})

const streamChunkArbitrary = fc.string({ minLength: 1, maxLength: 100 })

describe('insight QA properties', () => {
  it('appends messages in the order they are added', () => {
    fc.assert(
      fc.property(fc.array(qaMessageArbitrary, { minLength: 0, maxLength: 20 }), messages => {
        const store = createStore()

        addMessages(store, messages)

        expect(store.qaHistory).toEqual(messages)
        expect(store.qaHistory).toHaveLength(messages.length)
      }),
    )
  })

  it('accumulates streamed assistant content through the last assistant message', () => {
    fc.assert(
      fc.property(fc.array(streamChunkArbitrary, { minLength: 1, maxLength: 20 }), chunks => {
        const store = createStore()
        store.addQAMessage({
          id: 'assistant-1',
          role: 'assistant',
          content: '',
          timestamp: new Date().toISOString(),
        })

        let accumulatedContent = ''
        for (const chunk of chunks) {
          accumulatedContent += chunk
          store.updateLastAssistantMessage(accumulatedContent)
        }

        expect(store.qaHistory.at(-1)?.content).toBe(accumulatedContent)
        expect(store.qaHistory.at(-1)?.content).toBe(chunks.join(''))
      }),
    )
  })

  it('updates only the final assistant message', () => {
    fc.assert(
      fc.property(
        fc.array(qaMessageArbitrary, { minLength: 2, maxLength: 10 }),
        messageContentArbitrary,
        (messages, newContent) => {
          const store = createStore()
          const messagesWithAssistantLast = messages.map((message, index) =>
            index === messages.length - 1 ? { ...message, role: 'assistant' as const } : message,
          )

          addMessages(store, messagesWithAssistantLast)
          const previousMessages = store.qaHistory.slice(0, -1).map(message => ({ ...message }))

          store.updateLastAssistantMessage(newContent)

          expect(store.qaHistory.slice(0, -1)).toEqual(previousMessages)
          expect(store.qaHistory.at(-1)?.content).toBe(newContent)
        },
      ),
    )
  })

  it('ignores last-message updates when the final message is not from the assistant', () => {
    fc.assert(
      fc.property(
        fc.array(qaMessageArbitrary, { minLength: 1, maxLength: 10 }),
        messageContentArbitrary,
        (messages, newContent) => {
          const store = createStore()
          const messagesWithUserLast = messages.map((message, index) =>
            index === messages.length - 1 ? { ...message, role: 'user' as const } : message,
          )

          addMessages(store, messagesWithUserLast)
          const previousMessages = store.qaHistory.map(message => ({ ...message }))

          store.updateLastAssistantMessage(newContent)

          expect(store.qaHistory).toEqual(previousMessages)
        },
      ),
    )
  })

  it('clears all QA history entries', () => {
    fc.assert(
      fc.property(fc.array(qaMessageArbitrary, { minLength: 0, maxLength: 20 }), messages => {
        const store = createStore()
        addMessages(store, messages)

        store.clearQAHistory()

        expect(store.qaHistory).toEqual([])
      }),
    )
  })

  it('keeps streaming state equal to the last assigned value', () => {
    fc.assert(
      fc.property(fc.array(fc.boolean(), { minLength: 1, maxLength: 20 }), streamingSequence => {
        const store = createStore()

        for (const streaming of streamingSequence) {
          store.setStreaming(streaming)
        }

        expect(store.isStreaming).toBe(streamingSequence[streamingSequence.length - 1])
      }),
    )
  })

  it('keeps generated user and assistant pairs in alternating order', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 10 }),
        messageContentArbitrary,
        messageContentArbitrary,
        (pairCount, userContent, assistantContent) => {
          const store = createStore()

          for (let index = 0; index < pairCount; index += 1) {
            store.addQAMessage({
              id: `user-${index}`,
              role: 'user',
              content: userContent,
              timestamp: new Date().toISOString(),
            })
            store.addQAMessage({
              id: `assistant-${index}`,
              role: 'assistant',
              content: assistantContent,
              timestamp: new Date().toISOString(),
            })
          }

          expect(store.qaHistory).toHaveLength(pairCount * 2)
          for (let index = 0; index < pairCount; index += 1) {
            expect(store.qaHistory[index * 2]?.role).toBe('user')
            expect(store.qaHistory[index * 2 + 1]?.role).toBe('assistant')
          }
        },
      ),
    )
  })
})
