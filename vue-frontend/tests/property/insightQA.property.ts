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
const qaMessageArbitrary: fc.Arbitrary<QAMessage> = fc.record({
  id: messageIdArbitrary,
  role: messageRoleArbitrary,
  content: messageContentArbitrary,
})

describe('insight QA properties', () => {
  it('appends messages in the order they are added', () => {
    fc.assert(
      fc.property(fc.array(qaMessageArbitrary, { minLength: 0, maxLength: 20 }), messages => {
        const store = createStore()

        addMessages(store, messages)

        expect(store.qaHistory).toEqual(messages)
        expect(store.qaHistory).toHaveLength(messages.length)
      })
    )
  })

  it('clears all QA history entries', () => {
    fc.assert(
      fc.property(fc.array(qaMessageArbitrary, { minLength: 0, maxLength: 20 }), messages => {
        const store = createStore()
        addMessages(store, messages)

        store.clearQAHistory()

        expect(store.qaHistory).toEqual([])
      })
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
      })
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
            })
            store.addQAMessage({
              id: `assistant-${index}`,
              role: 'assistant',
              content: assistantContent,
            })
          }

          expect(store.qaHistory).toHaveLength(pairCount * 2)
          for (let index = 0; index < pairCount; index += 1) {
            expect(store.qaHistory[index * 2]?.role).toBe('user')
            expect(store.qaHistory[index * 2 + 1]?.role).toBe('assistant')
          }
        }
      )
    )
  })
})
