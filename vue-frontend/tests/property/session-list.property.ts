import { describe, expect, it } from 'vitest'
import * as fc from 'fast-check'
import { createPinia, setActivePinia } from 'pinia'
import { useSessionStore } from '@/stores/sessionStore'
import type { SessionListItem } from '@/types/api'

function createSessionStore() {
  setActivePinia(createPinia())
  return useSessionStore()
}

function createSessionList(count: number, uniqueId: string): SessionListItem[] {
  return Array.from({ length: count }, (_, index) => ({
    name: `session_${uniqueId}_${index}`,
    savedAt: new Date().toISOString(),
    imageCount: index * 10,
    version: '2.0',
  }))
}

describe('session list properties', () => {
  it('adds and removes sessions by name while preserving the existing list', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 5 }),
        fc.uuid(),
        (count, uniqueId) => {
          const sessionStore = createSessionStore()
          const sessions = createSessionList(count, uniqueId)
          const newSession: SessionListItem = {
            name: `new_session_${uniqueId}`,
            savedAt: new Date().toISOString(),
            imageCount: 5,
            version: '2.0',
          }

          sessionStore.setSessionList(sessions)
          expect(sessionStore.sessionList).toHaveLength(count)

          sessionStore.addToSessionList(newSession)
          expect(sessionStore.sessionList).toHaveLength(count + 1)
          expect(sessionStore.sessionList[0]).toEqual(newSession)

          sessionStore.removeFromSessionList(newSession.name)
          expect(sessionStore.sessionList).toHaveLength(count)
          expect(sessionStore.sessionList.map(session => session.name)).not.toContain(newSession.name)
        }
      ),
      { numRuns: 20 }
    )
  })
})
