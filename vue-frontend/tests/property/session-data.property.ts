import { describe, expect, it } from 'vitest'
import * as fc from 'fast-check'
import { createPinia, setActivePinia } from 'pinia'
import { useSessionStore } from '@/stores/sessionStore'

const validSessionNameArb = fc.string({ minLength: 1, maxLength: 30 })
  .filter(value => /^[a-z0-9_-]+$/i.test(value))

function createSessionStore() {
  setActivePinia(createPinia())
  return useSessionStore()
}

describe('session data properties', () => {
  it('creates the current session payload shape from store inputs', () => {
    fc.assert(
      fc.property(
        validSessionNameArb,
        fc.integer({ min: -1, max: 10 }),
        (name, currentIndex) => {
          const sessionStore = createSessionStore()
          const uiSettings = { testSetting: 'value' }
          const sessionData = sessionStore.createSessionData(name, [], currentIndex, uiSettings)

          expect(sessionData.name).toBe(name)
          expect(sessionData.version).toBe('2.0')
          expect(sessionData.imageCount).toBe(0)
          expect(sessionData.currentImageIndex).toBe(currentIndex)
          expect(sessionData.ui_settings).toEqual(uiSettings)
          expect(sessionData.images).toHaveLength(0)
          expect(new Date(sessionData.savedAt).toString()).not.toBe('Invalid Date')
        }
      ),
      { numRuns: 20 }
    )
  })
})
