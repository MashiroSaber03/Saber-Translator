import { beforeEach, describe, expect, it } from 'vitest'
import * as fc from 'fast-check'
import {
  DISMISS_SETUP_REMINDER_KEY,
  dismissFirstTimeGuide,
  resetFirstTimeGuideDismissal,
  shouldShowFirstTimeGuide,
} from '@/components/translate/firstTimeGuideState'

function createMockStorage(): Storage {
  let store: Record<string, string> = {}

  return {
    getItem: (key: string) => store[key] ?? null,
    setItem: (key: string, value: string) => {
      store[key] = value
    },
    removeItem: (key: string) => {
      delete store[key]
    },
    clear: () => {
      store = {}
    },
    get length() {
      return Object.keys(store).length
    },
    key: (index: number) => Object.keys(store)[index] ?? null,
  }
}

function createThrowingStorage(): Storage {
  return {
    getItem: () => {
      throw new Error('storage unavailable')
    },
    setItem: () => {
      throw new Error('storage unavailable')
    },
    removeItem: () => {
      throw new Error('storage unavailable')
    },
    clear: () => {
      throw new Error('storage unavailable')
    },
    get length() {
      return 0
    },
    key: () => null,
  }
}

describe('first-time guide storage contract', () => {
  let storage: Storage

  beforeEach(() => {
    storage = createMockStorage()
  })

  it('shows the guide until the exact dismissed marker is present', () => {
    expect(shouldShowFirstTimeGuide(storage)).toBe(true)

    storage.setItem(DISMISS_SETUP_REMINDER_KEY, 'true')
    expect(shouldShowFirstTimeGuide(storage)).toBe(false)

    storage.setItem(DISMISS_SETUP_REMINDER_KEY, 'false')
    expect(shouldShowFirstTimeGuide(storage)).toBe(true)
  })

  it('persists and resets dismissal idempotently', () => {
    dismissFirstTimeGuide(storage)
    dismissFirstTimeGuide(storage)

    expect(storage.getItem(DISMISS_SETUP_REMINDER_KEY)).toBe('true')
    expect(shouldShowFirstTimeGuide(storage)).toBe(false)

    resetFirstTimeGuideDismissal(storage)
    resetFirstTimeGuideDismissal(storage)

    expect(storage.getItem(DISMISS_SETUP_REMINDER_KEY)).toBeNull()
    expect(shouldShowFirstTimeGuide(storage)).toBe(true)
  })

  it('treats only the current dismissed marker as hidden', () => {
    fc.assert(
      fc.property(fc.string().filter((value) => value !== 'true'), (value) => {
        storage.clear()
        storage.setItem(DISMISS_SETUP_REMINDER_KEY, value)

        expect(shouldShowFirstTimeGuide(storage)).toBe(true)
      }),
      { numRuns: 100 }
    )
  })

  it('keeps dismiss and reset round-trips stable', () => {
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 20 }), (times) => {
        storage.clear()

        for (let index = 0; index < times; index += 1) {
          dismissFirstTimeGuide(storage)
          expect(shouldShowFirstTimeGuide(storage)).toBe(false)

          resetFirstTimeGuideDismissal(storage)
          expect(shouldShowFirstTimeGuide(storage)).toBe(true)
        }
      }),
      { numRuns: 50 }
    )
  })

  it('falls back safely when browser storage is unavailable', () => {
    const restrictedStorage = createThrowingStorage()

    expect(shouldShowFirstTimeGuide(restrictedStorage)).toBe(true)
    expect(() => dismissFirstTimeGuide(restrictedStorage)).not.toThrow()
    expect(() => resetFirstTimeGuideDismissal(restrictedStorage)).not.toThrow()
  })
})
