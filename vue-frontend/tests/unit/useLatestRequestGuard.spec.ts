import { describe, expect, it } from 'vitest'

import {
  useKeyedLatestRequestGuard,
  useLatestRequestGuard,
} from '@/composables/useLatestRequestGuard'

describe('useLatestRequestGuard', () => {
  it('invalidates earlier single-owner requests and supports snapshot predicates', () => {
    const guard = useLatestRequestGuard()

    const firstRequest = guard.next()
    expect(guard.isCurrent(firstRequest)).toBe(true)

    const secondRequest = guard.next()
    expect(guard.isCurrent(firstRequest)).toBe(false)
    expect(guard.isCurrent(secondRequest)).toBe(true)
    expect(guard.isCurrent(secondRequest, () => false)).toBe(false)

    guard.invalidate()
    expect(guard.isCurrent(secondRequest)).toBe(false)
  })

  it('tracks latest requests independently for keyed owners', () => {
    const guard = useKeyedLatestRequestGuard<number>()

    const firstRoundRequest = guard.next(0)
    const secondRoundRequest = guard.next(1)

    expect(guard.isCurrent(0, firstRoundRequest)).toBe(true)
    expect(guard.isCurrent(1, secondRoundRequest)).toBe(true)

    const newerFirstRoundRequest = guard.next(0)
    expect(guard.isCurrent(0, firstRoundRequest)).toBe(false)
    expect(guard.isCurrent(0, newerFirstRoundRequest)).toBe(true)
    expect(guard.isCurrent(1, secondRoundRequest)).toBe(true)

    guard.invalidate(1)
    expect(guard.isCurrent(1, secondRoundRequest)).toBe(false)
    expect(guard.isCurrent(0, newerFirstRoundRequest, () => false)).toBe(false)
  })
})
