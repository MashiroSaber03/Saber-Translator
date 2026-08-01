import { describe, expect, it } from 'vitest'

import { useLatestRequestGuard } from '@/composables/useLatestRequestGuard'

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
})
