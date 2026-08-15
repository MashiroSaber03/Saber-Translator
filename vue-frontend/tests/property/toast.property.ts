import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import * as fc from 'fast-check'
import { toastService, type ToastType } from '@/utils/toast'

const toastTypeArb: fc.Arbitrary<ToastType> = fc.constantFrom('success', 'error', 'info', 'warning')
const messageArb = fc.string({ minLength: 1, maxLength: 120 })
const addOperationArb = fc.record({
  message: messageArb,
  type: toastTypeArb,
})

function resetToastState(): void {
  toastService.clearAll()
  vi.clearAllTimers()
}

describe('toast service property contracts', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    resetToastState()
  })

  afterEach(() => {
    resetToastState()
    vi.useRealTimers()
  })

  it('adds all messages with unique ids and preserves message content', () => {
    fc.assert(
      fc.property(fc.array(addOperationArb, { minLength: 0, maxLength: 20 }), (operations) => {
        resetToastState()

        const ids = operations.map((operation) => toastService.addToast(operation.message, operation.type, 0))

        expect(toastService.toasts.value).toHaveLength(operations.length)
        expect(new Set(ids).size).toBe(ids.length)
        expect(toastService.toasts.value.map(({ message, type }) => ({ message, type }))).toEqual(operations)
      }),
      { numRuns: 100 }
    )
  })

  it('removes exactly the requested toast id', () => {
    fc.assert(
      fc.property(
        fc.array(addOperationArb, { minLength: 1, maxLength: 12 }),
        fc.integer({ min: 0, max: 11 }),
        (operations, removeIndex) => {
          resetToastState()

          const ids = operations.map((operation) => toastService.addToast(operation.message, operation.type, 0))
          const idToRemove = ids[removeIndex % ids.length]
          toastService.removeToast(idToRemove!)

          expect(toastService.toasts.value).toHaveLength(operations.length - 1)
          expect(toastService.toasts.value.some((toast) => toast.id === idToRemove)).toBe(false)
        }
      ),
      { numRuns: 100 }
    )
  })

  it('ignores missing ids and clears all queued messages', () => {
    fc.assert(
      fc.property(fc.array(addOperationArb, { minLength: 1, maxLength: 20 }), (operations) => {
        resetToastState()

        operations.forEach((operation) => toastService.addToast(operation.message, operation.type, 0))
        const beforeMissingRemove = [...toastService.toasts.value]

        toastService.removeToast(Number.MAX_SAFE_INTEGER)
        expect(toastService.toasts.value).toEqual(beforeMissingRemove)

        toastService.clearAll()
        expect(toastService.toasts.value).toEqual([])
      }),
      { numRuns: 100 }
    )
  })

  it('expires finite-duration toasts and leaves duration-zero toasts active', () => {
    const finiteId = toastService.addToast('finite', 'info', 1000)
    const persistentId = toastService.addToast('persistent', 'warning', 0)

    vi.advanceTimersByTime(999)
    expect(toastService.toasts.value.map((toast) => toast.id)).toEqual([finiteId, persistentId])

    vi.advanceTimersByTime(1)
    expect(toastService.toasts.value.map((toast) => toast.id)).toEqual([persistentId])

    vi.advanceTimersByTime(100000)
    expect(toastService.toasts.value.map((toast) => toast.id)).toEqual([persistentId])
  })

  it('manual removal cancels pending timer cleanup', () => {
    const id = toastService.addToast('pending', 'info', 5000)

    toastService.removeToast(id)
    vi.advanceTimersByTime(5000)

    expect(toastService.toasts.value).toEqual([])
  })

})
