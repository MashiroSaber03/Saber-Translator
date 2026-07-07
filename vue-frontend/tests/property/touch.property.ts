import { describe, expect, it } from 'vitest'
import * as fc from 'fast-check'
import { calculatePinchScale, detectSwipe } from '@/composables/useTouch'

const coordArb = fc.integer({ min: -2000, max: 2000 })
const thresholdArb = fc.integer({ min: 1, max: 500 })

const belowThresholdMoveArb = thresholdArb.chain(threshold =>
  fc.record({
    threshold: fc.constant(threshold),
    dx: fc.integer({ min: 1 - threshold, max: threshold - 1 }),
    dy: fc.integer({ min: 1 - threshold, max: threshold - 1 }),
  })
)

describe('touch helper property contracts', () => {
  it('detects dominant horizontal swipes', () => {
    fc.assert(
      fc.property(
        coordArb,
        coordArb,
        fc.constantFrom('right' as const, 'left' as const),
        fc.integer({ min: 50, max: 1000 }),
        fc.integer({ min: -49, max: 49 }),
        (startX, startY, expectedDirection, magnitude, verticalDelta) => {
          const horizontalDelta = expectedDirection === 'right' ? magnitude : -magnitude

          expect(detectSwipe(startX, startY, startX + horizontalDelta, startY + verticalDelta)).toBe(expectedDirection)
        },
      ),
      { numRuns: 100 },
    )
  })

  it('detects dominant vertical swipes', () => {
    fc.assert(
      fc.property(
        coordArb,
        coordArb,
        fc.constantFrom('down' as const, 'up' as const),
        fc.integer({ min: -49, max: 49 }),
        fc.integer({ min: 50, max: 1000 }),
        (startX, startY, expectedDirection, horizontalDelta, magnitude) => {
          const verticalDelta = expectedDirection === 'down' ? magnitude : -magnitude

          expect(detectSwipe(startX, startY, startX + horizontalDelta, startY + verticalDelta)).toBe(expectedDirection)
        },
      ),
      { numRuns: 100 },
    )
  })

  it('returns null when both axes stay below the active threshold', () => {
    fc.assert(
      fc.property(coordArb, coordArb, belowThresholdMoveArb, (startX, startY, move) => {
        expect(detectSwipe(startX, startY, startX + move.dx, startY + move.dy, move.threshold)).toBeNull()
      }),
      { numRuns: 100 },
    )
  })

  it('uses threshold-inclusive detection and vertical tie-breaking', () => {
    expect(detectSwipe(0, 0, 50, 0)).toBe('right')
    expect(detectSwipe(0, 0, 49, 0)).toBeNull()
    expect(detectSwipe(0, 0, 100, 100)).toBe('down')
    expect(detectSwipe(0, 0, -100, -100)).toBe('up')
  })

  it('calculates pinch scale from positive distances', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 1000 }),
        fc.integer({ min: 0, max: 2000 }),
        (initialDistance, currentDistance) => {
          expect(calculatePinchScale(initialDistance, currentDistance)).toBeCloseTo(
            currentDistance / initialDistance,
            10,
          )
        },
      ),
      { numRuns: 100 },
    )
  })

  it('keeps non-positive initial pinch distances neutral', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: -1000, max: 0 }),
        fc.integer({ min: 0, max: 2000 }),
        (initialDistance, currentDistance) => {
          expect(calculatePinchScale(initialDistance, currentDistance)).toBe(1)
        },
      ),
      { numRuns: 100 },
    )
  })

  it('orders pinch scale around the neutral value', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 2, max: 1000 }),
        fc.integer({ min: 1, max: 999 }),
        fc.integer({ min: 1, max: 1000 }),
        (initialDistance, reduction, growth) => {
          const smallerDistance = Math.max(1, initialDistance - reduction)
          const largerDistance = initialDistance + growth

          expect(calculatePinchScale(initialDistance, initialDistance)).toBe(1)
          expect(calculatePinchScale(initialDistance, smallerDistance)).toBeGreaterThan(0)
          expect(calculatePinchScale(initialDistance, smallerDistance)).toBeLessThanOrEqual(1)
          expect(calculatePinchScale(initialDistance, largerDistance)).toBeGreaterThan(1)
        },
      ),
      { numRuns: 100 },
    )
  })
})
