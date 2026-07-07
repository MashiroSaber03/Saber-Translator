import { describe, expect, it } from 'vitest'
import * as fc from 'fast-check'
import {
  BREAKPOINTS,
  getLayoutMode,
} from '@/composables/useResponsive'

const screenWidthArb = fc.integer({ min: 1, max: 3000 })

describe('responsive layout mode contracts', () => {
  it('classifies widths through the exported layout helper', () => {
    fc.assert(
      fc.property(screenWidthArb, (width) => {
        const layoutMode = getLayoutMode(width)

        if (width < BREAKPOINTS.MD) {
          expect(layoutMode).toBe('compact')
        } else if (width < BREAKPOINTS.XL) {
          expect(layoutMode).toBe('normal')
        } else {
          expect(layoutMode).toBe('wide')
        }
      }),
      { numRuns: 100 },
    )
  })

  it('keeps boundary widths in the expected layout bands', () => {
    expect(getLayoutMode(BREAKPOINTS.MD - 1)).toBe('compact')
    expect(getLayoutMode(BREAKPOINTS.MD)).toBe('normal')
    expect(getLayoutMode(BREAKPOINTS.XL - 1)).toBe('normal')
    expect(getLayoutMode(BREAKPOINTS.XL)).toBe('wide')
  })
})
