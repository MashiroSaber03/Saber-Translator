import { describe, expect, it } from 'vitest'
import { BREAKPOINTS } from '@/composables/useResponsive'

const orderedBreakpoints = [
  BREAKPOINTS.XS,
  BREAKPOINTS.SM,
  BREAKPOINTS.MD,
  BREAKPOINTS.LG,
  BREAKPOINTS.XL,
  BREAKPOINTS.XXL,
]

describe('responsive breakpoint contracts', () => {
  it('keeps breakpoint values ordered from small to large', () => {
    for (let index = 1; index < orderedBreakpoints.length; index++) {
      expect(orderedBreakpoints[index]).toBeGreaterThan(orderedBreakpoints[index - 1]!)
    }
  })

  it('keeps breakpoint values as positive integers', () => {
    for (const value of orderedBreakpoints) {
      expect(value).toBeGreaterThan(0)
      expect(Number.isInteger(value)).toBe(true)
    }
  })
})
