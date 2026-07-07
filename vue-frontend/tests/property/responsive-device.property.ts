import { describe, expect, it } from 'vitest'
import * as fc from 'fast-check'
import {
  BREAKPOINTS,
  getDeviceType,
  isMobileViewport,
} from '@/composables/useResponsive'

const screenWidthArb = fc.integer({ min: 1, max: 3000 })

describe('responsive device contracts', () => {
  it('classifies every width into exactly one device type', () => {
    fc.assert(
      fc.property(screenWidthArb, (width) => {
        const deviceType = getDeviceType(width)
        const matches = [
          deviceType === 'mobile',
          deviceType === 'tablet',
          deviceType === 'desktop',
        ]

        expect(matches.filter(Boolean)).toHaveLength(1)
        expect(isMobileViewport(width)).toBe(deviceType === 'mobile')
      }),
      { numRuns: 100 },
    )
  })

  it('keeps device boundaries aligned with the shared breakpoints', () => {
    const boundaryCases = [
      { width: BREAKPOINTS.MD - 1, deviceType: 'mobile', mobile: true },
      { width: BREAKPOINTS.MD, deviceType: 'tablet', mobile: false },
      { width: BREAKPOINTS.LG - 1, deviceType: 'tablet', mobile: false },
      { width: BREAKPOINTS.LG, deviceType: 'desktop', mobile: false },
    ] as const

    for (const testCase of boundaryCases) {
      expect(getDeviceType(testCase.width)).toBe(testCase.deviceType)
      expect(isMobileViewport(testCase.width)).toBe(testCase.mobile)
    }
  })
})
