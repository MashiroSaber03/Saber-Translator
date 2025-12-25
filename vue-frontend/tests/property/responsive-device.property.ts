/**
 * 响应式布局 - 设备类型判断属性测试
 * 
 * **Feature: vue-frontend-migration, Property 43: 视口尺寸计算一致性**
 * **Validates: Requirements 25.1, 25.4**
 */

import { describe, it, expect } from 'vitest'
import * as fc from 'fast-check'
import { BREAKPOINTS } from '@/composables/useResponsive'

function isMobile(width: number): boolean {
  return width < BREAKPOINTS.MD
}

function isTablet(width: number): boolean {
  return width >= BREAKPOINTS.MD && width < BREAKPOINTS.LG
}

function isDesktop(width: number): boolean {
  return width >= BREAKPOINTS.LG
}

function getDeviceType(width: number): 'mobile' | 'tablet' | 'desktop' {
  if (width < BREAKPOINTS.MD) return 'mobile'
  if (width < BREAKPOINTS.LG) return 'tablet'
  return 'desktop'
}

describe('响应式布局 - 设备类型判断', () => {
  const screenWidthArb = fc.integer({ min: 1, max: 3000 })
  
  it('设备类型判断应该互斥且完整覆盖', () => {
    fc.assert(
      fc.property(screenWidthArb, (width) => {
        const mobile = isMobile(width)
        const tablet = isTablet(width)
        const desktop = isDesktop(width)
        
        const count = [mobile, tablet, desktop].filter(Boolean).length
        expect(count).toBe(1)
        expect(mobile || tablet || desktop).toBe(true)
      }),
      { numRuns: 100 }
    )
  })
  
  it('设备类型应该与断点值一致', () => {
    fc.assert(
      fc.property(screenWidthArb, (width) => {
        const deviceType = getDeviceType(width)
        
        if (width < BREAKPOINTS.MD) {
          expect(deviceType).toBe('mobile')
        } else if (width < BREAKPOINTS.LG) {
          expect(deviceType).toBe('tablet')
        } else {
          expect(deviceType).toBe('desktop')
        }
      }),
      { numRuns: 100 }
    )
  })
  
  it('断点边界值应该正确判断', () => {
    const boundaryTests = [
      { width: BREAKPOINTS.XS - 1, expectedMobile: true },
      { width: BREAKPOINTS.XS, expectedMobile: true },
      { width: BREAKPOINTS.SM - 1, expectedMobile: true },
      { width: BREAKPOINTS.SM, expectedMobile: true },
      { width: BREAKPOINTS.MD - 1, expectedMobile: true },
      { width: BREAKPOINTS.MD, expectedMobile: false, expectedTablet: true },
      { width: BREAKPOINTS.LG - 1, expectedTablet: true },
      { width: BREAKPOINTS.LG, expectedDesktop: true },
      { width: BREAKPOINTS.XL - 1, expectedDesktop: true },
      { width: BREAKPOINTS.XL, expectedDesktop: true },
    ]
    
    boundaryTests.forEach(({ width, expectedMobile, expectedTablet, expectedDesktop }) => {
      if (expectedMobile !== undefined) {
        expect(isMobile(width)).toBe(expectedMobile)
      }
      if (expectedTablet !== undefined) {
        expect(isTablet(width)).toBe(expectedTablet)
      }
      if (expectedDesktop !== undefined) {
        expect(isDesktop(width)).toBe(expectedDesktop)
      }
    })
  })
})
