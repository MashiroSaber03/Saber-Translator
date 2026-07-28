import { describe, expect, it } from 'vitest'

import {
  fixedVirtualWindow,
  variableVirtualWindow,
} from '@/components/virtual/virtualWindow'

describe('virtual windows', () => {
  it('keeps a 1000-page thumbnail list bounded', () => {
    const window = fixedVirtualWindow(1000, 164, 164 * 500, 820, 5)
    expect(window.start).toBe(495)
    expect(window.end - window.start).toBeLessThanOrEqual(15)
    expect(window.totalSize).toBe(164000)
  })

  it('uses page aspect-derived sizes without mounting the whole stream', () => {
    const sizes = Array.from({ length: 1000 }, (_, index) => 500 + index % 5)
    const window = variableVirtualWindow(sizes, 250000, 1000, 2000)
    expect(window.start).toBeGreaterThan(400)
    expect(window.end - window.start).toBeLessThan(15)
    expect(window.totalSize).toBeGreaterThan(500000)
  })
})
