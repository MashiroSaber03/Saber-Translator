import { describe, expect, it } from 'vitest'
import fc from 'fast-check'
import { hexToHsv, hsvToHex } from '@/utils/colorConversion'

describe('color conversion', () => {
  it.each([
    ['#000000', 0, 0, 0], ['#ffffff', 0, 0, 1],
    ['#ff0000', 0, 1, 1], ['#ffff00', 60, 1, 1], ['#00ff00', 120, 1, 1],
    ['#00ffff', 180, 1, 1], ['#0000ff', 240, 1, 1], ['#ff00ff', 300, 1, 1],
  ] as const)('converts %s and its HSV coordinates exactly', (hex, h, s, v) => {
    expect(hexToHsv(hex)).toEqual({ h, s, v })
    expect(hsvToHex({ h, s, v })).toBe(hex)
  })

  it('preserves all RGB channels through an HSV round-trip', () => {
    fc.assert(fc.property(fc.integer({ min: 0, max: 0xffffff }), rgb => {
      const hex = `#${rgb.toString(16).padStart(6, '0')}`
      expect(hsvToHex(hexToHsv(hex))).toBe(hex)
    }), { numRuns: 1000 })
    expect(hsvToHex({ h: 360, s: 1, v: 1 })).toBe('#ff0000')
  })
})
