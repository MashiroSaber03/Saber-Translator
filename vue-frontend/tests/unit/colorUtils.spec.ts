import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import {
  colorDifference,
  formatConfidence,
  formatRgb,
  getContrastColor,
  hexToRgbArray,
  isRgbEqualToHex,
  isSameColor,
  normalizeHex,
  rgbArrayToHex,
} from '@/utils'

function source(file: string): string {
  return readFileSync(resolve(process.cwd(), file), 'utf8')
}

describe('color utilities', () => {
  it('converts, compares, and formats current RGB color values', () => {
    expect(rgbArrayToHex([15.2, 20.6, 300])).toBe('#0f15ff')
    expect(hexToRgbArray('#0f1419')).toEqual([15, 20, 25])
    expect(normalizeHex('FF0000')).toBe('#ff0000')
    expect(isSameColor('#FF0000', 'ff0000')).toBe(true)
    expect(isRgbEqualToHex([255, 0, 0], '#ff0000')).toBe(true)
    expect(isRgbEqualToHex(null, '#ff0000')).toBe(false)
    expect(colorDifference([0, 0, 0], [3, 4, 0])).toBe(5)
    expect(getContrastColor([20, 20, 20])).toBe('#ffffff')
    expect(getContrastColor([240, 240, 240])).toBe('#000000')
    expect(formatRgb([15, 20, 25])).toBe('RGB(15, 20, 25)')
    expect(formatConfidence(0.924)).toBe('92%')
  })

  it('keeps the source compact and free of tutorial narration', () => {
    const content = source('src/utils/colorUtils.ts')

    for (const staleNarration of [
      '/**',
      '@example',
      '颜色工具函数',
      'RGB 数组类型',
      '将 RGB 数组转换为 Hex 字符串',
      '将 Hex 字符串转换为 RGB 数组',
      '验证 Hex 颜色格式',
      '确保 Hex 颜色带有 # 前缀',
      '简化版欧几里得距离',
      '使用感知亮度公式',
      '获取对比色',
      '格式化 RGB',
      '格式化置信度',
    ]) {
      expect(content).not.toContain(staleNarration)
    }
  })
})
