import { describe, expect, it } from 'vitest'

import {
  clampPageSelection,
  createPageSelectionSummary,
  normalizePageSelection,
  parsePageSelectionText,
  pageSelectionToPageIndexes,
} from '@/utils/pageSelection'

describe('pageSelection utilities', () => {
  it('normalizes page selections to unique ascending 1-based pages', () => {
    expect(normalizePageSelection([5, 3, 5, 1, 0, -1, 2.8, Number.NaN])).toEqual([1, 3, 5])
  })

  it('clamps selections by total images', () => {
    expect(clampPageSelection([1, 4, 10, 12], 10)).toEqual([1, 4, 10])
    expect(clampPageSelection([1, 2], 0)).toEqual([])
  })

  it('converts 1-based selections to 0-based indexes', () => {
    expect(pageSelectionToPageIndexes([1, 3, 8, 10])).toEqual([0, 2, 7, 9])
  })

  it('creates compact summaries for selected pages', () => {
    expect(createPageSelectionSummary([1, 3, 8, 10])).toBe('第 1、3、8、10 页')
    expect(createPageSelectionSummary([1, 2, 3, 4, 5, 6, 7])).toBe('已选 7 页')
    expect(createPageSelectionSummary([])).toBe('未选择页码')
  })

  it('parses mixed page numbers and ranges into a normalized selection', () => {
    expect(parsePageSelectionText('5-7，1、3 6', 10)).toEqual({
      pages: [1, 3, 5, 6, 7],
      error: '',
    })
    expect(parsePageSelectionText('2 ～ 4', 10)).toEqual({
      pages: [2, 3, 4],
      error: '',
    })
  })

  it('reports invalid or out-of-range page input without silently clamping it', () => {
    expect(parsePageSelectionText('4-2', 10).error).toContain('起始页')
    expect(parsePageSelectionText('1,11', 10).error).toBe('页码 11 超出当前总页数 10')
    expect(parsePageSelectionText('1,a', 10).error).toBe('无法识别页码“a”')
  })
})
