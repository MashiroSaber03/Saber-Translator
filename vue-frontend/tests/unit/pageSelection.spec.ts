import { describe, expect, it } from 'vitest'

import {
  clampPageSelection,
  createPageSelectionSummary,
  normalizePageSelection,
  pageSelectionToPageIndexes,
} from '@/utils/pageSelection'

describe('pageSelection utilities', () => {
  it('normalizes page selections to unique ascending 1-based pages', () => {
    expect(normalizePageSelection([5, 3, 5, 1, 0, -1, 2.8, Number.NaN])).toEqual([1, 2, 3, 5])
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
})
