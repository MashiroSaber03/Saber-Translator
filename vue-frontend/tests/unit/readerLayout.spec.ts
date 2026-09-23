import { describe, expect, it, beforeEach, vi, afterEach } from 'vitest'
import * as fc from 'fast-check'
import { fitScale, pageGroups, pageIndexAtOffset, resolvePosition } from '@/components/reader/readerLayout'
import { loadReaderRecord, readerHistoryKey, saveReaderRecord } from '@/components/reader/readerHistory'
const pages = (count: number) => Array.from({ length: count }, (_, i) => ({ id: String(i), width: 800, height: 1200 }))

describe('reader grouping and geometry', () => {
  it('pairs pages with and without the first-page offset', () => {
    expect(pageGroups(pages(5), true, false)).toEqual([[0, 1], [2, 3], [4]])
    expect(pageGroups(pages(5), true, true)).toEqual([[0], [1, 2], [3, 4]])
    expect(pageGroups(pages(3), false, true)).toEqual([[0], [1], [2]])
  })
  it('keeps landscape images independent without dropping adjacent portrait pages', () => {
    const input = pages(7)
    input[3]!.width = 2000
    expect(pageGroups(input, true, false)).toEqual([[0, 1], [2], [3], [4, 5], [6]])
    expect(pageGroups(input, true, true)).toEqual([[0], [1, 2], [3], [4, 5], [6]])
  })
  it('covers every page exactly once for arbitrary chapters', () => {
    fc.assert(fc.property(fc.array(fc.boolean(), { maxLength: 1000 }), fc.boolean(), (wide, offset) => {
      const input = wide.map((landscape, i) => ({ id: String(i), width: landscape ? 1600 : 800, height: 1200 }))
      const groups = pageGroups(input, true, offset)
      expect(groups.flat()).toEqual(input.map((_, i) => i))
      expect(groups.every(g => g.length === 1 || g.every(i => !wide[i]))).toBe(true)
    }))
  })
  it('assigns exact scroll boundaries to the new page, not the preceding page', () => {
    expect(pageIndexAtOffset([0, 800, 1600, 2400], 1600)).toBe(2)
    expect(pageIndexAtOffset([0, 800, 1600, 2400], 1590)).toBe(1)
    expect(pageIndexAtOffset([0, 800, 1600, 2400], 1599.9)).toBe(2)
    expect(pageIndexAtOffset([0], 0)).toBe(0)
  })
  it('keeps page identity across reordering and clamps deleted-page fallback', () => {
    expect(resolvePosition(pages(5), { pageId: '3', index: 0, fraction: 0 })).toBe(3)
    expect(resolvePosition(pages(2), { pageId: 'deleted', index: 8, fraction: 0 })).toBe(1)
  })
  it('fits the entire spread while allowing overflow for width or height only', () => {
    expect(fitScale(1600, 1200, 1000, 600, 'screen')).toBe(.5)
    expect(fitScale(1600, 1200, 1000, 600, 'width')).toBe(.625)
    expect(fitScale(1600, 1200, 1000, 600, 'height')).toBe(.5)
    expect(fitScale(1600, 1200, 1000, 600, 'original')).toBe(1)
  })
})
describe('local reading history', () => {
  beforeEach(() => localStorage.clear())
  afterEach(() => vi.restoreAllMocks())
  it('isolates users and chapters, stores offsets and keeps only 100 recent entries', () => {
    const key = readerHistoryKey('alice', 'book', 'chapter')
    const record = { key, pageId: 'p4', index: 3, fraction: .4, offset: true }
    saveReaderRecord(record)
    expect(loadReaderRecord(key)).toEqual(record)
    expect(loadReaderRecord(readerHistoryKey('bob', 'book', 'chapter'))).toBeUndefined()
    for (let i = 0; i < 101; i++) saveReaderRecord({ ...record, key: String(i) })
    expect(JSON.parse(localStorage.getItem('readerHistory')!).length).toBe(100)
    expect(loadReaderRecord(key)).toBeUndefined()
  })
  it('ignores corrupt history and tolerates unavailable storage', () => {
    localStorage.setItem('readerHistory', '[{"key":"x","fraction":"oops"}]')
    expect(loadReaderRecord('x')).toBeUndefined()
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw Error('quota') })
    expect(() => saveReaderRecord({ key: 'x', pageId: 'p', index: 0, fraction: 0, offset: false })).not.toThrow()
  })
})
