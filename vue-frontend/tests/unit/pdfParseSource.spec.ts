import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import {
  buildDocumentParseBatches,
  calculateDocumentParseProgress,
  createDocumentPageFileName,
} from '@/utils/pdfParse'

describe('document parse helpers', () => {
  it('plans parse batches in order with stable has-more flags', () => {
    const batches = buildDocumentParseBatches(7, 3)

    expect(batches).toEqual([
      { batchIndex: 0, startIndex: 0, count: 3, processedPages: 3, hasMore: true },
      { batchIndex: 1, startIndex: 3, count: 3, processedPages: 6, hasMore: true },
      { batchIndex: 2, startIndex: 6, count: 1, processedPages: 7, hasMore: false },
    ])
  })

  it('clamps document parse progress to the progressbar range', () => {
    expect(calculateDocumentParseProgress(0, 0)).toBe(100)
    expect(calculateDocumentParseProgress(-2, 10)).toBe(0)
    expect(calculateDocumentParseProgress(5, 10)).toBe(50)
    expect(calculateDocumentParseProgress(12, 10)).toBe(100)
  })

  it('formats page filenames from the current source document name', () => {
    expect(createDocumentPageFileName('chapter.final.PDF', 7)).toBe('chapter.final_page_007.png')
    expect(createDocumentPageFileName('comic', 12)).toBe('comic_page_012.png')
    expect(createDocumentPageFileName('.pdf', 1)).toBe('document_page_001.png')
  })

  it('keeps pdf parse property tests on production helpers', () => {
    const source = readFileSync(resolve(process.cwd(), 'tests/property/pdfParse.property.ts'), 'utf8')

    for (const staleSource of [
      '/' + '**',
      'PDF 解析' + '属性测试',
      '测试内容',
      'MockPdf',
      'simulateBatchParsing',
      'calculateBatchPageCount',
      'hasMoreBatches',
      'Property 27',
      '验证',
      '生成页面文件名',
      'function generatePageFileName',
    ]) {
      expect(source).not.toContain(staleSource)
    }

    expect(source).toContain("from '@/utils/pdfParse'")
    expect(source).toContain('buildDocumentParseBatches')
    expect(source).toContain('calculateDocumentParseProgress')
    expect(source).toContain('createDocumentPageFileName')
  })
})
