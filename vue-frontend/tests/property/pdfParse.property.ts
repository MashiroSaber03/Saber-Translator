import { describe, expect, it } from 'vitest'
import * as fc from 'fast-check'
import {
  buildDocumentParseBatches,
  calculateDocumentParseProgress,
  createDocumentPageFileName,
} from '@/utils/pdfParse'

const pageCountArbitrary = fc.integer({ min: 1, max: 200 })
const batchSizeArbitrary = fc.integer({ min: 1, max: 50 })

describe('document parse properties', () => {
  it('plans batches that cover every page exactly once in order', () => {
    fc.assert(
      fc.property(pageCountArbitrary, batchSizeArbitrary, (totalPages, batchSize) => {
        const batches = buildDocumentParseBatches(totalPages, batchSize)

        expect(batches.length).toBe(Math.ceil(totalPages / batchSize))
        expect(batches.at(0)?.startIndex).toBe(0)
        expect(batches.at(-1)?.processedPages).toBe(totalPages)

        batches.forEach((batch, index) => {
          const previous = batches[index - 1]
          expect(batch.batchIndex).toBe(index)
          expect(batch.count).toBeGreaterThan(0)
          expect(batch.count).toBeLessThanOrEqual(batchSize)
          expect(batch.startIndex).toBe(previous ? previous.processedPages : 0)
          expect(batch.processedPages).toBe(batch.startIndex + batch.count)
          expect(batch.hasMore).toBe(index < batches.length - 1)
        })

        expect(batches.reduce((sum, batch) => sum + batch.count, 0)).toBe(totalPages)
      }),
    )
  })

  it('normalizes empty and invalid parse batch inputs without producing work', () => {
    fc.assert(
      fc.property(
        fc.oneof(
          fc.integer({ min: -200, max: 0 }),
          fc.constant(Number.NaN),
          fc.constant(Number.POSITIVE_INFINITY),
        ),
        fc.integer({ min: -10, max: 10 }),
        (totalPages, batchSize) => {
          expect(buildDocumentParseBatches(totalPages, batchSize)).toEqual([])
        },
      ),
    )
  })

  it('keeps parse progress inside the progressbar range and monotonic', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: -200, max: 1200 }),
        fc.integer({ min: 1, max: 1000 }),
        (processedPages, totalPages) => {
          const progress = calculateDocumentParseProgress(processedPages, totalPages)
          const previousProgress = calculateDocumentParseProgress(processedPages - 1, totalPages)

          expect(progress).toBeGreaterThanOrEqual(0)
          expect(progress).toBeLessThanOrEqual(100)
          expect(progress).toBeGreaterThanOrEqual(previousProgress)
          if (processedPages <= 0) {
            expect(progress).toBe(0)
          }
          if (processedPages >= totalPages) {
            expect(progress).toBe(100)
          }
        },
      ),
    )
  })

  it('treats zero or invalid totals as complete progress', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: -100, max: 100 }),
        fc.oneof(
          fc.integer({ min: -200, max: 0 }),
          fc.constant(Number.NaN),
          fc.constant(Number.POSITIVE_INFINITY),
        ),
        (processedPages, totalPages) => {
          expect(calculateDocumentParseProgress(processedPages, totalPages)).toBe(100)
        },
      ),
    )
  })

  it('formats one-based page filenames with a stable document base name', () => {
    fc.assert(
      fc.property(
        fc.stringOf(fc.constantFrom(...'abcdefghijklmnopqrstuvwxyz0123456789.-_'.split('')), {
          minLength: 1,
          maxLength: 40,
        }),
        fc.constantFrom('', '.pdf', '.PDF', '.mobi', '.zip'),
        fc.integer({ min: 1, max: 9999 }),
        (rawBaseName, extension, pageNumber) => {
          const sourceName = `${rawBaseName}${extension}`
          const fileName = createDocumentPageFileName(sourceName, pageNumber)
          const expectedPage = String(pageNumber).padStart(3, '0')

          expect(fileName).toMatch(/^.+_page_\d{3,}\.png$/)
          expect(fileName).toContain(`_page_${expectedPage}.png`)
          expect(fileName.endsWith('.png')).toBe(true)
        },
      ),
    )
  })
})
