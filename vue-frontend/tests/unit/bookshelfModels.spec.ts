import { describe, expect, it } from 'vitest'
import { normalizeBookData } from '@/utils/bookshelfModels'
import type { BookData } from '@/types/api'

describe('bookshelf model normalization', () => {
  it('normalizes only the current bookshelf model', () => {
    const book = normalizeBookData({
      id: 'book-wire',
      title: 'Wire Book',
      totalPages: 7,
      chapterCount: 2,
      createdAt: '2026-03-01T00:00:00.000Z',
      updatedAt: '2026-03-02T00:00:00.000Z',
      chapters: [
        {
          id: 'chapter-1',
          title: 'Chapter 1',
          order: 0,
          imageCount: 3,
        },
        {
          id: 'chapter-2',
          title: 'Chapter 2',
          order: 1,
          imageCount: 4,
        },
      ],
    } satisfies BookData)

    expect(book).toMatchObject({
      id: 'book-wire',
      totalPages: 7,
      chapterCount: 2,
      createdAt: '2026-03-01T00:00:00.000Z',
      updatedAt: '2026-03-02T00:00:00.000Z',
    })
    expect(book.chapters).toEqual([
      expect.objectContaining({
        id: 'chapter-1',
        imageCount: 3,
      }),
      expect.objectContaining({
        id: 'chapter-2',
        imageCount: 4,
      }),
    ])
  })
})
