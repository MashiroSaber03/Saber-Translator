import { describe, expect, it } from 'vitest'
import { normalizedTaskPageUrl, stablePageTitle } from './pageIdentity'

describe('browser page identity', () => {
  it('keeps every MangaDex image index in one chapter task', () => {
    const first = normalizedTaskPageUrl(
      'https://mangadex.org/chapter/a3355473-49f5-4533-b81d-175c25c77b42/1',
    )
    const second = normalizedTaskPageUrl(
      'https://mangadex.org/chapter/a3355473-49f5-4533-b81d-175c25c77b42/2',
    )

    expect(first).toBe(
      'https://mangadex.org/chapter/a3355473-49f5-4533-b81d-175c25c77b42',
    )
    expect(second).toBe(first)
  })

  it('still distinguishes different MangaDex chapters', () => {
    expect(normalizedTaskPageUrl('https://mangadex.org/chapter/chapter-a/19'))
      .not.toBe(normalizedTaskPageUrl('https://mangadex.org/chapter/chapter-b/1'))
  })

  it('does not reinterpret numbered paths on unrelated sites', () => {
    expect(normalizedTaskPageUrl('https://reader.example/chapter/example/2#panel'))
      .toBe('https://reader.example/chapter/example/2')
  })

  it('removes MangaDex image numbers from the panel title', () => {
    expect(stablePageTitle(
      'https://mangadex.org/chapter/example/2',
      '2 | Chapter 6 - Example - MangaDex',
      'mangadex.org',
    )).toBe('Chapter 6 - Example - MangaDex')
  })
})
