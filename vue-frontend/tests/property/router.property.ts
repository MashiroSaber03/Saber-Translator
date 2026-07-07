import { describe, expect, it } from 'vitest'
import * as fc from 'fast-check'
import router from '@/router'
import { APP_ROUTE_PATHS } from '@/constants/routes'

const namedRouteCases = [
  { path: APP_ROUTE_PATHS.bookshelf, name: 'bookshelf' },
  { path: APP_ROUTE_PATHS.translate, name: 'translate' },
  { path: APP_ROUTE_PATHS.reader, name: 'reader' },
  { path: APP_ROUTE_PATHS.insight, name: 'insight' },
  { path: APP_ROUTE_PATHS.characterStudio, name: 'character-studio' },
] as const

function routeQuery(params: { book?: string; chapter?: string; doc?: string }): Record<string, string> {
  return Object.fromEntries(Object.entries(params).filter((entry): entry is [string, string] => Boolean(entry[1])))
}

describe('router properties', () => {
  it('resolves current app route paths to their route names', () => {
    fc.assert(
      fc.property(fc.constantFrom(...namedRouteCases), routeCase => {
        const resolved = router.resolve(routeCase.path)

        expect(resolved.name).toBe(routeCase.name)
      }),
      { numRuns: 100 }
    )
  })

  it('routes undefined top-level paths through the bookshelf catch-all redirect', () => {
    const reservedSegments = new Set(['translate', 'reader', 'insight', 'api', 'static', 'assets', 'js'])

    fc.assert(
      fc.property(
        fc.stringOf(fc.constantFrom(...'abcdefghijklmnopqrstuvwxyz0123456789'.split('')), {
          minLength: 1,
          maxLength: 20,
        }).filter(segment => !reservedSegments.has(segment)),
        randomSegment => {
          const resolved = router.resolve(`/${randomSegment}`)

          expect(resolved.matched).toHaveLength(1)
          expect(resolved.matched[0]?.redirect).toEqual({ name: 'bookshelf' })
        }
      ),
      { numRuns: 100 }
    )
  })

  it('preserves translate route query parameters', () => {
    fc.assert(
      fc.property(
        fc.record({
          book: fc.option(fc.string({ minLength: 1, maxLength: 20 }), { nil: undefined }),
          chapter: fc.option(fc.string({ minLength: 1, maxLength: 20 }), { nil: undefined }),
        }),
        params => {
          const query = routeQuery(params)
          const resolved = router.resolve({ name: 'translate', query })

          expect(resolved.name).toBe('translate')
          expect(resolved.query).toMatchObject(query)
        }
      ),
      { numRuns: 100 }
    )
  })

  it('preserves insight route query parameters', () => {
    fc.assert(
      fc.property(
        fc.option(fc.string({ minLength: 1, maxLength: 20 }), { nil: undefined }),
        bookId => {
          const query = routeQuery({ book: bookId })
          const resolved = router.resolve({ name: 'insight', query })

          expect(resolved.name).toBe('insight')
          expect(resolved.query).toMatchObject(query)
        }
      ),
      { numRuns: 100 }
    )
  })

  it('preserves character studio route query parameters', () => {
    fc.assert(
      fc.property(
        fc.record({
          book: fc.option(fc.string({ minLength: 1, maxLength: 20 }), { nil: undefined }),
          doc: fc.option(fc.string({ minLength: 1, maxLength: 20 }), { nil: undefined }),
        }),
        params => {
          const query = routeQuery(params)
          const resolved = router.resolve({ name: 'character-studio', query })

          expect(resolved.name).toBe('character-studio')
          expect(resolved.query).toMatchObject(query)
        }
      ),
      { numRuns: 100 }
    )
  })
})
