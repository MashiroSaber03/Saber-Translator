import { describe, expect, it } from 'vitest'
import * as fc from 'fast-check'
import { FRONTEND_ROUTE_PATHS } from '@/constants/routes'
import {
  buildApiPath,
  buildStaticPath,
  buildVueStaticAssetPath,
  classifyAppPath,
  isKnownFrontendRoute,
  normalizeAppPath,
} from '@/utils/routePath'

describe('route path properties', () => {
  it('classifies API paths as API routes', () => {
    fc.assert(
      fc.property(
        fc.constantFrom(
          'books',
          'jobs',
          'plugins',
          'settings',
          'insight/runs',
          'system/health'
        ),
        endpoint => {
          expect(classifyAppPath(buildApiPath(endpoint))).toEqual({
            isFrontendRoute: false,
            isApiRoute: true,
            isStaticRoute: false,
          })
        }
      ),
      { numRuns: 100 }
    )
  })

  it('classifies backend static resource paths as static routes', () => {
    fc.assert(
      fc.property(
        fc.constantFrom('css', 'js', 'fonts', 'pic', 'vue'),
        fc.constantFrom('layout.css', 'bundle.js', 'font.woff2', 'logo.png', 'index.html'),
        (resourceType, filename) => {
          expect(classifyAppPath(buildStaticPath(resourceType, filename))).toEqual({
            isFrontendRoute: false,
            isApiRoute: false,
            isStaticRoute: true,
          })
        }
      ),
      { numRuns: 100 }
    )
  })

  it('classifies frontend routes as app routes', () => {
    fc.assert(
      fc.property(fc.constantFrom(...FRONTEND_ROUTE_PATHS), path => {
        expect(classifyAppPath(path)).toEqual({
          isFrontendRoute: true,
          isApiRoute: false,
          isStaticRoute: false,
        })
        expect(isKnownFrontendRoute(path)).toBe(true)
      }),
      { numRuns: 100 }
    )
  })

  it('keeps API and static classifications mutually exclusive', () => {
    fc.assert(
      fc.property(
        fc.constantFrom(
          '/',
          '/translate',
          '/reader',
          '/insight',
          '/insight/character-studio',
          '/api/v2/books',
          '/static/css/layout.css',
          '/js/bundle.abc123.js',
          '/assets/index.abc123.css',
          '/unknown/path'
        ),
        path => {
          const classification = classifyAppPath(path)
          const exclusiveCount = Number(classification.isApiRoute) + Number(classification.isStaticRoute)

          expect(exclusiveCount).toBeLessThanOrEqual(1)
          expect(classification.isFrontendRoute).toBe(!classification.isApiRoute && !classification.isStaticRoute)
        }
      ),
      { numRuns: 100 }
    )
  })

  it('builds Vue asset paths for the served js and assets roots', () => {
    fc.assert(
      fc.property(fc.constantFrom('main.abc123.js', 'vue-vendor.def456.js', 'js/app.js'), filename => {
        const path = buildVueStaticAssetPath(filename)

        expect(path).toMatch(/^\/js\//)
        expect(path.endsWith(filename.replace(/^js\//, ''))).toBe(true)
        expect(classifyAppPath(path).isStaticRoute).toBe(true)
      }),
      { numRuns: 100 }
    )

    fc.assert(
      fc.property(
        fc.constantFrom('index.abc123.css', 'logo.ghi789.png', 'assets/chunk.css'),
        filename => {
          const path = buildVueStaticAssetPath(filename)

          expect(path).toMatch(/^\/assets\//)
          expect(path.endsWith(filename.replace(/^assets\//, ''))).toBe(true)
          expect(classifyAppPath(path).isStaticRoute).toBe(true)
        }
      ),
      { numRuns: 100 }
    )
  })

  it('recognizes query-bearing app routes without treating unknown paths as known routes', () => {
    fc.assert(
      fc.property(
        fc.constantFrom('/translate', '/reader', '/insight', '/insight/character-studio'),
        fc.record({
          book: fc.option(fc.hexaString({ minLength: 8, maxLength: 8 }), { nil: undefined }),
          chapter: fc.option(fc.hexaString({ minLength: 8, maxLength: 8 }), { nil: undefined }),
        }),
        (basePath, params) => {
          const queryParts = [
            params.book ? `book=${params.book}` : '',
            params.chapter ? `chapter=${params.chapter}` : '',
          ].filter(Boolean)
          const fullPath = queryParts.length > 0 ? `${basePath}?${queryParts.join('&')}` : basePath
          const unknownPath = queryParts.length > 0 ? `${basePath}/missing?${queryParts.join('&')}` : `${basePath}/missing`

          expect(isKnownFrontendRoute(fullPath)).toBe(true)
          expect(isKnownFrontendRoute(unknownPath)).toBe(false)
        }
      ),
      { numRuns: 100 }
    )
  })

  it('normalizes leading slashes while building API paths consistently', () => {
    fc.assert(
      fc.property(
        fc.stringOf(fc.constantFrom(...'abcdefghijklmnopqrstuvwxyz0123456789_/'.split('')), {
          minLength: 1,
          maxLength: 30,
        }).filter(value => !value.startsWith('/') && !value.endsWith('/') && !value.includes('//')),
        endpoint => {
          expect(buildApiPath(endpoint)).toBe(buildApiPath(`/${endpoint}`))
          expect(normalizeAppPath(endpoint)).toBe(`/${endpoint}`)
        }
      ),
      { numRuns: 100 }
    )
  })
})
