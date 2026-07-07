import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { describe, expect, it } from 'vitest'
import router from '@/router'
import { APP_ROUTE_PATHS, FRONTEND_ROUTE_PATHS } from '@/constants/routes'
import {
  buildApiPath,
  buildStaticPath,
  buildVueStaticAssetPath,
  classifyAppPath,
  isKnownFrontendRoute,
  normalizeAppPath,
} from '@/utils/routePath'

function source(file: string): string {
  return readFileSync(resolve(process.cwd(), file), 'utf8')
}

describe('route path source contracts', () => {
  it('keeps frontend route paths in a production contract used by the router', () => {
    expect(FRONTEND_ROUTE_PATHS).toEqual([
      APP_ROUTE_PATHS.bookshelf,
      APP_ROUTE_PATHS.translate,
      APP_ROUTE_PATHS.reader,
      APP_ROUTE_PATHS.insight,
      APP_ROUTE_PATHS.characterStudio,
    ])

    const routerPaths = router
      .getRoutes()
      .filter(route => route.name && route.name !== 'pathMatch')
      .map(route => route.path)

    for (const routePath of FRONTEND_ROUTE_PATHS) {
      expect(routerPaths).toContain(routePath)
    }
  })

  it('routes app, api, and static paths through production helpers', () => {
    expect(normalizeAppPath('translate?book=book-1')).toBe('/translate?book=book-1')
    expect(isKnownFrontendRoute('/insight/character-studio?book=book-1')).toBe(true)
    expect(isKnownFrontendRoute('/missing')).toBe(false)

    expect(classifyAppPath('/api/bookshelf/books')).toEqual({
      isFrontendRoute: false,
      isApiRoute: true,
      isStaticRoute: false,
    })
    expect(classifyAppPath('/assets/index.css')).toEqual({
      isFrontendRoute: false,
      isApiRoute: false,
      isStaticRoute: true,
    })
    expect(classifyAppPath('/reader?book=book-1&chapter=chapter-1')).toEqual({
      isFrontendRoute: true,
      isApiRoute: false,
      isStaticRoute: false,
    })

    expect(buildApiPath('/bookshelf/books')).toBe('/api/bookshelf/books')
    expect(buildStaticPath('/css', '/layout.css')).toBe('/static/css/layout.css')
    expect(buildVueStaticAssetPath('main.abc123.js')).toBe('/js/main.abc123.js')
    expect(buildVueStaticAssetPath('assets/index.abc123.css')).toBe('/assets/index.abc123.css')
  })

  it('keeps route path properties free of shadow helper implementations', () => {
    const propertyFile = source('tests/property/routePath.property.ts')

    expect(propertyFile).toContain("from '@/utils/routePath'")
    expect(propertyFile).not.toMatch(/interface\s+RouteClassification/)
    expect(propertyFile).not.toMatch(/function\s+(classifyRoute|isValidFrontendRoute|buildApiPath|buildStaticPath|buildVueStaticPath)/)
    expect(propertyFile).not.toContain('export type { RouteClassification }')
    expect(propertyFile).not.toContain('export {')
    expect(propertyFile).not.toContain('/**')
  })
})
