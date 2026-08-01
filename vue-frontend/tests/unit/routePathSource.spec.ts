import { describe, expect, it } from 'vitest'
import router from '@/router'
import { APP_ROUTE_PATHS } from '@/constants/routes'

describe('route path source contracts', () => {
  it('keeps frontend route paths in a production contract used by the router', () => {
    const routePaths = [
      APP_ROUTE_PATHS.bookshelf,
      APP_ROUTE_PATHS.translate,
      APP_ROUTE_PATHS.reader,
      APP_ROUTE_PATHS.insight,
      APP_ROUTE_PATHS.characterStudio,
    ]

    const routerPaths = router
      .getRoutes()
      .filter(route => route.name && route.name !== 'pathMatch')
      .map(route => route.path)

    for (const routePath of routePaths) {
      expect(routerPaths).toContain(routePath)
    }
  })
})
