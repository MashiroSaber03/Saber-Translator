import { describe, expect, it } from 'vitest'
import router from '@/router'
import { APP_ROUTE_PATHS } from '@/constants/routes'

const routeCases = [
  { label: 'bookshelf', name: 'bookshelf', path: APP_ROUTE_PATHS.bookshelf },
  { label: 'translate', name: 'translate', path: APP_ROUTE_PATHS.translate },
  { label: 'reader', name: 'reader', path: APP_ROUTE_PATHS.reader },
  { label: 'insight', name: 'insight', path: APP_ROUTE_PATHS.insight },
  { label: 'character studio', name: 'character-studio', path: APP_ROUTE_PATHS.characterStudio },
] as const

describe('router config', () => {
  it.each(routeCases)('contains the $label route', ({ name, path }) => {
    const route = router.getRoutes().find(candidate => candidate.name === name)

    expect(route).toBeDefined()
    expect(route?.path).toBe(path)
  })
})
