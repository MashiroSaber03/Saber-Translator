import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'
import router from '@/router'
import { APP_ROUTE_PATHS } from '@/constants/routes'
import { useRuntimeStore } from '@/stores/runtimeStore'

const routeCases = [
  { label: 'bookshelf', name: 'bookshelf', path: APP_ROUTE_PATHS.bookshelf },
  { label: 'translate', name: 'translate', path: APP_ROUTE_PATHS.translate },
  { label: 'reader', name: 'reader', path: APP_ROUTE_PATHS.reader },
  { label: 'insight', name: 'insight', path: APP_ROUTE_PATHS.insight },
  { label: 'character studio', name: 'character-studio', path: APP_ROUTE_PATHS.characterStudio },
] as const

describe('router config', () => {
  beforeEach(async () => {
    setActivePinia(createPinia())
    useRuntimeStore().assumeLocalForTests()
    await router.push(APP_ROUTE_PATHS.bookshelf)
  })

  it.each(routeCases)('contains the $label route', ({ name, path }) => {
    const route = router.getRoutes().find(candidate => candidate.name === name)

    expect(route).toBeDefined()
    expect(route?.path).toBe(path)
  })

  it('redirects the account-only route out of the local profile', async () => {
    await router.push(APP_ROUTE_PATHS.account)

    expect(router.currentRoute.value.name).toBe('bookshelf')
  })
})
