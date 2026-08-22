import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'
import { APP_ROUTE_PATHS } from '@/constants/routes'

const routes: RouteRecordRaw[] = [
  {
    path: APP_ROUTE_PATHS.login,
    name: 'login',
    component: () => import('@/views/AuthView.vue'),
    meta: { title: '登录', guestOnly: true },
  },
  {
    path: APP_ROUTE_PATHS.register,
    name: 'register',
    component: () => import('@/views/AuthView.vue'),
    meta: { title: '注册', guestOnly: true },
  },
  {
    path: APP_ROUTE_PATHS.recover,
    name: 'recover',
    component: () => import('@/views/AuthView.vue'),
    meta: { title: '恢复账号', guestOnly: true },
  },
  {
    path: APP_ROUTE_PATHS.account,
    name: 'account',
    component: () => import('@/views/AccountView.vue'),
    meta: { title: '账户', standalone: true },
  },
  {
    path: APP_ROUTE_PATHS.admin,
    name: 'admin',
    component: () => import('@/views/AdminView.vue'),
    meta: { title: '管理后台', requiresAdmin: true, standalone: true },
  },
  {
    path: APP_ROUTE_PATHS.bookshelf,
    name: 'bookshelf',
    component: () => import('@/views/BookshelfView.vue'),
    meta: { title: '书架' }
  },
  {
    path: APP_ROUTE_PATHS.translate,
    name: 'translate',
    component: () => import('@/views/TranslateView.vue'),
    meta: { title: '翻译', publicFeature: 'translation' }
  },
  {
    path: APP_ROUTE_PATHS.reader,
    name: 'reader',
    component: () => import('@/views/ReaderView.vue'),
    meta: { title: '阅读器' },
    props: (route) => ({
      bookId: route.query.book as string,
      chapterId: route.query.chapter as string
    }),
    beforeEnter: (to, _from, next) => {
      if (!to.query.book || !to.query.chapter) {
        next({ name: 'bookshelf' })
      } else {
        next()
      }
    }
  },
  {
    path: APP_ROUTE_PATHS.insight,
    name: 'insight',
    component: () => import('@/views/InsightView.vue'),
    meta: { title: '漫画分析', publicFeature: 'insight' }
  },
  {
    path: APP_ROUTE_PATHS.characterStudio,
    name: 'character-studio',
    component: () => import('@/views/CharacterStudioView.vue'),
    meta: { title: '角色工坊', publicFeature: 'characterStudio' },
    props: (route) => ({
      bookId: route.query.book as string | undefined,
      docId: route.query.doc as string | undefined,
    })
  },
  {
    path: '/:pathMatch(.*)*',
    redirect: { name: 'bookshelf' }
  }
]

const router = createRouter({
  history: createWebHistory('/'),
  routes
})

router.beforeEach(async (to, _from, next) => {
  const { useRuntimeStore } = await import('@/stores/runtimeStore')
  const { useAuthStore } = await import('@/stores/authStore')
  const runtime = useRuntimeStore()
  const auth = useAuthStore()
  try {
    await runtime.load()
    if (runtime.capabilities?.requiresAuth) {
      await auth.restore()
      if (!auth.authenticated && !to.meta.guestOnly) {
        next({ name: 'login', query: { redirect: to.fullPath } })
        return
      }
      if (auth.authenticated && to.meta.guestOnly) {
        next({ name: 'bookshelf' })
        return
      }
      if (to.meta.requiresAdmin && !auth.isAdmin) {
        next({ name: 'bookshelf' })
        return
      }
      const publicFeature = to.meta.publicFeature
      if (
        !auth.isAdmin
        && typeof publicFeature === 'string'
        && publicFeature in runtime.capabilities.publicUserPolicy.features
        && runtime.capabilities.publicUserPolicy.features[
          publicFeature as keyof typeof runtime.capabilities.publicUserPolicy.features
        ] === false
      ) {
        next({
          name: 'bookshelf',
          query: { disabledFeature: publicFeature },
        })
        return
      }
    } else if (to.meta.guestOnly || to.meta.requiresAdmin) {
      next({ name: 'bookshelf' })
      return
    }
  } catch {
    // Let the page mount so the existing backend error UI can explain outages.
  }
  const title = to.meta.title as string | undefined
  document.title = title ? `${title} - Saber Translator` : 'Saber Translator'
  next()
})

export default router
