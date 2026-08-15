import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router'
import { APP_ROUTE_PATHS } from '@/constants/routes'

const routes: RouteRecordRaw[] = [
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
    meta: { title: '翻译' }
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
    meta: { title: '漫画分析' }
  },
  {
    path: APP_ROUTE_PATHS.characterStudio,
    name: 'character-studio',
    component: () => import('@/views/CharacterStudioView.vue'),
    meta: { title: '角色工坊' },
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

router.beforeEach((to, _from, next) => {
  const title = to.meta.title as string | undefined
  document.title = title ? `${title} - Saber Translator` : 'Saber Translator'
  next()
})

export default router
